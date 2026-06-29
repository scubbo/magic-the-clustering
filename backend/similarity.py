"""
Loads the FAISS index and card metadata at startup; provides similarity lookups.
"""
import json
import random
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"


class SimilarityIndex:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        import faiss

        embeddings_path = artifacts_dir / "embeddings.npy"
        index_path = artifacts_dir / "index.faiss"
        cards_path = artifacts_dir / "cards.json"

        self.embeddings = np.load(embeddings_path)
        self.index = faiss.read_index(str(index_path))
        with open(cards_path) as f:
            self.cards: list[dict] = json.load(f)

        # Build lookup maps
        self.oracle_id_to_row: dict[str, int] = {
            c["oracle_id"]: i for i, c in enumerate(self.cards) if c.get("oracle_id")
        }
        self.name_to_oracle_id: dict[str, str] = {
            c["name"].lower(): c["oracle_id"] for c in self.cards if c.get("name") and c.get("oracle_id")
        }

    def similarity(self, oracle_id_a: str, oracle_id_b: str) -> Optional[float]:
        """Cosine similarity in [0, 1] between two cards. None if either id unknown."""
        row_a = self.oracle_id_to_row.get(oracle_id_a)
        row_b = self.oracle_id_to_row.get(oracle_id_b)
        if row_a is None or row_b is None:
            return None
        # Vectors are already L2-normalized, so dot product = cosine similarity
        score = float(np.dot(self.embeddings[row_a], self.embeddings[row_b]))
        # Clamp to [0, 1] — should already be there for non-adversarial inputs
        return max(0.0, min(1.0, score))

    def rank_of(self, guess_id: str, target_id: str) -> Optional[int]:
        """
        How many cards are MORE similar to target_id than guess_id is?
        Returns 1 if guess is the most similar card (i.e. guess == target),
        or None if either id is unknown.
        """
        row_guess = self.oracle_id_to_row.get(guess_id)
        row_target = self.oracle_id_to_row.get(target_id)
        if row_guess is None or row_target is None:
            return None

        target_vec = self.embeddings[row_target].reshape(1, -1)
        guess_score = float(np.dot(self.embeddings[row_guess], self.embeddings[row_target]))

        # Query all neighbours; count how many beat the guess score
        # We use the full index (exact search) so this is O(N) — fine for ~6-18k cards
        scores, _ = self.index.search(target_vec, self.index.ntotal)
        rank = int(np.sum(scores[0] > guess_score)) + 1
        return rank

    def similar_to(self, oracle_id: str, limit: int = 10) -> list[dict]:
        """Returns the `limit` most similar cards to oracle_id, excluding itself."""
        row = self.oracle_id_to_row.get(oracle_id)
        if row is None:
            return []
        vec = self.embeddings[row].reshape(1, -1)
        scores, indices = self.index.search(vec, limit + 1)  # +1 because self is always #1
        results = []
        for score, idx in zip(scores[0], indices[0]):
            card = self.cards[idx]
            if card.get("oracle_id") == oracle_id:
                continue
            results.append({**card, "similarity_pct": round(float(score) * 100)})
            if len(results) >= limit:
                break
        return results

    def card_by_oracle_id(self, oracle_id: str) -> Optional[dict]:
        row = self.oracle_id_to_row.get(oracle_id)
        return self.cards[row] if row is not None else None

    def search_by_name(self, query: str, limit: int = 10) -> list[dict]:
        """Prefix/substring search on card names (case-insensitive)."""
        q = query.lower()
        results = [c for c in self.cards if q in (c.get("name") or "").lower()]
        return results[:limit]

    def daily_target(self, for_date: Optional[date] = None) -> dict:
        """Deterministic daily card, seeded by date. Same card for all players."""
        d = for_date or date.today()
        rng = random.Random(d.isoformat())
        return rng.choice(self.cards)


@lru_cache(maxsize=1)
def get_index() -> SimilarityIndex:
    """Singleton — loaded once per process (Vercel cold start)."""
    return SimilarityIndex()
