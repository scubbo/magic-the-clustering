"""
Loads card embeddings and metadata at startup; provides similarity lookups.
Uses numpy for all similarity operations — fast enough for ~22k cards without FAISS.
"""
import json
import random
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent

_MAIN_TYPES = {
    "Creature", "Instant", "Sorcery", "Enchantment",
    "Artifact", "Land", "Planeswalker", "Battle", "Kindred",
}


def _parse_types(type_line: str) -> set:
    main_part = type_line.split("—")[0] if "—" in type_line else type_line
    return {w for w in main_part.split() if w in _MAIN_TYPES}


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0
ARTIFACTS_DIR = ROOT / "artifacts"


class SimilarityIndex:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        embeddings_path = artifacts_dir / "embeddings.npy"
        cards_path = artifacts_dir / "cards.json"

        self.embeddings = np.load(embeddings_path)
        with open(cards_path) as f:
            self.cards: list[dict] = json.load(f)

        self.oracle_id_to_row: dict[str, int] = {
            c["oracle_id"]: i for i, c in enumerate(self.cards) if c.get("oracle_id")
        }

    def _scores_against(self, target_row: int) -> np.ndarray:
        """Dot products of every card against target (= cosine sim on L2-normalised vecs)."""
        return self.embeddings @ self.embeddings[target_row]

    def similarity(self, oracle_id_a: str, oracle_id_b: str) -> Optional[float]:
        """Cosine similarity in [0, 1] between two cards. None if either id unknown."""
        row_a = self.oracle_id_to_row.get(oracle_id_a)
        row_b = self.oracle_id_to_row.get(oracle_id_b)
        if row_a is None or row_b is None:
            return None
        score = float(np.dot(self.embeddings[row_a], self.embeddings[row_b]))
        return max(0.0, min(1.0, score))

    def rank_of(self, guess_id: str, target_id: str) -> Optional[int]:
        """
        Rank of the guess by similarity to target (1 = most similar = correct answer).
        None if either id is unknown.
        """
        row_guess = self.oracle_id_to_row.get(guess_id)
        row_target = self.oracle_id_to_row.get(target_id)
        if row_guess is None or row_target is None:
            return None
        all_scores = self._scores_against(row_target)
        guess_score = all_scores[row_guess]
        return int(np.sum(all_scores > guess_score)) + 1

    def similar_to(self, oracle_id: str, limit: int = 10) -> list[dict]:
        """Returns the `limit` most similar cards to oracle_id, excluding itself."""
        row = self.oracle_id_to_row.get(oracle_id)
        if row is None:
            return []
        all_scores = self._scores_against(row)
        all_scores[row] = -1.0  # exclude self
        top_indices = np.argpartition(all_scores, -limit)[-limit:]
        top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]
        return [
            {**self.cards[i], "similarity_pct": round(float(all_scores[i]) * 100)}
            for i in top_indices
        ]

    def card_by_oracle_id(self, oracle_id: str) -> Optional[dict]:
        row = self.oracle_id_to_row.get(oracle_id)
        return self.cards[row] if row is not None else None

    def search_by_name(self, query: str, limit: int = 10) -> list[dict]:
        """Substring search on card names (case-insensitive)."""
        q = query.lower()
        results = [c for c in self.cards if q in (c.get("name") or "").lower()]
        return results[:limit]

    def daily_target(self, for_date: Optional[date] = None) -> dict:
        """Deterministic daily card, seeded by date. Same card for all players."""
        d = for_date or date.today()
        rng = random.Random(d.isoformat())
        return rng.choice(self.cards)

    def random_card(self) -> dict:
        """Non-deterministic random card for practice mode."""
        return random.choice(self.cards)

    def feature_hints(self, guess_id: str, target_id: str) -> list[dict]:
        """
        Returns the two most-similar named feature groups between guess and target.
        Computed from card metadata; does not use embeddings.
        """
        guess = self.card_by_oracle_id(guess_id)
        target = self.card_by_oracle_id(target_id)
        if not guess or not target:
            return []

        scores: dict[str, float] = {}

        g_colors = set(guess.get("colors") or [])
        t_colors = set(target.get("colors") or [])
        scores["Colors"] = _jaccard(g_colors, t_colors)

        g_types = _parse_types(guess.get("type_line") or "")
        t_types = _parse_types(target.get("type_line") or "")
        scores["Card type"] = _jaccard(g_types, t_types)

        g_cmc = float(guess.get("cmc") or 0)
        t_cmc = float(target.get("cmc") or 0)
        max_cmc = max(g_cmc, t_cmc, 1.0)
        scores["Mana value"] = 1.0 - abs(g_cmc - t_cmc) / max_cmc

        g_kw = set(guess.get("keywords") or [])
        t_kw = set(target.get("keywords") or [])
        if g_kw or t_kw:
            scores["Keywords"] = _jaccard(g_kw, t_kw)

        # Sort descending by score, then alphabetically for deterministic tie-breaking
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [{"feature": k, "similarity_pct": round(v * 100)} for k, v in ranked[:2]]


@lru_cache(maxsize=1)
def get_index() -> SimilarityIndex:
    """Singleton — loaded once per process (Vercel cold start)."""
    return SimilarityIndex()
