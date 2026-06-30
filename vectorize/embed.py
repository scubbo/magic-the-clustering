"""
Builds combined (text + structured) embeddings for all cards and saves artifacts.

Usage:
    python -m vectorize.embed [--legality modern|commander|all] [--cards data/cards.json]

Outputs (to artifacts/):
    embeddings.npy     float32 array of shape (N, D)
    cards.json         card metadata in the same row order as embeddings.npy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from vectorize.features import extract, build_text_input

ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"

# Cards excluded from the game regardless of legality filter.
_EXCLUDED_LAYOUTS = {"art_series", "token", "double_faced_token", "emblem"}
_EXCLUDED_SET_TYPES = {"funny"}  # silver-bordered / acorn


def _is_basic_land(card: dict) -> bool:
    return "Basic" in (card.get("supertypes") or []) or "Basic Land" in (card.get("type_line") or "")


def _is_english(card: dict) -> bool:
    return (card.get("lang") or "en") == "en"


def _passes_legality(card: dict, legality: str) -> bool:
    if legality == "all":
        return True
    legalities = card.get("legalities") or {}
    return legalities.get(legality) == "legal"


def load_cards(cards_path: Path, legality: str) -> list[dict]:
    print(f"Loading cards from {cards_path} ...", flush=True)
    with open(cards_path) as f:
        all_cards = json.load(f)

    kept = []
    for card in all_cards:
        if not _is_english(card):
            continue
        if card.get("layout") in _EXCLUDED_LAYOUTS:
            continue
        if card.get("set_type") in _EXCLUDED_SET_TYPES:
            continue
        if _is_basic_land(card):
            continue
        if not _passes_legality(card, legality):
            continue
        kept.append(card)

    print(f"  {len(all_cards)} total → {len(kept)} after filtering (legality={legality})", flush=True)
    return kept


def build_embeddings(cards: list[dict], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)

    print(f"Loading sentence-transformer model '{model_name}' ...", flush=True)
    model = SentenceTransformer(model_name)

    texts = [build_text_input(c) for c in cards]
    print(f"Encoding {len(texts)} cards ...", flush=True)
    text_vecs = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    # text_vecs shape: (N, 768)

    print("Extracting structured features ...", flush=True)
    struct_vecs = np.array([extract(c) for c in cards], dtype=np.float32)
    # struct_vecs shape: (N, 38)

    # Combine: 75% text, 25% structured, then L2-normalize
    combined = np.concatenate([text_vecs * 0.75, struct_vecs * 0.25], axis=1).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid divide-by-zero for degenerate cards
    combined = combined / norms

    return combined


def save_artifacts(cards: list[dict], embeddings: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings to {emb_path}  shape={embeddings.shape}", flush=True)

    # Minimal card metadata for the frontend and backend
    metadata = [
        {
            "oracle_id": c.get("oracle_id"),
            "name": c.get("name"),
            "type_line": c.get("type_line"),
            "oracle_text": c.get("oracle_text"),
            "mana_cost": c.get("mana_cost"),
            "image_uri": (c.get("image_uris") or {}).get("normal"),
            "rarity": c.get("rarity"),
            "set": c.get("set"),
            "set_name": c.get("set_name"),
            "colors": c.get("colors") or [],
            "color_identity": c.get("color_identity") or [],
            "cmc": c.get("cmc") or 0.0,
            "keywords": c.get("keywords") or [],
        }
        for c in cards
    ]
    cards_path = out_dir / "cards.json"
    with open(cards_path, "w") as f:
        json.dump(metadata, f, separators=(",", ":"))
    print(f"Saved card metadata to {cards_path}  ({len(metadata)} cards)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MtG card embeddings")
    parser.add_argument("--legality", default="modern", choices=["modern", "commander", "all"],
                        help="Filter to cards legal in this format (default: modern)")
    parser.add_argument("--cards", default=str(DATA_DIR / "cards.json"),
                        help="Path to Scryfall oracle_cards JSON")
    parser.add_argument("--model", default="all-mpnet-base-v2",
                        help="Sentence-transformer model name")
    parser.add_argument("--out", default=str(ARTIFACTS_DIR),
                        help="Directory to write artifacts to")
    args = parser.parse_args()

    cards = load_cards(Path(args.cards), args.legality)
    embeddings = build_embeddings(cards, args.model)
    save_artifacts(cards, embeddings, Path(args.out))
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
