"""
Extracts a structured feature vector from a Scryfall card object.
The text embedding (oracle text, type line, name) is handled separately in embed.py.
These structured features cover information not reliably encoded in text:
mana cost shape, color identity, card type, P/T, loyalty, rarity, and set era.
"""
import re
from datetime import date
from typing import Optional

import numpy as np

COLORS = ["W", "U", "B", "R", "G"]
CARD_TYPES = ["Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Land", "Planeswalker", "Battle"]
SET_TYPES = ["core", "expansion", "masters", "commander", "draft_innovation", "spellbook", "starter", "memorabilia"]

# Magic's first release; used to normalize set release year to [0, 1]
_MAGIC_EPOCH = date(1993, 8, 5)
_NORMALIZATION_SPAN_YEARS = 40.0  # generous ceiling so new sets don't exceed 1.0


def _parse_pip_counts(mana_cost: str) -> dict[str, float]:
    """Count colored and colorless pips in a mana cost string like '{3}{U}{U}'."""
    counts: dict[str, float] = {c: 0.0 for c in COLORS}
    counts["C"] = 0.0
    if not mana_cost:
        return counts
    for token in re.findall(r"\{([^}]+)\}", mana_cost):
        if token in COLORS:
            counts[token] += 1.0
        elif token == "C":
            counts["C"] += 1.0
        # generic mana (numbers, X, Y, etc.) is already captured by CMC
    return counts


def _parse_power_or_toughness(value: Optional[str]) -> float:
    """Convert P/T string to float. '*' → -1.0, missing → 0.0."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return -1.0  # handles "*", "1+*", "∞", etc.


def _set_release_year_normalized(released_at: Optional[str]) -> float:
    """Normalize a 'YYYY-MM-DD' release date to [0, 1] relative to Magic's launch."""
    if not released_at:
        return 0.0
    try:
        release = date.fromisoformat(released_at)
        days_since_epoch = (release - _MAGIC_EPOCH).days
        years = days_since_epoch / 365.25
        return min(max(years / _NORMALIZATION_SPAN_YEARS, 0.0), 1.0)
    except ValueError:
        return 0.0


def extract(card: dict) -> np.ndarray:
    """
    Returns a float32 feature vector for one Scryfall card object.
    Dimension layout (38 total):
      [0]     CMC normalized
      [1-6]   mana pip counts (W U B R G C)
      [7-11]  colors multi-hot (W U B R G)
      [12-16] color identity multi-hot (W U B R G)
      [17-24] card type multi-hot (8 types)
      [25]    power
      [26]    toughness
      [27]    loyalty
      [28]    rarity ordinal normalized
      [29]    set release year normalized
      [30-37] set type multi-hot (8 types)
    """
    vec = np.zeros(38, dtype=np.float32)

    # CMC
    vec[0] = min(float(card.get("cmc") or 0.0), 20.0) / 20.0

    # Mana pip counts
    pip_counts = _parse_pip_counts(card.get("mana_cost") or "")
    for i, c in enumerate(COLORS):
        vec[1 + i] = pip_counts[c]
    vec[6] = pip_counts["C"]

    # Colors multi-hot
    colors = set(card.get("colors") or [])
    for i, c in enumerate(COLORS):
        vec[7 + i] = 1.0 if c in colors else 0.0

    # Color identity multi-hot
    identity = set(card.get("color_identity") or [])
    for i, c in enumerate(COLORS):
        vec[12 + i] = 1.0 if c in identity else 0.0

    # Card types multi-hot (parsed from type_line)
    type_line = card.get("type_line") or ""
    for i, t in enumerate(CARD_TYPES):
        vec[17 + i] = 1.0 if t in type_line else 0.0

    # Power, toughness, loyalty
    vec[25] = _parse_power_or_toughness(card.get("power"))
    vec[26] = _parse_power_or_toughness(card.get("toughness"))
    loyalty = card.get("loyalty")
    vec[27] = float(loyalty) if loyalty and loyalty.isdigit() else 0.0

    # Rarity ordinal
    rarity_map = {"common": 0.0, "uncommon": 1 / 3, "rare": 2 / 3, "mythic": 1.0}
    vec[28] = rarity_map.get(card.get("rarity") or "", 0.0)

    # Set release year
    vec[29] = _set_release_year_normalized(card.get("released_at"))

    # Set type multi-hot
    set_type = card.get("set_type") or ""
    for i, st in enumerate(SET_TYPES):
        vec[30 + i] = 1.0 if set_type == st else 0.0

    return vec


def build_text_input(card: dict) -> str:
    """Construct the text string fed to the sentence-transformer for this card."""
    parts = [card.get("name") or "", card.get("type_line") or "", card.get("oracle_text") or ""]
    return ". ".join(p for p in parts if p).strip()
