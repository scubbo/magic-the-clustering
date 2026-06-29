import numpy as np
import pytest
from vectorize.features import (
    CARD_TYPES,
    COLORS,
    SET_TYPES,
    _parse_pip_counts,
    _parse_power_or_toughness,
    _set_release_year_normalized,
    build_text_input,
    extract,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIGHTNING_BOLT = {
    "name": "Lightning Bolt",
    "mana_cost": "{R}",
    "cmc": 1.0,
    "colors": ["R"],
    "color_identity": ["R"],
    "type_line": "Instant",
    "oracle_text": "Lightning Bolt deals 3 damage to any target.",
    "power": None,
    "toughness": None,
    "loyalty": None,
    "rarity": "common",
    "released_at": "1993-08-05",
    "set_type": "core",
}

GRIZZLY_BEARS = {
    "name": "Grizzly Bears",
    "mana_cost": "{1}{G}",
    "cmc": 2.0,
    "colors": ["G"],
    "color_identity": ["G"],
    "type_line": "Creature — Bear",
    "oracle_text": "",
    "power": "2",
    "toughness": "2",
    "loyalty": None,
    "rarity": "common",
    "released_at": "1993-08-05",
    "set_type": "core",
}

JACE_MIND_SCULPTOR = {
    "name": "Jace, the Mind Sculptor",
    "mana_cost": "{2}{U}{U}",
    "cmc": 4.0,
    "colors": ["U"],
    "color_identity": ["U"],
    "type_line": "Legendary Planeswalker — Jace",
    "oracle_text": "+2: Look at the top card...",
    "power": None,
    "toughness": None,
    "loyalty": "3",
    "rarity": "mythic",
    "released_at": "2010-02-05",
    "set_type": "expansion",
}

BALEFUL_STRIX = {
    "name": "Baleful Strix",
    "mana_cost": "{U}{B}",
    "cmc": 2.0,
    "colors": ["U", "B"],
    "color_identity": ["U", "B"],
    "type_line": "Artifact Creature — Bird",
    "oracle_text": "Flying, deathtouch\nWhen Baleful Strix enters the battlefield, draw a card.",
    "power": "1",
    "toughness": "1",
    "loyalty": None,
    "rarity": "rare",
    "released_at": "2012-06-01",
    "set_type": "masters",
}


# ---------------------------------------------------------------------------
# _parse_pip_counts
# ---------------------------------------------------------------------------

def test_pip_counts_single_red():
    counts = _parse_pip_counts("{R}")
    assert counts["R"] == 1.0
    assert counts["U"] == 0.0

def test_pip_counts_multicolor():
    counts = _parse_pip_counts("{U}{U}{B}{B}")
    assert counts["U"] == 2.0
    assert counts["B"] == 2.0
    assert counts["R"] == 0.0

def test_pip_counts_with_generic():
    counts = _parse_pip_counts("{2}{U}{U}")
    # generic cost is not tracked as pips
    assert counts["U"] == 2.0
    assert counts["C"] == 0.0

def test_pip_counts_colorless_pip():
    counts = _parse_pip_counts("{C}{C}{G}")
    assert counts["C"] == 2.0
    assert counts["G"] == 1.0

def test_pip_counts_empty():
    counts = _parse_pip_counts("")
    assert all(v == 0.0 for v in counts.values())

def test_pip_counts_none():
    counts = _parse_pip_counts(None)
    assert all(v == 0.0 for v in counts.values())


# ---------------------------------------------------------------------------
# _parse_power_or_toughness
# ---------------------------------------------------------------------------

def test_power_numeric():
    assert _parse_power_or_toughness("3") == 3.0

def test_power_star():
    assert _parse_power_or_toughness("*") == -1.0

def test_power_star_plus():
    assert _parse_power_or_toughness("1+*") == -1.0

def test_power_none():
    assert _parse_power_or_toughness(None) == 0.0


# ---------------------------------------------------------------------------
# _set_release_year_normalized
# ---------------------------------------------------------------------------

def test_release_year_alpha():
    # Alpha is year 0 → 0.0
    assert _set_release_year_normalized("1993-08-05") == pytest.approx(0.0, abs=0.01)

def test_release_year_later():
    # 2013 is 20 years in, span is 40 → ~0.5
    val = _set_release_year_normalized("2013-08-05")
    assert 0.45 < val < 0.55

def test_release_year_none():
    assert _set_release_year_normalized(None) == 0.0

def test_release_year_clamp():
    # Far future should clamp to 1.0
    assert _set_release_year_normalized("2100-01-01") == 1.0


# ---------------------------------------------------------------------------
# extract — vector shape and dtype
# ---------------------------------------------------------------------------

def test_extract_shape():
    vec = extract(LIGHTNING_BOLT)
    assert vec.shape == (38,)

def test_extract_dtype():
    vec = extract(LIGHTNING_BOLT)
    assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# extract — Lightning Bolt spot checks
# ---------------------------------------------------------------------------

def test_lightning_bolt_cmc():
    vec = extract(LIGHTNING_BOLT)
    assert vec[0] == pytest.approx(1.0 / 20.0)

def test_lightning_bolt_red_pip():
    vec = extract(LIGHTNING_BOLT)
    # pip counts: W=1, U=2, B=3, R=4, G=5, C=6  (1-indexed from 1 in the vector)
    r_idx = 1 + COLORS.index("R")  # index 4
    assert vec[r_idx] == 1.0
    assert vec[1] == 0.0  # W
    assert vec[2] == 0.0  # U

def test_lightning_bolt_color_red():
    vec = extract(LIGHTNING_BOLT)
    r_idx = 7 + COLORS.index("R")
    assert vec[r_idx] == 1.0
    assert vec[7] == 0.0  # W

def test_lightning_bolt_instant_type():
    vec = extract(LIGHTNING_BOLT)
    instant_idx = 17 + CARD_TYPES.index("Instant")
    assert vec[instant_idx] == 1.0
    assert vec[17 + CARD_TYPES.index("Creature")] == 0.0

def test_lightning_bolt_no_power():
    vec = extract(LIGHTNING_BOLT)
    assert vec[25] == 0.0
    assert vec[26] == 0.0

def test_lightning_bolt_rarity_common():
    vec = extract(LIGHTNING_BOLT)
    assert vec[28] == pytest.approx(0.0)

def test_lightning_bolt_set_type_core():
    vec = extract(LIGHTNING_BOLT)
    core_idx = 30 + SET_TYPES.index("core")
    assert vec[core_idx] == 1.0


# ---------------------------------------------------------------------------
# extract — Grizzly Bears spot checks
# ---------------------------------------------------------------------------

def test_grizzly_bears_power_toughness():
    vec = extract(GRIZZLY_BEARS)
    assert vec[25] == 2.0
    assert vec[26] == 2.0

def test_grizzly_bears_creature_type():
    vec = extract(GRIZZLY_BEARS)
    assert vec[17 + CARD_TYPES.index("Creature")] == 1.0

def test_grizzly_bears_green_pip():
    vec = extract(GRIZZLY_BEARS)
    g_idx = 1 + COLORS.index("G")
    assert vec[g_idx] == 1.0

def test_grizzly_bears_cmc():
    vec = extract(GRIZZLY_BEARS)
    assert vec[0] == pytest.approx(2.0 / 20.0)


# ---------------------------------------------------------------------------
# extract — Jace the Mind Sculptor spot checks
# ---------------------------------------------------------------------------

def test_jace_loyalty():
    vec = extract(JACE_MIND_SCULPTOR)
    assert vec[27] == 3.0

def test_jace_mythic_rarity():
    vec = extract(JACE_MIND_SCULPTOR)
    assert vec[28] == pytest.approx(1.0)

def test_jace_two_blue_pips():
    vec = extract(JACE_MIND_SCULPTOR)
    u_idx = 1 + COLORS.index("U")
    assert vec[u_idx] == 2.0

def test_jace_planeswalker_type():
    vec = extract(JACE_MIND_SCULPTOR)
    assert vec[17 + CARD_TYPES.index("Planeswalker")] == 1.0

def test_jace_set_type_expansion():
    vec = extract(JACE_MIND_SCULPTOR)
    exp_idx = 30 + SET_TYPES.index("expansion")
    assert vec[exp_idx] == 1.0


# ---------------------------------------------------------------------------
# extract — Baleful Strix (multicolor artifact creature)
# ---------------------------------------------------------------------------

def test_baleful_strix_two_colors():
    vec = extract(BALEFUL_STRIX)
    u_idx = 7 + COLORS.index("U")
    b_idx = 7 + COLORS.index("B")
    assert vec[u_idx] == 1.0
    assert vec[b_idx] == 1.0
    r_idx = 7 + COLORS.index("R")
    assert vec[r_idx] == 0.0

def test_baleful_strix_artifact_and_creature():
    vec = extract(BALEFUL_STRIX)
    assert vec[17 + CARD_TYPES.index("Artifact")] == 1.0
    assert vec[17 + CARD_TYPES.index("Creature")] == 1.0

def test_baleful_strix_rare():
    vec = extract(BALEFUL_STRIX)
    assert vec[28] == pytest.approx(2 / 3)

def test_baleful_strix_masters_set():
    vec = extract(BALEFUL_STRIX)
    masters_idx = 30 + SET_TYPES.index("masters")
    assert vec[masters_idx] == 1.0


# ---------------------------------------------------------------------------
# build_text_input
# ---------------------------------------------------------------------------

def test_build_text_input_combines_parts():
    text = build_text_input(LIGHTNING_BOLT)
    assert "Lightning Bolt" in text
    assert "Instant" in text
    assert "3 damage" in text

def test_build_text_input_skips_empty_oracle():
    text = build_text_input(GRIZZLY_BEARS)
    assert "Grizzly Bears" in text
    # No trailing ". " from empty oracle text
    assert not text.endswith(". ")

def test_build_text_input_missing_fields():
    text = build_text_input({"name": "Test Card"})
    assert text == "Test Card"
