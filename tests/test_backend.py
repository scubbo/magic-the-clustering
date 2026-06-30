"""
Tests for the FastAPI backend using TestClient with a mock SimilarityIndex.
These tests do not require the FAISS artifacts to exist.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Patch get_index before importing the app so the real index is never loaded
MOCK_CARDS = [
    {
        "oracle_id": "aaa",
        "name": "Lightning Bolt",
        "type_line": "Instant",
        "oracle_text": "Deal 3 damage.",
        "mana_cost": "{R}",
        "image_uri": "https://example.com/bolt.jpg",
        "rarity": "common",
        "set": "lea",
        "set_name": "Limited Edition Alpha",
        "colors": ["R"],
        "color_identity": ["R"],
        "cmc": 1.0,
        "keywords": [],
    },
    {
        "oracle_id": "bbb",
        "name": "Lightning Strike",
        "type_line": "Instant",
        "oracle_text": "Deal 3 damage.",
        "mana_cost": "{1}{R}",
        "image_uri": None,
        "rarity": "common",
        "set": "m13",
        "set_name": "Magic 2013",
        "colors": ["R"],
        "color_identity": ["R"],
        "cmc": 2.0,
        "keywords": [],
    },
    {
        "oracle_id": "ccc",
        "name": "Forest",
        "type_line": "Basic Land — Forest",
        "oracle_text": "{T}: Add {G}.",
        "mana_cost": None,
        "image_uri": None,
        "rarity": "common",
        "set": "lea",
        "set_name": "Limited Edition Alpha",
        "colors": [],
        "color_identity": ["G"],
        "cmc": 0.0,
        "keywords": [],
    },
]


def _make_mock_similar(cards, exclude_id, limit):
    return [
        {**c, "similarity_pct": 80}
        for c in cards
        if c["oracle_id"] != exclude_id
    ][:limit]


def _make_mock_index(daily_target_id: str = "aaa"):
    mock = MagicMock()
    mock.cards = MOCK_CARDS
    mock.oracle_id_to_row = {c["oracle_id"]: i for i, c in enumerate(MOCK_CARDS)}

    dim = 10
    mock.embeddings = np.eye(len(MOCK_CARDS), dim, dtype=np.float32)

    def _card_by_oracle_id(oid):
        return next((c for c in MOCK_CARDS if c["oracle_id"] == oid), None)

    mock.card_by_oracle_id.side_effect = _card_by_oracle_id

    def _similarity(a, b):
        if a == b:
            return 1.0
        return 0.5

    mock.similarity.side_effect = _similarity

    def _rank_of(guess_id, target_id):
        if guess_id == target_id:
            return 1
        return 2

    mock.rank_of.side_effect = _rank_of

    def _search(q, limit=10):
        return [c for c in MOCK_CARDS if q.lower() in c["name"].lower()][:limit]

    mock.search_by_name.side_effect = _search

    mock.similar_to.side_effect = lambda oid, limit=10: _make_mock_similar(MOCK_CARDS, oid, limit)

    mock.feature_hints.return_value = [
        {"feature": "Card type", "similarity_pct": 100},
        {"feature": "Colors", "similarity_pct": 100},
        {"feature": "Oracle text", "similarity_pct": 72},
    ]

    mock.random_card.return_value = MOCK_CARDS[1]

    target_card = _card_by_oracle_id(daily_target_id)
    mock.daily_target.return_value = target_card

    return mock


@pytest.fixture()
def client():
    mock_idx = _make_mock_index(daily_target_id="aaa")
    with patch("api.index.get_index", return_value=mock_idx):
        from api.index import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# /api/daily
# ---------------------------------------------------------------------------

def test_daily_returns_date_and_count(client):
    resp = client.get("/api/daily")
    assert resp.status_code == 200
    data = resp.json()
    assert "date" in data
    assert data["card_count"] == len(MOCK_CARDS)


# ---------------------------------------------------------------------------
# /api/guess
# ---------------------------------------------------------------------------

def test_guess_correct(client):
    resp = client.post("/api/guess", json={"oracle_id": "aaa"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is True
    assert data["rank"] == 1
    assert data["similarity_pct"] == 100


def test_guess_wrong(client):
    resp = client.post("/api/guess", json={"oracle_id": "bbb"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is False
    assert data["similarity_pct"] == 50


def test_guess_unknown_id(client):
    resp = client.post("/api/guess", json={"oracle_id": "zzz"})
    assert resp.status_code == 404


def test_guess_returns_top_features(client):
    resp = client.post("/api/guess", json={"oracle_id": "bbb"})
    assert resp.status_code == 200
    data = resp.json()
    assert "top_features" in data
    assert len(data["top_features"]) > 0
    for hint in data["top_features"]:
        assert "feature" in hint
        assert "similarity_pct" in hint


def test_guess_with_practice_target(client):
    resp = client.post("/api/guess", json={"oracle_id": "bbb", "target_id": "bbb"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is True
    assert data["oracle_id"] == "bbb"


def test_guess_with_practice_target_wrong(client):
    resp = client.post("/api/guess", json={"oracle_id": "aaa", "target_id": "ccc"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is False


def test_guess_with_unknown_practice_target(client):
    resp = client.post("/api/guess", json={"oracle_id": "aaa", "target_id": "zzz"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/card/{oracle_id}
# ---------------------------------------------------------------------------

def test_get_card_known(client):
    resp = client.get("/api/card/aaa")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Lightning Bolt"
    assert data["oracle_id"] == "aaa"


def test_get_card_unknown(client):
    resp = client.get("/api/card/zzz")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/cards/search
# ---------------------------------------------------------------------------

def test_search_returns_matches(client):
    resp = client.get("/api/cards/search?q=lightning")
    assert resp.status_code == 200
    names = [r["name"] for r in resp.json()]
    assert "Lightning Bolt" in names
    assert "Lightning Strike" in names
    assert "Forest" not in names


def test_search_requires_query(client):
    resp = client.get("/api/cards/search")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/similar
# ---------------------------------------------------------------------------

def test_similar_excludes_target(client):
    resp = client.get("/api/similar")
    assert resp.status_code == 200
    ids = [r["oracle_id"] for r in resp.json()]
    assert "aaa" not in ids  # target itself excluded

def test_similar_returns_other_cards(client):
    resp = client.get("/api/similar")
    assert len(resp.json()) > 0

def test_similar_limit(client):
    resp = client.get("/api/similar?limit=1")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

def test_similar_includes_similarity_pct(client):
    resp = client.get("/api/similar")
    for card in resp.json():
        assert "similarity_pct" in card


# ---------------------------------------------------------------------------
# /api/surrender
# ---------------------------------------------------------------------------

def test_surrender_returns_target(client):
    resp = client.post("/api/surrender")
    assert resp.status_code == 200
    data = resp.json()
    assert data["oracle_id"] == "aaa"
    assert data["name"] == "Lightning Bolt"


def test_surrender_with_practice_target(client):
    resp = client.post("/api/surrender", json={"target_id": "bbb"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["oracle_id"] == "bbb"
    assert data["name"] == "Lightning Strike"


def test_surrender_with_unknown_practice_target(client):
    resp = client.post("/api/surrender", json={"target_id": "zzz"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/practice/new
# ---------------------------------------------------------------------------

def test_practice_new_returns_oracle_id(client):
    resp = client.get("/api/practice/new")
    assert resp.status_code == 200
    data = resp.json()
    assert "oracle_id" in data
    assert data["oracle_id"] in [c["oracle_id"] for c in MOCK_CARDS]


# ---------------------------------------------------------------------------
# feature_hints unit tests (directly on SimilarityIndex, no API layer)
# ---------------------------------------------------------------------------

FEATURE_HINT_CARDS = [
    {
        "oracle_id": "bolt",
        "name": "Lightning Bolt",
        "type_line": "Instant",
        "colors": ["R"],
        "color_identity": ["R"],
        "cmc": 1.0,
        "keywords": [],
    },
    {
        "oracle_id": "strike",
        "name": "Lightning Strike",
        "type_line": "Instant",
        "colors": ["R"],
        "color_identity": ["R"],
        "cmc": 2.0,
        "keywords": [],
    },
    {
        "oracle_id": "birds",
        "name": "Birds of Paradise",
        "type_line": "Creature — Bird",
        "colors": ["G"],
        "color_identity": ["G"],
        "cmc": 1.0,
        "keywords": ["Flying"],
    },
    {
        "oracle_id": "archangel",
        "name": "Archangel of Thune",
        "type_line": "Legendary Creature — Angel",
        "colors": ["W"],
        "color_identity": ["W"],
        "cmc": 5.0,
        "keywords": ["Flying", "Lifelink"],
    },
    {
        "oracle_id": "sol_ring",
        "name": "Sol Ring",
        "type_line": "Artifact",
        "colors": [],
        "color_identity": [],
        "cmc": 1.0,
        "keywords": [],
    },
]


@pytest.fixture()
def hint_idx():
    """Minimal SimilarityIndex for feature_hints — uses small identity embeddings."""
    from backend.similarity import SimilarityIndex
    instance = SimilarityIndex.__new__(SimilarityIndex)
    instance.cards = FEATURE_HINT_CARDS
    instance.oracle_id_to_row = {c["oracle_id"]: i for i, c in enumerate(FEATURE_HINT_CARDS)}
    # Identity matrix: each card has orthogonal text vectors → text sim = 0 between different cards
    instance.embeddings = np.eye(len(FEATURE_HINT_CARDS), 10, dtype=np.float32)
    return instance


def test_hints_returns_list(hint_idx):
    hints = hint_idx.feature_hints("bolt", "strike")
    assert isinstance(hints, list)


def test_hints_oracle_text_always_present(hint_idx):
    hints = hint_idx.feature_hints("bolt", "strike")
    names = [h["feature"] for h in hints]
    assert "Oracle text" in names


def test_hints_oracle_text_not_duplicated_when_in_top_two(hint_idx):
    # When oracle text would naturally be in the top 2, it should appear exactly once
    hints = hint_idx.feature_hints("bolt", "bolt")  # self-comparison: all features 100%
    names = [h["feature"] for h in hints]
    assert names.count("Oracle text") == 1


def test_hints_each_has_feature_and_pct(hint_idx):
    hints = hint_idx.feature_hints("bolt", "strike")
    for h in hints:
        assert "feature" in h
        assert "similarity_pct" in h
        assert 0 <= h["similarity_pct"] <= 100


def test_hints_sorted_descending(hint_idx):
    hints = hint_idx.feature_hints("bolt", "archangel")
    pcts = [h["similarity_pct"] for h in hints]
    assert pcts == sorted(pcts, reverse=True)


def test_hints_same_type_included(hint_idx):
    # Both Lightning Bolt and Lightning Strike are Instants
    hints = hint_idx.feature_hints("bolt", "strike")
    names = [h["feature"] for h in hints]
    assert "Card type" in names


def test_hints_same_color_included(hint_idx):
    # Both are Red
    hints = hint_idx.feature_hints("bolt", "strike")
    names = [h["feature"] for h in hints]
    assert "Colors" in names


def test_hints_keywords_included_when_present(hint_idx):
    # Birds has [Flying], Archangel has [Flying, Lifelink] — keyword overlap exists
    hints = hint_idx.feature_hints("birds", "archangel")
    names = [h["feature"] for h in hints]
    assert "Keywords" in names


def test_hints_keywords_omitted_when_both_empty(hint_idx):
    # Lightning Bolt and Strike both have no keywords
    hints = hint_idx.feature_hints("bolt", "strike")
    names = [h["feature"] for h in hints]
    assert "Keywords" not in names


def test_hints_returns_empty_for_unknown_card(hint_idx):
    hints = hint_idx.feature_hints("bolt", "unknown")
    assert hints == []


def test_hints_both_colorless_color_similarity_is_100(hint_idx):
    # Sol Ring is colorless; two colorless cards should have 100% color similarity
    # Add a second colorless card temporarily
    from backend.similarity import SimilarityIndex
    instance = SimilarityIndex.__new__(SimilarityIndex)
    cards = [
        {"oracle_id": "sol", "name": "Sol Ring", "type_line": "Artifact",
         "colors": [], "color_identity": [], "cmc": 1.0, "keywords": []},
        {"oracle_id": "vault", "name": "Black Lotus", "type_line": "Artifact",
         "colors": [], "color_identity": [], "cmc": 0.0, "keywords": []},
    ]
    instance.cards = cards
    instance.oracle_id_to_row = {c["oracle_id"]: i for i, c in enumerate(cards)}
    instance.embeddings = np.eye(len(cards), 10, dtype=np.float32)
    hints = instance.feature_hints("sol", "vault")
    color_hint = next((h for h in hints if h["feature"] == "Colors"), None)
    if color_hint:
        assert color_hint["similarity_pct"] == 100
