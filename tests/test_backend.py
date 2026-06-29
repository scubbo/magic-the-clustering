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
