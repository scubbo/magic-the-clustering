"""
FastAPI application — Vercel Python serverless entry point.
Route all /api/* traffic here via vercel.json.
"""
import sys
from pathlib import Path

# Make sure the repo root is importable when running as a Vercel function
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.similarity import get_index

app = FastAPI(title="Magic The Clustering")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GuessRequest(BaseModel):
    oracle_id: str


class GuessResponse(BaseModel):
    oracle_id: str
    similarity: float       # 0.0 – 1.0
    similarity_pct: int     # 0 – 100 (for display)
    rank: int               # 1 = most similar (= correct)
    is_correct: bool


class DailyResponse(BaseModel):
    date: str
    card_count: int


class CardResponse(BaseModel):
    oracle_id: str
    name: str
    type_line: Optional[str]
    oracle_text: Optional[str]
    mana_cost: Optional[str]
    image_uri: Optional[str]
    rarity: Optional[str]
    set: Optional[str]
    set_name: Optional[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/daily", response_model=DailyResponse)
def daily_info():
    """Returns today's challenge metadata (no card spoilers)."""
    idx = get_index()
    return DailyResponse(date=date.today().isoformat(), card_count=len(idx.cards))


@app.post("/api/guess", response_model=GuessResponse)
def guess(req: GuessRequest):
    idx = get_index()
    target = idx.daily_target()
    target_id = target["oracle_id"]
    guess_id = req.oracle_id

    if not idx.card_by_oracle_id(guess_id):
        raise HTTPException(status_code=404, detail=f"Unknown oracle_id: {guess_id}")

    sim = idx.similarity(guess_id, target_id)
    rank = idx.rank_of(guess_id, target_id)

    if sim is None or rank is None:
        raise HTTPException(status_code=500, detail="Similarity computation failed")

    is_correct = guess_id == target_id
    return GuessResponse(
        oracle_id=guess_id,
        similarity=round(sim, 4),
        similarity_pct=round(sim * 100),
        rank=rank,
        is_correct=is_correct,
    )


@app.get("/api/card/{oracle_id}", response_model=CardResponse)
def get_card(oracle_id: str):
    """Returns full card info. Called only on win or surrender."""
    idx = get_index()
    card = idx.card_by_oracle_id(oracle_id)
    if not card:
        raise HTTPException(status_code=404, detail=f"Unknown oracle_id: {oracle_id}")
    return CardResponse(**{k: card.get(k) for k in CardResponse.model_fields})


@app.get("/api/cards/search")
def search_cards(q: str = Query(..., min_length=1), limit: int = Query(10, le=50)):
    """Name autocomplete for the guess input."""
    idx = get_index()
    results = idx.search_by_name(q, limit=limit)
    return [{"oracle_id": c.get("oracle_id"), "name": c.get("name")} for c in results]


@app.post("/api/surrender", response_model=CardResponse)
def surrender():
    """Returns today's target card. Called only when the player gives up."""
    idx = get_index()
    target = idx.daily_target()
    return CardResponse(**{k: target.get(k) for k in CardResponse.model_fields})
