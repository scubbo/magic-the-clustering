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
from fastapi.staticfiles import StaticFiles
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
    target_id: Optional[str] = None  # practice mode: explicit target instead of daily


class GuessResponse(BaseModel):
    oracle_id: str
    similarity: float       # 0.0 – 1.0
    similarity_pct: int     # 0 – 100 (for display)
    rank: int               # 1 = most similar (= correct)
    is_correct: bool
    top_features: list[dict]  # two most-similar named feature groups


class SurrenderRequest(BaseModel):
    target_id: Optional[str] = None  # practice mode: explicit target instead of daily


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

@app.get("/api/debug")
def debug():
    task_dir = Path(__file__).parent.parent
    try:
        top_level = sorted(str(p.relative_to(task_dir)) for p in task_dir.iterdir())
        api_dir = sorted(str(p.relative_to(task_dir)) for p in (task_dir / "api").iterdir()) if (task_dir / "api").exists() else []
    except Exception as e:
        top_level = [f"ERROR: {e}"]
        api_dir = []
    return {
        "file": str(Path(__file__)),
        "task_dir": str(task_dir),
        "public_exists": (task_dir / "public").exists(),
        "top_level": top_level,
        "api_dir": api_dir,
    }


@app.get("/api/daily", response_model=DailyResponse)
def daily_info():
    """Returns today's challenge metadata (no card spoilers)."""
    idx = get_index()
    return DailyResponse(date=date.today().isoformat(), card_count=len(idx.cards))


@app.post("/api/guess", response_model=GuessResponse)
def guess(req: GuessRequest):
    idx = get_index()

    if req.target_id:
        target = idx.card_by_oracle_id(req.target_id)
        if not target:
            raise HTTPException(status_code=404, detail=f"Unknown target_id: {req.target_id}")
    else:
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
        top_features=idx.feature_hints(guess_id, target_id),
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


@app.get("/api/similar")
def similar_to_target(limit: int = Query(10, le=20), target_id: Optional[str] = Query(None)):
    """Returns the most similar cards to the target, excluding the target itself."""
    idx = get_index()
    if target_id:
        target = idx.card_by_oracle_id(target_id)
        if not target:
            raise HTTPException(status_code=404, detail=f"Unknown target_id: {target_id}")
    else:
        target = idx.daily_target()
    return idx.similar_to(target["oracle_id"], limit=limit)


@app.get("/api/practice/new")
def practice_new():
    """Returns a randomly selected card oracle_id for use as a practice target."""
    idx = get_index()
    card = idx.random_card()
    return {"oracle_id": card["oracle_id"]}


@app.post("/api/surrender", response_model=CardResponse)
def surrender(req: SurrenderRequest = SurrenderRequest()):
    """Returns the target card. Called only when the player gives up."""
    idx = get_index()
    if req.target_id:
        target = idx.card_by_oracle_id(req.target_id)
        if not target:
            raise HTTPException(status_code=404, detail=f"Unknown target_id: {req.target_id}")
    else:
        target = idx.daily_target()
    return CardResponse(**{k: target.get(k) for k in CardResponse.model_fields})


_static = Path(__file__).parent / "static"
if _static.exists():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
