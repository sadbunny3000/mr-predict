import pickle
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.tennis_alert_engine import run_tennis_alert_engine
from tennis.ingestion.sackmann_ingestion import SackmannIngestion
from tennis.features.rebuild import run_full_feature_rebuild

router = APIRouter()

try:
    with open('/app/ml/saved_models/tennis_winner_model.pkl', 'rb') as f:
        _saved = pickle.load(f)
    _model = _saved['model']
    _features = _saved['features']
except Exception as e:
    _model = None
    _features = []

class TennisPredictionRequest(BaseModel):
    p1_name: str
    p2_name: str
    p1_rank: int
    p2_rank: int
    p1_elo: float
    p2_elo: float
    p1_surface_elo: float
    p2_surface_elo: float
    surface: str = 'Hard'
    tourney_level: str = 'A'
    round: str = 'R32'
    best_of: int = 3

class TennisIngestRequest(BaseModel):
    year: int

@router.post('/tennis/predict')
async def tennis_predict(req: TennisPredictionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail='Tennis model not loaded')

    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    level_map   = {'G': 4, 'M': 3, 'A': 2, 'F': 1, 'D': 0}
    round_map   = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1, 'RR': 3}

    elo_diff         = req.p1_elo - req.p2_elo
    surface_elo_diff = req.p1_surface_elo - req.p2_surface_elo
    rank_diff        = req.p2_rank - req.p1_rank
    elo_win_prob     = 1 / (1 + 10 ** (-elo_diff / 400))

    features = {
        'elo_diff':         elo_diff,
        'surface_elo_diff': surface_elo_diff,
        'rank_diff':        rank_diff,
        'elo_win_prob':     elo_win_prob,
        'surface_enc':      surface_map.get(req.surface, 0),
        'level_enc':        level_map.get(req.tourney_level, 2),
        'round_enc':        round_map.get(req.round, 3),
        'best_of_enc':      req.best_of,
    }

    X = np.array([[features[f] for f in _features]])
    prob_p1 = float(_model.predict_proba(X)[0][1])
    prob_p2 = 1 - prob_p1

    if prob_p1 >= prob_p2:
        predicted_winner = req.p1_name
        confidence = prob_p1
    else:
        predicted_winner = req.p2_name
        confidence = prob_p2

    return {
        'p1': req.p1_name,
        'p2': req.p2_name,
        'surface': req.surface,
        'predicted_winner': predicted_winner,
        'confidence': round(confidence * 100, 1),
        'p1_win_probability': round(prob_p1 * 100, 1),
        'p2_win_probability': round(prob_p2 * 100, 1),
    }

@router.get('/tennis/model/status')
async def tennis_model_status():
    return {
        'model_loaded': _model is not None,
        'features': _features,
        'feature_count': len(_features),
    }

@router.post('/tennis/alerts/run')
async def trigger_tennis_alert_engine(db: AsyncSession = Depends(get_db)):
    """Manually trigger the tennis alert engine (for testing)."""
    result = await run_tennis_alert_engine(db)
    return result

@router.post('/tennis/ingest/run')
async def trigger_tennis_ingestion(req: TennisIngestRequest, db: AsyncSession = Depends(get_db)):
    """Manually trigger a live ingestion pull from JeffSackmann/tennis_atp on GitHub
    for a given season. Upserts, so safe to re-run against a season already in the DB."""
    ingestion = SackmannIngestion(db)
    try:
        result = await ingestion.ingest_season_from_github(req.year)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub ingestion failed: {e}")
    return result

@router.post('/tennis/features/rebuild')
async def trigger_feature_rebuild(db: AsyncSession = Depends(get_db)):
    """Manually trigger a full feature rebuild (Elo/serve-stats + rolling windows)
    after new match data has landed. Safe to re-run — everything upserts."""
    try:
        result = await run_full_feature_rebuild(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature rebuild failed: {e}")
    return result
