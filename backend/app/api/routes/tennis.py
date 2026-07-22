import os
import pickle
import asyncio
from datetime import datetime, timezone
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.tennis_alert_engine import (
    run_tennis_alert_engine,
    _get_player_data,
    _get_rolling_features,
    _predict_winner,
    _predict_total_games,
)
from tennis.ingestion.sackmann_ingestion import SackmannIngestion
from tennis.features.rebuild import run_full_feature_rebuild, _get_sync_database_url
from tennis.training.retrain_total_games import train_total_games_candidate

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

class TennisRetrainRequest(BaseModel):
    test_cutoff_date: str = '2023-01-01'

class TennisMatchupRequest(BaseModel):
    p1_name: str
    p2_name: str
    surface: str = 'Hard'
    level: str = 'A'
    round: str = 'R32'
    best_of: int = 3

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

@router.post('/tennis/predict/matchup')
async def predict_matchup(req: TennisMatchupRequest, db: AsyncSession = Depends(get_db)):
    """Predict a real matchup using each player's actual historical data —
    Elo, rank, surface-Elo, and rolling serve stats pulled from tennis_matches.
    No live odds needed, no Telegram send. Player names are matched loosely
    (by last name) against your historical data, same as the alert engine."""
    p1_data = await _get_player_data(db, req.p1_name)
    p2_data = await _get_player_data(db, req.p2_name)
    if not p1_data or not p2_data:
        missing = req.p1_name if not p1_data else req.p2_name
        raise HTTPException(status_code=404, detail=f"No historical match data found for '{missing}'")

    match_date = datetime.now(timezone.utc).date()
    p1_data['rolling'] = await _get_rolling_features(db, p1_data['player_id'], match_date, req.surface)
    p2_data['rolling'] = await _get_rolling_features(db, p2_data['player_id'], match_date, req.surface)

    p1_prob, p2_prob = _predict_winner(
        p1_data, p2_data, surface=req.surface, level=req.level, round_=req.round, best_of=req.best_of
    )
    if p1_prob is None:
        raise HTTPException(status_code=503, detail="Winner model not loaded")

    winner = req.p1_name if p1_prob >= p2_prob else req.p2_name
    tg_result = _predict_total_games(
        p1_data, p2_data, surface=req.surface, level=req.level, round_=req.round, best_of=req.best_of
    )

    return {
        'p1': req.p1_name,
        'p2': req.p2_name,
        'p1_matched_name': p1_data['name'],
        'p2_matched_name': p2_data['name'],
        'p1_elo': round(p1_data['elo'], 1),
        'p2_elo': round(p2_data['elo'], 1),
        'p1_rank': p1_data['rank'],
        'p2_rank': p2_data['rank'],
        'predicted_winner': winner,
        'p1_win_probability': round(p1_prob * 100, 1),
        'p2_win_probability': round(p2_prob * 100, 1),
        'confidence': round(max(p1_prob, p2_prob) * 100, 1),
        'total_games': tg_result,
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

@router.post('/tennis/retrain/run')
async def trigger_retrain_candidate(req: TennisRetrainRequest):
    """Retrains the total-games quantile models (GBR/XGBoost/LightGBM per
    segment) on current data and saves the result to a timestamped CANDIDATE
    file under ml/saved_models/candidates/. Does NOT touch the live model."""
    candidates_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ml', 'saved_models', 'candidates')
    os.makedirs(candidates_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    candidate_path = os.path.join(candidates_dir, f'tennis_total_games_candidate_{timestamp}.pkl')

    try:
        result = await asyncio.to_thread(
            train_total_games_candidate,
            _get_sync_database_url(),
            candidate_path,
            req.test_cutoff_date,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")
    return result

@router.get('/tennis/predictions/upcoming')
async def get_tennis_upcoming_predictions(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Upcoming tennis matches that cleared the alert engine's 62% confidence
    bar (that's the only reason a row exists in tennis_upcoming_matches at all).
    total_games fields are only populated for matches where an alert was
    actually sent — the alert engine doesn't persist that calculation otherwise."""
    now = datetime.now(timezone.utc)
    result = await db.execute(text('''
        SELECT u.odds_api_id, u.p1_name, u.p2_name, u.tournament, u.sport_key,
               u.commence_time, u.p1_odds, u.p2_odds, u.predicted_winner,
               u.p1_win_prob, u.p2_win_prob, u.confidence, u.alert_sent,
               c.model_line, c.model_prob_over, c.model_prob_under
        FROM tennis_upcoming_matches u
        LEFT JOIN tennis_clv_tracking c ON c.odds_api_id = u.odds_api_id
        WHERE u.commence_time >= :now
        ORDER BY u.commence_time ASC
        LIMIT :limit
    '''), {'now': now, 'limit': limit})
    rows = result.fetchall()

    data = []
    for r in rows:
        (odds_api_id, p1, p2, tournament, sport_key, commence_time,
         p1_odds, p2_odds, predicted_winner, p1_prob, p2_prob, confidence,
         alert_sent, model_line, prob_over, prob_under) = r
        data.append({
            'id': odds_api_id,
            'player1': p1,
            'player2': p2,
            'tour': 'ATP' if 'atp' in sport_key else 'WTA',
            'tournament': tournament,
            'commence_time': commence_time,
            'market_odds': {'p1': p1_odds, 'p2': p2_odds},
            'model_prob': {'p1': p1_prob, 'p2': p2_prob},
            'predicted_winner': 'player1' if predicted_winner == p1 else 'player2',
            'confidence': confidence,
            'is_alert': alert_sent,
            'total_games_line': model_line,
            'prob_over': prob_over,
            'prob_under': prob_under,
        })
    return data