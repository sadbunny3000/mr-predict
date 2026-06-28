import pickle
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# Load model once when the route file is imported
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
