import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.match import Match
from app.models.prediction import Prediction, Odds
from app.schemas.prediction import PredictRequestSchema, PredictionSchema
from app.services.prediction_service import prediction_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=list[PredictionSchema])
async def get_predictions(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/upcoming")
async def get_upcoming_predictions(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Upcoming football matches with predictions, latest odds snapshot, and
    the four prop fields (corners/throw-ins) — those come back as null until
    a model that predicts them actually exists."""
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(Match, Prediction)
        .join(Prediction, Prediction.match_id == Match.id)
        .options(selectinload(Match.home_team), selectinload(Match.away_team))
        .where(Match.status == "NS", Match.match_date >= now)
        .order_by(Match.match_date)
        .limit(limit)
    )
    rows = result.all()

    match_ids = [m.id for m, _ in rows]
    latest_odds = {}
    if match_ids:
        odds_result = await db.execute(
            select(Odds).where(Odds.match_id.in_(match_ids)).order_by(Odds.recorded_at.desc())
        )
        for o in odds_result.scalars().all():
            if o.match_id not in latest_odds:
                latest_odds[o.match_id] = o

    outcome_map = {"H": "home_win", "D": "draw", "A": "away_win"}
    data = []
    for match, pred in rows:
        odds = latest_odds.get(match.id)
        probs = [pred.home_win_prob, pred.draw_prob, pred.away_win_prob]
        confidence = max([p for p in probs if p is not None], default=None)
        data.append({
            "id": match.id,
            "home_team": match.home_team.name,
            "away_team": match.away_team.name,
            "competition": match.league_name,
            "round_label": match.season,
            "match_date": match.match_date,
            "market_odds": {
                "home": odds.home_odds if odds else None,
                "draw": odds.draw_odds if odds else None,
                "away": odds.away_odds if odds else None,
            },
            "model_confidence": confidence,
            "predicted_result": outcome_map.get(pred.predicted_outcome),
            "props": {
                "total_corners": pred.corners_ft_pred,
                "corners_first_half": pred.corners_ht_pred,
                "corners_second_half": pred.corners_2h_pred,
                "total_throw_ins": pred.throw_ins_pred,
            },
        })
    return data


@router.get("/{match_id}", response_model=PredictionSchema)
async def get_prediction_for_match(match_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Prediction).where(Prediction.match_id == match_id)
    )
    prediction = result.scalar_one_or_none()
    if not prediction:
        raise HTTPException(status_code=404, detail="No prediction found for this match")
    return prediction


@router.post("/predict", response_model=PredictionSchema)
async def predict_match(
    request: PredictRequestSchema,
    db: AsyncSession = Depends(get_db),
):
    """Run all ML models for a match and save the prediction."""
    try:
        prediction = await prediction_service.predict_for_match(
            match_api_id=request.match_api_id,
            db=db,
        )
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    except Exception as e:
        logger.exception(f"Prediction failed for match {request.match_api_id}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
