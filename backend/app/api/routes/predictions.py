import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.match import Match
from app.models.prediction import Prediction
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


@router.post("/predict-all", response_model=dict)
async def predict_all_upcoming(db: AsyncSession = Depends(get_db)):
    """Run predictions for all upcoming (NS) matches that don't have one yet."""
    result = await db.execute(
        select(Match)
        .options(selectinload(Match.home_team), selectinload(Match.away_team))
        .where(Match.status == "NS")
        .order_by(Match.match_date)
        .limit(20)
    )
    matches = result.scalars().all()

    if not matches:
        return {"message": "No upcoming matches found", "predicted": 0}

    success = 0
    failed = 0
    errors = []

    for match in matches:
        try:
            await prediction_service.predict_for_match(
                match_api_id=match.api_id,
                db=db,
            )
            success += 1
        except Exception as e:
            failed += 1
            errors.append(f"Match {match.api_id}: {str(e)}")
            logger.warning(f"Failed to predict match {match.api_id}: {e}")

    return {
        "message": f"Predicted {success} matches, {failed} failed",
        "predicted": success,
        "failed": failed,
        "errors": errors[:5],
    }
