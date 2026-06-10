import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.alert_engine import run_alert_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("/run")
async def trigger_alert_engine(db: AsyncSession = Depends(get_db)):
    """Scan all recent predictions for value bets and send Telegram alerts."""
    result = await run_alert_engine(db)
    return result


@router.get("/health")
async def alerts_health():
    return {"status": "ok", "bot": "MRXD3000 Predict"}
