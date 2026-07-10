import logging
import sys
import asyncio
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import alerts, health, matches, predictions, tennis
from app.config import settings
from app.redis_client import close_redis
from app.database import AsyncSessionLocal
from app.services.alert_engine import run_alert_engine
from app.services.tennis_alert_engine import run_tennis_alert_engine, run_tennis_clv_closing_job, run_tennis_result_check_job

# ─── Logging setup ──────────────────────────────────────────
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    ),
)
logger = structlog.get_logger()

# ─── App ────────────────────────────────────────────────────
app = FastAPI(
    title="Football Predictor API",
    description="ML-powered football match prediction platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ──────────────────────────────────────────────────
app.include_router(health.router, prefix="/api/v1")
app.include_router(matches.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")
app.include_router(tennis.router, prefix="/api/v1")


# ─── Scheduler ───────────────────────────────────────────────
async def alert_scheduler():
    """Run the alert engine every hour automatically."""
    logger.info("Alert scheduler started — runs every hour")
    while True:
        await asyncio.sleep(3600)  # wait 1 hour
        try:
            logger.info("Scheduler: running alert engine")
            async with AsyncSessionLocal() as db:
                result = await run_alert_engine(db)
                logger.info(f"Scheduler: alert engine done — {result}")
                tennis_result = await run_tennis_alert_engine(db)
                logger.info(f"Scheduler: tennis alert engine done — {tennis_result}")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")


async def tennis_clv_closing_scheduler():
    """Check for matches starting soon and record closing odds every 5 minutes."""
    logger.info("Tennis CLV closing scheduler started — runs every 5 minutes")
    while True:
        await asyncio.sleep(300)  # wait 5 minutes
        try:
            async with AsyncSessionLocal() as db:
                result = await run_tennis_clv_closing_job(db)
                if result.get('checked', 0) > 0:
                    logger.info(f"Scheduler: tennis CLV closing job done — {result}")
        except Exception as e:
            logger.error(f"CLV closing scheduler error: {e}")

async def tennis_result_check_scheduler():
    """Check completed matches for winner-prediction accuracy every 30 minutes."""
    logger.info("Tennis result-check scheduler started — runs every 30 minutes")
    while True:
        await asyncio.sleep(1800)  # wait 30 minutes
        try:
            async with AsyncSessionLocal() as db:
                result = await run_tennis_result_check_job(db)
                if result.get('checked', 0) > 0:
                    logger.info(f"Scheduler: tennis result-check job done — {result}")
        except Exception as e:
            logger.error(f"Result-check scheduler error: {e}")


# ─── Lifecycle ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    asyncio.create_task(alert_scheduler())
    asyncio.create_task(tennis_clv_closing_scheduler())
    asyncio.create_task(tennis_result_check_scheduler())
    logger.info("Application started — alert scheduler running")


@app.on_event("shutdown")
async def shutdown():
    await close_redis()
    logger.info("Application shutdown complete")


@app.get("/")
async def root():
    return {
        "service": "Football Predictor API",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "running",
    }