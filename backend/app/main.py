import logging
import sys

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import alerts, health, matches, predictions
from app.config import settings
from app.redis_client import close_redis

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


# ─── Lifecycle ───────────────────────────────────────────────
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
