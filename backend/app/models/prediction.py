from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class MatchStats(Base):
    """Per-team statistics for a finished match."""

    __tablename__ = "match_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False, index=True
    )
    is_home: Mapped[bool] = mapped_column(Boolean, default=True)

    # Possession & Passing
    possession: Mapped[float | None] = mapped_column(Float)
    passes_total: Mapped[int | None] = mapped_column(Integer)
    passes_accurate: Mapped[int | None] = mapped_column(Integer)
    pass_accuracy: Mapped[float | None] = mapped_column(Float)

    # Shots
    shots_total: Mapped[int | None] = mapped_column(Integer)
    shots_on_target: Mapped[int | None] = mapped_column(Integer)
    shots_off_target: Mapped[int | None] = mapped_column(Integer)
    shots_blocked: Mapped[int | None] = mapped_column(Integer)

    # Expected Goals
    xg: Mapped[float | None] = mapped_column(Float)

    # Corners
    corners_total: Mapped[int | None] = mapped_column(Integer)
    corners_ht: Mapped[int | None] = mapped_column(Integer)
    corners_ft: Mapped[int | None] = mapped_column(Integer)

    # Other
    throw_ins: Mapped[int | None] = mapped_column(Integer)
    fouls: Mapped[int | None] = mapped_column(Integer)
    yellow_cards: Mapped[int | None] = mapped_column(Integer)
    red_cards: Mapped[int | None] = mapped_column(Integer)
    offsides: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    match: Mapped["Match"] = relationship("Match", back_populates="stats")  # noqa: F821


class Prediction(Base):
    """ML model prediction for a match."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), unique=True, nullable=False, index=True
    )

    # Outcome probabilities
    home_win_prob: Mapped[float | None] = mapped_column(Float)
    draw_prob: Mapped[float | None] = mapped_column(Float)
    away_win_prob: Mapped[float | None] = mapped_column(Float)
    predicted_outcome: Mapped[str | None] = mapped_column(String(1))  # H, D, A

    # Goals
    total_goals_pred: Mapped[float | None] = mapped_column(Float)
    over_25_prob: Mapped[float | None] = mapped_column(Float)
    over_35_prob: Mapped[float | None] = mapped_column(Float)

    # Corners
    corners_ht_pred: Mapped[float | None] = mapped_column(Float)
    corners_2h_pred: Mapped[float | None] = mapped_column(Float)
    corners_ft_pred: Mapped[float | None] = mapped_column(Float)

    # Throw-ins
    throw_ins_pred: Mapped[float | None] = mapped_column(Float)

    # Meta
    confidence: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(50))
    features_used: Mapped[str | None] = mapped_column(Text)  # JSON string

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    match: Mapped["Match"] = relationship("Match", back_populates="prediction")  # noqa: F821
    alerts: Mapped[list["Alert"]] = relationship(  # noqa: F821
        "Alert", back_populates="prediction", cascade="all, delete-orphan"
    )


class Odds(Base):
    """Bookmaker odds snapshots for a match."""

    __tablename__ = "odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    bookmaker: Mapped[str] = mapped_column(String(100), nullable=False)
    market: Mapped[str] = mapped_column(String(50), default="h2h")  # h2h, totals

    home_odds: Mapped[float | None] = mapped_column(Float)
    draw_odds: Mapped[float | None] = mapped_column(Float)
    away_odds: Mapped[float | None] = mapped_column(Float)
    over_25_odds: Mapped[float | None] = mapped_column(Float)
    under_25_odds: Mapped[float | None] = mapped_column(Float)

    is_opening: Mapped[bool] = mapped_column(Boolean, default=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    match: Mapped["Match"] = relationship("Match", back_populates="odds")  # noqa: F821


class Alert(Base):
    """Value bet or odds movement alerts."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    prediction_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("predictions.id"), index=True
    )

    alert_type: Mapped[str] = mapped_column(String(50))  # VALUE_BET, ODDS_MOVEMENT
    outcome: Mapped[str | None] = mapped_column(String(10))  # H, D, A, OVER, UNDER
    message: Mapped[str] = mapped_column(Text, nullable=False)
    model_prob: Mapped[float | None] = mapped_column(Float)
    implied_prob: Mapped[float | None] = mapped_column(Float)
    edge_pct: Mapped[float | None] = mapped_column(Float)

    sent_telegram: Mapped[bool] = mapped_column(Boolean, default=False)
    sent_sms: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    prediction: Mapped["Prediction | None"] = relationship(  # noqa: F821
        "Prediction", back_populates="alerts"
    )
