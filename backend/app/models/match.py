from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    api_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    home_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False, index=True
    )
    away_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False, index=True
    )
    league_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    league_name: Mapped[str | None] = mapped_column(String(200))
    season: Mapped[str | None] = mapped_column(String(10))
    match_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(50), default="NS")  # NS, 1H, HT, 2H, FT
    venue: Mapped[str | None] = mapped_column(String(200))
    referee: Mapped[str | None] = mapped_column(String(200))

    # Scores
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    home_score_ht: Mapped[int | None] = mapped_column(Integer)
    away_score_ht: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    home_team: Mapped["Team"] = relationship(  # noqa: F821
        "Team", back_populates="home_matches", foreign_keys=[home_team_id]
    )
    away_team: Mapped["Team"] = relationship(  # noqa: F821
        "Team", back_populates="away_matches", foreign_keys=[away_team_id]
    )
    stats: Mapped[list["MatchStats"]] = relationship(  # noqa: F821
        "MatchStats", back_populates="match", cascade="all, delete-orphan"
    )
    prediction: Mapped["Prediction | None"] = relationship(  # noqa: F821
        "Prediction", back_populates="match", uselist=False
    )
    odds: Mapped[list["Odds"]] = relationship(  # noqa: F821
        "Odds", back_populates="match", cascade="all, delete-orphan"
    )

    @property
    def result(self) -> str | None:
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "H"
        if self.home_score < self.away_score:
            return "A"
        return "D"

    def __repr__(self) -> str:
        return f"<Match {self.api_id}: {self.home_team_id} vs {self.away_team_id}>"
