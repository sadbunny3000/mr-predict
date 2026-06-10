"""Initial tables

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── teams ───────────────────────────────────────────────
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("api_id", sa.BigInteger(), nullable=False, unique=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("short_name", sa.String(10)),
        sa.Column("logo_url", sa.String(500)),
        sa.Column("country", sa.String(100)),
        sa.Column("league_id", sa.Integer()),
        sa.Column("founded", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    op.create_index("ix_teams_api_id", "teams", ["api_id"])

    # ─── matches ─────────────────────────────────────────────
    op.create_table(
        "matches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("api_id", sa.BigInteger(), nullable=False, unique=True),
        sa.Column("home_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("away_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("league_id", sa.Integer(), nullable=False),
        sa.Column("league_name", sa.String(200)),
        sa.Column("season", sa.String(10)),
        sa.Column("match_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(50), default="NS"),
        sa.Column("venue", sa.String(200)),
        sa.Column("referee", sa.String(200)),
        sa.Column("home_score", sa.Integer()),
        sa.Column("away_score", sa.Integer()),
        sa.Column("home_score_ht", sa.Integer()),
        sa.Column("away_score_ht", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    op.create_index("ix_matches_api_id", "matches", ["api_id"])
    op.create_index("ix_matches_match_date", "matches", ["match_date"])
    op.create_index("ix_matches_league_id", "matches", ["league_id"])

    # ─── match_stats ─────────────────────────────────────────
    op.create_table(
        "match_stats",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("is_home", sa.Boolean(), default=True),
        sa.Column("possession", sa.Float()),
        sa.Column("passes_total", sa.Integer()),
        sa.Column("passes_accurate", sa.Integer()),
        sa.Column("pass_accuracy", sa.Float()),
        sa.Column("shots_total", sa.Integer()),
        sa.Column("shots_on_target", sa.Integer()),
        sa.Column("shots_off_target", sa.Integer()),
        sa.Column("shots_blocked", sa.Integer()),
        sa.Column("xg", sa.Float()),
        sa.Column("corners_total", sa.Integer()),
        sa.Column("corners_ht", sa.Integer()),
        sa.Column("corners_ft", sa.Integer()),
        sa.Column("throw_ins", sa.Integer()),
        sa.Column("fouls", sa.Integer()),
        sa.Column("yellow_cards", sa.Integer()),
        sa.Column("red_cards", sa.Integer()),
        sa.Column("offsides", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_match_stats_match_id", "match_stats", ["match_id"])

    # ─── predictions ─────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), unique=True, nullable=False),
        sa.Column("home_win_prob", sa.Float()),
        sa.Column("draw_prob", sa.Float()),
        sa.Column("away_win_prob", sa.Float()),
        sa.Column("predicted_outcome", sa.String(1)),
        sa.Column("total_goals_pred", sa.Float()),
        sa.Column("over_25_prob", sa.Float()),
        sa.Column("over_35_prob", sa.Float()),
        sa.Column("corners_ht_pred", sa.Float()),
        sa.Column("corners_2h_pred", sa.Float()),
        sa.Column("corners_ft_pred", sa.Float()),
        sa.Column("throw_ins_pred", sa.Float()),
        sa.Column("confidence", sa.Float()),
        sa.Column("model_version", sa.String(50)),
        sa.Column("features_used", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )

    # ─── odds ────────────────────────────────────────────────
    op.create_table(
        "odds",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("bookmaker", sa.String(100), nullable=False),
        sa.Column("market", sa.String(50), default="h2h"),
        sa.Column("home_odds", sa.Float()),
        sa.Column("draw_odds", sa.Float()),
        sa.Column("away_odds", sa.Float()),
        sa.Column("over_25_odds", sa.Float()),
        sa.Column("under_25_odds", sa.Float()),
        sa.Column("is_opening", sa.Boolean(), default=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_odds_match_id", "odds", ["match_id"])
    op.create_index("ix_odds_recorded_at", "odds", ["recorded_at"])

    # ─── alerts ──────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("prediction_id", sa.Integer(), sa.ForeignKey("predictions.id")),
        sa.Column("alert_type", sa.String(50)),
        sa.Column("outcome", sa.String(10)),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("model_prob", sa.Float()),
        sa.Column("implied_prob", sa.Float()),
        sa.Column("edge_pct", sa.Float()),
        sa.Column("sent_telegram", sa.Boolean(), default=False),
        sa.Column("sent_sms", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_alerts_match_id", "alerts", ["match_id"])
    op.create_index("ix_alerts_created_at", "alerts", ["created_at"])


def downgrade() -> None:
    op.drop_table("alerts")
    op.drop_table("odds")
    op.drop_table("predictions")
    op.drop_table("match_stats")
    op.drop_table("matches")
    op.drop_table("teams")
