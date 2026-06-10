from datetime import datetime

from pydantic import BaseModel


class PredictionSchema(BaseModel):
    id: int
    match_id: int
    home_win_prob: float | None = None
    draw_prob: float | None = None
    away_win_prob: float | None = None
    predicted_outcome: str | None = None
    total_goals_pred: float | None = None
    over_25_prob: float | None = None
    over_35_prob: float | None = None
    corners_ht_pred: float | None = None
    corners_2h_pred: float | None = None
    corners_ft_pred: float | None = None
    throw_ins_pred: float | None = None
    confidence: float | None = None
    model_version: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionWithMatchSchema(PredictionSchema):
    home_team_name: str
    away_team_name: str
    match_date: datetime
    league_name: str | None = None


class PredictRequestSchema(BaseModel):
    match_api_id: int


class AlertSchema(BaseModel):
    id: int
    match_id: int
    alert_type: str
    outcome: str | None = None
    message: str
    model_prob: float | None = None
    implied_prob: float | None = None
    edge_pct: float | None = None
    sent_telegram: bool
    sent_sms: bool
    created_at: datetime

    model_config = {"from_attributes": True}
