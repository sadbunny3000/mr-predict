from datetime import datetime

from pydantic import BaseModel


class TeamSchema(BaseModel):
    id: int
    api_id: int
    name: str
    short_name: str | None = None
    logo_url: str | None = None
    country: str | None = None

    model_config = {"from_attributes": True}


class MatchStatsSchema(BaseModel):
    team_id: int
    is_home: bool
    possession: float | None = None
    pass_accuracy: float | None = None
    passes_total: int | None = None
    shots_total: int | None = None
    shots_on_target: int | None = None
    xg: float | None = None
    corners_ht: int | None = None
    corners_ft: int | None = None
    throw_ins: int | None = None

    model_config = {"from_attributes": True}


class MatchSchema(BaseModel):
    id: int
    api_id: int
    league_id: int
    league_name: str | None = None
    season: str | None = None
    match_date: datetime
    status: str
    venue: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    home_team: TeamSchema
    away_team: TeamSchema

    model_config = {"from_attributes": True}


class MatchListSchema(BaseModel):
    id: int
    api_id: int
    match_date: datetime
    status: str
    league_name: str | None = None
    home_team_name: str
    away_team_name: str
    home_score: int | None = None
    away_score: int | None = None

    model_config = {"from_attributes": True}
