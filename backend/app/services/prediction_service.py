import logging
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.match import Match
from app.models.prediction import Prediction, MatchStats

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/backend/ml/saved_models")


def _load(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


class PredictionService:
    def __init__(self):
        self._models_loaded = False

    def _ensure_models(self):
        if self._models_loaded:
            return
        logger.info("Loading ML models...")
        self.outcome_model = _load("outcome_model.pkl")
        self.outcome_features = _load("outcome_features.pkl")
        self.outcome_classes = _load("outcome_classes.pkl")

        self.goals_regressor = _load("goals_regressor.pkl")
        self.over25_classifier = _load("over25_classifier.pkl")
        self.goals_features = _load("goals_features.pkl")

        self.corners_ft_model = _load("corners_ft_model.pkl")
        self.corners_ft_features = _load("corners_ft_model_features.pkl")

        self.corners_ht_model = _load("corners_ht_model.pkl")
        self.corners_ht_features = _load("corners_ht_model_features.pkl")

        self.corners_2h_model = _load("corners_2h_model.pkl")
        self.corners_2h_features = _load("corners_2h_model_features.pkl")

        self.throw_ins_model = _load("throw_ins_model.pkl")
        self.throw_ins_features = _load("throw_ins_model_features.pkl")

        self._models_loaded = True
        logger.info("All models loaded.")

    async def predict_for_match(self, match_api_id: int, db: AsyncSession) -> Prediction:
        """Build features from DB and run all models for a match."""
        self._ensure_models()

        # Load match with teams
        result = await db.execute(
            select(Match)
            .options(selectinload(Match.home_team), selectinload(Match.away_team))
            .where(Match.api_id == match_api_id)
        )
        match = result.scalar_one_or_none()
        if not match:
            raise ValueError(f"Match {match_api_id} not found in DB")

        # Check for existing prediction
        existing = await db.execute(
            select(Prediction).where(Prediction.match_id == match.id)
        )
        existing_pred = existing.scalar_one_or_none()

        # Build feature row from DB
        features = await self._build_features(match, db)

        # --- Outcome prediction ---
        outcome_row = self._align(features, self.outcome_features)
        outcome_probs = self.outcome_model.predict_proba(outcome_row)[0]
        classes = list(self.outcome_classes)
        prob_map = dict(zip(classes, outcome_probs))
        home_prob = float(prob_map.get("H", 0.0))
        draw_prob = float(prob_map.get("D", 0.0))
        away_prob = float(prob_map.get("A", 0.0))
        predicted_outcome = classes[int(np.argmax(outcome_probs))]

        # --- Goals prediction ---
        goals_row = self._align(features, self.goals_features)
        total_goals_pred = float(self.goals_regressor.predict(goals_row)[0])
        over25_prob = float(self.over25_classifier.predict_proba(goals_row)[0][1])
        over35_prob = max(0.0, over25_prob - 0.22)  # heuristic offset

        # --- Corners predictions ---
        corners_ft = self._predict_safe(self.corners_ft_model, features, self.corners_ft_features)
        corners_ht = self._predict_safe(self.corners_ht_model, features, self.corners_ht_features)
        corners_2h = self._predict_safe(self.corners_2h_model, features, self.corners_2h_features)
        throw_ins  = self._predict_safe(self.throw_ins_model,  features, self.throw_ins_features)

        # Confidence = max outcome probability
        confidence = float(max(home_prob, draw_prob, away_prob))

        # Save or update prediction
        if existing_pred:
            pred = existing_pred
        else:
            pred = Prediction(match_id=match.id)
            db.add(pred)

        pred.home_win_prob     = home_prob
        pred.draw_prob         = draw_prob
        pred.away_win_prob     = away_prob
        pred.predicted_outcome = predicted_outcome
        pred.total_goals_pred  = total_goals_pred
        pred.over_25_prob      = over25_prob
        pred.over_35_prob      = over35_prob
        pred.corners_ht_pred   = corners_ht
        pred.corners_2h_pred   = corners_2h
        pred.corners_ft_pred   = corners_ft
        pred.throw_ins_pred    = throw_ins
        pred.confidence        = confidence
        pred.model_version     = "v1.0"
        pred.features_used     = json.dumps(list(features.keys()))

        await db.commit()
        await db.refresh(pred)
        logger.info(
            f"Prediction saved for match {match_api_id}: "
            f"{predicted_outcome} (H={home_prob:.2f} D={draw_prob:.2f} A={away_prob:.2f})"
        )
        return pred

    def _align(self, features: dict, feature_names: list) -> pd.DataFrame:
        """Align feature dict to the exact columns the model was trained on."""
        row = {col: features.get(col, 0.0) for col in feature_names}
        return pd.DataFrame([row])[feature_names]

    def _predict_safe(self, model, features: dict, feature_names: list) -> float:
        """Run a regression model, return 0.0 if features are missing."""
        try:
            row = self._align(features, feature_names)
            return float(model.predict(row)[0])
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return 0.0

    async def _build_features(self, match: Match, db: AsyncSession) -> dict:
        """
        Build the feature dict for a match using historical data from DB.
        Mirrors the logic in feature_builder.py but queries live from DB.
        """
        from sqlalchemy import text

        home_id = match.home_team_id
        away_id = match.away_team_id
        match_date = match.match_date

        features = {}

        # Helper: recent form for a team (last N matches before this one)
        async def team_form(team_id: int, n: int = 5):
            q = await db.execute(text("""
                SELECT m.home_team_id, m.away_team_id, m.home_score, m.away_score,
                       ms.corners_total, ms.shots_total, ms.xg
                FROM matches m
                LEFT JOIN match_stats ms ON ms.match_id = m.id AND ms.team_id = :tid
                WHERE m.status = 'FT'
                  AND (m.home_team_id = :tid OR m.away_team_id = :tid)
                  AND m.match_date < :dt
                ORDER BY m.match_date DESC
                LIMIT :n
            """), {"tid": team_id, "dt": match_date, "n": n})
            return q.fetchall()

        home_form = await team_form(home_id)
        away_form = await team_form(away_id)

        def form_stats(rows, team_id):
            wins = draws = losses = goals_for = goals_against = corners = shots = xg_sum = 0
            for r in rows:
                is_home = r.home_team_id == team_id
                gf = r.home_score if is_home else r.away_score
                ga = r.away_score if is_home else r.home_score
                if gf is None or ga is None:
                    continue
                goals_for += gf
                goals_against += ga
                if gf > ga: wins += 1
                elif gf == ga: draws += 1
                else: losses += 1
                if r.corners_total: corners += r.corners_total
                if r.shots_total: shots += r.shots_total
                if r.xg: xg_sum += r.xg
            n = max(len(rows), 1)
            return {
                "wins": wins / n, "draws": draws / n, "losses": losses / n,
                "goals_for": goals_for / n, "goals_against": goals_against / n,
                "corners": corners / n, "shots": shots / n, "xg": xg_sum / n,
            }

        hf = form_stats(home_form, home_id)
        af = form_stats(away_form, away_id)

        features.update({
            "home_form_wins": hf["wins"],
            "home_form_draws": hf["draws"],
            "home_form_losses": hf["losses"],
            "home_form_goals_scored": hf["goals_for"],
            "home_form_goals_conceded": hf["goals_against"],
            "home_form_corners": hf["corners"],
            "home_form_shots": hf["shots"],
            "home_form_xg": hf["xg"],
            "away_form_wins": af["wins"],
            "away_form_draws": af["draws"],
            "away_form_losses": af["losses"],
            "away_form_goals_scored": af["goals_for"],
            "away_form_goals_conceded": af["goals_against"],
            "away_form_corners": af["corners"],
            "away_form_shots": af["shots"],
            "away_form_xg": af["xg"],
        })

        # Season averages
        async def season_avg(team_id: int):
            q = await db.execute(text("""
                SELECT AVG(CASE WHEN m.home_team_id=:tid THEN m.home_score ELSE m.away_score END) as avg_gf,
                       AVG(CASE WHEN m.home_team_id=:tid THEN m.away_score ELSE m.home_score END) as avg_ga,
                       COUNT(*) as played
                FROM matches m
                WHERE m.status='FT'
                  AND (m.home_team_id=:tid OR m.away_team_id=:tid)
                  AND m.match_date < :dt
                  AND m.season = :season
            """), {"tid": team_id, "dt": match_date, "season": match.season})
            return q.fetchone()

        hs = await season_avg(home_id)
        as_ = await season_avg(away_id)

        features.update({
            "home_season_avg_goals_scored": float(hs.avg_gf or 0),
            "home_season_avg_goals_conceded": float(hs.avg_ga or 0),
            "home_season_played": int(hs.played or 0),
            "away_season_avg_goals_scored": float(as_.avg_gf or 0),
            "away_season_avg_goals_conceded": float(as_.avg_ga or 0),
            "away_season_played": int(as_.played or 0),
        })

        # H2H last 5
        h2h_q = await db.execute(text("""
            SELECT home_score, away_score, home_team_id
            FROM matches
            WHERE status='FT'
              AND ((home_team_id=:h AND away_team_id=:a) OR (home_team_id=:a AND away_team_id=:h))
              AND match_date < :dt
            ORDER BY match_date DESC LIMIT 5
        """), {"h": home_id, "a": away_id, "dt": match_date})
        h2h = h2h_q.fetchall()

        h2h_home_wins = h2h_draws = h2h_away_wins = h2h_avg_goals = 0
        for r in h2h:
            if r.home_score is None: continue
            total = r.home_score + r.away_score
            h2h_avg_goals += total
            if r.home_team_id == home_id:
                if r.home_score > r.away_score: h2h_home_wins += 1
                elif r.home_score == r.away_score: h2h_draws += 1
                else: h2h_away_wins += 1
            else:
                if r.away_score > r.home_score: h2h_home_wins += 1
                elif r.home_score == r.away_score: h2h_draws += 1
                else: h2h_away_wins += 1

        n_h2h = max(len(h2h), 1)
        features.update({
            "h2h_home_wins": h2h_home_wins / n_h2h,
            "h2h_draws": h2h_draws / n_h2h,
            "h2h_away_wins": h2h_away_wins / n_h2h,
            "h2h_avg_goals": h2h_avg_goals / n_h2h,
        })

        # Strength differentials
        features["goal_diff_home_away"] = (
            features["home_season_avg_goals_scored"] - features["away_season_avg_goals_scored"]
        )
        features["defense_diff"] = (
            features["away_season_avg_goals_conceded"] - features["home_season_avg_goals_conceded"]
        )
        features["form_diff_wins"] = features["home_form_wins"] - features["away_form_wins"]
        features["form_diff_goals"] = features["home_form_goals_scored"] - features["away_form_goals_scored"]

        # Corners season averages
        async def corner_avg(team_id: int):
            q = await db.execute(text("""
                SELECT AVG(ms.corners_total) as avg_corners
                FROM match_stats ms
                JOIN matches m ON m.id = ms.match_id
                WHERE ms.team_id = :tid AND m.status='FT' AND m.match_date < :dt
            """), {"tid": team_id, "dt": match_date})
            r = q.fetchone()
            return float(r.avg_corners or 0)

        home_corners_avg = await corner_avg(home_id)
        away_corners_avg = await corner_avg(away_id)

        features.update({
            "home_corners_avg": home_corners_avg,
            "away_corners_avg": away_corners_avg,
            "total_corners_avg": home_corners_avg + away_corners_avg,
            "corner_diff": home_corners_avg - away_corners_avg,
        })

        # Placeholder features (weather, referee — default 0 for now)
        features.update({
            "weather_temp": 15.0,
            "weather_humidity": 60.0,
            "referee_avg_yellows": 3.5,
            "referee_avg_fouls": 22.0,
            "is_derby": 0,
            "league_id": match.league_id or 39,
            "season": int(str(match.season)[:4]) if match.season else 2024,
            "match_week": 20,
        })

        return features


# Singleton instance
prediction_service = PredictionService()
