"""
Day 4 — Model Training
Trains three models:
  1. Outcome classifier (H/D/A)
  2. Goals regressor + Over/Under 2.5 classifier
  3. Corners regressor (total FT now; HT/2H/throw-ins auto-enabled when paid plan active)
"""
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

OUTCOME_FEATURES = [
    "home_form_wins", "home_form_draws", "home_form_losses",
    "home_form_goals_scored", "home_form_goals_conceded", "home_form_points",
    "away_form_wins", "away_form_draws", "away_form_losses",
    "away_form_goals_scored", "away_form_goals_conceded", "away_form_points",
    "home_season_avg_scored", "home_season_avg_conceded",
    "home_season_win_rate", "home_season_draw_rate", "home_season_home_avg_scored",
    "away_season_avg_scored", "away_season_avg_conceded",
    "away_season_win_rate", "away_season_draw_rate", "away_season_away_avg_scored",
    "home_attack_vs_away_defence", "away_attack_vs_home_defence",
    "form_points_diff", "season_win_rate_diff",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_home_goals", "h2h_away_goals", "h2h_matches",
    "referee_avg_corners", "referee_avg_cards",
    "temperature", "precipitation", "wind_speed",
]

CORNERS_FEATURES = OUTCOME_FEATURES + [
    "home_avg_corners_ht", "home_avg_corners_ft", "home_avg_corners_total",
    "home_avg_throw_ins",
    "away_avg_corners_ht", "away_avg_corners_ft", "away_avg_corners_total",
    "away_avg_throw_ins",
    "expected_total_corners_ht", "expected_total_corners_ft",
    "expected_total_throw_ins",
]

# Targets — free plan has corners_total only.
# HT, 2H, throw-ins auto-activate when paid plan data is present.
CORNER_TARGETS = {
    "total_corners_ft":  "corners_ft_model",   # free plan ✅
    "total_corners_ht":  "corners_ht_model",   # paid plan (auto)
    "total_corners_2h":  "corners_2h_model",   # paid plan (auto)
    "total_throw_ins":   "throw_ins_model",    # paid plan (auto)
}


class ModelTrainer:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}

    def _save(self, name: str, obj) -> None:
        path = os.path.join(self.model_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved {name} → {path}")

    def _get_available_features(self, df: pd.DataFrame, feature_list: list) -> list:
        return [f for f in feature_list if f in df.columns]

    def train_outcome(self, df: pd.DataFrame) -> dict:
        logger.info("Training outcome model (H/D/A)...")
        feat_cols = self._get_available_features(df, OUTCOME_FEATURES)
        df = df.dropna(subset=["result"])
        X = df[feat_cols].fillna(0)
        y = df["result"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
        rf_cal = CalibratedClassifierCV(rf, cv=5)
        rf_cal.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf_cal.predict(X_test))

        # XGBoost with encoded labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                             random_state=42, eval_metric="mlogloss", verbosity=0)
        xgb_cal = CalibratedClassifierCV(xgb, cv=5)
        xgb_cal.fit(X_train, y_train_enc)
        xgb_preds = le.inverse_transform(xgb_cal.predict(X_test))
        xgb_acc = accuracy_score(y_test, xgb_preds)

        logger.info(f"Outcome — RF: {rf_acc:.3f}, XGB: {xgb_acc:.3f}")

        if rf_acc >= xgb_acc:
            best = rf_cal
            classes = list(rf_cal.classes_)
            logger.info(f"Using Random Forest")
            logger.info(f"\n{classification_report(y_test, rf_cal.predict(X_test))}")
        else:
            best = {"model": xgb_cal, "encoder": le}
            classes = list(le.classes_)
            logger.info(f"Using XGBoost")
            logger.info(f"\n{classification_report(y_test, xgb_preds)}")

        self._save("outcome_model", best)
        self._save("outcome_features", feat_cols)
        self._save("outcome_classes", classes)
        self.models["outcome"] = best
        return {"rf_accuracy": rf_acc, "xgb_accuracy": xgb_acc, "best_accuracy": max(rf_acc, xgb_acc)}

    def train_goals(self, df: pd.DataFrame) -> dict:
        logger.info("Training goals model...")
        feat_cols = self._get_available_features(df, OUTCOME_FEATURES)
        df = df.dropna(subset=["total_goals"])
        X = df[feat_cols].fillna(0)
        y_goals = df["total_goals"]
        y_over25 = (df["total_goals"] > 2.5).astype(int)

        X_train, X_test, yg_train, yg_test, yo_train, yo_test = train_test_split(
            X, y_goals, y_over25, test_size=0.2, random_state=42
        )

        rf_reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        rf_reg.fit(X_train, yg_train)
        goals_mae = mean_absolute_error(yg_test, rf_reg.predict(X_test))

        xgb_cls = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                 random_state=42, eval_metric="logloss", verbosity=0)
        xgb_cal = CalibratedClassifierCV(xgb_cls, cv=5)
        xgb_cal.fit(X_train, yo_train)
        over25_acc = accuracy_score(yo_test, xgb_cal.predict(X_test))

        logger.info(f"Goals MAE: {goals_mae:.2f}, Over2.5 Accuracy: {over25_acc:.3f}")
        self._save("goals_regressor", rf_reg)
        self._save("over25_classifier", xgb_cal)
        self._save("goals_features", feat_cols)
        self.models["goals"] = rf_reg
        self.models["over25"] = xgb_cal
        return {"goals_mae": goals_mae, "over25_accuracy": over25_acc}

    def train_corners(self, df: pd.DataFrame) -> dict:
        logger.info("Training corners + throw-ins model...")
        results = {}

        for target_col, model_name in CORNER_TARGETS.items():
            sub = df.dropna(subset=[target_col])
            feat_cols = self._get_available_features(sub, CORNERS_FEATURES)
            # Only drop rows where ALL selected features are null
            sub = sub.dropna(subset=feat_cols, how='all')

            if len(sub) < 10:
                logger.warning(f"Not enough data for {target_col} ({len(sub)} rows) — skipping")
                results[target_col] = {"status": "skipped", "rows": len(sub)}
                continue

            X = sub[feat_cols].fillna(0)
            y = sub[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                  random_state=42, verbosity=0)
            model.fit(X_train, y_train)
            mae = mean_absolute_error(y_test, model.predict(X_test))

            logger.info(f"{target_col} MAE: {mae:.2f} ({len(sub)} rows)")
            self._save(model_name, model)
            self._save(f"{model_name}_features", feat_cols)
            self.models[model_name] = model
            results[target_col] = {"mae": mae, "rows": len(sub)}

        return results
