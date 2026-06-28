"""
Model Trainer v2 - Corners-focused with Elo, rest days, probability classifiers
"""
import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report,
                              mean_absolute_error, brier_score_loss)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

BASE_FEATURES = [
    "home_elo", "away_elo", "elo_diff", "elo_expected_home",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "home_congestion_14d", "away_congestion_14d",
    "home_form_wins", "home_form_draws", "home_form_losses",
    "home_form_goals_scored", "home_form_goals_conceded", "home_form_points",
    "home_form_clean_sheets",
    "away_form_wins", "away_form_draws", "away_form_losses",
    "away_form_goals_scored", "away_form_goals_conceded", "away_form_points",
    "away_form_clean_sheets",
    "home_season_avg_scored", "home_season_avg_conceded",
    "home_season_win_rate", "home_season_draw_rate",
    "away_season_avg_scored", "away_season_avg_conceded",
    "away_season_win_rate", "away_season_draw_rate",
    "home_attack_vs_away_defence", "away_attack_vs_home_defence",
    "form_points_diff", "season_win_rate_diff",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_home_goals", "h2h_away_goals", "h2h_matches", "h2h_avg_total_goals",
    "is_premier_league", "is_championship", "is_la_liga",
]

CORNER_FEATURES = BASE_FEATURES + [
    "home_avg_corners_for", "home_avg_corners_against",
    "home_avg_corners_total", "home_avg_corners_ht",
    "home_corner_rate_over_9", "home_corner_rate_over_10",
    "home_corner_rate_over_11", "home_std_corners",
    "away_avg_corners_for", "away_avg_corners_against",
    "away_avg_corners_total", "away_avg_corners_ht",
    "away_corner_rate_over_9", "away_corner_rate_over_10",
    "away_corner_rate_over_11", "away_std_corners",
    "expected_total_corners",
    "combined_corner_rate_over_9", "combined_corner_rate_over_10",
    "referee_avg_corners", "referee_avg_yellows",
]


class ModelTrainer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def _save(self, name, obj):
        path = os.path.join(self.model_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved {name}")

    def _get_features(self, df, feature_list):
        return [f for f in feature_list if f in df.columns]

    def _time_series_split(self, df, test_size=0.2):
        df = df.sort_values("match_date").reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))
        return df.iloc[:split_idx], df.iloc[split_idx:]

    def train_outcome(self, df):
        logger.info("Training outcome model (H/D/A) with Elo + rest days...")
        feat_cols = self._get_features(df, BASE_FEATURES)
        df = df.dropna(subset=["result"]).sort_values("match_date")

        train_df, test_df = self._time_series_split(df)
        X_train = train_df[feat_cols].fillna(0)
        X_test = test_df[feat_cols].fillna(0)
        y_train = train_df["result"]
        y_test = test_df["result"]

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        class_counts = pd.Series(y_train_enc).value_counts()
        total = len(y_train_enc)
        n_classes = len(class_counts)
        class_weights = {cls: total / (n_classes * count) for cls, count in class_counts.items()}
        sample_weights = pd.Series(y_train_enc).map(class_weights).values.astype('float32')
        logger.info(f"Class weights: {class_weights}")

        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="mlogloss", verbosity=0
        )
        xgb_cal = CalibratedClassifierCV(xgb, cv=3, method="sigmoid")
        xgb_cal.fit(X_train, y_train_enc, sample_weight=sample_weights)

        preds_enc = xgb_cal.predict(X_test)
        preds = le.inverse_transform(preds_enc)
        acc = accuracy_score(y_test, preds)

        logger.info(f"Outcome accuracy (chronological): {acc:.3f}")
        logger.info(f"\n{classification_report(y_test, preds)}")

        self._save("outcome_model", {"model": xgb_cal, "encoder": le})
        self._save("outcome_features", feat_cols)
        self._save("outcome_classes", list(le.classes_))

        return {"accuracy": acc, "train_size": len(train_df), "test_size": len(test_df)}

    def train_goals(self, df):
        logger.info("Training goals model...")
        feat_cols = self._get_features(df, BASE_FEATURES)
        df = df.dropna(subset=["total_goals"]).sort_values("match_date")

        train_df, test_df = self._time_series_split(df)
        X_train = train_df[feat_cols].fillna(0)
        X_test = test_df[feat_cols].fillna(0)
        y_goals_train = train_df["total_goals"]
        y_goals_test = test_df["total_goals"]
        y_over25_train = (train_df["total_goals"] > 2.5).astype(int)
        y_over25_test = (test_df["total_goals"] > 2.5).astype(int)

        goals_reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  random_state=42, verbosity=0)
        goals_reg.fit(X_train, y_goals_train)
        goals_mae = mean_absolute_error(y_goals_test, goals_reg.predict(X_test))

        over25_cls = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    random_state=42, eval_metric="logloss", verbosity=0)
        over25_cal = CalibratedClassifierCV(over25_cls, cv=3, method="sigmoid")
        over25_cal.fit(X_train, y_over25_train)
        over25_preds = over25_cal.predict(X_test)
        over25_acc = accuracy_score(y_over25_test, over25_preds)

        over25_probs = over25_cal.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_over25_test, over25_probs)

        logger.info(f"Goals MAE: {goals_mae:.2f}, Over2.5 Acc: {over25_acc:.3f}, Brier: {brier:.3f}")

        self._save("goals_regressor", goals_reg)
        self._save("over25_classifier", over25_cal)
        self._save("goals_features", feat_cols)

        return {"goals_mae": goals_mae, "over25_accuracy": over25_acc, "brier_score": brier}

    def train_corners(self, df):
        logger.info("Training corners models - regression + probability classifiers...")
        results = {}

        for target_col, model_name in [
            ("total_corners_ft", "corners_ft_model"),
            ("total_corners_ht", "corners_ht_model"),
            ("total_corners_2h", "corners_2h_model"),
        ]:
            sub = df.dropna(subset=[target_col]).sort_values("match_date")
            available_feats = self._get_features(sub, CORNER_FEATURES)

            if len(sub) < 20:
                logger.warning(f"Not enough data for {target_col} ({len(sub)} rows) - skipping")
                results[target_col] = {"status": "skipped"}
                continue

            train_df, test_df = self._time_series_split(sub)
            X_train = train_df[available_feats].fillna(0)
            X_test = test_df[available_feats].fillna(0)
            y_train = train_df[target_col]
            y_test = test_df[target_col]

            model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  random_state=42, verbosity=0)
            model.fit(X_train, y_train)
            mae = mean_absolute_error(y_test, model.predict(X_test))

            logger.info(f"{target_col} MAE: {mae:.2f} ({len(sub)} rows)")
            self._save(model_name, model)
            self._save(f"{model_name}_features", available_feats)
            results[target_col] = {"mae": mae, "rows": len(sub)}

        for target_col, model_name, threshold in [
            ("over_8_5_corners", "corners_over_8_5_model", 8.5),
            ("over_9_5_corners", "corners_over_9_5_model", 9.5),
            ("over_10_5_corners", "corners_over_10_5_model", 10.5),
        ]:
            sub = df.dropna(subset=["total_corners_ft"]).sort_values("match_date").copy()
            sub[target_col] = (sub["total_corners_ft"] > threshold).astype(int)
            available_feats = self._get_features(sub, CORNER_FEATURES)

            if len(sub) < 30:
                logger.warning(f"Not enough data for {target_col} - skipping")
                continue

            train_df, test_df = self._time_series_split(sub)
            X_train = train_df[available_feats].fillna(0)
            X_test = test_df[available_feats].fillna(0)
            y_train = train_df[target_col]
            y_test = test_df[target_col]

            base_rate = y_train.mean()
            logger.info(f"{target_col} base rate: {base_rate:.3f}")

            if base_rate == 0 or base_rate == 1:
                logger.warning(f"{target_col} has no variance - skipping")
                continue

            xgb = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                scale_pos_weight=(1 - base_rate) / base_rate,
                random_state=42, eval_metric="logloss", verbosity=0
            )
            xgb_cal = CalibratedClassifierCV(xgb, cv=3, method="sigmoid")
            xgb_cal.fit(X_train, y_train)

            preds = xgb_cal.predict(X_test)
            probs = xgb_cal.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, preds)
            brier = brier_score_loss(y_test, probs)

            logger.info(f"{target_col} - Acc: {acc:.3f}, Brier: {brier:.3f}")

            self._save(model_name, xgb_cal)
            self._save(f"{model_name}_features", available_feats)
            results[target_col] = {"accuracy": acc, "brier": brier, "rows": len(sub)}

        try:
            sub = df.dropna(subset=["total_corners_ft"]).sort_values("match_date")
            available_feats = self._get_features(sub, CORNER_FEATURES)
            X = sub[available_feats].fillna(0)
            y = sub["total_corners_ft"]
            importance_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            importance_model.fit(X, y)
            importances = pd.Series(
                importance_model.feature_importances_,
                index=available_feats
            ).sort_values(ascending=False).head(15)
            logger.info(f"\nTop 15 corner features:\n{importances.to_string()}")
        except Exception as e:
            logger.warning(f"Feature importance failed: {e}")

        return results
