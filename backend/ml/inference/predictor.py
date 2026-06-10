"""
Prediction engine — loads trained models and generates predictions for a match.
"""
import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = {}
        self.features = {}
        self._load_models()

    def _load(self, name: str):
        path = os.path.join(self.model_dir, f"{name}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_models(self):
        model_names = [
            "outcome_model", "outcome_features", "outcome_classes",
            "goals_regressor", "over25_classifier", "goals_features",
            "corners_ht_model", "corners_ht_model_features",
            "corners_ft_model", "corners_ft_model_features",
            "corners_2h_model", "corners_2h_model_features",
            "throw_ins_model", "throw_ins_model_features",
        ]
        for name in model_names:
            obj = self._load(name)
            if obj is not None:
                self.models[name] = obj
        logger.info(f"Loaded models: {[k for k in self.models if not k.endswith('features') and not k.endswith('classes')]}")

    def predict(self, features: dict) -> dict:
        """
        Generate full prediction for a match.
        features: dict of feature name → value (same as feature matrix columns)
        Returns dict with all predictions and probabilities.
        """
        result = {}

        # ── Outcome prediction ────────────────────────────────
        if "outcome_model" in self.models:
            feat_cols = self.models.get("outcome_features", [])
            X = pd.DataFrame([{f: features.get(f, 0) for f in feat_cols}])
            probs = self.models["outcome_model"].predict_proba(X)[0]
            classes = self.models["outcome_classes"]
            prob_map = dict(zip(classes, probs))

            result["home_win_prob"] = round(float(prob_map.get("H", 0)), 4)
            result["draw_prob"] = round(float(prob_map.get("D", 0)), 4)
            result["away_win_prob"] = round(float(prob_map.get("A", 0)), 4)
            result["predicted_outcome"] = max(prob_map, key=prob_map.get)
            result["outcome_confidence"] = round(float(max(probs)), 4)

        # ── Goals prediction ──────────────────────────────────
        if "goals_regressor" in self.models:
            feat_cols = self.models.get("goals_features", [])
            X = pd.DataFrame([{f: features.get(f, 0) for f in feat_cols}])
            total_goals = float(self.models["goals_regressor"].predict(X)[0])
            result["total_goals_pred"] = round(total_goals, 2)

        if "over25_classifier" in self.models:
            feat_cols = self.models.get("goals_features", [])
            X = pd.DataFrame([{f: features.get(f, 0) for f in feat_cols}])
            over25_prob = float(self.models["over25_classifier"].predict_proba(X)[0][1])
            result["over_25_prob"] = round(over25_prob, 4)
            result["under_25_prob"] = round(1 - over25_prob, 4)

        # ── Corners HT prediction ─────────────────────────────
        if "corners_ht_model" in self.models:
            feat_cols = self.models.get("corners_ht_model_features", [])
            X = pd.DataFrame([{f: features.get(f, 0) for f in feat_cols}])
            result["corners_ht_pred"] = round(float(self.models["corners_ht_model"].predict(X)[0]), 1)

        # ── Corners 2H prediction ─────────────────────────────
        if "corners_2h_model" in self.models:
            feat_cols = self.models.get("corners_2h_model_features", [])
            X = pd.DataFrame([{f: features.get(f, 0) for f in feat_cols}])
            result["corners_2h_pred"] = round(float(self.models["corners_2h_model"].predict(X)[0]), 1)

        # ── Corners FT prediction ─────────────────────────────
        if "corners_ft_model" in self.models:
            feat_cols = self.models.get("corners_ft_model_features", [])
            X = pd.DataFrame([{f: features.get(f, 0) for f in feat_cols}])
            result["corners_ft_pred"] = round(float(self.models["corners_ft_model"].predict(X)[0]), 1)
        elif "corners_ht_pred" in result and "corners_2h_pred" in result:
            result["corners_ft_pred"] = round(result["corners_ht_pred"] + result["corners_2h_pred"], 1)

        # ── Throw-ins prediction ──────────────────────────────
        if "throw_ins_model" in self.models:
            feat_cols = self.models.get("throw_ins_model_features", [])
            X = pd.DataFrame([{f: features.get(f, 0) for f in feat_cols}])
            result["throw_ins_pred"] = round(float(self.models["throw_ins_model"].predict(X)[0]), 1)

        return result

    def models_available(self) -> list[str]:
        return [k for k in self.models if not k.endswith("features") and not k.endswith("classes")]
