#!/usr/bin/env python3
"""
Day 4 — Train ML Models
Loads the feature matrix and trains all prediction models.

Usage:
    python scripts/train_models.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import pandas as pd
    from ml.training.trainer import ModelTrainer

    # Load feature matrix
    features_path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
    if not os.path.exists(features_path):
        logger.error("features.csv not found. Run build_features.py first.")
        sys.exit(1)

    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Model directory
    model_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "ml", "saved_models")
    trainer = ModelTrainer(model_dir)

    print("\n" + "="*55)
    print("TRAINING ML MODELS")
    print("="*55)

    # 1. Outcome model
    print("\n[1/3] Training Outcome Model (Home/Draw/Away)...")
    outcome_results = trainer.train_outcome(df)
    print(f"  ✅ Best accuracy: {outcome_results['best_accuracy']*100:.1f}%")
    print(f"  RF: {outcome_results['rf_accuracy']*100:.1f}%  XGB: {outcome_results['xgb_accuracy']*100:.1f}%")

    # 2. Goals model
    print("\n[2/3] Training Goals Model (total goals + over/under 2.5)...")
    goals_results = trainer.train_goals(df)
    print(f"  ✅ Goals MAE: {goals_results['goals_mae']:.2f} goals")
    print(f"  ✅ Over 2.5 accuracy: {goals_results['over25_accuracy']*100:.1f}%")

    # 3. Corners + throw-ins model
    print("\n[3/3] Training Corners + Throw-ins Models...")
    corners_results = trainer.train_corners(df)
    for target, result in corners_results.items():
        if result.get("status") == "skipped":
            print(f"  ⏭  {target}: skipped (only {result['rows']} rows, need 10+)")
        else:
            print(f"  ✅ {target}: MAE {result['mae']:.2f} ({result['rows']} rows)")

    print("\n" + "="*55)
    print("✅ All models trained and saved to:")
    print(f"   {model_dir}")
    print("="*55)
    print("\nNext step: Run the prediction API (Day 5)")


if __name__ == "__main__":
    main()
