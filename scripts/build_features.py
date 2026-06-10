#!/usr/bin/env python3
"""
Day 3 — Build Feature Matrix
Reads matches + stats from PostgreSQL and generates features for ML training.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    db_url = os.getenv("DATABASE_URL_SYNC")
    if not db_url:
        logger.error("DATABASE_URL_SYNC not set in .env")
        sys.exit(1)

    from ml.features.feature_builder import FeatureBuilder
    builder = FeatureBuilder(db_url)

    matches_df = builder.load_matches()
    if len(matches_df) == 0:
        logger.error("No finished matches found. Run ingest.py first.")
        sys.exit(1)

    # Load stats (corners, throw-ins) — may be empty if not yet ingested
    stats_df = builder.load_stats()

    features_df = builder.build_feature_matrix(matches_df, stats_df)

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
    builder.save_features(features_df, output_path)
    builder.print_summary(features_df)
    logger.info("✅ Feature engineering complete")


if __name__ == "__main__":
    main()
