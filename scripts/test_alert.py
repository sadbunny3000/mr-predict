#!/usr/bin/env python3
"""
Test the alert format with a sample prediction.
Run this after training to see what alerts look like.

Usage:
    python scripts/test_alert.py
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.alert_service import build_alert, format_telegram_message


def main():
    # Sample prediction (as would come from the ML model)
    prediction = {
        "home_win_prob": 0.58,
        "draw_prob": 0.22,
        "away_win_prob": 0.20,
        "predicted_outcome": "H",
        "outcome_confidence": 0.58,
        "total_goals_pred": 2.8,
        "over_25_prob": 0.62,
        "under_25_prob": 0.38,
        "corners_ht_pred": 4.2,
        "corners_2h_pred": 5.8,
        "corners_ft_pred": 10.0,
        "throw_ins_pred": 28.5,
    }

    # Sample bookmaker odds (as would come from Odds API)
    odds_by_bookmaker = {
        "castlebet": {
            "home": 2.20,
            "draw": 3.40,
            "away": 3.10,
            "over25": 1.85,
            "under25": 1.95,
            "corners_ft_over": 1.80,
            "corners_ht_over": 1.75,
        },
        "easybetnam": {
            "home": 2.15,
            "draw": 3.30,
            "away": 3.20,
            "over25": 1.90,
            "corners_ft_over": 1.85,
        },
        "williamhill": {
            "home": 2.25,
            "draw": 3.50,
            "away": 3.00,
            "over25": 1.83,
            "corners_ft_over": 1.95,
            "corners_ht_over": 1.80,
        },
        "1xbet": {
            "home": 2.30,
            "draw": 3.60,
            "away": 3.10,
            "over25": 1.95,
            "corners_ft_over": 2.00,
            "corners_ht_over": 1.90,
            "corners_2h_over": 1.85,
        },
    }

    alert = build_alert(
        home_team="Arsenal",
        away_team="Chelsea",
        match_date=datetime(2026, 6, 14, 15, 0, tzinfo=timezone.utc),
        league="Premier League",
        prediction=prediction,
        odds_by_bookmaker=odds_by_bookmaker,
    )

    if alert:
        message = format_telegram_message(alert, prediction)
        print("\n" + "="*55)
        print("SAMPLE TELEGRAM ALERT:")
        print("="*55)
        print(message)
        print("="*55)
    else:
        print("No value found in this sample — adjust odds or thresholds.")


if __name__ == "__main__":
    main()
