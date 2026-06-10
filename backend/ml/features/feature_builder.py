"""
Day 3 — Feature Engineering
Reads matches from PostgreSQL and builds the feature matrix for ML training.
Includes: outcome, goals, corners (HT/FT), throw-ins features.
"""
import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class FeatureBuilder:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def load_matches(self) -> pd.DataFrame:
        query = """
            SELECT
                m.id, m.api_id, m.match_date, m.league_id, m.league_name,
                m.season, m.status, m.home_team_id, m.away_team_id,
                ht.name AS home_team, at.name AS away_team,
                m.home_score, m.away_score,
                m.home_score_ht, m.away_score_ht
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.status = 'FT'
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
            ORDER BY m.match_date ASC
        """
        df = pd.read_sql(query, self.engine)
        df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
        df["total_goals"] = df["home_score"] + df["away_score"]
        df["result"] = df.apply(
            lambda r: "H" if r.home_score > r.away_score
            else ("A" if r.home_score < r.away_score else "D"), axis=1
        )
        logger.info(f"Loaded {len(df)} finished matches")
        return df

    def load_stats(self) -> pd.DataFrame:
        """Load per-match team statistics including corners and throw-ins."""
        query = """
            SELECT
                ms.match_id, ms.team_id, ms.is_home,
                ms.corners_ht, ms.corners_ft, ms.corners_total,
                ms.throw_ins,
                ms.shots_total, ms.shots_on_target,
                ms.possession, ms.pass_accuracy
            FROM match_stats ms
        """
        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df)} match stat rows")
        return df

    def compute_team_stats_history(self, stats_df: pd.DataFrame, matches_df: pd.DataFrame,
                                    team_id: int, before_date, n: int = 10) -> dict:
        """Compute average corners and throw-ins for a team from recent matches."""
        mask = (
            ((matches_df["home_team_id"] == team_id) | (matches_df["away_team_id"] == team_id)) &
            (matches_df["match_date"] < before_date)
        )
        recent_matches = matches_df[mask].tail(n)

        if len(recent_matches) == 0:
            return {
                "avg_corners_ht": None, "avg_corners_ft": None,
                "avg_corners_total": None, "avg_throw_ins": None,
            }

        match_ids = recent_matches["id"].tolist()
        team_stats = stats_df[
            (stats_df["match_id"].isin(match_ids)) &
            (stats_df["team_id"] == team_id)
        ]

        if len(team_stats) == 0:
            return {
                "avg_corners_ht": None, "avg_corners_ft": None,
                "avg_corners_total": None, "avg_throw_ins": None,
            }

        return {
            "avg_corners_ht": team_stats["corners_ht"].mean(),
            "avg_corners_ft": team_stats["corners_ft"].mean(),
            "avg_corners_total": team_stats["corners_total"].mean(),
            "avg_throw_ins": team_stats["throw_ins"].mean(),
        }

    def compute_match_corners(self, stats_df: pd.DataFrame, match_id: int) -> dict:
        """Get actual corners and throw-ins for a finished match (used as targets)."""
        match_stats = stats_df[stats_df["match_id"] == match_id]
        if len(match_stats) == 0:
            return {
                "total_corners_ht": None, "total_corners_2h": None,
                "total_corners_ft": None, "total_throw_ins": None,
            }

        corners_ht = match_stats["corners_ht"].sum()
        corners_ft = match_stats["corners_total"].sum()  # free plan: use corners_total as FT
        corners_total = match_stats["corners_total"].sum()
        throw_ins = match_stats["throw_ins"].sum()

        # corners_total = full game corners (free plan)
        # corners_ht, corners_ft, throw_ins = paid plan only
        total_ft = corners_total if not pd.isna(corners_total) else None
        total_ht = corners_ht if not pd.isna(corners_ht) else None
        total_2h = (corners_total - corners_ht) if (not pd.isna(corners_total) and not pd.isna(corners_ht)) else None
        total_ti = throw_ins if not pd.isna(throw_ins) else None
        return {
            "total_corners_ht": total_ht,
            "total_corners_2h": total_2h,
            "total_corners_ft": total_ft,
            "total_throw_ins": total_ti,
        }

    def compute_form(self, df: pd.DataFrame, team_id: int, before_date, n: int = 5) -> dict:
        mask = (
            ((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)) &
            (df["match_date"] < before_date)
        )
        recent = df[mask].tail(n)

        if len(recent) == 0:
            return {
                "form_wins": 0, "form_draws": 0, "form_losses": 0,
                "form_goals_scored": 0, "form_goals_conceded": 0,
                "form_points": 0,
            }

        wins = draws = losses = goals_scored = goals_conceded = 0
        for _, row in recent.iterrows():
            if row["home_team_id"] == team_id:
                gf, ga = row["home_score"], row["away_score"]
            else:
                gf, ga = row["away_score"], row["home_score"]
            goals_scored += gf
            goals_conceded += ga
            if gf > ga: wins += 1
            elif gf == ga: draws += 1
            else: losses += 1

        return {
            "form_wins": wins, "form_draws": draws, "form_losses": losses,
            "form_goals_scored": goals_scored, "form_goals_conceded": goals_conceded,
            "form_points": wins * 3 + draws,
        }

    def compute_season_stats(self, df: pd.DataFrame, team_id: int, before_date) -> dict:
        mask = (
            ((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)) &
            (df["match_date"] < before_date)
        )
        played = df[mask]

        if len(played) == 0:
            return {
                "season_avg_scored": 0, "season_avg_conceded": 0,
                "season_win_rate": 0, "season_draw_rate": 0,
                "season_home_avg_scored": 0, "season_away_avg_scored": 0,
            }

        gf_list, ga_list, results = [], [], []
        home_gf, away_gf = [], []

        for _, row in played.iterrows():
            if row["home_team_id"] == team_id:
                gf, ga = row["home_score"], row["away_score"]
                home_gf.append(gf)
            else:
                gf, ga = row["away_score"], row["home_score"]
                away_gf.append(gf)
            gf_list.append(gf)
            ga_list.append(ga)
            res = "H" if gf > ga else ("D" if gf == ga else "A")
            results.append(res)

        n = len(played)
        return {
            "season_avg_scored": np.mean(gf_list),
            "season_avg_conceded": np.mean(ga_list),
            "season_win_rate": results.count("H") / n,
            "season_draw_rate": results.count("D") / n,
            "season_home_avg_scored": np.mean(home_gf) if home_gf else 0,
            "season_away_avg_scored": np.mean(away_gf) if away_gf else 0,
        }

    def compute_h2h(self, df: pd.DataFrame, home_id: int, away_id: int, before_date, n: int = 5) -> dict:
        mask = (
            (
                ((df["home_team_id"] == home_id) & (df["away_team_id"] == away_id)) |
                ((df["home_team_id"] == away_id) & (df["away_team_id"] == home_id))
            ) &
            (df["match_date"] < before_date)
        )
        h2h = df[mask].tail(n)

        if len(h2h) == 0:
            return {
                "h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0,
                "h2h_home_goals": 0, "h2h_away_goals": 0, "h2h_matches": 0,
            }

        home_wins = draws = away_wins = home_goals = away_goals = 0
        for _, row in h2h.iterrows():
            if row["home_team_id"] == home_id:
                hg, ag = row["home_score"], row["away_score"]
            else:
                hg, ag = row["away_score"], row["home_score"]
            home_goals += hg
            away_goals += ag
            if hg > ag: home_wins += 1
            elif hg == ag: draws += 1
            else: away_wins += 1

        return {
            "h2h_home_wins": home_wins, "h2h_draws": draws, "h2h_away_wins": away_wins,
            "h2h_home_goals": home_goals, "h2h_away_goals": away_goals,
            "h2h_matches": len(h2h),
        }

    def build_feature_matrix(self, df: pd.DataFrame, stats_df: pd.DataFrame = None) -> pd.DataFrame:
        logger.info("Building feature matrix...")
        if stats_df is None:
            stats_df = pd.DataFrame()

        rows = []
        for _, match in df.iterrows():
            date = match["match_date"]
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]

            home_form = self.compute_form(df, home_id, date)
            home_season = self.compute_season_stats(df, home_id, date)
            away_form = self.compute_form(df, away_id, date)
            away_season = self.compute_season_stats(df, away_id, date)
            h2h = self.compute_h2h(df, home_id, away_id, date)

            # Corner/throw-in history features
            if len(stats_df) > 0:
                home_corner_stats = self.compute_team_stats_history(stats_df, df, home_id, date)
                away_corner_stats = self.compute_team_stats_history(stats_df, df, away_id, date)
                match_corners = self.compute_match_corners(stats_df, match["id"])
            else:
                home_corner_stats = {"avg_corners_ht": None, "avg_corners_ft": None, "avg_corners_total": None, "avg_throw_ins": None}
                away_corner_stats = {"avg_corners_ht": None, "avg_corners_ft": None, "avg_corners_total": None, "avg_throw_ins": None}
                match_corners = {"total_corners_ht": None, "total_corners_2h": None, "total_corners_ft": None, "total_throw_ins": None}

            row = {
                # Identifiers
                "match_id": match["id"],
                "match_date": date,
                "league_id": match["league_id"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],

                # ── TARGET VARIABLES ──────────────────────────
                "result": match["result"],
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "total_goals": match["total_goals"],
                "total_corners_ht": match_corners["total_corners_ht"],
                "total_corners_2h": match_corners["total_corners_2h"],
                "total_corners_ft": match_corners["total_corners_ft"],
                "total_throw_ins": match_corners["total_throw_ins"],

                # ── HOME FORM FEATURES ────────────────────────
                "home_form_wins": home_form["form_wins"],
                "home_form_draws": home_form["form_draws"],
                "home_form_losses": home_form["form_losses"],
                "home_form_goals_scored": home_form["form_goals_scored"],
                "home_form_goals_conceded": home_form["form_goals_conceded"],
                "home_form_points": home_form["form_points"],

                # ── AWAY FORM FEATURES ────────────────────────
                "away_form_wins": away_form["form_wins"],
                "away_form_draws": away_form["form_draws"],
                "away_form_losses": away_form["form_losses"],
                "away_form_goals_scored": away_form["form_goals_scored"],
                "away_form_goals_conceded": away_form["form_goals_conceded"],
                "away_form_points": away_form["form_points"],

                # ── HOME SEASON STATS ─────────────────────────
                "home_season_avg_scored": home_season["season_avg_scored"],
                "home_season_avg_conceded": home_season["season_avg_conceded"],
                "home_season_win_rate": home_season["season_win_rate"],
                "home_season_draw_rate": home_season["season_draw_rate"],
                "home_season_home_avg_scored": home_season["season_home_avg_scored"],

                # ── AWAY SEASON STATS ─────────────────────────
                "away_season_avg_scored": away_season["season_avg_scored"],
                "away_season_avg_conceded": away_season["season_avg_conceded"],
                "away_season_win_rate": away_season["season_win_rate"],
                "away_season_draw_rate": away_season["season_draw_rate"],
                "away_season_away_avg_scored": away_season["season_away_avg_scored"],

                # ── DERIVED STRENGTH FEATURES ─────────────────
                "home_attack_vs_away_defence": home_season["season_avg_scored"] - away_season["season_avg_conceded"],
                "away_attack_vs_home_defence": away_season["season_avg_scored"] - home_season["season_avg_conceded"],
                "form_points_diff": home_form["form_points"] - away_form["form_points"],
                "season_win_rate_diff": home_season["season_win_rate"] - away_season["season_win_rate"],

                # ── H2H FEATURES ──────────────────────────────
                "h2h_home_wins": h2h["h2h_home_wins"],
                "h2h_draws": h2h["h2h_draws"],
                "h2h_away_wins": h2h["h2h_away_wins"],
                "h2h_home_goals": h2h["h2h_home_goals"],
                "h2h_away_goals": h2h["h2h_away_goals"],
                "h2h_matches": h2h["h2h_matches"],

                # ── CORNER HISTORY FEATURES ───────────────────
                "home_avg_corners_ht": home_corner_stats["avg_corners_ht"],
                "home_avg_corners_ft": home_corner_stats["avg_corners_ft"],
                "home_avg_corners_total": home_corner_stats["avg_corners_total"],
                "home_avg_throw_ins": home_corner_stats["avg_throw_ins"],
                "away_avg_corners_ht": away_corner_stats["avg_corners_ht"],
                "away_avg_corners_ft": away_corner_stats["avg_corners_ft"],
                "away_avg_corners_total": away_corner_stats["avg_corners_total"],
                "away_avg_throw_ins": away_corner_stats["avg_throw_ins"],

                # ── COMBINED CORNER PREDICTIONS ───────────────
                "expected_total_corners_ht": (
                    (home_corner_stats["avg_corners_ht"] or 0) +
                    (away_corner_stats["avg_corners_ht"] or 0)
                ),
                "expected_total_corners_ft": (
                    (home_corner_stats["avg_corners_total"] or 0) +
                    (away_corner_stats["avg_corners_total"] or 0)
                ),
                "expected_total_throw_ins": (
                    (home_corner_stats["avg_throw_ins"] or 0) +
                    (away_corner_stats["avg_throw_ins"] or 0)
                ),
            }
            rows.append(row)

        feature_df = pd.DataFrame(rows)
        logger.info(f"Feature matrix built: {len(feature_df)} rows, {len(feature_df.columns)} columns")
        return feature_df

    def save_features(self, df: pd.DataFrame, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")

    def print_summary(self, df: pd.DataFrame) -> None:
        print("\n" + "="*55)
        print("FEATURE MATRIX SUMMARY")
        print("="*55)
        print(f"Total matches:     {len(df)}")
        print(f"Total features:    {len(df.columns)}")
        print(f"Date range:        {df['match_date'].min()} → {df['match_date'].max()}")
        print(f"\nResult distribution:")
        print(df["result"].value_counts().to_string())
        print(f"\nAvg total goals:       {df['total_goals'].mean():.2f}")
        print(f"Over 2.5 goals:        {(df['total_goals'] > 2.5).mean()*100:.1f}%")
        has_corners = df["total_corners_ft"].notna().sum()
        print(f"\nMatches with corner data: {has_corners}/{len(df)}")
        if has_corners > 0:
            print(f"Avg corners FT:        {df['total_corners_ft'].mean():.1f}")
            print(f"Avg corners HT:        {df['total_corners_ht'].mean():.1f}")
            print(f"Avg throw-ins:         {df['total_throw_ins'].mean():.1f}")
        print("="*55)
