"""
Feature Engineering v2 — Corners-focused with Elo, rest days, probability outputs
"""
import logging
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class EloSystem:
    """Dynamic Elo rating system updated after each match."""

    def __init__(self, k=32, default=1500):
        self.k = k
        self.default = default
        self.ratings = {}

    def get(self, team_id):
        return self.ratings.get(team_id, self.default)

    def expected(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, home_id, away_id, home_score, away_score):
        home_rating = self.get(home_id)
        away_rating = self.get(away_id)

        exp_home = self.expected(home_rating, away_rating)
        exp_away = 1 - exp_home

        if home_score > away_score:
            actual_home, actual_away = 1.0, 0.0
        elif home_score == away_score:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        self.ratings[home_id] = home_rating + self.k * (actual_home - exp_home)
        self.ratings[away_id] = away_rating + self.k * (actual_away - exp_away)


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
                m.home_score_ht, m.away_score_ht,
                m.referee
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.status = \'FT\'
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
        query = """
            SELECT
                ms.match_id, ms.team_id, ms.is_home,
                ms.corners_ht, ms.corners_ft, ms.corners_total,
                ms.throw_ins,
                ms.shots_total, ms.shots_on_target,
                ms.possession, ms.pass_accuracy,
                ms.fouls, ms.yellow_cards, ms.red_cards
            FROM match_stats ms
        """
        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df)} match stat rows")
        return df

    def compute_rest_days(self, df, team_id, before_date):
        mask = (
            ((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)) &
            (df["match_date"] < before_date)
        )
        recent = df[mask].tail(1)
        if len(recent) == 0:
            return 7
        last_match = recent["match_date"].iloc[0]
        delta = before_date - last_match
        return int(delta.days)

    def compute_fixture_congestion(self, df, team_id, before_date, days=14):
        cutoff = before_date - pd.Timedelta(days=days)
        mask = (
            ((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)) &
            (df["match_date"] >= cutoff) &
            (df["match_date"] < before_date)
        )
        return len(df[mask])

    def compute_form(self, df, team_id, before_date, n=5):
        mask = (
            ((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)) &
            (df["match_date"] < before_date)
        )
        recent = df[mask].tail(n)
        if len(recent) == 0:
            return {
                "form_wins": 0, "form_draws": 0, "form_losses": 0,
                "form_goals_scored": 0, "form_goals_conceded": 0,
                "form_points": 0, "form_clean_sheets": 0,
            }
        wins = draws = losses = goals_scored = goals_conceded = clean_sheets = 0
        for _, row in recent.iterrows():
            if row["home_team_id"] == team_id:
                gf, ga = row["home_score"], row["away_score"]
            else:
                gf, ga = row["away_score"], row["home_score"]
            goals_scored += gf
            goals_conceded += ga
            if ga == 0:
                clean_sheets += 1
            if gf > ga: wins += 1
            elif gf == ga: draws += 1
            else: losses += 1
        return {
            "form_wins": wins, "form_draws": draws, "form_losses": losses,
            "form_goals_scored": goals_scored, "form_goals_conceded": goals_conceded,
            "form_points": wins * 3 + draws, "form_clean_sheets": clean_sheets,
        }

    def compute_corner_form(self, stats_df, df, team_id, before_date, n=8):
        mask = (
            ((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)) &
            (df["match_date"] < before_date)
        )
        recent_matches = df[mask].tail(n)
        if len(recent_matches) == 0:
            return {
                "avg_corners_for": 0, "avg_corners_against": 0,
                "avg_corners_total": 0, "avg_corners_ht": 0,
                "corner_rate_over_9": 0, "corner_rate_over_10": 0,
                "corner_rate_over_11": 0, "std_corners": 0,
            }

        match_ids = recent_matches["id"].tolist()
        team_stats = stats_df[
            (stats_df["match_id"].isin(match_ids)) &
            (stats_df["team_id"] == team_id)
        ]
        opp_stats = stats_df[
            (stats_df["match_id"].isin(match_ids)) &
            (stats_df["team_id"] != team_id) &
            (stats_df["match_id"].isin(team_stats["match_id"]))
        ]

        corners_for = team_stats["corners_total"].dropna().tolist()
        corners_against = opp_stats["corners_total"].dropna().tolist()
        corners_ht = team_stats["corners_ht"].dropna().tolist()

        match_totals = []
        for mid in match_ids:
            m_stats = stats_df[stats_df["match_id"] == mid]
            total = m_stats["corners_total"].sum()
            if total > 0:
                match_totals.append(total)

        if not match_totals:
            return {
                "avg_corners_for": np.mean(corners_for) if corners_for else 0,
                "avg_corners_against": np.mean(corners_against) if corners_against else 0,
                "avg_corners_total": 0, "avg_corners_ht": np.mean(corners_ht) if corners_ht else 0,
                "corner_rate_over_9": 0, "corner_rate_over_10": 0,
                "corner_rate_over_11": 0, "std_corners": 0,
            }

        return {
            "avg_corners_for": np.mean(corners_for) if corners_for else 0,
            "avg_corners_against": np.mean(corners_against) if corners_against else 0,
            "avg_corners_total": np.mean(match_totals),
            "avg_corners_ht": np.mean(corners_ht) if corners_ht else 0,
            "corner_rate_over_9": np.mean([1 if t > 9 else 0 for t in match_totals]),
            "corner_rate_over_10": np.mean([1 if t > 10 else 0 for t in match_totals]),
            "corner_rate_over_11": np.mean([1 if t > 11 else 0 for t in match_totals]),
            "std_corners": np.std(match_totals) if len(match_totals) > 1 else 0,
        }

    def compute_referee_stats(self, df, stats_df, referee, before_date):
        if not referee or len(stats_df) == 0:
            return {"referee_avg_corners": 10.0, "referee_avg_yellows": 3.5}

        ref_matches = df[
            (df["referee"] == referee) &
            (df["match_date"] < before_date)
        ].tail(20)

        if len(ref_matches) == 0:
            return {"referee_avg_corners": 10.0, "referee_avg_yellows": 3.5}

        match_ids = ref_matches["id"].tolist()
        ref_stats = stats_df[stats_df["match_id"].isin(match_ids)]

        corner_totals = []
        for mid in match_ids:
            m_stats = ref_stats[ref_stats["match_id"] == mid]
            total = m_stats["corners_total"].sum()
            if total > 0:
                corner_totals.append(total)

        avg_yellows = ref_stats["yellow_cards"].mean() if len(ref_stats) > 0 else 3.5

        return {
            "referee_avg_corners": np.mean(corner_totals) if corner_totals else 10.0,
            "referee_avg_yellows": float(avg_yellows) if not pd.isna(avg_yellows) else 3.5,
        }

    def compute_season_stats(self, df, team_id, before_date):
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

    def compute_h2h(self, df, home_id, away_id, before_date, n=5):
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
                "h2h_avg_total_goals": 0,
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
        n_matches = len(h2h)
        return {
            "h2h_home_wins": home_wins, "h2h_draws": draws, "h2h_away_wins": away_wins,
            "h2h_home_goals": home_goals, "h2h_away_goals": away_goals,
            "h2h_matches": n_matches,
            "h2h_avg_total_goals": (home_goals + away_goals) / n_matches,
        }

    def compute_corner_targets(self, stats_df, match_id):
        match_stats = stats_df[stats_df["match_id"] == match_id]
        if len(match_stats) == 0:
            return {
                "total_corners_ft": None, "total_corners_ht": None,
                "total_corners_2h": None, "total_throw_ins": None,
                "over_8_5_corners": None, "over_9_5_corners": None,
                "over_10_5_corners": None,
            }
        corners_ft = match_stats["corners_total"].sum()
        corners_ht = match_stats["corners_ht"].sum()
        throw_ins = match_stats["throw_ins"].sum()

        corners_ft = float(corners_ft) if not pd.isna(corners_ft) else None
        corners_ht = float(corners_ht) if not pd.isna(corners_ht) else None
        corners_2h = (corners_ft - corners_ht) if (corners_ft and corners_ht) else None

        return {
            "total_corners_ft": corners_ft,
            "total_corners_ht": corners_ht,
            "total_corners_2h": corners_2h,
            "total_throw_ins": float(throw_ins) if not pd.isna(throw_ins) else None,
            "over_8_5_corners": int(corners_ft > 8.5) if corners_ft else None,
            "over_9_5_corners": int(corners_ft > 9.5) if corners_ft else None,
            "over_10_5_corners": int(corners_ft > 10.5) if corners_ft else None,
        }

    def build_feature_matrix(self, df, stats_df=None):
        logger.info("Building feature matrix v2 with Elo + rest days + corner probabilities...")
        if stats_df is None:
            stats_df = pd.DataFrame()

        elo = EloSystem(k=32, default=1500)
        elo_ratings_before = {}

        sorted_df = df.sort_values("match_date")
        for _, match in sorted_df.iterrows():
            mid = match["id"]
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]
            elo_ratings_before[mid] = (elo.get(home_id), elo.get(away_id))
            if match["home_score"] is not None and match["away_score"] is not None:
                elo.update(home_id, away_id, match["home_score"], match["away_score"])

        rows = []
        for _, match in df.iterrows():
            date = match["match_date"]
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]
            mid = match["id"]

            home_form = self.compute_form(df, home_id, date)
            away_form = self.compute_form(df, away_id, date)
            home_season = self.compute_season_stats(df, home_id, date)
            away_season = self.compute_season_stats(df, away_id, date)
            h2h = self.compute_h2h(df, home_id, away_id, date)

            home_elo, away_elo = elo_ratings_before.get(mid, (1500, 1500))
            elo_diff = home_elo - away_elo
            elo_expected_home = 1 / (1 + 10 ** (-elo_diff / 400))

            home_rest_days = self.compute_rest_days(df, home_id, date)
            away_rest_days = self.compute_rest_days(df, away_id, date)
            home_congestion = self.compute_fixture_congestion(df, home_id, date)
            away_congestion = self.compute_fixture_congestion(df, away_id, date)

            if len(stats_df) > 0:
                home_cf = self.compute_corner_form(stats_df, df, home_id, date)
                away_cf = self.compute_corner_form(stats_df, df, away_id, date)
                referee = match.get("referee", "")
                ref_stats = self.compute_referee_stats(df, stats_df, referee, date)
                corner_targets = self.compute_corner_targets(stats_df, mid)
            else:
                home_cf = away_cf = {k: 0 for k in ["avg_corners_for", "avg_corners_against",
                    "avg_corners_total", "avg_corners_ht", "corner_rate_over_9",
                    "corner_rate_over_10", "corner_rate_over_11", "std_corners"]}
                ref_stats = {"referee_avg_corners": 10.0, "referee_avg_yellows": 3.5}
                corner_targets = {k: None for k in ["total_corners_ft", "total_corners_ht",
                    "total_corners_2h", "total_throw_ins", "over_8_5_corners",
                    "over_9_5_corners", "over_10_5_corners"]}

            expected_corners = (
                home_cf["avg_corners_for"] + away_cf["avg_corners_for"] +
                home_cf["avg_corners_against"] + away_cf["avg_corners_against"]
            ) / 2

            row = {
                "match_id": mid,
                "match_date": date,
                "league_id": match["league_id"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "result": match["result"],
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "total_goals": match["total_goals"],
                **corner_targets,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": elo_diff,
                "elo_expected_home": elo_expected_home,
                "home_rest_days": home_rest_days,
                "away_rest_days": away_rest_days,
                "rest_days_diff": home_rest_days - away_rest_days,
                "home_congestion_14d": home_congestion,
                "away_congestion_14d": away_congestion,
                "home_form_wins": home_form["form_wins"],
                "home_form_draws": home_form["form_draws"],
                "home_form_losses": home_form["form_losses"],
                "home_form_goals_scored": home_form["form_goals_scored"],
                "home_form_goals_conceded": home_form["form_goals_conceded"],
                "home_form_points": home_form["form_points"],
                "home_form_clean_sheets": home_form["form_clean_sheets"],
                "away_form_wins": away_form["form_wins"],
                "away_form_draws": away_form["form_draws"],
                "away_form_losses": away_form["form_losses"],
                "away_form_goals_scored": away_form["form_goals_scored"],
                "away_form_goals_conceded": away_form["form_goals_conceded"],
                "away_form_points": away_form["form_points"],
                "away_form_clean_sheets": away_form["form_clean_sheets"],
                "home_season_avg_scored": home_season["season_avg_scored"],
                "home_season_avg_conceded": home_season["season_avg_conceded"],
                "home_season_win_rate": home_season["season_win_rate"],
                "home_season_draw_rate": home_season["season_draw_rate"],
                "home_season_home_avg_scored": home_season["season_home_avg_scored"],
                "away_season_avg_scored": away_season["season_avg_scored"],
                "away_season_avg_conceded": away_season["season_avg_conceded"],
                "away_season_win_rate": away_season["season_win_rate"],
                "away_season_draw_rate": away_season["season_draw_rate"],
                "away_season_away_avg_scored": away_season["season_away_avg_scored"],
                "home_attack_vs_away_defence": home_season["season_avg_scored"] - away_season["season_avg_conceded"],
                "away_attack_vs_home_defence": away_season["season_avg_scored"] - home_season["season_avg_conceded"],
                "form_points_diff": home_form["form_points"] - away_form["form_points"],
                "season_win_rate_diff": home_season["season_win_rate"] - away_season["season_win_rate"],
                "h2h_home_wins": h2h["h2h_home_wins"],
                "h2h_draws": h2h["h2h_draws"],
                "h2h_away_wins": h2h["h2h_away_wins"],
                "h2h_home_goals": h2h["h2h_home_goals"],
                "h2h_away_goals": h2h["h2h_away_goals"],
                "h2h_matches": h2h["h2h_matches"],
                "h2h_avg_total_goals": h2h["h2h_avg_total_goals"],
                "home_avg_corners_for": home_cf["avg_corners_for"],
                "home_avg_corners_against": home_cf["avg_corners_against"],
                "home_avg_corners_total": home_cf["avg_corners_total"],
                "home_avg_corners_ht": home_cf["avg_corners_ht"],
                "home_corner_rate_over_9": home_cf["corner_rate_over_9"],
                "home_corner_rate_over_10": home_cf["corner_rate_over_10"],
                "home_corner_rate_over_11": home_cf["corner_rate_over_11"],
                "home_std_corners": home_cf["std_corners"],
                "away_avg_corners_for": away_cf["avg_corners_for"],
                "away_avg_corners_against": away_cf["avg_corners_against"],
                "away_avg_corners_total": away_cf["avg_corners_total"],
                "away_avg_corners_ht": away_cf["avg_corners_ht"],
                "away_corner_rate_over_9": away_cf["corner_rate_over_9"],
                "away_corner_rate_over_10": away_cf["corner_rate_over_10"],
                "away_corner_rate_over_11": away_cf["corner_rate_over_11"],
                "away_std_corners": away_cf["std_corners"],
                "expected_total_corners": expected_corners,
                "combined_corner_rate_over_9": (home_cf["corner_rate_over_9"] + away_cf["corner_rate_over_9"]) / 2,
                "combined_corner_rate_over_10": (home_cf["corner_rate_over_10"] + away_cf["corner_rate_over_10"]) / 2,
                "referee_avg_corners": ref_stats["referee_avg_corners"],
                "referee_avg_yellows": ref_stats["referee_avg_yellows"],
                "is_premier_league": int(match["league_id"] == 39),
                "is_championship": int(match["league_id"] == 40),
                "is_la_liga": int(match["league_id"] == 140),
            }
            rows.append(row)

        feature_df = pd.DataFrame(rows)
        logger.info(f"Feature matrix v2: {len(feature_df)} rows, {len(feature_df.columns)} columns")
        return feature_df

    def save_features(self, df, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")

    def print_summary(self, df):
        print("=" * 60)
        print("FEATURE MATRIX v2 SUMMARY")
        print("=" * 60)
        print("Total matches:     " + str(len(df)))
        print("Total features:    " + str(len(df.columns)))
        print("Date range:        " + str(df["match_date"].min()) + " to " + str(df["match_date"].max()))
        print("Result distribution:")
        print(df["result"].value_counts().to_string())
        print("Avg total goals:   " + str(round(df["total_goals"].mean(), 2)))
        print("Over 2.5 goals:    " + str(round((df["total_goals"] > 2.5).mean() * 100, 1)) + "%")
        has_corners = df["total_corners_ft"].notna().sum()
        print("Matches with corner data: " + str(has_corners))
        if has_corners > 0:
            valid = df[df["total_corners_ft"].notna()]
            print("Avg corners FT:    " + str(round(valid["total_corners_ft"].mean(), 1)))
            print("Over 9.5 corners:  " + str(round((valid["total_corners_ft"] > 9.5).mean() * 100, 1)) + "%")
            print("Over 10.5 corners: " + str(round((valid["total_corners_ft"] > 10.5).mean() * 100, 1)) + "%")
        print("Avg home Elo:      " + str(round(df["home_elo"].mean(), 0)))
        print("Avg home rest days:" + str(round(df["home_rest_days"].mean(), 1)))
        print("=" * 60)