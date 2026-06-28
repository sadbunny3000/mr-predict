"""
Tennis Feature Builder — Elo, Serve Stats, Fatigue, Head-to-Head
------------------------------------------------------------------
Walks every match chronologically, computing pre-match-only features.
Nothing here ever uses information from the match being predicted.
"""

import logging
from collections import deque
from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)

STARTING_ELO = 1500.0
K_FACTOR = 32.0


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating, expected, actual, k=K_FACTOR):
    return rating + k * (actual - expected)


def safe_div(num, denom):
    if not denom:
        return None
    return num / denom


class TennisEloBuilder:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.overall_elo = {}
        self.surface_elo = {}
        self.match_count = {}
        self.serve_totals = {}
        self.match_dates = {}          # player_id -> deque of past match dates
        self.h2h_wins = {}              # (player_a, player_b) -> wins for player_a over player_b

    def _get_overall(self, pid):
        return self.overall_elo.get(pid, STARTING_ELO)

    def _get_surface(self, pid, surface):
        return self.surface_elo.get((pid, surface), STARTING_ELO)

    def _get_match_count(self, pid):
        return self.match_count.get(pid, 0)

    def _get_serve_totals(self, pid):
        return self.serve_totals.setdefault(pid, {
            "ace": 0, "df": 0, "svpt": 0, "first_in": 0,
            "first_won": 0, "second_won": 0, "bp_saved": 0, "bp_faced": 0,
        })

    def _get_serve_rates_pre(self, pid):
        t = self._get_serve_totals(pid)
        second_in = t["svpt"] - t["first_in"]
        return {
            "ace_rate": safe_div(t["ace"], t["svpt"]),
            "first_in_pct": safe_div(t["first_in"], t["svpt"]),
            "first_won_pct": safe_div(t["first_won"], t["first_in"]),
            "second_won_pct": safe_div(t["second_won"], second_in),
            "bp_saved_pct": safe_div(t["bp_saved"], t["bp_faced"]),
        }

    def _accumulate_serve_stats(self, pid, ace, df, svpt, first_in, first_won, second_won, bp_saved, bp_faced):
        t = self._get_serve_totals(pid)
        for key, val in [
            ("ace", ace), ("df", df), ("svpt", svpt), ("first_in", first_in),
            ("first_won", first_won), ("second_won", second_won),
            ("bp_saved", bp_saved), ("bp_faced", bp_faced),
        ]:
            if val is not None:
                t[key] += val

    def _get_fatigue_pre(self, pid, current_date: date):
        history = self.match_dates.get(pid, deque())
        if not history:
            return {"days_since_last": None, "matches_last14": 0}
        days_since_last = (current_date - history[-1]).days
        matches_last14 = sum(1 for d in history if (current_date - d).days <= 14)
        return {"days_since_last": days_since_last, "matches_last14": matches_last14}

    def _record_match_date(self, pid, match_date: date):
        self.match_dates.setdefault(pid, deque(maxlen=30)).append(match_date)

    def _get_h2h_wins_pre(self, pid, opponent_id):
        return self.h2h_wins.get((pid, opponent_id), 0)

    def _record_h2h_result(self, winner_id, loser_id):
        key = (winner_id, loser_id)
        self.h2h_wins[key] = self.h2h_wins.get(key, 0) + 1

    async def build(self) -> int:
        result = await self.session.execute(
            text("""
                SELECT id, winner_id, loser_id, surface, tourney_date, match_num,
                       w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced,
                       l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced
                FROM tennis_matches
                ORDER BY tourney_date ASC, match_num ASC
            """)
        )
        rows = result.fetchall()

        updated = 0
        for row in rows:
            (match_id, winner_id, loser_id, surface, tourney_date, match_num,
             w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced,
             l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced) = row

            surface_key = surface if surface in ("Hard", "Clay", "Grass", "Carpet") else "Unknown"

            winner_elo_pre = self._get_overall(winner_id)
            loser_elo_pre = self._get_overall(loser_id)
            winner_surface_elo_pre = self._get_surface(winner_id, surface_key)
            loser_surface_elo_pre = self._get_surface(loser_id, surface_key)
            winner_total_matches = self._get_match_count(winner_id)
            loser_total_matches = self._get_match_count(loser_id)
            w_rates = self._get_serve_rates_pre(winner_id)
            l_rates = self._get_serve_rates_pre(loser_id)
            w_fatigue = self._get_fatigue_pre(winner_id, tourney_date)
            l_fatigue = self._get_fatigue_pre(loser_id, tourney_date)
            winner_h2h = self._get_h2h_wins_pre(winner_id, loser_id)
            loser_h2h = self._get_h2h_wins_pre(loser_id, winner_id)

            await self.session.execute(
                text("""
                    UPDATE tennis_matches
                    SET winner_elo_pre = :w_elo, loser_elo_pre = :l_elo,
                        winner_surface_elo_pre = :w_surf_elo, loser_surface_elo_pre = :l_surf_elo,
                        winner_total_matches_pre = :w_matches, loser_total_matches_pre = :l_matches,
                        winner_ace_rate_pre = :w_ace_rate, loser_ace_rate_pre = :l_ace_rate,
                        winner_1st_in_pct_pre = :w_1st_in, loser_1st_in_pct_pre = :l_1st_in,
                        winner_1st_won_pct_pre = :w_1st_won, loser_1st_won_pct_pre = :l_1st_won,
                        winner_2nd_won_pct_pre = :w_2nd_won, loser_2nd_won_pct_pre = :l_2nd_won,
                        winner_bp_saved_pct_pre = :w_bp_saved, loser_bp_saved_pct_pre = :l_bp_saved,
                        winner_days_since_last_pre = :w_days, loser_days_since_last_pre = :l_days,
                        winner_matches_last14_pre = :w_last14, loser_matches_last14_pre = :l_last14,
                        winner_h2h_wins_pre = :w_h2h, loser_h2h_wins_pre = :l_h2h
                    WHERE id = :match_id
                """),
                {
                    "w_elo": winner_elo_pre, "l_elo": loser_elo_pre,
                    "w_surf_elo": winner_surface_elo_pre, "l_surf_elo": loser_surface_elo_pre,
                    "w_matches": winner_total_matches, "l_matches": loser_total_matches,
                    "w_ace_rate": w_rates["ace_rate"], "l_ace_rate": l_rates["ace_rate"],
                    "w_1st_in": w_rates["first_in_pct"], "l_1st_in": l_rates["first_in_pct"],
                    "w_1st_won": w_rates["first_won_pct"], "l_1st_won": l_rates["first_won_pct"],
                    "w_2nd_won": w_rates["second_won_pct"], "l_2nd_won": l_rates["second_won_pct"],
                    "w_bp_saved": w_rates["bp_saved_pct"], "l_bp_saved": l_rates["bp_saved_pct"],
                    "w_days": w_fatigue["days_since_last"], "l_days": l_fatigue["days_since_last"],
                    "w_last14": w_fatigue["matches_last14"], "l_last14": l_fatigue["matches_last14"],
                    "w_h2h": winner_h2h, "l_h2h": loser_h2h,
                    "match_id": match_id,
                },
            )

            expected_winner = expected_score(winner_elo_pre, loser_elo_pre)
            self.overall_elo[winner_id] = update_elo(winner_elo_pre, expected_winner, 1.0)
            self.overall_elo[loser_id] = update_elo(loser_elo_pre, 1.0 - expected_winner, 0.0)

            expected_winner_surf = expected_score(winner_surface_elo_pre, loser_surface_elo_pre)
            self.surface_elo[(winner_id, surface_key)] = update_elo(winner_surface_elo_pre, expected_winner_surf, 1.0)
            self.surface_elo[(loser_id, surface_key)] = update_elo(loser_surface_elo_pre, 1.0 - expected_winner_surf, 0.0)

            self.match_count[winner_id] = winner_total_matches + 1
            self.match_count[loser_id] = loser_total_matches + 1

            self._accumulate_serve_stats(winner_id, w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced)
            self._accumulate_serve_stats(loser_id, l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced)

            self._record_match_date(winner_id, tourney_date)
            self._record_match_date(loser_id, tourney_date)
            self._record_h2h_result(winner_id, loser_id)

            updated += 1
            if updated % 5000 == 0:
                print(f"Processed {updated}/{len(rows)} matches")

        await self.session.commit()
        print(f"Full feature build complete: {updated} matches processed")
        return updated