"""
Rolling-window feature builder for the tennis total-games model.
------------------------------------------------------------------
Replays tennis_matches chronologically, computing PRE-MATCH-ONLY rolling
stats per player (5/10/20-match windows, streaks, 90-day stats, surface
form), then parses the score string to compute the actual total games
played. Writes everything into tennis_match_features using batched inserts.

Callable as build_rolling_features(database_url) from in-app code (see
tennis/features/rebuild.py), or run standalone via:
    DATABASE_PUBLIC_URL=... python -m tennis.features.rolling_features
"""
import os
import re
import sys
from collections import deque
import psycopg2
import psycopg2.extras

BATCH_SIZE = 500


def parse_total_games(score):
    if not score:
        return None, False
    if any(tag in score.upper() for tag in ['RET', 'W/O', 'WO', 'DEF', 'ABN']):
        return None, False
    sets = score.strip().split()
    total = 0
    found_any = False
    for s in sets:
        s = re.sub(r'\(\d+\)', '', s)
        m = re.match(r'^(\d+)-(\d+)$', s)
        if not m:
            continue
        g1, g2 = int(m.group(1)), int(m.group(2))
        total += g1 + g2
        found_any = True
    if not found_any:
        return None, False
    return total, True


def safe_div(num, denom):
    if not denom:
        return None
    return num / denom


class PlayerHistory:
    def __init__(self):
        self.matches = deque(maxlen=50)

    def _recent(self, n):
        return list(self.matches)[-n:] if self.matches else []

    def _mean(self, values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def get_features_pre(self, current_date):
        last5  = self._recent(5)
        last10 = self._recent(10)
        last20 = self._recent(20)

        roll5_ace      = self._mean([m[3] for m in last5])
        roll5_1stwon   = self._mean([m[4] for m in last5])
        roll5_2ndwon   = self._mean([m[5] for m in last5])
        roll5_bpsaved  = self._mean([m[6] for m in last5])

        roll10_ace     = self._mean([m[3] for m in last10])
        roll10_1stwon  = self._mean([m[4] for m in last10])
        roll10_2ndwon  = self._mean([m[5] for m in last10])
        roll10_bpsaved = self._mean([m[6] for m in last10])
        roll10_winrate = self._mean([1.0 if m[2] else 0.0 for m in last10])

        roll20_bpsaved = self._mean([m[6] for m in last20])

        streak = 0
        for m in reversed(self.matches):
            if m[2]:
                streak += 1
            else:
                break

        wins_last10 = sum(1 for m in last10 if m[2])

        d90 = [m for m in self.matches if (current_date - m[0]).days <= 90]
        d90_ace    = self._mean([m[3] for m in d90])
        d90_1stwon = self._mean([m[4] for m in d90])

        return {
            'roll5_ace': roll5_ace, 'roll5_1stwon': roll5_1stwon,
            'roll5_2ndwon': roll5_2ndwon, 'roll5_bpsaved': roll5_bpsaved,
            'roll10_ace': roll10_ace, 'roll10_1stwon': roll10_1stwon,
            'roll10_2ndwon': roll10_2ndwon, 'roll10_bpsaved': roll10_bpsaved,
            'roll10_winrate': roll10_winrate, 'roll20_bpsaved': roll20_bpsaved,
            'win_streak': streak, 'wins_last10': wins_last10,
            'd90_ace': d90_ace, 'd90_1stwon': d90_1stwon,
        }

    def get_surf_form10(self, surface):
        surf_matches = [m for m in self.matches if m[1] == surface]
        last10_surf = surf_matches[-10:] if surf_matches else []
        if not last10_surf:
            return None
        return sum(1.0 if m[2] else 0.0 for m in last10_surf) / len(last10_surf)

    def record(self, date, surface, won, ace_rate, first_won_pct, second_won_pct, bp_saved_pct):
        self.matches.append((date, surface, won, ace_rate, first_won_pct, second_won_pct, bp_saved_pct))


INSERT_SQL = """
    INSERT INTO tennis_match_features (
        match_id,
        winner_roll5_ace, loser_roll5_ace,
        winner_roll5_1stwon, loser_roll5_1stwon,
        winner_roll5_2ndwon, loser_roll5_2ndwon,
        winner_roll5_bpsaved, loser_roll5_bpsaved,
        winner_roll10_ace, loser_roll10_ace,
        winner_roll10_1stwon, loser_roll10_1stwon,
        winner_roll10_2ndwon, loser_roll10_2ndwon,
        winner_roll10_bpsaved, loser_roll10_bpsaved,
        winner_roll10_winrate, loser_roll10_winrate,
        winner_roll20_bpsaved, loser_roll20_bpsaved,
        winner_win_streak, loser_win_streak,
        winner_wins_last10, loser_wins_last10,
        winner_d90_ace, loser_d90_ace,
        winner_d90_1stwon, loser_d90_1stwon,
        winner_surf_form10, loser_surf_form10,
        total_games, target_valid
    ) VALUES %s
    ON CONFLICT (match_id) DO UPDATE SET
        winner_roll5_ace = EXCLUDED.winner_roll5_ace,
        loser_roll5_ace = EXCLUDED.loser_roll5_ace,
        winner_roll5_1stwon = EXCLUDED.winner_roll5_1stwon,
        loser_roll5_1stwon = EXCLUDED.loser_roll5_1stwon,
        winner_roll5_2ndwon = EXCLUDED.winner_roll5_2ndwon,
        loser_roll5_2ndwon = EXCLUDED.loser_roll5_2ndwon,
        winner_roll5_bpsaved = EXCLUDED.winner_roll5_bpsaved,
        loser_roll5_bpsaved = EXCLUDED.loser_roll5_bpsaved,
        winner_roll10_ace = EXCLUDED.winner_roll10_ace,
        loser_roll10_ace = EXCLUDED.loser_roll10_ace,
        winner_roll10_1stwon = EXCLUDED.winner_roll10_1stwon,
        loser_roll10_1stwon = EXCLUDED.loser_roll10_1stwon,
        winner_roll10_2ndwon = EXCLUDED.winner_roll10_2ndwon,
        loser_roll10_2ndwon = EXCLUDED.loser_roll10_2ndwon,
        winner_roll10_bpsaved = EXCLUDED.winner_roll10_bpsaved,
        loser_roll10_bpsaved = EXCLUDED.loser_roll10_bpsaved,
        winner_roll10_winrate = EXCLUDED.winner_roll10_winrate,
        loser_roll10_winrate = EXCLUDED.loser_roll10_winrate,
        winner_roll20_bpsaved = EXCLUDED.winner_roll20_bpsaved,
        loser_roll20_bpsaved = EXCLUDED.loser_roll20_bpsaved,
        winner_win_streak = EXCLUDED.winner_win_streak,
        loser_win_streak = EXCLUDED.loser_win_streak,
        winner_wins_last10 = EXCLUDED.winner_wins_last10,
        loser_wins_last10 = EXCLUDED.loser_wins_last10,
        winner_d90_ace = EXCLUDED.winner_d90_ace,
        loser_d90_ace = EXCLUDED.loser_d90_ace,
        winner_d90_1stwon = EXCLUDED.winner_d90_1stwon,
        loser_d90_1stwon = EXCLUDED.loser_d90_1stwon,
        winner_surf_form10 = EXCLUDED.winner_surf_form10,
        loser_surf_form10 = EXCLUDED.loser_surf_form10,
        total_games = EXCLUDED.total_games,
        target_valid = EXCLUDED.target_valid
"""


def build_rolling_features(database_url: str) -> dict:
    """Replays all tennis_matches chronologically and rebuilds tennis_match_features.
    Safe to re-run — ON CONFLICT upserts every row. Returns a summary dict.
    Blocking/synchronous (psycopg2) — call via asyncio.to_thread() from async code."""
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    read_cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    read_cur.execute("""
        SELECT id, winner_id, loser_id, surface, tourney_date, score, best_of, match_num,
               w_ace, w_svpt, w_1stin, w_1stwon, w_2ndwon, w_bpsaved, w_bpfaced,
               l_ace, l_svpt, l_1stin, l_1stwon, l_2ndwon, l_bpsaved, l_bpfaced
        FROM tennis_matches
        ORDER BY tourney_date ASC, match_num ASC
    """)
    rows = read_cur.fetchall()
    total_rows = len(rows)

    histories = {}

    def get_hist(pid):
        return histories.setdefault(pid, PlayerHistory())

    batch = []
    processed = 0

    try:
        for row in rows:
            match_id      = row['id']
            winner_id     = row['winner_id']
            loser_id      = row['loser_id']
            surface       = row['surface'] or 'Unknown'
            tourney_date  = row['tourney_date']
            score         = row['score']

            total_games, valid = parse_total_games(score)

            w_hist = get_hist(winner_id)
            l_hist = get_hist(loser_id)

            w_feat = w_hist.get_features_pre(tourney_date)
            l_feat = l_hist.get_features_pre(tourney_date)
            w_surf_form10 = w_hist.get_surf_form10(surface)
            l_surf_form10 = l_hist.get_surf_form10(surface)

            batch.append((
                match_id,
                w_feat['roll5_ace'], l_feat['roll5_ace'],
                w_feat['roll5_1stwon'], l_feat['roll5_1stwon'],
                w_feat['roll5_2ndwon'], l_feat['roll5_2ndwon'],
                w_feat['roll5_bpsaved'], l_feat['roll5_bpsaved'],
                w_feat['roll10_ace'], l_feat['roll10_ace'],
                w_feat['roll10_1stwon'], l_feat['roll10_1stwon'],
                w_feat['roll10_2ndwon'], l_feat['roll10_2ndwon'],
                w_feat['roll10_bpsaved'], l_feat['roll10_bpsaved'],
                w_feat['roll10_winrate'], l_feat['roll10_winrate'],
                w_feat['roll20_bpsaved'], l_feat['roll20_bpsaved'],
                w_feat['win_streak'], l_feat['win_streak'],
                w_feat['wins_last10'], l_feat['wins_last10'],
                w_feat['d90_ace'], l_feat['d90_ace'],
                w_feat['d90_1stwon'], l_feat['d90_1stwon'],
                w_surf_form10, l_surf_form10,
                total_games, valid,
            ))

            w_ace_rate    = safe_div(row['w_ace'], row['w_svpt'])
            w_1stwon_pct  = safe_div(row['w_1stwon'], row['w_1stin'])
            w_2ndwon_pct  = safe_div(row['w_2ndwon'], (row['w_svpt'] or 0) - (row['w_1stin'] or 0))
            w_bpsaved_pct = safe_div(row['w_bpsaved'], row['w_bpfaced'])
            l_ace_rate    = safe_div(row['l_ace'], row['l_svpt'])
            l_1stwon_pct  = safe_div(row['l_1stwon'], row['l_1stin'])
            l_2ndwon_pct  = safe_div(row['l_2ndwon'], (row['l_svpt'] or 0) - (row['l_1stin'] or 0))
            l_bpsaved_pct = safe_div(row['l_bpsaved'], row['l_bpfaced'])

            w_hist.record(tourney_date, surface, True,  w_ace_rate, w_1stwon_pct, w_2ndwon_pct, w_bpsaved_pct)
            l_hist.record(tourney_date, surface, False, l_ace_rate, l_1stwon_pct, l_2ndwon_pct, l_bpsaved_pct)

            processed += 1

            if len(batch) >= BATCH_SIZE:
                psycopg2.extras.execute_values(cur, INSERT_SQL, batch)
                conn.commit()
                batch = []

        if batch:
            psycopg2.extras.execute_values(cur, INSERT_SQL, batch)
            conn.commit()

        return {"total_matches": total_rows, "processed": processed}
    finally:
        cur.close()
        read_cur.close()
        conn.close()


if __name__ == '__main__':
    db_url = os.environ.get('DATABASE_PUBLIC_URL')
    if not db_url:
        print("ERROR: set DATABASE_PUBLIC_URL environment variable before running.")
        sys.exit(1)
    result = build_rolling_features(db_url)
    print(f"Done. Processed {result['processed']}/{result['total_matches']} matches.")
