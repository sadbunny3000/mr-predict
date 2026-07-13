"""
Quantifies the impact of the placeholder-vs-real rolling feature bug.
Loads the current production total-games model and generates predictions
on the same 2023+ holdout matches using:
  (a) REAL rolling features (from tennis_match_features)
  (b) PLACEHOLDER features (matching what live inference currently does)
Then compares both against actual total games played.
"""
import os
import pickle
import random
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ.get('DATABASE_PUBLIC_URL')
if not DATABASE_URL:
    raise SystemExit("ERROR: set DATABASE_PUBLIC_URL first")

MODEL_PATH = 'backend/ml/saved_models/tennis_total_games_model_v2.pkl'

SURFACE_MAP = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
LEVEL_MAP   = {'G': 4, 'M': 3, 'A': 2, 'F': 1, 'D': 0}
ROUND_MAP   = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1, 'RR': 3}


def load_data():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT tm.id AS match_id, tm.winner_id, tm.loser_id, tm.surface, tm.tourney_level,
               tm.round, tm.best_of, tm.tourney_date, tm.winner_rank, tm.loser_rank,
               tm.winner_elo_pre, tm.loser_elo_pre,
               tm.winner_surface_elo_pre, tm.loser_surface_elo_pre,
               tm.winner_ace_rate_pre, tm.loser_ace_rate_pre,
               tm.winner_1st_in_pct_pre, tm.loser_1st_in_pct_pre,
               tm.winner_1st_won_pct_pre, tm.loser_1st_won_pct_pre,
               tm.winner_2nd_won_pct_pre, tm.loser_2nd_won_pct_pre,
               tm.winner_bp_saved_pct_pre, tm.loser_bp_saved_pct_pre,
               mf.winner_roll5_bpsaved, mf.loser_roll5_bpsaved,
               mf.winner_roll10_ace, mf.loser_roll10_ace,
               mf.winner_roll10_1stwon, mf.loser_roll10_1stwon,
               mf.winner_roll10_2ndwon, mf.loser_roll10_2ndwon,
               mf.winner_roll10_bpsaved, mf.loser_roll10_bpsaved,
               mf.winner_roll10_winrate, mf.loser_roll10_winrate,
               mf.winner_roll20_bpsaved, mf.loser_roll20_bpsaved,
               mf.winner_win_streak, mf.loser_win_streak,
               mf.winner_wins_last10, mf.loser_wins_last10,
               mf.winner_d90_ace, mf.loser_d90_ace,
               mf.winner_d90_1stwon, mf.loser_d90_1stwon,
               mf.winner_surf_form10, mf.loser_surf_form10,
               mf.total_games, mf.target_valid
        FROM tennis_matches tm
        JOIN tennis_match_features mf ON mf.match_id = tm.id
        WHERE mf.target_valid = TRUE
          AND tm.tourney_date >= '2023-01-01'
        ORDER BY tm.tourney_date ASC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def build_feature_row(row, rng_state, use_real):
    p1_is_winner = rng_state.random() < 0.5

    def pick(w_val, l_val):
        return (w_val, l_val) if p1_is_winner else (l_val, w_val)

    p1_elo, p2_elo = pick(row['winner_elo_pre'], row['loser_elo_pre'])
    p1_surf_elo, p2_surf_elo = pick(row['winner_surface_elo_pre'], row['loser_surface_elo_pre'])
    p1_rank, p2_rank = pick(row['winner_rank'], row['loser_rank'])
    p1_ace, p2_ace = pick(row['winner_ace_rate_pre'], row['loser_ace_rate_pre'])
    p1_1stin, p2_1stin = pick(row['winner_1st_in_pct_pre'], row['loser_1st_in_pct_pre'])
    p1_1stwon, p2_1stwon = pick(row['winner_1st_won_pct_pre'], row['loser_1st_won_pct_pre'])
    p1_2ndwon, p2_2ndwon = pick(row['winner_2nd_won_pct_pre'], row['loser_2nd_won_pct_pre'])
    p1_bpsaved, p2_bpsaved = pick(row['winner_bp_saved_pct_pre'], row['loser_bp_saved_pct_pre'])

    p1_elo = float(p1_elo) if p1_elo is not None else 1500.0
    p2_elo = float(p2_elo) if p2_elo is not None else 1500.0
    p1_surf_elo = float(p1_surf_elo) if p1_surf_elo is not None else 1500.0
    p2_surf_elo = float(p2_surf_elo) if p2_surf_elo is not None else 1500.0
    p1_rank = p1_rank or 100
    p2_rank = p2_rank or 100
    p1_ace = float(p1_ace) if p1_ace is not None else 0.06
    p2_ace = float(p2_ace) if p2_ace is not None else 0.06
    p1_1stin = float(p1_1stin) if p1_1stin is not None else 0.60
    p2_1stin = float(p2_1stin) if p2_1stin is not None else 0.60
    p1_1stwon = float(p1_1stwon) if p1_1stwon is not None else 0.70
    p2_1stwon = float(p2_1stwon) if p2_1stwon is not None else 0.70
    p1_2ndwon = float(p1_2ndwon) if p1_2ndwon is not None else 0.50
    p2_2ndwon = float(p2_2ndwon) if p2_2ndwon is not None else 0.50
    p1_bpsaved = float(p1_bpsaved) if p1_bpsaved is not None else 0.60
    p2_bpsaved = float(p2_bpsaved) if p2_bpsaved is not None else 0.60

    elo_diff = p1_elo - p2_elo
    surface_elo_diff = p1_surf_elo - p2_surf_elo
    elo_mismatch = abs(elo_diff)
    rank_diff = abs(p1_rank - p2_rank)
    elo_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))

    avg_ace = (p1_ace + p2_ace) / 2
    avg_1stin = (p1_1stin + p2_1stin) / 2
    avg_1stwon = (p1_1stwon + p2_1stwon) / 2
    avg_2ndwon = (p1_2ndwon + p2_2ndwon) / 2
    avg_bpsv = (p1_bpsaved + p2_bpsaved) / 2
    serve_dom = avg_1stwon * avg_1stin + avg_2ndwon * (1 - avg_1stin)

    surface = row['surface'] or 'Hard'
    surf_enc = SURFACE_MAP.get(surface, 2)
    level_enc = LEVEL_MAP.get(row['tourney_level'], 4)
    round_enc = ROUND_MAP.get(row['round'], 2)

    if use_real:
        p1_roll10_ace, p2_roll10_ace = pick(row['winner_roll10_ace'], row['loser_roll10_ace'])
        p1_roll10_1stwon, p2_roll10_1stwon = pick(row['winner_roll10_1stwon'], row['loser_roll10_1stwon'])
        p1_roll10_2ndwon, p2_roll10_2ndwon = pick(row['winner_roll10_2ndwon'], row['loser_roll10_2ndwon'])
        p1_roll10_bpsaved, p2_roll10_bpsaved = pick(row['winner_roll10_bpsaved'], row['loser_roll10_bpsaved'])
        p1_roll10_winrate, p2_roll10_winrate = pick(row['winner_roll10_winrate'], row['loser_roll10_winrate'])
        p1_roll20_bpsaved, p2_roll20_bpsaved = pick(row['winner_roll20_bpsaved'], row['loser_roll20_bpsaved'])
        p1_streak, p2_streak = pick(row['winner_win_streak'], row['loser_win_streak'])
        p1_wins10, p2_wins10 = pick(row['winner_wins_last10'], row['loser_wins_last10'])
        p1_d90ace, p2_d90ace = pick(row['winner_d90_ace'], row['loser_d90_ace'])
        p1_d901stwon, p2_d901stwon = pick(row['winner_d90_1stwon'], row['loser_d90_1stwon'])
        p1_surfform, p2_surfform = pick(row['winner_surf_form10'], row['loser_surf_form10'])

        def d(v, default):
            return float(v) if v is not None else default

        roll10_ace_diff     = d(p1_roll10_ace, avg_ace) - d(p2_roll10_ace, avg_ace)
        roll10_1stwon_diff  = d(p1_roll10_1stwon, avg_1stwon) - d(p2_roll10_1stwon, avg_1stwon)
        roll10_2ndwon_diff  = d(p1_roll10_2ndwon, avg_2ndwon) - d(p2_roll10_2ndwon, avg_2ndwon)
        roll10_bpsaved_diff = d(p1_roll10_bpsaved, avg_bpsv) - d(p2_roll10_bpsaved, avg_bpsv)
        roll10_winrate_diff = d(p1_roll10_winrate, 0.5) - d(p2_roll10_winrate, 0.5)
        roll10_ace_avg     = (d(p1_roll10_ace, avg_ace) + d(p2_roll10_ace, avg_ace)) / 2
        roll10_1stwon_avg  = (d(p1_roll10_1stwon, avg_1stwon) + d(p2_roll10_1stwon, avg_1stwon)) / 2
        roll10_2ndwon_avg  = (d(p1_roll10_2ndwon, avg_2ndwon) + d(p2_roll10_2ndwon, avg_2ndwon)) / 2
        roll10_bpsaved_avg = (d(p1_roll10_bpsaved, avg_bpsv) + d(p2_roll10_bpsaved, avg_bpsv)) / 2
        roll20_bpsaved_avg = (d(p1_roll20_bpsaved, avg_bpsv) + d(p2_roll20_bpsaved, avg_bpsv)) / 2
        roll5_bpsaved_avg  = roll10_bpsaved_avg
        win_streak_diff    = (p1_streak or 0) - (p2_streak or 0)
        wins_last10_avg    = ((p1_wins10 if p1_wins10 is not None else 5) + (p2_wins10 if p2_wins10 is not None else 5)) / 2
        d90_ace_avg    = (d(p1_d90ace, avg_ace) + d(p2_d90ace, avg_ace)) / 2
        d90_1stwon_avg = (d(p1_d901stwon, avg_1stwon) + d(p2_d901stwon, avg_1stwon)) / 2
        surf_form10_avg = (d(p1_surfform, 0.5) + d(p2_surfform, 0.5)) / 2
        momentum_x_level = win_streak_diff * level_enc
    else:
        roll10_ace_diff = roll10_1stwon_diff = roll10_2ndwon_diff = 0
        roll10_bpsaved_diff = roll10_winrate_diff = 0
        roll10_ace_avg = avg_ace
        roll10_1stwon_avg = avg_1stwon
        roll10_2ndwon_avg = avg_2ndwon
        roll10_bpsaved_avg = avg_bpsv
        roll20_bpsaved_avg = avg_bpsv
        roll5_bpsaved_avg = avg_bpsv
        win_streak_diff = 0
        wins_last10_avg = 5
        d90_ace_avg = avg_ace
        d90_1stwon_avg = avg_1stwon
        surf_form10_avg = 0.5
        momentum_x_level = 0

    feat_row = {
        'elo_diff': elo_diff, 'surface_elo_diff': surface_elo_diff,
        'elo_mismatch': elo_mismatch, 'rank_diff': rank_diff, 'elo_win_prob': elo_win_prob,
        'surface_enc': surf_enc, 'level_enc': level_enc, 'round_enc': round_enc,
        'roll10_ace_diff': roll10_ace_diff, 'roll10_1stwon_diff': roll10_1stwon_diff,
        'roll10_2ndwon_diff': roll10_2ndwon_diff, 'roll10_bpsaved_diff': roll10_bpsaved_diff,
        'roll10_winrate_diff': roll10_winrate_diff,
        'roll10_ace_avg': roll10_ace_avg, 'roll10_1stwon_avg': roll10_1stwon_avg,
        'roll10_2ndwon_avg': roll10_2ndwon_avg, 'roll10_bpsaved_avg': roll10_bpsaved_avg,
        'roll20_bpsaved_avg': roll20_bpsaved_avg, 'roll5_bpsaved_avg': roll5_bpsaved_avg,
        'win_streak_diff': win_streak_diff, 'wins_last10_avg': wins_last10_avg,
        '90d_ace_avg': d90_ace_avg, '90d_1stwon_avg': d90_1stwon_avg,
        'surf_form10_avg': surf_form10_avg, 'serve_dominance': serve_dom,
        'surface_elo_diff_x_surf': surface_elo_diff * surf_enc,
        'rankdiff_x_level': rank_diff * level_enc,
        'servedom_x_surface': serve_dom * surf_enc,
        'momentum_x_level': momentum_x_level,
        'bpsaved_x_elo_mismatch': avg_bpsv * elo_mismatch,
    }
    return feat_row


def main():
    rows = load_data()
    print(f"Loaded {len(rows)} holdout matches (2023+)")
    with open(MODEL_PATH, 'rb') as f:
        models = pickle.load(f)

    results = {}

    for row in rows:
        surface = row['surface'] or 'Hard'
        best_of = row['best_of'] or 3
        key = f'bo{best_of}_{surface}'
        if key not in models:
            key = 'fallback' if 'fallback' in models else None
        if key is None:
            continue

        sub = models[key]
        qm = sub['quantile_models']
        features = sub['features']

        rng_real = random.Random(row['match_id'])
        real_feat = build_feature_row(row, rng_real, use_real=True)
        rng_fake = random.Random(row['match_id'])
        fake_feat = build_feature_row(row, rng_fake, use_real=False)

        X_real = pd.DataFrame([{f: real_feat.get(f, 0) for f in features}], columns=features)
        X_fake = pd.DataFrame([{f: fake_feat.get(f, 0) for f in features}], columns=features)

        pred_real = float(qm[0.50].predict(X_real)[0])
        pred_fake = float(qm[0.50].predict(X_fake)[0])
        actual = row['total_games']

        results.setdefault(key, []).append((pred_real, pred_fake, actual))

    print("\n=== Impact of Real vs Placeholder Features ===")
    print(f"{'Segment':<12} {'N':>6} {'MAE(real)':>10} {'MAE(fake)':>10} {'RMSE(real)':>11} {'RMSE(fake)':>11}")
    for key, vals in results.items():
        arr = np.array(vals)
        preds_real, preds_fake, actuals = arr[:, 0], arr[:, 1], arr[:, 2]
        mae_real = np.mean(np.abs(preds_real - actuals))
        mae_fake = np.mean(np.abs(preds_fake - actuals))
        rmse_real = np.sqrt(np.mean((preds_real - actuals) ** 2))
        rmse_fake = np.sqrt(np.mean((preds_fake - actuals) ** 2))
        print(f"{key:<12} {len(vals):>6} {mae_real:>10.3f} {mae_fake:>10.3f} {rmse_real:>11.3f} {rmse_fake:>11.3f}")


if __name__ == '__main__':
    main()