"""
Retrains tennis total-games quantile-regression models on the current
tennis_matches + tennis_match_features data, comparing GradientBoostingRegressor,
XGBoost, and LightGBM per segment (same approach as train_ensemble_compare.py),
and saves the best-per-segment result to a CANDIDATE file — never the live
model. Promotion to production is a separate manual step (rename the file
yourself once you've reviewed the metrics).
"""
import pickle
import random
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
SEGMENTS = ['bo3_Hard', 'bo3_Clay', 'bo3_Grass', 'bo5_Hard', 'bo5_Clay', 'bo5_Grass']

SURFACE_MAP = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
LEVEL_MAP   = {'G': 4, 'M': 3, 'A': 2, 'F': 1, 'D': 0}
ROUND_MAP   = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1, 'RR': 3}

FEATURES = [
    'elo_diff', 'surface_elo_diff', 'elo_mismatch', 'rank_diff', 'elo_win_prob',
    'surface_enc', 'level_enc', 'round_enc',
    'roll10_ace_diff', 'roll10_1stwon_diff', 'roll10_2ndwon_diff',
    'roll10_bpsaved_diff', 'roll10_winrate_diff',
    'roll10_ace_avg', 'roll10_1stwon_avg', 'roll10_2ndwon_avg', 'roll10_bpsaved_avg',
    'roll20_bpsaved_avg', 'roll5_bpsaved_avg',
    'win_streak_diff', 'wins_last10_avg',
    '90d_ace_avg', '90d_1stwon_avg',
    'surf_form10_avg', 'serve_dominance',
    'surface_elo_diff_x_surf', 'rankdiff_x_level', 'servedom_x_surface',
    'momentum_x_level', 'bpsaved_x_elo_mismatch',
]


def _load_data(database_url: str):
    conn = psycopg2.connect(database_url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT tm.id AS match_id, tm.surface, tm.tourney_level, tm.round, tm.best_of, tm.tourney_date,
               tm.winner_rank, tm.loser_rank,
               tm.winner_elo_pre, tm.loser_elo_pre,
               tm.winner_surface_elo_pre, tm.loser_surface_elo_pre,
               tm.winner_ace_rate_pre, tm.loser_ace_rate_pre,
               tm.winner_1st_in_pct_pre, tm.loser_1st_in_pct_pre,
               tm.winner_1st_won_pct_pre, tm.loser_1st_won_pct_pre,
               tm.winner_2nd_won_pct_pre, tm.loser_2nd_won_pct_pre,
               tm.winner_bp_saved_pct_pre, tm.loser_bp_saved_pct_pre,
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
        ORDER BY tm.tourney_date ASC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def _build_feature_row(row, rng_state):
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

    return {
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


def _get_segment_key(row):
    surface = row['surface'] or 'Hard'
    best_of = row['best_of'] or 3
    key = f'bo{best_of}_{surface}'
    return key if key in SEGMENTS else None


def _mae_rmse(preds, actuals):
    preds = np.array(preds)
    actuals = np.array(actuals)
    mae = np.mean(np.abs(preds - actuals))
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    return float(mae), float(rmse)


def train_total_games_candidate(database_url: str, candidate_output_path: str,
                                 test_cutoff_date: str = '2023-01-01') -> dict:
    """Trains GBR, XGBoost, and LightGBM quantile models per segment, picks the
    best by test MAE, and writes the result to candidate_output_path — never
    touches the live model. Returns a summary dict with per-segment metrics
    so you can review before manually promoting the candidate file.
    Blocking/synchronous — call via asyncio.to_thread() from async code."""
    rows = _load_data(database_url)
    train_rows = [r for r in rows if r['tourney_date'].isoformat() < test_cutoff_date]
    test_rows  = [r for r in rows if r['tourney_date'].isoformat() >= test_cutoff_date]

    final_models = {}
    segment_summary = {}

    for seg in SEGMENTS:
        seg_train = [r for r in train_rows if _get_segment_key(r) == seg]
        seg_test  = [r for r in test_rows if _get_segment_key(r) == seg]
        if len(seg_train) < 50 or not seg_test:
            segment_summary[seg] = {"status": "skipped", "train_n": len(seg_train), "test_n": len(seg_test)}
            continue

        X_train, y_train = [], []
        for r in seg_train:
            rng = random.Random(r['match_id'])
            feat = _build_feature_row(r, rng)
            X_train.append([feat.get(f, 0) for f in FEATURES])
            y_train.append(r['total_games'])
        X_train = pd.DataFrame(X_train, columns=FEATURES)
        y_train = np.array(y_train)

        X_test, y_test = [], []
        for r in seg_test:
            rng = random.Random(r['match_id'])
            feat = _build_feature_row(r, rng)
            X_test.append([feat.get(f, 0) for f in FEATURES])
            y_test.append(r['total_games'])
        X_test = pd.DataFrame(X_test, columns=FEATURES)
        y_test = np.array(y_test)

        # --- GradientBoostingRegressor ---
        gbr_models = {}
        for q in QUANTILES:
            model = GradientBoostingRegressor(
                loss='quantile', alpha=q,
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            model.fit(X_train, y_train)
            gbr_models[q] = model
        gbr_mae, gbr_rmse = _mae_rmse(gbr_models[0.50].predict(X_test), y_test)

        # --- XGBoost ---
        xgb_models = {}
        for q in QUANTILES:
            model = xgb.XGBRegressor(
                objective='reg:quantileerror', quantile_alpha=q,
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            model.fit(X_train, y_train)
            xgb_models[q] = model
        xgb_mae, xgb_rmse = _mae_rmse(xgb_models[0.50].predict(X_test), y_test)

        # --- LightGBM ---
        lgb_models = {}
        for q in QUANTILES:
            model = lgb.LGBMRegressor(
                objective='quantile', alpha=q,
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42, verbose=-1,
            )
            model.fit(X_train, y_train)
            lgb_models[q] = model
        lgb_mae, lgb_rmse = _mae_rmse(lgb_models[0.50].predict(X_test), y_test)

        scores = {'gbr': gbr_mae, 'xgboost': xgb_mae, 'lightgbm': lgb_mae}
        winner = min(scores, key=scores.get)

        if winner == 'xgboost':
            final_models[seg] = {'quantile_models': xgb_models, 'features': FEATURES, 'model_type': 'xgboost'}
        elif winner == 'lightgbm':
            final_models[seg] = {'quantile_models': lgb_models, 'features': FEATURES, 'model_type': 'lightgbm'}
        else:
            final_models[seg] = {'quantile_models': gbr_models, 'features': FEATURES, 'model_type': 'gbr'}

        segment_summary[seg] = {
            "status": "trained",
            "train_n": len(seg_train),
            "test_n": len(seg_test),
            "mae": {"gbr": round(gbr_mae, 3), "xgboost": round(xgb_mae, 3), "lightgbm": round(lgb_mae, 3)},
            "rmse": {"gbr": round(gbr_rmse, 3), "xgboost": round(xgb_rmse, 3), "lightgbm": round(lgb_rmse, 3)},
            "winner": winner,
        }

    with open(candidate_output_path, 'wb') as f:
        pickle.dump(final_models, f)

    return {
        "candidate_path": candidate_output_path,
        "segments": segment_summary,
    }
