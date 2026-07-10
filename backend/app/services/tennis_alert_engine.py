"""
Tennis Alert Engine v2
- Match winner prediction (65% accuracy)
- Total games prediction with full probability distribution + REAL rolling features
- Real totals odds from The Odds API
- Never sends duplicates, never alerts on past matches
- Caches match discovery to reduce Odds API quota usage
"""
import logging
import os
import pickle
import numpy as np
import pandas as pd
import httpx
from datetime import datetime, timezone, timedelta
from scipy.stats import norm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
 
logger = logging.getLogger(__name__)
 
TENNIS_SPORT_KEYS = ['tennis_atp_wimbledon', 'tennis_wta_wimbledon']
WINNER_MODEL_PATH      = '/app/ml/saved_models/tennis_winner_model.pkl'
TOTAL_GAMES_MODEL_PATH = '/app/ml/saved_models/tennis_total_games_model_v4_ensemble.pkl'
 
TOTAL_GAMES_LINES = {
    'bo3_Hard': 21.5, 'bo3_Clay': 20.5, 'bo3_Grass': 21.5,
    'bo5_Hard': 33.5, 'bo5_Clay': 32.5, 'bo5_Grass': 34.5,
}
 
# ─── Match discovery cache (reduces Odds API quota usage) ────
CACHE_TTL_HOURS = 3
_match_cache = {}  # sport_key -> {'fetched_at': datetime, 'matches': list}
 
try:
    with open(WINNER_MODEL_PATH, 'rb') as f:
        _winner_saved = pickle.load(f)
    _winner_model    = _winner_saved['model']
    _winner_features = _winner_saved['features']
    logger.info('Tennis winner model loaded')
except Exception as e:
    _winner_model = None
    _winner_features = []
    logger.error(f'Winner model failed: {e}')
 
try:
    with open(TOTAL_GAMES_MODEL_PATH, 'rb') as f:
        _tg_models = pickle.load(f)
    logger.info('Tennis total games model v4 (ensemble) loaded')
except Exception as e:
    _tg_models = None
    logger.error(f'Total games model failed: {e}')
 
 
async def _get_matches_cached(client, sport_key, api_key):
    """Fetch matches for a sport_key, using a cached copy if still fresh."""
    now = datetime.now(timezone.utc)
    cached = _match_cache.get(sport_key)
    if cached and (now - cached['fetched_at']) < timedelta(hours=CACHE_TTL_HOURS):
        age_min = round((now - cached['fetched_at']).total_seconds() / 60)
        logger.info(f'Using cached matches for {sport_key} (age: {age_min}m)')
        return cached['matches']
 
    try:
        r = await client.get(
            f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds',
            params={'apiKey': api_key, 'regions': 'eu,uk', 'markets': 'h2h,totals', 'oddsFormat': 'decimal'}
        )
        matches = r.json()
        if not isinstance(matches, list):
            logger.error(f'Odds API error: {matches}')
            return cached['matches'] if cached else []
        _match_cache[sport_key] = {'fetched_at': now, 'matches': matches}
        logger.info(f'Fetched fresh matches for {sport_key} ({len(matches)} matches)')
        return matches
    except Exception as e:
        logger.error(f'Odds API failed: {e}')
        return cached['matches'] if cached else []
 
 
def _predict_winner(p1_data, p2_data, surface='Grass', level='G', round_='R64', best_of=3):
    if _winner_model is None:
        return None, None
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    level_map   = {'G': 4, 'M': 3, 'A': 2, 'F': 1, 'D': 0}
    round_map   = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1, 'RR': 3}
    elo_diff         = p1_data['elo'] - p2_data['elo']
    surface_elo_diff = p1_data['surface_elo'] - p2_data['surface_elo']
    rank_diff        = p2_data['rank'] - p1_data['rank']
    elo_win_prob     = 1 / (1 + 10 ** (-elo_diff / 400))
    feature_vals = {
        'elo_diff': elo_diff, 'surface_elo_diff': surface_elo_diff,
        'rank_diff': rank_diff, 'elo_win_prob': elo_win_prob,
        'surface_enc': surface_map.get(surface, 2),
        'level_enc': level_map.get(level, 4),
        'round_enc': round_map.get(round_, 2),
        'best_of_enc': best_of,
        'days_since_diff': p1_data.get('days_since', 7) - p2_data.get('days_since', 7),
        'last14_diff': p1_data.get('last14', 3) - p2_data.get('last14', 3),
        'h2h_diff': p1_data.get('h2h_wins', 0) - p2_data.get('h2h_wins', 0),
        'h2h_total': p1_data.get('h2h_wins', 0) + p2_data.get('h2h_wins', 0),
        'ace_rate_diff': p1_data.get('ace_rate', 0.06) - p2_data.get('ace_rate', 0.06),
        '1st_in_diff': p1_data.get('1st_in', 0.60) - p2_data.get('1st_in', 0.60),
        '1st_won_diff': p1_data.get('1st_won', 0.70) - p2_data.get('1st_won', 0.70),
        'bp_saved_diff': p1_data.get('bp_saved', 0.60) - p2_data.get('bp_saved', 0.60),
        'exp_diff': p1_data.get('exp', 100) - p2_data.get('exp', 100),
    }
    X = np.array([[feature_vals[f] for f in _winner_features]])
    prob_p1 = float(_winner_model.predict_proba(X)[0][1])
    return prob_p1, 1 - prob_p1
 
 
async def _get_rolling_features(session, player_id, current_date, surface):
    """Compute REAL rolling-window stats for a player from their actual match history."""
    result = await session.execute(text('''
        SELECT tourney_date, surface,
               CASE WHEN winner_id = :pid THEN TRUE ELSE FALSE END AS won,
               CASE WHEN winner_id = :pid THEN w_ace ELSE l_ace END AS ace,
               CASE WHEN winner_id = :pid THEN w_svpt ELSE l_svpt END AS svpt,
               CASE WHEN winner_id = :pid THEN w_1stin ELSE l_1stin END AS first_in,
               CASE WHEN winner_id = :pid THEN w_1stwon ELSE l_1stwon END AS first_won,
               CASE WHEN winner_id = :pid THEN w_2ndwon ELSE l_2ndwon END AS second_won,
               CASE WHEN winner_id = :pid THEN w_bpsaved ELSE l_bpsaved END AS bp_saved,
               CASE WHEN winner_id = :pid THEN w_bpfaced ELSE l_bpfaced END AS bp_faced
        FROM tennis_matches
        WHERE (winner_id = :pid OR loser_id = :pid) AND tourney_date < :cur_date
        ORDER BY tourney_date DESC, match_num DESC
        LIMIT 50
    '''), {'pid': player_id, 'cur_date': current_date})
    rows = result.fetchall()
 
    def safe_div(num, denom):
        if not num or not denom:
            return None
        return num / denom
 
    matches = []
    for row in rows:
        t_date, m_surface, won, ace, svpt, first_in, first_won, second_won, bp_saved, bp_faced = row
        ace_rate = safe_div(ace, svpt)
        first_won_pct = safe_div(first_won, first_in)
        second_won_pct = safe_div(second_won, (svpt or 0) - (first_in or 0))
        bp_saved_pct = safe_div(bp_saved, bp_faced)
        matches.append((t_date, m_surface, won, ace_rate, first_won_pct, second_won_pct, bp_saved_pct))
 
    if not matches:
        return {}
 
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
 
    last5, last10, last20 = matches[:5], matches[:10], matches[:20]
 
    streak = 0
    for m in matches:
        if m[2]:
            streak += 1
        else:
            break
 
    d90 = [m for m in matches if (current_date - m[0]).days <= 90]
    surf_matches = [m for m in matches if m[1] == surface][:10]
 
    return {
        'roll5_bpsaved': mean([m[6] for m in last5]),
        'roll10_ace': mean([m[3] for m in last10]),
        'roll10_1stwon': mean([m[4] for m in last10]),
        'roll10_2ndwon': mean([m[5] for m in last10]),
        'roll10_bpsaved': mean([m[6] for m in last10]),
        'roll10_winrate': mean([1.0 if m[2] else 0.0 for m in last10]),
        'roll20_bpsaved': mean([m[6] for m in last20]),
        'win_streak': streak,
        'wins_last10': sum(1 for m in last10 if m[2]),
        'd90_ace': mean([m[3] for m in d90]),
        'd90_1stwon': mean([m[4] for m in d90]),
        'surf_form10': mean([1.0 if m[2] else 0.0 for m in surf_matches]) if surf_matches else None,
    }
 
 
def _predict_total_games(p1_data, p2_data, surface='Grass', level='G',
                         round_='R64', best_of=3, over_price=None, under_price=None):
    if _tg_models is None:
        return None
    key = f'bo{best_of}_{surface}'
    if key not in _tg_models:
        key = 'fallback' if 'fallback' in _tg_models else None
    if key is None:
        return None
    sub      = _tg_models[key]
    qm       = sub['quantile_models']
    features = sub['features']
    line     = TOTAL_GAMES_LINES.get(f'bo{best_of}_{surface}', 21.5)
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    level_map   = {'G': 4, 'M': 3, 'A': 2, 'F': 1, 'D': 0}
    round_map   = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1, 'RR': 3}
    elo_diff         = p1_data['elo'] - p2_data['elo']
    surface_elo_diff = p1_data['surface_elo'] - p2_data['surface_elo']
    elo_mismatch     = abs(elo_diff)
    rank_diff        = abs(p1_data['rank'] - p2_data['rank'])
    elo_win_prob     = 1 / (1 + 10 ** (-elo_diff / 400))
    avg_ace    = (p1_data.get('ace_rate', 0.06) + p2_data.get('ace_rate', 0.06)) / 2
    avg_1stin  = (p1_data.get('1st_in', 0.60)  + p2_data.get('1st_in', 0.60))  / 2
    avg_1stwon = (p1_data.get('1st_won', 0.70) + p2_data.get('1st_won', 0.70)) / 2
    avg_2ndwon = (p1_data.get('2nd_won', 0.50) + p2_data.get('2nd_won', 0.50)) / 2
    avg_bpsv   = (p1_data.get('bp_saved', 0.60)+ p2_data.get('bp_saved', 0.60)) / 2
    serve_dom  = avg_1stwon * avg_1stin + avg_2ndwon * (1 - avg_1stin)
    surf_enc   = surface_map.get(surface, 2)
    level_enc  = level_map.get(level, 4)
    round_enc  = round_map.get(round_, 2)
 
    def d(v, default):
        return float(v) if v is not None else default
 
    p1_roll = p1_data.get('rolling', {})
    p2_roll = p2_data.get('rolling', {})
 
    roll10_ace_diff     = d(p1_roll.get('roll10_ace'), avg_ace) - d(p2_roll.get('roll10_ace'), avg_ace)
    roll10_1stwon_diff  = d(p1_roll.get('roll10_1stwon'), avg_1stwon) - d(p2_roll.get('roll10_1stwon'), avg_1stwon)
    roll10_2ndwon_diff  = d(p1_roll.get('roll10_2ndwon'), avg_2ndwon) - d(p2_roll.get('roll10_2ndwon'), avg_2ndwon)
    roll10_bpsaved_diff = d(p1_roll.get('roll10_bpsaved'), avg_bpsv) - d(p2_roll.get('roll10_bpsaved'), avg_bpsv)
    roll10_winrate_diff = d(p1_roll.get('roll10_winrate'), 0.5) - d(p2_roll.get('roll10_winrate'), 0.5)
    roll10_ace_avg     = (d(p1_roll.get('roll10_ace'), avg_ace) + d(p2_roll.get('roll10_ace'), avg_ace)) / 2
    roll10_1stwon_avg  = (d(p1_roll.get('roll10_1stwon'), avg_1stwon) + d(p2_roll.get('roll10_1stwon'), avg_1stwon)) / 2
    roll10_2ndwon_avg  = (d(p1_roll.get('roll10_2ndwon'), avg_2ndwon) + d(p2_roll.get('roll10_2ndwon'), avg_2ndwon)) / 2
    roll10_bpsaved_avg = (d(p1_roll.get('roll10_bpsaved'), avg_bpsv) + d(p2_roll.get('roll10_bpsaved'), avg_bpsv)) / 2
    roll20_bpsaved_avg = (d(p1_roll.get('roll20_bpsaved'), avg_bpsv) + d(p2_roll.get('roll20_bpsaved'), avg_bpsv)) / 2
    roll5_bpsaved_avg  = roll10_bpsaved_avg
    win_streak_diff    = (p1_roll.get('win_streak') or 0) - (p2_roll.get('win_streak') or 0)
    wins_last10_avg    = ((p1_roll.get('wins_last10') if p1_roll.get('wins_last10') is not None else 5) +
                          (p2_roll.get('wins_last10') if p2_roll.get('wins_last10') is not None else 5)) / 2
    d90_ace_avg    = (d(p1_roll.get('d90_ace'), avg_ace) + d(p2_roll.get('d90_ace'), avg_ace)) / 2
    d90_1stwon_avg = (d(p1_roll.get('d90_1stwon'), avg_1stwon) + d(p2_roll.get('d90_1stwon'), avg_1stwon)) / 2
    surf_form10_avg = (d(p1_roll.get('surf_form10'), 0.5) + d(p2_roll.get('surf_form10'), 0.5)) / 2
    momentum_x_level = win_streak_diff * level_enc
 
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
    X = pd.DataFrame([{f: feat_row.get(f, 0) for f in features}], columns=features)
    median = float(qm[0.50].predict(X)[0])
    p05    = float(qm[0.05].predict(X)[0])
    p95    = float(qm[0.95].predict(X)[0])
    std    = (p95 - p05) / (2 * 1.645)
    p90_low  = norm.ppf(0.05, median, std)
    p90_high = norm.ppf(0.95, median, std)
    p95_low  = norm.ppf(0.025, median, std)
    p95_high = norm.ppf(0.975, median, std)
    prob_over  = 1 - norm.cdf(line, median, std)
    prob_under = norm.cdf(line, median, std)
    confidence = round(abs(prob_over - 0.5) * 200)
    result = {
        'key': key, 'line': line,
        'expected_games': round(median, 1),
        'std_dev': round(std, 1),
        'p90_interval': f'{p90_low:.1f}-{p90_high:.1f}',
        'p95_interval': f'{p95_low:.1f}-{p95_high:.1f}',
        'prob_over': round(prob_over * 100, 1),
        'prob_under': round(prob_under * 100, 1),
        'confidence': confidence,
        'recommendation': 'No Bet',
        'ev_pct': None, 'edge_pct': None, 'kelly_pct': None,
    }
    if over_price and under_price:
        implied_over  = 1 / over_price
        implied_under = 1 / under_price
        total_implied = implied_over + implied_under
        fair_over     = implied_over / total_implied
        fair_under    = implied_under / total_implied
        edge_over  = prob_over  - fair_over
        edge_under = prob_under - fair_under
        if edge_over > edge_under:
            side, model_prob, edge, price = 'Over', prob_over, edge_over, over_price
        else:
            side, model_prob, edge, price = 'Under', prob_under, edge_under, under_price
        ev    = (model_prob * (price - 1)) - (1 - model_prob)
        kelly = max(0, (model_prob * (price - 1) - (1 - model_prob)) / (price - 1))
        result['edge_pct']  = round(edge * 100, 1)
        result['ev_pct']    = round(ev * 100, 1)
        result['kelly_pct'] = round(kelly / 4 * 100, 2)
        result['bet_side']  = side
        result['bet_price'] = price
        if confidence >= 70 and ev > 0.03:
            result['recommendation'] = f'STRONG {side.upper()}'
        elif confidence >= 50 and ev > 0.01:
            result['recommendation'] = f'LEAN {side.upper()}'
    return result
 
 
async def _get_player_data(session, player_name):
    name_parts = player_name.strip().split()
    if not name_parts:
        return None
    last_name = name_parts[-1]
    for role in ['winner', 'loser']:
        if role == 'winner':
            cols     = 'tp.id, tm.winner_elo_pre, tm.winner_surface_elo_pre, tm.winner_rank, tm.winner_ace_rate_pre, tm.winner_1st_in_pct_pre, tm.winner_1st_won_pct_pre, tm.winner_2nd_won_pct_pre, tm.winner_bp_saved_pct_pre, tm.winner_days_since_last_pre, tm.winner_matches_last14_pre, tm.winner_h2h_wins_pre, tm.winner_total_matches_pre'
            join_col = 'winner_id'
        else:
            cols     = 'tp.id, tm.loser_elo_pre, tm.loser_surface_elo_pre, tm.loser_rank, tm.loser_ace_rate_pre, tm.loser_1st_in_pct_pre, tm.loser_1st_won_pct_pre, tm.loser_2nd_won_pct_pre, tm.loser_bp_saved_pct_pre, tm.loser_days_since_last_pre, tm.loser_matches_last14_pre, tm.loser_h2h_wins_pre, tm.loser_total_matches_pre'
            join_col = 'loser_id'
        result = await session.execute(text(f'''
            SELECT tp.name, {cols}
            FROM tennis_matches tm
            JOIN tennis_players tp ON tp.id = tm.{join_col}
            WHERE tp.name ILIKE :pattern
            ORDER BY tm.tourney_date DESC LIMIT 1
        '''), {'pattern': f'%{last_name}%'})
        row = result.fetchone()
        if row:
            return {
                'name': row[0], 'player_id': row[1],
                'elo': float(row[2]) if row[2] else 1500.0,
                'surface_elo': float(row[3]) if row[3] else 1500.0,
                'rank': int(row[4]) if row[4] else 100,
                'ace_rate': float(row[5]) if row[5] else 0.06,
                '1st_in': float(row[6]) if row[6] else 0.60,
                '1st_won': float(row[7]) if row[7] else 0.70,
                '2nd_won': float(row[8]) if row[8] else 0.50,
                'bp_saved': float(row[9]) if row[9] else 0.60,
                'days_since': int(row[10]) if row[10] else 7,
                'last14': int(row[11]) if row[11] else 3,
                'h2h_wins': int(row[12]) if row[12] else 0,
                'exp': int(row[13]) if row[13] else 100,
            }
    return None
 
 
async def _already_alerted(session, odds_api_id):
    result = await session.execute(text(
        'SELECT alert_sent FROM tennis_upcoming_matches WHERE odds_api_id = :oid AND alert_sent = TRUE'
    ), {'oid': odds_api_id})
    return result.fetchone() is not None
 
 
async def _record_clv_opening(db, odds_api_id, p1_name, p2_name, tournament,
                               surface, best_of, commence_time, tg_result,
                               totals_over_price, totals_under_price):
    if not tg_result:
        return
    opening_implied_over = None
    if totals_over_price:
        implied_over  = 1 / totals_over_price
        implied_under = 1 / totals_under_price if totals_under_price else None
        if implied_under:
            opening_implied_over = implied_over / (implied_over + implied_under)
    try:
        await db.execute(text('''
            INSERT INTO tennis_clv_tracking
                (odds_api_id, p1_name, p2_name, tournament, surface, best_of,
                 commence_time, model_line, model_prob_over, model_prob_under,
                 model_confidence, model_recommendation, model_ev_pct, model_edge_pct,
                 opening_line, opening_over_price, opening_under_price, opening_implied_over)
            VALUES
                (:oid, :p1, :p2, :tour, :surf, :bo,
                 :ct, :line, :prob_over, :prob_under,
                 :conf, :rec, :ev, :edge,
                 :line, :over_price, :under_price, :impl_over)
            ON CONFLICT DO NOTHING
        '''), {
            'oid': odds_api_id, 'p1': p1_name, 'p2': p2_name,
            'tour': tournament, 'surf': surface, 'bo': best_of,
            'ct': commence_time, 'line': tg_result.get('line'),
            'prob_over': tg_result.get('prob_over'), 'prob_under': tg_result.get('prob_under'),
            'conf': tg_result.get('confidence'), 'rec': tg_result.get('recommendation'),
            'ev': tg_result.get('ev_pct'), 'edge': tg_result.get('edge_pct'),
            'over_price': totals_over_price, 'under_price': totals_under_price,
            'impl_over': opening_implied_over,
        })
    except Exception as e:
        logger.error(f'CLV record failed for {odds_api_id}: {e}')
 
 
async def run_tennis_clv_closing_job(db):
    """Fetch closing odds for matches starting soon and record CLV."""
    api_key = os.getenv('TENNIS_ODDS_API_KEY') or os.getenv('ODDS_API_KEY', '')
    now          = datetime.now(timezone.utc)
    window_start = now
    window_end   = now + timedelta(minutes=10)
 
    result = await db.execute(text('''
        SELECT id, odds_api_id, tournament, best_of,
               model_recommendation, opening_over_price, opening_under_price, model_line
        FROM tennis_clv_tracking
        WHERE closing_line IS NULL
          AND commence_time BETWEEN :ws AND :we
    '''), {'ws': window_start, 'we': window_end})
    rows = result.fetchall()
    if not rows:
        return {'checked': 0, 'updated': 0}
 
    updated = 0
    async with httpx.AsyncClient(timeout=15.0) as client:
        for row in rows:
            row_id, odds_api_id, tournament, best_of, rec, open_over, open_under, model_line = row
            sport_key = 'tennis_atp_wimbledon' if best_of == 5 else 'tennis_wta_wimbledon'
            try:
                r = await client.get(
                    f'https://api.the-odds-api.com/v4/sports/{sport_key}/events/{odds_api_id}/odds',
                    params={'apiKey': api_key, 'regions': 'eu,uk', 'markets': 'totals', 'oddsFormat': 'decimal'}
                )
                data = r.json()
                closing_over = closing_under = None
                for bk in data.get('bookmakers', []):
                    for mkt in bk.get('markets', []):
                        if mkt['key'] == 'totals':
                            for oc in mkt['outcomes']:
                                if oc['name'] == 'Over':
                                    closing_over = oc['price']
                                elif oc['name'] == 'Under':
                                    closing_under = oc['price']
                    if closing_over:
                        break
                if not closing_over or not closing_under:
                    continue
 
                implied_over = 1 / closing_over
                implied_under = 1 / closing_under
                closing_implied_over = implied_over / (implied_over + implied_under)
 
                side = None
                if rec and 'OVER' in rec.upper():
                    side = 'Over'
                elif rec and 'UNDER' in rec.upper():
                    side = 'Under'
 
                clv = beat = None
                if side == 'Over' and open_over:
                    clv  = (open_over / closing_over - 1) * 100
                    beat = open_over > closing_over
                elif side == 'Under' and open_under:
                    clv  = (open_under / closing_under - 1) * 100
                    beat = open_under > closing_under
 
                await db.execute(text('''
                    UPDATE tennis_clv_tracking
                    SET closing_line = :cl, closing_over_price = :co,
                        closing_under_price = :cu, closing_implied_over = :cio,
                        clv = :clv, beat_closing_line = :beat, updated_at = NOW()
                    WHERE id = :id
                '''), {
                    'cl': model_line, 'co': closing_over, 'cu': closing_under,
                    'cio': closing_implied_over, 'clv': clv, 'beat': beat, 'id': row_id,
                })
                updated += 1
            except Exception as e:
                logger.error(f'CLV closing fetch failed for {odds_api_id}: {e}')
 
    await db.commit()
    return {'checked': len(rows), 'updated': updated}
 
 
async def _send_telegram(message, bot_token, chat_id):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f'https://api.telegram.org/bot{bot_token}/sendMessage',
                json={'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
            )
            return r.status_code == 200
    except Exception as e:
        logger.error(f'Telegram send failed: {e}')
        return False
 
 
def _format_message(p1, p2, winner, p1_prob, p2_prob,
                    p1_odds, p2_odds, commence_time, tournament, tg_result=None):
    time_str    = commence_time.strftime('%A %d %B, %H:%M UTC')
    winner_conf = round(max(p1_prob, p2_prob))
    edge_str    = ''
    if p1_odds and p2_odds:
        w_odds      = p1_odds if winner == p1 else p2_odds
        market_prob = 1 / w_odds
        model_prob  = max(p1_prob, p2_prob) / 100
        edge        = (model_prob - market_prob) * 100
        edge_emoji  = '✅' if edge >= 5 else '⚠️' if edge >= 2 else '➖'
        edge_str    = f'{edge_emoji} Winner edge: {edge:+.1f}%'
    msg = (
        f'🎾 <b>TENNIS PREDICTION</b>\n'
        f'{"─"*28}\n'
        f'🏆 {tournament}\n'
        f'📅 {time_str}\n\n'
        f'<b>{p1}</b> vs <b>{p2}</b>\n\n'
        f'━━ MATCH WINNER ━━\n'
        f'🎯 <b>Pick: {winner}</b> ({winner_conf}% confidence)\n'
        f'📊 {p1}: {p1_prob:.1f}% | {p2}: {p2_prob:.1f}%\n'
        f'💰 Odds: {p1} {p1_odds or "N/A"} | {p2} {p2_odds or "N/A"}\n'
        f'{edge_str}\n'
    )
    if tg_result:
        tg        = tg_result
        rec       = tg['recommendation']
        rec_emoji = '🔥' if 'STRONG' in rec else '📌' if 'LEAN' in rec else '➖'
        msg += (
            f'\n━━ TOTAL GAMES ━━\n'
            f'📏 Expected: <b>{tg["expected_games"]}</b> games (±{tg["std_dev"]})\n'
            f'📐 90% range: {tg["p90_interval"]}\n'
            f'📐 95% range: {tg["p95_interval"]}\n'
            f'📊 Line {tg["line"]}: Over {tg["prob_over"]}% | Under {tg["prob_under"]}%\n'
            f'🎯 Confidence: {tg["confidence"]}/100\n'
        )
        if tg.get('ev_pct') is not None:
            msg += (
                f'💡 Edge: {tg["edge_pct"]:+.1f}% | EV: {tg["ev_pct"]:+.1f}%\n'
                f'💰 Kelly stake: {tg["kelly_pct"]}% of bankroll\n'
                f'{rec_emoji} <b>Totals: {rec}</b>\n'
            )
        else:
            msg += f'{rec_emoji} Totals: {rec} (no live totals odds)\n'
    return msg
 
 
async def run_tennis_alert_engine(db):
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat_id   = os.getenv('TELEGRAM_CHAT_ID', '')
    api_key   = os.getenv('TENNIS_ODDS_API_KEY') or os.getenv('ODDS_API_KEY', '')
    now          = datetime.now(timezone.utc)
    window_start = now
    window_end   = now + timedelta(hours=8)
    alerts_sent = skipped_duplicate = no_data = 0
    errors = []
 
    async with httpx.AsyncClient(timeout=15.0) as client:
        for sport_key in TENNIS_SPORT_KEYS:
            matches = await _get_matches_cached(client, sport_key, api_key)
            if not matches:
                continue
 
            tournament = 'ATP Wimbledon' if 'atp' in sport_key else 'WTA Wimbledon'
            best_of    = 5 if 'atp' in sport_key else 3
 
            for match in matches:
                try:
                    odds_api_id   = match.get('id', '')
                    p1_name       = match.get('home_team', '')
                    p2_name       = match.get('away_team', '')
                    commence_str  = match.get('commence_time', '')
                    commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                    if commence_time <= now:
                        continue
                    if not (window_start <= commence_time <= window_end):
                        continue
                    if await _already_alerted(db, odds_api_id):
                        skipped_duplicate += 1
                        continue
 
                    p1_odds = p2_odds = totals_over_price = totals_under_price = None
                    for bk in match.get('bookmakers', []):
                        for mkt in bk.get('markets', []):
                            if mkt['key'] == 'h2h' and not p1_odds:
                                for oc in mkt['outcomes']:
                                    if oc['name'] == p1_name:
                                        p1_odds = oc['price']
                                    elif oc['name'] == p2_name:
                                        p2_odds = oc['price']
                            elif mkt['key'] == 'totals' and not totals_over_price:
                                for oc in mkt['outcomes']:
                                    if oc['name'] == 'Over':
                                        totals_over_price = oc['price']
                                    elif oc['name'] == 'Under':
                                        totals_under_price = oc['price']
                        if p1_odds and totals_over_price:
                            break
 
                    p1_data = await _get_player_data(db, p1_name)
                    p2_data = await _get_player_data(db, p2_name)
                    if not p1_data or not p2_data:
                        no_data += 1
                        logger.info(f"Skipping: no ELO data for {p1_name} or {p2_name}")
                        continue
 
                    # fetch REAL rolling features for both players
                    match_date = commence_time.date()
                    p1_data['rolling'] = await _get_rolling_features(db, p1_data['player_id'], match_date, 'Grass')
                    p2_data['rolling'] = await _get_rolling_features(db, p2_data['player_id'], match_date, 'Grass')
 
                    p1_prob, p2_prob = _predict_winner(
                        p1_data, p2_data, surface='Grass', level='G', round_='R64', best_of=best_of
                    )
                    if p1_prob is None:
                        errors.append(f'Winner model failed: {p1_name} vs {p2_name}')
                        continue
 
                    winner     = p1_name if p1_prob >= p2_prob else p2_name
                    confidence = max(p1_prob, p2_prob) * 100
 
                    # Skip low confidence matches — model has no meaningful edge
                    if confidence < 62.0:
                        logger.info(f"Skipping low confidence: {p1_name} vs {p2_name} ({confidence:.1f}%)")
                        continue
 
                    tg_result = _predict_total_games(
                        p1_data, p2_data, surface='Grass', level='G',
                        round_='R64', best_of=best_of,
                        over_price=totals_over_price, under_price=totals_under_price
                    )
 
                    await db.execute(text('''
                        INSERT INTO tennis_upcoming_matches
                            (odds_api_id, p1_name, p2_name, commence_time, tournament,
                             sport_key, p1_odds, p2_odds, predicted_winner,
                             p1_win_prob, p2_win_prob, confidence, alert_sent)
                        VALUES
                            (:oid, :p1, :p2, :ct, :tour, :sk, :p1o, :p2o, :pw,
                             :p1p, :p2p, :conf, FALSE)
                        ON CONFLICT (odds_api_id) DO UPDATE SET
                            p1_odds=EXCLUDED.p1_odds, p2_odds=EXCLUDED.p2_odds,
                            predicted_winner=EXCLUDED.predicted_winner,
                            p1_win_prob=EXCLUDED.p1_win_prob,
                            p2_win_prob=EXCLUDED.p2_win_prob,
                            confidence=EXCLUDED.confidence
                    '''), {
                        'oid': odds_api_id, 'p1': p1_name, 'p2': p2_name,
                        'ct': commence_time, 'tour': tournament, 'sk': sport_key,
                        'p1o': p1_odds, 'p2o': p2_odds, 'pw': winner,
                        'p1p': p1_prob * 100, 'p2p': p2_prob * 100, 'conf': confidence,
                    })
 
                    message = _format_message(
                        p1=p1_name, p2=p2_name, winner=winner,
                        p1_prob=p1_prob * 100, p2_prob=p2_prob * 100,
                        p1_odds=p1_odds, p2_odds=p2_odds,
                        commence_time=commence_time, tournament=tournament,
                        tg_result=tg_result,
                    )
 
                    sent = False
                    if bot_token and chat_id:
                        sent = await _send_telegram(message, bot_token, chat_id)
                    if sent:
                        await db.execute(text(
                            'UPDATE tennis_upcoming_matches SET alert_sent=TRUE WHERE odds_api_id=:oid'
                        ), {'oid': odds_api_id})
                        await _record_clv_opening(
                            db, odds_api_id, p1_name, p2_name, tournament,
                            'Grass', best_of, commence_time, tg_result,
                            totals_over_price, totals_under_price
                        )
                        alerts_sent += 1
                        logger.info(f'Alert sent: {p1_name} vs {p2_name}')
 
                except Exception as e:
                    logger.error(f'Error processing match: {e}')
                    errors.append(str(e))
 
    await db.commit()
    return {
        'engine': 'Tennis Alert Engine v2',
        'window': f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')} UTC",
        'alerts_sent': alerts_sent,
        'skipped_duplicate': skipped_duplicate,
        'no_elo_data': no_data,
        'errors': errors[:5],
    }