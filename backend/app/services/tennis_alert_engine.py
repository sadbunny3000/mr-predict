"""
Tennis Alert Engine
- Fetches upcoming Wimbledon matches from The Odds API
- Runs our ML model on each match
- Sends Telegram alerts 3 hours before match starts
- Never sends duplicates, never alerts on past matches
"""
import logging
import os
import pickle
import numpy as np
import httpx
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)

TENNIS_SPORT_KEYS = ['tennis_atp_wimbledon', 'tennis_wta_wimbledon']
MODEL_PATH = '/app/ml/saved_models/tennis_winner_model.pkl'

# Load model once at import time
try:
    with open(MODEL_PATH, 'rb') as f:
        _saved = pickle.load(f)
    _model = _saved['model']
    _features = _saved['features']
    logger.info('Tennis model loaded successfully')
except Exception as e:
    _model = None
    _features = []
    logger.error(f'Tennis model failed to load: {e}')


def _predict(p1_elo, p2_elo, p1_surface_elo, p2_surface_elo,
             p1_rank, p2_rank,
             p1_ace_rate=0.06, p2_ace_rate=0.06,
             p1_1st_in=0.60, p2_1st_in=0.60,
             p1_1st_won=0.70, p2_1st_won=0.70,
             p1_bp_saved=0.60, p2_bp_saved=0.60,
             p1_days_since=7, p2_days_since=7,
             p1_last14=3, p2_last14=3,
             p1_h2h_wins=0, p2_h2h_wins=0,
             p1_exp=100, p2_exp=100,
             surface='Grass', level='G',
             round_='R64', best_of=3):
    if _model is None:
        return None, None

    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    level_map   = {'G': 4, 'M': 3, 'A': 2, 'F': 1, 'D': 0}
    round_map   = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3,
                   'R64': 2, 'R128': 1, 'RR': 3}

    elo_diff         = p1_elo - p2_elo
    surface_elo_diff = p1_surface_elo - p2_surface_elo
    rank_diff        = p2_rank - p1_rank
    elo_win_prob     = 1 / (1 + 10 ** (-elo_diff / 400))

    feature_vals = {
        'elo_diff':         elo_diff,
        'surface_elo_diff': surface_elo_diff,
        'rank_diff':        rank_diff,
        'elo_win_prob':     elo_win_prob,
        'surface_enc':      surface_map.get(surface, 2),
        'level_enc':        level_map.get(level, 4),
        'round_enc':        round_map.get(round_, 2),
        'best_of_enc':      best_of,
        'days_since_diff':  p1_days_since - p2_days_since,
        'last14_diff':      p1_last14 - p2_last14,
        'h2h_diff':         p1_h2h_wins - p2_h2h_wins,
        'h2h_total':        p1_h2h_wins + p2_h2h_wins,
        'ace_rate_diff':    p1_ace_rate - p2_ace_rate,
        '1st_in_diff':      p1_1st_in - p2_1st_in,
        '1st_won_diff':     p1_1st_won - p2_1st_won,
        'bp_saved_diff':    p1_bp_saved - p2_bp_saved,
        'exp_diff':         p1_exp - p2_exp,
    }

    X = np.array([[feature_vals[f] for f in _features]])
    prob_p1 = float(_model.predict_proba(X)[0][1])
    return prob_p1, 1 - prob_p1


async def _get_player_elo(session: AsyncSession, player_name: str):
    """Look up latest ELO and pre-match stats for a player by name."""
    name_parts = player_name.strip().split()
    if not name_parts:
        return None
    last_name = name_parts[-1]

    result = await session.execute(text('''
        SELECT
            tp.name,
            tm.winner_elo_pre, tm.winner_surface_elo_pre, tm.winner_rank,
            tm.winner_ace_rate_pre, tm.winner_1st_in_pct_pre,
            tm.winner_1st_won_pct_pre, tm.winner_bp_saved_pct_pre,
            tm.winner_days_since_last_pre, tm.winner_matches_last14_pre,
            tm.winner_h2h_wins_pre, tm.winner_total_matches_pre
        FROM tennis_matches tm
        JOIN tennis_players tp ON tp.id = tm.winner_id
        WHERE tp.name ILIKE :pattern
        ORDER BY tm.tourney_date DESC
        LIMIT 1
    '''), {'pattern': f'%{last_name}%'})
    row = result.fetchone()
    if row:
        return {
            'name': row[0],
            'elo': float(row[1]) if row[1] else 1500.0,
            'surface_elo': float(row[2]) if row[2] else 1500.0,
            'rank': int(row[3]) if row[3] else 100,
            'ace_rate': float(row[4]) if row[4] else 0.06,
            '1st_in': float(row[5]) if row[5] else 0.60,
            '1st_won': float(row[6]) if row[6] else 0.70,
            'bp_saved': float(row[7]) if row[7] else 0.60,
            'days_since': int(row[8]) if row[8] else 7,
            'last14': int(row[9]) if row[9] else 3,
            'h2h_wins': int(row[10]) if row[10] else 0,
            'exp': int(row[11]) if row[11] else 100,
        }

    result2 = await session.execute(text('''
        SELECT
            tp.name,
            tm.loser_elo_pre, tm.loser_surface_elo_pre, tm.loser_rank,
            tm.loser_ace_rate_pre, tm.loser_1st_in_pct_pre,
            tm.loser_1st_won_pct_pre, tm.loser_bp_saved_pct_pre,
            tm.loser_days_since_last_pre, tm.loser_matches_last14_pre,
            tm.loser_h2h_wins_pre, tm.loser_total_matches_pre
        FROM tennis_matches tm
        JOIN tennis_players tp ON tp.id = tm.loser_id
        WHERE tp.name ILIKE :pattern
        ORDER BY tm.tourney_date DESC
        LIMIT 1
    '''), {'pattern': f'%{last_name}%'})
    row2 = result2.fetchone()
    if row2:
        return {
            'name': row2[0],
            'elo': float(row2[1]) if row2[1] else 1500.0,
            'surface_elo': float(row2[2]) if row2[2] else 1500.0,
            'rank': int(row2[3]) if row2[3] else 100,
            'ace_rate': float(row2[4]) if row2[4] else 0.06,
            '1st_in': float(row2[5]) if row2[5] else 0.60,
            '1st_won': float(row2[6]) if row2[6] else 0.70,
            'bp_saved': float(row2[7]) if row2[7] else 0.60,
            'days_since': int(row2[8]) if row2[8] else 7,
            'last14': int(row2[9]) if row2[9] else 3,
            'h2h_wins': int(row2[10]) if row2[10] else 0,
            'exp': int(row2[11]) if row2[11] else 100,
        }

    return None


async def _already_alerted(session: AsyncSession, odds_api_id: str) -> bool:
    result = await session.execute(text('''
        SELECT alert_sent FROM tennis_upcoming_matches
        WHERE odds_api_id = :oid AND alert_sent = TRUE
    '''), {'oid': odds_api_id})
    return result.fetchone() is not None


async def _send_telegram(message: str, bot_token: str, chat_id: str) -> bool:
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


def _format_message(p1, p2, winner, confidence, p1_prob, p2_prob,
                    p1_odds, p2_odds, commence_time, tournament):
    loser = p2 if winner == p1 else p1
    winner_odds = p1_odds if winner == p1 else p2_odds

    # Edge = model prob vs market implied prob
    if winner == p1 and p1_odds:
        market_prob = 1 / p1_odds
        edge = (p1_prob - market_prob) * 100
    elif winner == p2 and p2_odds:
        market_prob = 1 / p2_odds
        edge = (p2_prob - market_prob) * 100
    else:
        edge = 0

    edge_emoji = '✅' if edge >= 5 else '⚠️' if edge >= 2 else '➖'
    time_str = commence_time.strftime('%A %d %B, %H:%M UTC')

    return (
        f'🎾 <b>TENNIS PREDICTION</b>\n'
        f'{'─'*30}\n'
        f'🏆 {tournament}\n'
        f'📅 {time_str}\n\n'
        f'<b>{p1}</b> vs <b>{p2}</b>\n\n'
        f'🎯 <b>Model picks: {winner}</b>\n'
        f'📊 Confidence: {confidence:.1f}%\n'
        f'📈 {p1}: {p1_prob:.1f}% | {p2}: {p2_prob:.1f}%\n\n'
        f'💰 Odds: {p1} {p1_odds or "N/A"} | {p2} {p2_odds or "N/A"}\n'
        f'{edge_emoji} Model edge: {edge:+.1f}%\n'
    )


async def run_tennis_alert_engine(db: AsyncSession) -> dict:
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat_id   = os.getenv('TELEGRAM_CHAT_ID', '')
    api_key   = os.getenv('ODDS_API_KEY', '')

    now          = datetime.now(timezone.utc)
    window_start = now
    window_end   = now + timedelta(hours=3)

    alerts_sent = 0
    skipped_duplicate = 0
    no_data = 0
    errors = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        for sport_key in TENNIS_SPORT_KEYS:
            try:
                r = await client.get(
                    f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds',
                    params={
                        'apiKey': api_key,
                        'regions': 'eu,uk',
                        'markets': 'h2h',
                        'oddsFormat': 'decimal',
                    }
                )
                matches = r.json()
            except Exception as e:
                logger.error(f'Odds API failed for {sport_key}: {e}')
                continue

            tournament = 'ATP Wimbledon' if 'atp' in sport_key else 'WTA Wimbledon'

            for match in matches:
                try:
                    odds_api_id  = match.get('id', '')
                    p1_name      = match.get('home_team', '')
                    p2_name      = match.get('away_team', '')
                    commence_str = match.get('commence_time', '')

                    # Parse match time
                    commence_time = datetime.fromisoformat(
                        commence_str.replace('Z', '+00:00')
                    )

                    # Skip past matches
                    if commence_time <= now:
                        continue

                    # Only alert if match is within 3 hours
                    if not (window_start <= commence_time <= window_end):
                        continue

                    # Skip duplicates
                    if await _already_alerted(db, odds_api_id):
                        skipped_duplicate += 1
                        continue

                    # Get odds
                    p1_odds = p2_odds = None
                    for bk in match.get('bookmakers', []):
                        for mkt in bk.get('markets', []):
                            if mkt['key'] == 'h2h':
                                for outcome in mkt['outcomes']:
                                    if outcome['name'] == p1_name:
                                        p1_odds = outcome['price']
                                    elif outcome['name'] == p2_name:
                                        p2_odds = outcome['price']
                        if p1_odds and p2_odds:
                            break

                    # Look up ELO from historical data
                    p1_data = await _get_player_elo(db, p1_name)
                    p2_data = await _get_player_elo(db, p2_name)

                    if not p1_data or not p2_data:
                        logger.info(f'No ELO data for {p1_name} or {p2_name} — using defaults')
                        no_data += 1
                        p1_data = p1_data or {'elo': 1500.0, 'surface_elo': 1500.0, 'rank': 100}
                        p2_data = p2_data or {'elo': 1500.0, 'surface_elo': 1500.0, 'rank': 100}

                    # Run model
                    p1_prob, p2_prob = _predict(
                        p1_elo=p1_data['elo'],
                        p2_elo=p2_data['elo'],
                        p1_surface_elo=p1_data['surface_elo'],
                        p2_surface_elo=p2_data['surface_elo'],
                        p1_rank=p1_data['rank'],
                        p2_rank=p2_data['rank'],
                        p1_ace_rate=p1_data.get('ace_rate', 0.06),
                        p2_ace_rate=p2_data.get('ace_rate', 0.06),
                        p1_1st_in=p1_data.get('1st_in', 0.60),
                        p2_1st_in=p2_data.get('1st_in', 0.60),
                        p1_1st_won=p1_data.get('1st_won', 0.70),
                        p2_1st_won=p2_data.get('1st_won', 0.70),
                        p1_bp_saved=p1_data.get('bp_saved', 0.60),
                        p2_bp_saved=p2_data.get('bp_saved', 0.60),
                        p1_days_since=p1_data.get('days_since', 7),
                        p2_days_since=p2_data.get('days_since', 7),
                        p1_last14=p1_data.get('last14', 3),
                        p2_last14=p2_data.get('last14', 3),
                        p1_h2h_wins=p1_data.get('h2h_wins', 0),
                        p2_h2h_wins=p2_data.get('h2h_wins', 0),
                        p1_exp=p1_data.get('exp', 100),
                        p2_exp=p2_data.get('exp', 100),
                        surface='Grass',
                        level='G',
                        round_='R64',
                        best_of=3,
                    )

                    if p1_prob is None:
                        errors.append(f'Model failed for {p1_name} vs {p2_name}')
                        continue

                    winner     = p1_name if p1_prob >= p2_prob else p2_name
                    confidence = max(p1_prob, p2_prob) * 100

                    # Store in DB
                    await db.execute(text('''
                        INSERT INTO tennis_upcoming_matches
                            (odds_api_id, p1_name, p2_name, commence_time, tournament,
                             sport_key, p1_odds, p2_odds, predicted_winner,
                             p1_win_prob, p2_win_prob, confidence, alert_sent)
                        VALUES
                            (:oid, :p1, :p2, :ct, :tour, :sk, :p1o, :p2o, :pw,
                             :p1p, :p2p, :conf, FALSE)
                        ON CONFLICT (odds_api_id) DO UPDATE SET
                            p1_odds = EXCLUDED.p1_odds,
                            p2_odds = EXCLUDED.p2_odds,
                            predicted_winner = EXCLUDED.predicted_winner,
                            p1_win_prob = EXCLUDED.p1_win_prob,
                            p2_win_prob = EXCLUDED.p2_win_prob,
                            confidence = EXCLUDED.confidence
                    '''), {
                        'oid': odds_api_id, 'p1': p1_name, 'p2': p2_name,
                        'ct': commence_time, 'tour': tournament, 'sk': sport_key,
                        'p1o': p1_odds, 'p2o': p2_odds, 'pw': winner,
                        'p1p': p1_prob * 100, 'p2p': p2_prob * 100, 'conf': confidence,
                    })

                    # Format and send Telegram message
                    message = _format_message(
                        p1=p1_name, p2=p2_name,
                        winner=winner, confidence=confidence,
                        p1_prob=p1_prob * 100, p2_prob=p2_prob * 100,
                        p1_odds=p1_odds, p2_odds=p2_odds,
                        commence_time=commence_time, tournament=tournament,
                    )

                    sent = False
                    if bot_token and chat_id:
                        sent = await _send_telegram(message, bot_token, chat_id)

                    if sent:
                        await db.execute(text('''
                            UPDATE tennis_upcoming_matches
                            SET alert_sent = TRUE
                            WHERE odds_api_id = :oid
                        '''), {'oid': odds_api_id})
                        alerts_sent += 1
                        logger.info(f'Tennis alert sent: {p1_name} vs {p2_name}')

                except Exception as e:
                    logger.error(f'Error processing match: {e}')
                    errors.append(str(e))

    await db.commit()

    return {
        'engine': 'Tennis Alert Engine',
        'window': f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')} UTC",
        'alerts_sent': alerts_sent,
        'skipped_duplicate': skipped_duplicate,
        'no_elo_data': no_data,
        'errors': errors[:5],
    }
