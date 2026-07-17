// Mock data shaped exactly like the real API responses should look once the
// matching backend read-endpoints exist. Swap api.js's USE_MOCK flag to false
// once those endpoints are live and this file becomes unused (safe to delete).

export const mockAccuracy = {
  tennis_hit_rate_30d: 64.2,
  football_hit_rate_30d: 58.7,
  alerts_sent_30d: 47,
}

export const mockTennisMatches = [
  {
    id: 'tm-1',
    player1: 'Sinner',
    player2: 'Alcaraz',
    tour: 'ATP',
    surface: 'HARD',
    round: 'R32',
    market_odds: { p1: 1.62, p2: 2.35 },
    model_prob: { p1: 72.4, p2: 27.6 },
    predicted_winner: 'player1',
    total_games: { low: 22, high: 24, median: 23 },
    is_alert: false,
  },
  {
    id: 'tm-2',
    player1: 'Swiatek',
    player2: 'Gauff',
    tour: 'WTA',
    surface: 'CLAY',
    round: 'QF',
    market_odds: { p1: 1.30, p2: 3.40 },
    model_prob: { p1: 86.1, p2: 13.9 },
    predicted_winner: 'player1',
    total_games: null,
    is_alert: true,
  },
]

export const mockFootballMatches = [
  {
    id: 'fm-1',
    home_team: 'Arsenal',
    away_team: 'Chelsea',
    competition: 'EPL',
    round_label: 'GW 3',
    market_odds: { home: 2.05, draw: 3.60, away: 3.80 },
    model_prob_home_win: 66.2,
    predicted_result: 'home_win',
    is_alert: false,
    props: {
      total_corners: null,
      corners_first_half: null,
      corners_second_half: null,
      total_throw_ins: null,
    },
  },
]

export const mockAlertHistory = [
  { id: 'a-1', match: 'Sinner vs Medvedev', confidence: 72.1, sent_ago: '2h ago', outcome: 'win' },
  { id: 'a-2', match: 'Liverpool vs Man City', confidence: 64.8, sent_ago: '5h ago', outcome: 'loss' },
  { id: 'a-3', match: 'Djokovic vs Zverev', confidence: 79.3, sent_ago: '8h ago', outcome: 'win' },
  { id: 'a-4', match: 'Swiatek vs Gauff', confidence: 86.1, sent_ago: '20m ago', outcome: 'pending' },
]
