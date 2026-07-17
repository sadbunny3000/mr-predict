import { useEffect, useState } from 'react'
import { getAccuracy, getTennisMatches, getFootballMatches, getAlertHistory } from './api.js'

const GAUGE_TICKS = [50, 60, 70, 80, 90]

function gaugePercent(confidence) {
  const clamped = Math.max(50, Math.min(90, confidence))
  return ((clamped - 50) / 40) * 100
}

function ConfidenceGauge({ confidence, sportClass, isAlert }) {
  const markerClass = isAlert ? 'signal' : sportClass
  const readoutClass = isAlert ? 'signal' : sportClass
  return (
    <div className="gauge-row">
      <div className="gauge">
        <div className="gauge-track"></div>
        {GAUGE_TICKS.map((tick, i) => (
          <div key={tick}>
            <div className="gauge-tick" style={{ left: `${(i / (GAUGE_TICKS.length - 1)) * 100}%` }}></div>
            <div className="gauge-tick-label" style={{ left: `${(i / (GAUGE_TICKS.length - 1)) * 100}%` }}>{tick}</div>
          </div>
        ))}
        <div className={`gauge-marker ${markerClass}`} style={{ left: `${gaugePercent(confidence)}%` }}></div>
      </div>
      <div className={`confidence-readout ${readoutClass}`}>{confidence.toFixed(1)}%</div>
    </div>
  )
}

function AccuracyStrip({ accuracy }) {
  if (!accuracy) return null
  return (
    <div className="accuracy-strip">
      <div className="accuracy-cell">
        <div className="accuracy-label">Tennis hit rate (30d)</div>
        <div className="accuracy-value tennis">{accuracy.tennis_hit_rate_30d.toFixed(1)}%</div>
      </div>
      <div className="accuracy-cell">
        <div className="accuracy-label">Football hit rate (30d)</div>
        <div className="accuracy-value football">{accuracy.football_hit_rate_30d.toFixed(1)}%</div>
      </div>
      <div className="accuracy-cell">
        <div className="accuracy-label">Alerts sent (30d)</div>
        <div className="accuracy-value" style={{ color: 'var(--parchment)' }}>{accuracy.alerts_sent_30d}</div>
      </div>
    </div>
  )
}

function TennisMatchCard({ match }) {
  const winnerName = match.predicted_winner === 'player1' ? match.player1 : match.player2
  const confidence = match.predicted_winner === 'player1' ? match.model_prob.p1 : match.model_prob.p2
  return (
    <div className="card">
      <div className="card-top">
        <div className="matchup">{match.player1}<span className="vs">vs</span>{match.player2}</div>
        <div className="meta">{match.tour} · {match.surface} · {match.round}</div>
      </div>
      <div className="odds-row">
        <div className="odds-item">Market: <strong>{match.market_odds.p1} / {match.market_odds.p2}</strong></div>
        <div className="odds-item">Model: <strong>{match.model_prob.p1.toFixed(1)}% / {match.model_prob.p2.toFixed(1)}%</strong></div>
      </div>
      <div className="subline">Winner prediction</div>
      <ConfidenceGauge confidence={confidence} sportClass="tennis" isAlert={match.is_alert} />
      <div className="pick-line">
        Model favors <strong>{winnerName}</strong>
        {match.total_games && ` · total games: ${match.total_games.low}–${match.total_games.high} (median ${match.total_games.median})`}
        {match.is_alert && ' · high-confidence alert sent'}
      </div>
    </div>
  )
}

function FootballPropRow({ label, value }) {
  return (
    <div className="prop-row">
      <div className="prop-name">{label}</div>
      {value == null
        ? <div className="prop-value pending">pending</div>
        : <div className="prop-value ready">{value}</div>}
    </div>
  )
}

function FootballMatchCard({ match }) {
  const resultLabel = match.predicted_result === 'home_win'
    ? `${match.home_team} win`
    : match.predicted_result === 'away_win'
      ? `${match.away_team} win`
      : 'Draw'
  return (
    <div className="card">
      <div className="card-top">
        <div className="matchup">{match.home_team}<span className="vs">vs</span>{match.away_team}</div>
        <div className="meta">{match.competition} · {match.round_label}</div>
      </div>
      <div className="odds-row">
        <div className="odds-item">Market: <strong>{match.market_odds.home} / {match.market_odds.draw} / {match.market_odds.away}</strong></div>
        <div className="odds-item">Model: <strong>{match.model_prob_home_win.toFixed(1)}%</strong></div>
      </div>
      <div className="subline">Match result prediction</div>
      <ConfidenceGauge confidence={match.model_prob_home_win} sportClass="football" isAlert={match.is_alert} />
      <div className="pick-line">Model favors <strong>{resultLabel}</strong></div>

      <div className="prop-list">
        <div className="prop-list-label">Match props</div>
        <FootballPropRow label="Total corners" value={match.props.total_corners} />
        <FootballPropRow label="Corners — 1st half" value={match.props.corners_first_half} />
        <FootballPropRow label="Corners — 2nd half" value={match.props.corners_second_half} />
        <FootballPropRow label="Total throw-ins" value={match.props.total_throw_ins} />
      </div>
    </div>
  )
}

function AlertHistoryList({ alerts }) {
  if (!alerts || alerts.length === 0) {
    return <div className="empty">— no alerts sent yet —</div>
  }
  return (
    <>
      {alerts.map((a) => (
        <div className="alert-item" key={a.id}>
          <div className="alert-left">
            <div className="alert-match">{a.match}</div>
            <div className="alert-meta">{a.confidence.toFixed(1)}% · sent {a.sent_ago}</div>
          </div>
          <div className={`alert-outcome ${a.outcome === 'pending' ? 'pending-outcome' : a.outcome}`}>
            {a.outcome}
          </div>
        </div>
      ))}
    </>
  )
}

export default function App() {
  const [sport, setSport] = useState('tennis')
  const [accuracy, setAccuracy] = useState(null)
  const [tennisMatches, setTennisMatches] = useState(null)
  const [footballMatches, setFootballMatches] = useState(null)
  const [alertHistory, setAlertHistory] = useState(null)

  useEffect(() => {
    getAccuracy().then(setAccuracy)
    getTennisMatches().then(setTennisMatches)
    getFootballMatches().then(setFootballMatches)
    getAlertHistory().then(setAlertHistory)
  }, [])

  return (
    <div className="wrap">
      <header>
        <div className="eyebrow">Mr Predict — Live Board</div>
        <h1>Today's picks</h1>
      </header>

      <AccuracyStrip accuracy={accuracy} />

      <div className="toggle">
        <button
          className={sport === 'tennis' ? 'active tennis' : ''}
          onClick={() => setSport('tennis')}
        >
          Tennis
        </button>
        <button
          className={sport === 'football' ? 'active football' : ''}
          onClick={() => setSport('football')}
        >
          Football
        </button>
      </div>

      <div className="section-label">Upcoming — {sport === 'tennis' ? 'Tennis' : 'Football'}</div>

      {sport === 'tennis' ? (
        tennisMatches === null
          ? <div className="loading-text">Loading…</div>
          : tennisMatches.length === 0
            ? <div className="empty">— no upcoming tennis matches —</div>
            : tennisMatches.map((m) => <TennisMatchCard match={m} key={m.id} />)
      ) : (
        footballMatches === null
          ? <div className="loading-text">Loading…</div>
          : footballMatches.length === 0
            ? <div className="empty">— no upcoming football matches —</div>
            : footballMatches.map((m) => <FootballMatchCard match={m} key={m.id} />)
      )}

      <div className="section-label">Alert history</div>
      <AlertHistoryList alerts={alertHistory} />
    </div>
  )
}
