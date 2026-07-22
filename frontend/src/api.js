import {
  mockAccuracy,
  mockTennisMatches,
  mockFootballMatches,
  mockAlertHistory,
} from './mockData.js'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

async function fetchJson(path) {
  if (!API_BASE) {
    throw new Error('No API_BASE configured — using mock data')
  }
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) {
    throw new Error(`Request to ${path} failed: ${res.status}`)
  }
  return res.json()
}

export async function getAccuracy() {
  try {
    return await fetchJson('/api/v1/accuracy')
  } catch {
    return mockAccuracy
  }
}

export async function getTennisMatches() {
  try {
    return await fetchJson('/api/v1/tennis/predictions/upcoming')
  } catch {
    return mockTennisMatches
  }
}

export async function getFootballMatches() {
  try {
    return await fetchJson('/api/v1/predictions/upcoming')
  } catch {
    return mockFootballMatches
  }
}

export async function getAlertHistory() {
  try {
    return await fetchJson('/api/v1/alerts/history')
  } catch {
    return mockAlertHistory
  }
}

export async function predictMatchup(payload) {
  const res = await fetch(`${API_BASE}/api/v1/tennis/predict/matchup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const data = await res.json()
  if (!res.ok) {
    throw new Error(data.detail || `Request failed: ${res.status}`)
  }
  return data
}
