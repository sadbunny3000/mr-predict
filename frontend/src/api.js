import {
  mockAccuracy,
  mockTennisMatches,
  mockFootballMatches,
  mockAlertHistory,
} from './mockData.js'

// Set this in a .env file as VITE_API_BASE_URL once Railway is back online,
// e.g. VITE_API_BASE_URL=https://authentic-flow-production-9812.up.railway.app
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

// TODO: point these at real read-endpoints once they exist on the backend.
// None of the four endpoints below exist yet — this whole app currently
// runs on mock data only. Planned routes (not yet built):
//   GET /api/v1/tennis/predictions/upcoming
//   GET /api/v1/football/predictions/upcoming
//   GET /api/v1/tennis/alerts/history   (or a combined /alerts/history)
//   GET /api/v1/accuracy

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
    return await fetchJson('/api/v1/football/predictions/upcoming')
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
