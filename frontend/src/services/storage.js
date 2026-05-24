const KEY = 'ecg_history'
const MAX_ENTRIES = 100

export function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '[]')
  } catch {
    return []
  }
}

export function saveAnalysis(report) {
  const history = getHistory()
  history.unshift(report)
  localStorage.setItem(KEY, JSON.stringify(history.slice(0, MAX_ENTRIES)))
}

export function deleteAnalysis(analysisId) {
  const updated = getHistory().filter(r => r.analysis_id !== analysisId)
  localStorage.setItem(KEY, JSON.stringify(updated))
}

export function clearHistory() {
  localStorage.removeItem(KEY)
}
