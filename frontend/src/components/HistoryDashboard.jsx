import React, { useState, useMemo } from 'react'
import { deleteAnalysis, clearHistory } from '../services/storage'
import { SEV_LABEL, fmtDate } from '../constants'
import '../styles/components.css'

export default function HistoryDashboard({ history, onHistoryChange }) {
  const [search, setSearch] = useState('')
  const [filterDiag, setFilterDiag] = useState('')

  // Memoize diagnoses list
  const diagnoses = useMemo(
    () => [...new Set(history.map(r => r.diagnosis).filter(Boolean))],
    [history]
  )

  // Normalize search term once
  const searchLower = useMemo(() => search.toLowerCase(), [search])

  // Memoize filtered results
  const filtered = useMemo(() => {
    return history.filter(r => {
      const matchSearch = !searchLower ||
        (r.patient_id || '').toLowerCase().includes(searchLower) ||
        (r.diagnosis || '').toLowerCase().includes(searchLower)
      const matchDiag = !filterDiag || r.diagnosis === filterDiag
      return matchSearch && matchDiag
    })
  }, [history, searchLower, filterDiag])

  // Memoize summary stats
  const { total, avgConf, diagCounts } = useMemo(() => {
    const total = history.length
    const avgConf = total > 0
      ? (history.reduce((s, r) => s + (r.confidence || 0), 0) / total * 100).toFixed(1)
      : '—'
    const diagCounts = history.reduce((acc, r) => {
      if (r.diagnosis) acc[r.diagnosis] = (acc[r.diagnosis] || 0) + 1
      return acc
    }, {})
    return { total, avgConf, diagCounts }
  }, [history])

  function handleDelete(id) {
    deleteAnalysis(id)
    onHistoryChange()
  }

  function handleClearAll() {
    if (window.confirm('Delete all analysis history? This cannot be undone.')) {
      clearHistory()
      onHistoryChange()
    }
  }

  function downloadReport(report) {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ecg_report_${report.patient_id}_${(report.analysis_id || 'x').slice(0, 8)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (history.length === 0) {
    return (
      <div className="card">
        <div className="empty-state">
          <div className="empty-icon" aria-hidden="true">📋</div>
          <p>No analyses yet. Upload an ECG file to get started.</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      {/* Stats */}
      <div className="stats-row" aria-label="Summary statistics">
        <div className="stat-card">
          <div className="stat-value">{total}</div>
          <div className="stat-label">Total Analyses</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{avgConf}%</div>
          <div className="stat-label">Avg Confidence</div>
        </div>
      </div>

      {/* Diagnosis distribution pills */}
      {Object.keys(diagCounts).length > 0 && (
        <div className="diag-counts" aria-label="Diagnosis breakdown">
          {Object.entries(diagCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([diag, count]) => (
              <span key={diag} className="diag-pill">{diag}: {count}</span>
            ))
          }
        </div>
      )}

      {/* Search / filter */}
      <div className="search-row">
        <input
          type="text"
          placeholder="Search by Patient ID or Diagnosis…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          aria-label="Search history"
        />
        <select
          value={filterDiag}
          onChange={e => setFilterDiag(e.target.value)}
          aria-label="Filter by diagnosis"
        >
          <option value="">All Diagnoses</option>
          {diagnoses.map(d => <option key={d} value={d}>{d}</option>)}
        </select>
        <button className="btn btn-danger btn-sm" onClick={handleClearAll}>
          Clear All
        </button>
      </div>

      {/* Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="history-table" aria-label="Analysis history">
          <thead>
            <tr>
              <th>Date / Time</th>
              <th>Patient ID</th>
              <th>Diagnosis</th>
              <th>Confidence</th>
              <th>Severity</th>
              <th>Duration</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', color: 'var(--color-text-muted)', padding: '2rem' }}>
                  No results match your search.
                </td>
              </tr>
            ) : (
              filtered.map(r => {
                const sev = r.severity || 'unknown'
                const conf = r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : '—'
                const dur = r.metrics?.duration_s != null ? r.metrics.duration_s.toFixed(1) + ' s' : '—'
                return (
                  <tr key={r.analysis_id || r.timestamp}>
                    <td>{fmtDate(r.timestamp)}</td>
                    <td>{r.patient_id || 'UNKNOWN'}</td>
                    <td><strong>{r.diagnosis || '—'}</strong></td>
                    <td>{conf}</td>
                    <td>
                      <span className={`badge badge-${sev}`} style={{ fontSize: '0.72rem' }}>
                        {SEV_LABEL[sev] || sev}
                      </span>
                    </td>
                    <td>{dur}</td>
                    <td>
                      <div className="table-actions">
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => downloadReport(r)}
                          aria-label={`Download report for ${r.patient_id}`}
                        >
                          Download
                        </button>
                        <button
                          className="btn btn-danger btn-sm"
                          onClick={() => handleDelete(r.analysis_id)}
                          aria-label={`Delete analysis for ${r.patient_id}`}
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
