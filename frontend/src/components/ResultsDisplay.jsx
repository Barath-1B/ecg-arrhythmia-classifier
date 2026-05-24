import React from 'react'
import MetricsGrid from './MetricsGrid'
import ProbabilityChart from './ProbabilityChart'
import { SEV_LABEL } from '../constants'
import '../styles/components.css'

export default function ResultsDisplay({ report, onNewAnalysis }) {
  if (!report) return null

  if (report.error) {
    return (
      <div className="results-container">
        <div className="card error-box" role="alert">
          <strong>Analysis Error:</strong> {report.error}
          {report.warnings?.length > 0 && (
            <ul style={{ marginTop: '0.5rem', paddingLeft: '1.2rem' }}>
              {report.warnings.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          )}
        </div>
        <div>
          <button className="btn btn-secondary" onClick={onNewAnalysis}>
            Try Another File
          </button>
        </div>
      </div>
    )
  }

  const sev = report.severity || 'unknown'
  const method = report.method === 'ml_fallback' ? 'ML model' : 'Clinical rules'

  function downloadReport() {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ecg_report_${report.patient_id}_${report.analysis_id?.slice(0,8) || 'result'}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="results-container">

      {/* ── Diagnosis card ─────────────────────────────────── */}
      <div className="card diagnosis-card">
        <div className="card-title">Diagnosis</div>
        <div className="diagnosis-header">
          <span className="diagnosis-name">{report.diagnosis}</span>
          <span className={`badge badge-${sev}`} aria-label={`Severity: ${SEV_LABEL[sev] || sev}`}>
            {SEV_LABEL[sev] || sev} severity
          </span>
          <span className="diagnosis-confidence">
            Confidence: {report.confidence != null ? (report.confidence * 100).toFixed(1) + '%' : 'N/A'}
          </span>
          <span className="diagnosis-method">{method}</span>
        </div>

        {report.recommendation && (
          <div className="recommendation-text" role="note">
            {report.recommendation}
          </div>
        )}

        {/* Warnings */}
        {report.warnings?.length > 0 && (
          <div className="warning-box" role="alert">
            <strong>Clinical Warnings:</strong>
            <ul className="warnings-list" style={{ marginTop: '0.4rem', paddingLeft: '1rem' }}>
              {report.warnings.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          </div>
        )}

        <div className="disclaimer" role="note">{report.disclaimer}</div>
      </div>

      {/* ── Metrics ────────────────────────────────────────── */}
      <MetricsGrid metrics={report.metrics} />

      {/* ── ML Probabilities ───────────────────────────────── */}
      {report.ml_probabilities && Object.keys(report.ml_probabilities).length > 0 && (
        <div className="card">
          <div className="card-title">ML Class Probabilities</div>
          <ProbabilityChart probs={report.ml_probabilities} winner={report.diagnosis} />
        </div>
      )}

      {/* ── Actions ────────────────────────────────────────── */}
      <div className="action-row">
        <button className="btn btn-primary" onClick={downloadReport}>
          Download Report (JSON)
        </button>
        <button className="btn btn-secondary" onClick={onNewAnalysis}>
          New Analysis
        </button>
      </div>

    </div>
  )
}
