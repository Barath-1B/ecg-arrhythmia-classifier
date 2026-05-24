import React from 'react'
import { fmt } from '../constants'
import '../styles/components.css'

export default function MetricsGrid({ metrics }) {
  if (!metrics) return null
  const m = metrics

  return (
    <div className="metrics-grid">

      {/* ── Heart Rate ──────────────────────────────────── */}
      <div className="card metric-card">
        <div className="card-title">Heart Rate</div>
        <div className="metric-value">
          {fmt(m.hr_mean, 0)}
          <span className="metric-unit"> bpm</span>
        </div>
        <div className="metric-target">Target: 60–100 bpm</div>
        <div style={{ marginTop: '0.75rem' }}>
          <div className="metric-row">
            <span className="mlabel">Std dev</span>
            <span className="mvalue">±{fmt(m.hr_std, 1)} bpm</span>
          </div>
          <div className="metric-row">
            <span className="mlabel">R-peaks</span>
            <span className="mvalue">{m.r_peaks_detected ?? 'N/A'}</span>
          </div>
          <div className="metric-row">
            <span className="mlabel">Duration</span>
            <span className="mvalue">{fmt(m.duration_s, 1)} s</span>
          </div>
        </div>
      </div>

      {/* ── Regularity / HRV ────────────────────────────── */}
      <div className="card metric-card">
        <div className="card-title">Regularity (HRV)</div>
        <div className="metric-value">
          {fmt(m.sdnn, 0)}
          <span className="metric-unit"> ms SDNN</span>
        </div>
        <div className="metric-target">Healthy: &gt; 50 ms</div>
        <div style={{ marginTop: '0.75rem' }}>
          <div className="metric-row">
            <span className="mlabel">RR CV</span>
            <span className="mvalue">{fmt(m.rr_cv, 3)}</span>
          </div>
          <div className="metric-row">
            <span className="mlabel">RR Entropy</span>
            <span className="mvalue">{fmt(m.rr_entropy, 3)}</span>
          </div>
          <div className="metric-row">
            <span className="mlabel">RMSSD</span>
            <span className="mvalue">{fmt(m.rmssd, 1)} ms</span>
          </div>
          <div className="metric-row">
            <span className="mlabel">pNN50</span>
            <span className="mvalue">{fmt(m.pnn50, 1)}%</span>
          </div>
        </div>
      </div>

      {/* ── Morphology ──────────────────────────────────── */}
      <div className="card metric-card">
        <div className="card-title">Morphology</div>
        <div className="metric-value">
          {fmt(m.qrs_dur_mean, 0)}
          <span className="metric-unit"> ms QRS</span>
        </div>
        <div className="metric-target">Normal: &lt; 120 ms</div>
        <div style={{ marginTop: '0.75rem' }}>
          <div className="metric-row">
            <span className="mlabel">QRS std</span>
            <span className="mvalue">±{fmt(m.qrs_dur_std, 1)} ms</span>
          </div>
          <div className="metric-row">
            <span className="mlabel">P-wave ratio</span>
            <span className="mvalue">{fmt(m.p_wave_ratio, 3)}</span>
          </div>
          <div className="metric-row">
            <span className="mlabel">ST elevation</span>
            <span className="mvalue">{fmt(m.st_elevation, 1)} μV</span>
          </div>
        </div>
      </div>

    </div>
  )
}
