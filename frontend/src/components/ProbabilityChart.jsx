import React from 'react'
import '../styles/components.css'

const CLASS_ORDER = ['Normal', 'AFib', 'PVC', 'Bradycardia', 'Tachycardia']

// Normalize key casing: api may return lowercase keys
function normalizeProbs(probs) {
  const map = {}
  for (const [k, v] of Object.entries(probs)) {
    // Capitalize first letter for display matching
    const norm = k.charAt(0).toUpperCase() + k.slice(1)
    map[norm] = v
  }
  return map
}

export default function ProbabilityChart({ probs, winner }) {
  if (!probs || Object.keys(probs).length === 0) return null

  const normalized = normalizeProbs(probs)

  // Determine display order: use CLASS_ORDER, then any extra keys
  const known = CLASS_ORDER.filter(c => normalized[c] != null)
  const extra = Object.keys(normalized).filter(c => !CLASS_ORDER.includes(c))
  const order = [...known, ...extra]

  return (
    <div className="prob-chart" role="list" aria-label="Class probabilities">
      {order.map(cls => {
        const p = normalized[cls] ?? 0
        const pct = (p * 100).toFixed(1)
        const isWinner = cls.toLowerCase() === (winner || '').toLowerCase()
        return (
          <div key={cls} className="prob-row" role="listitem">
            <span className="prob-label">{cls}</span>
            <div className="prob-bar-bg" aria-hidden="true">
              <div
                className={`prob-bar-fill${isWinner ? ' winner' : ''}`}
                style={{ width: `${Math.min(p * 100, 100)}%` }}
              />
            </div>
            <span className="prob-pct" aria-label={`${pct}%`}>
              {pct}%
            </span>
          </div>
        )
      })}
    </div>
  )
}
