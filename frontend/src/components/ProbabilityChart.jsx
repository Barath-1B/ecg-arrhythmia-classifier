import React from 'react'
import '../styles/components.css'

const CLASS_ORDER = ['Normal', 'AFib', 'PVC', 'Bradycardia', 'Tachycardia']

export default function ProbabilityChart({ probs, winner }) {
  if (!probs || Object.keys(probs).length === 0) return null

  return (
    <div className="prob-chart" role="list" aria-label="Class probabilities">
      {CLASS_ORDER.map(cls => {
        const p = probs[cls] ?? 0
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
