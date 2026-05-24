import React, { useState, useEffect } from 'react'
import HistoryDashboard from '../components/HistoryDashboard'
import { getHistory } from '../services/storage'

export default function DashboardPage() {
  const [history, setHistory] = useState([])

  function refresh() {
    setHistory(getHistory())
  }

  useEffect(() => {
    refresh()
  }, [])

  return (
    <div>
      <div className="dashboard-header">
        <h2>Analysis History</h2>
      </div>
      <HistoryDashboard history={history} onHistoryChange={refresh} />
    </div>
  )
}
