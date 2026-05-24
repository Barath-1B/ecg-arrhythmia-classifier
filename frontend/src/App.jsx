import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import AnalysisPage from './pages/AnalysisPage'
import DashboardPage from './pages/DashboardPage'

export default function App() {
  return (
    <div className="app">
      <Header />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<AnalysisPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
        </Routes>
      </main>
    </div>
  )
}
