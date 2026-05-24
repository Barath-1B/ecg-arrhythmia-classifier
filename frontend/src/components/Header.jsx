import React from 'react'
import { NavLink } from 'react-router-dom'
import '../styles/components.css'

export default function Header() {
  return (
    <header className="header">
      <div className="header-logo">
        <span aria-hidden="true">🫀</span>
        ECG Arrhythmia Monitor
      </div>
      <nav className="header-nav" aria-label="Main navigation">
        <NavLink
          to="/"
          end
          className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
        >
          New Analysis
        </NavLink>
        <NavLink
          to="/dashboard"
          className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
        >
          History
        </NavLink>
      </nav>
    </header>
  )
}
