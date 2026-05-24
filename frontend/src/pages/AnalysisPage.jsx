import React, { useState } from 'react'
import UploadForm from '../components/UploadForm'
import ResultsDisplay from '../components/ResultsDisplay'
import { analyzeECG } from '../services/api'
import { saveAnalysis } from '../services/storage'

export default function AnalysisPage() {
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState(null)
  const [apiError, setApiError] = useState('')

  async function handleSubmit(file, patientId, samplingRate) {
    setLoading(true)
    setApiError('')
    setReport(null)
    try {
      const result = await analyzeECG(file, patientId, samplingRate)
      setReport(result)
      saveAnalysis(result)
    } catch (err) {
      setApiError(err.message || 'Connection failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  function handleNewAnalysis() {
    setReport(null)
    setApiError('')
  }

  return (
    <div>
      <h1 style={{ marginBottom: '1.5rem', fontSize: '1.5rem' }}>
        ECG Arrhythmia Analysis
      </h1>

      {!report && (
        <>
          <UploadForm onSubmit={handleSubmit} loading={loading} />
          {apiError && (
            <div className="error-box" role="alert" style={{ marginTop: '1rem', maxWidth: 580 }}>
              <strong>Error:</strong> {apiError}
            </div>
          )}
        </>
      )}

      {report && (
        <ResultsDisplay report={report} onNewAnalysis={handleNewAnalysis} />
      )}
    </div>
  )
}
