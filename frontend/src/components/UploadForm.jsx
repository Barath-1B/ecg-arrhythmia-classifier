import React, { useState, useRef } from 'react'
import { DEFAULT_SAMPLING_RATE, FILE_ACCEPT_STRING } from '../constants'
import { validateAndParseECGFile } from '../utils/fileValidation'
import '../styles/components.css'

export default function UploadForm({ onSubmit, loading }) {
  const [file, setFile] = useState(null)
  const [fileInfo, setFileInfo] = useState(null)
  const [patientId, setPatientId] = useState('')
  const [samplingRate, setSamplingRate] = useState(DEFAULT_SAMPLING_RATE)
  const [validationError, setValidationError] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  async function handleFileSelect(selectedFile) {
    setValidationError('')

    if (!selectedFile) {
      setFile(null)
      setFileInfo(null)
      return
    }

    try {
      const info = await validateAndParseECGFile(selectedFile, samplingRate)
      setFileInfo(info)
      setFile(selectedFile)
    } catch (err) {
      setValidationError(err.message)
      setFile(null)
      setFileInfo(null)
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) handleFileSelect(dropped)
  }

  function handleSubmit(e) {
    e.preventDefault()
    if (!file) { setValidationError('Please select a file first.'); return }
    onSubmit(file, patientId, samplingRate)
  }

  return (
    <form className="upload-form card" onSubmit={handleSubmit} aria-label="ECG Analysis form">
      <h2>Upload ECG Recording</h2>

      {/* File drop zone */}
      <div className="form-group">
        <label>ECG File (CSV / TXT)</label>
        <div
          className={`file-drop-zone${dragging ? ' dragging' : ''}`}
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
          aria-label="Click or drag to upload ECG file"
          onKeyDown={(e) => e.key === 'Enter' && inputRef.current.click()}
        >
          <input
            type="file"
            ref={inputRef}
            accept={FILE_ACCEPT_STRING}
            onChange={(e) => handleFileSelect(e.target.files[0])}
            aria-hidden="true"
          />
          <div className="drop-icon" aria-hidden="true">📂</div>
          <div className="drop-label">
            <strong>Click to browse</strong> or drag and drop<br />
            Single-column CSV (mV values, one per row)
          </div>
        </div>

        {fileInfo && (
          <div className="file-info" role="status">
            <span className="file-name">📄 {fileInfo.name}</span>
            <span className="file-meta">
              {fileInfo.size} &bull; {fileInfo.rows.toLocaleString()} samples &bull; ~{fileInfo.duration} s
            </span>
          </div>
        )}
      </div>

      {/* Patient ID */}
      <div className="form-group">
        <label htmlFor="patientId">Patient ID (optional)</label>
        <input
          id="patientId"
          type="text"
          placeholder="e.g. P001"
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          maxLength={50}
        />
      </div>

      {/* Sampling rate */}
      <div className="form-group">
        <label htmlFor="sr">Sampling Rate (Hz)</label>
        <select
          id="sr"
          value={samplingRate}
          onChange={(e) => setSamplingRate(Number(e.target.value))}
        >
          <option value={256}>256 Hz</option>
          <option value={360}>360 Hz (MIT-BIH default)</option>
          <option value={500}>500 Hz</option>
        </select>
      </div>

      {validationError && (
        <div className="error-box" role="alert">{validationError}</div>
      )}

      <div className="submit-row">
        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading || !file}
          aria-busy={loading}
        >
          {loading ? 'Analyzing…' : 'Analyze ECG'}
        </button>
        {loading && (
          <span className="loading-msg" aria-live="polite">
            <span className="spinner" aria-hidden="true" />
            Running analysis, please wait…
          </span>
        )}
      </div>
    </form>
  )
}
