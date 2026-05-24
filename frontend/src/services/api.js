import { DEFAULT_SAMPLING_RATE } from '../constants'

const API_BASE = '/api'

export async function analyzeECG(file, patientId, samplingRate) {
  const form = new FormData()
  form.append('file', file)
  form.append('patient_id', patientId || 'UNKNOWN')
  form.append('sampling_rate', String(samplingRate || DEFAULT_SAMPLING_RATE))

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: form,
  })

  const data = await res.json()
  if (!res.ok) {
    throw new Error(data.error || `Server error ${res.status}`)
  }
  return data
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`)
  return res.json()
}
