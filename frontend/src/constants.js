// Severity labels used across components
export const SEV_LABEL = {
  NONE: 'None',
  LOW: 'Low',
  MODERATE: 'Moderate',
  HIGH: 'High',
}

// Default sampling rate for ECG analysis
export const DEFAULT_SAMPLING_RATE = 360

// Allowed file types for ECG upload
export const ALLOWED_FILE_TYPES = ['csv', 'txt']
export const FILE_ACCEPT_STRING = '.csv,.txt'

// Common format utilities
export function fmt(val, decimals = 1) {
  if (val == null || isNaN(val)) return 'N/A'
  return Number(val).toFixed(decimals)
}

export function fmtDate(ts) {
  if (!ts) return '—'
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return ts
  }
}
