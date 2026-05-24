// ECG file validation utilities
import { ALLOWED_FILE_TYPES } from '../constants'

const MIN_DURATION_SECONDS = 10

export function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => resolve(e.target.result)
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsText(file)
  })
}

export function validateECGFile(file, samplingRate) {
  // Check file extension
  const ext = file.name.split('.').pop().toLowerCase()
  if (!ALLOWED_FILE_TYPES.includes(ext)) {
    throw new Error('Invalid file type. Please upload a CSV or TXT file.')
  }

  // Check file size is reasonable
  if (file.size === 0) {
    throw new Error('File is empty.')
  }

  return ext
}

export async function parseAndValidateECGContent(text, samplingRate) {
  // Parse CSV: strip comments, split lines, filter empty
  const lines = text
    .split('\n')
    .map(l => l.trim())
    .filter(l => l && !l.startsWith('#'))

  // Check minimum duration (10 seconds at sampling rate)
  const minRows = Math.ceil(MIN_DURATION_SECONDS * samplingRate)
  if (lines.length < minRows) {
    throw new Error(
      `File too small: ${lines.length} samples detected. ` +
      `Need at least ${minRows} rows (${MIN_DURATION_SECONDS} seconds at ${samplingRate} Hz).`
    )
  }

  // Validate numeric data: check first 20 lines
  const sample = lines.slice(0, 20)
  const badLine = sample.find(l => {
    const first = l.split(',')[0].trim()
    return first && isNaN(parseFloat(first))
  })
  if (badLine) {
    throw new Error(
      `File contains non-numeric data: "${badLine.substring(0, 40)}"`
    )
  }

  // Calculate duration
  const durationSec = (lines.length / samplingRate).toFixed(1)

  return {
    lines,
    duration: durationSec,
    rowCount: lines.length,
  }
}

export async function validateAndParseECGFile(file, samplingRate) {
  // Validate file extension and properties
  validateECGFile(file, samplingRate)

  // Read file content
  const text = await readFileAsText(file)

  // Parse and validate content
  const contentInfo = await parseAndValidateECGContent(text, samplingRate)

  return {
    name: file.name,
    size: (file.size / 1024).toFixed(1) + ' KB',
    rows: contentInfo.rowCount,
    duration: contentInfo.duration,
  }
}
