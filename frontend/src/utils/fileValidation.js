// ECG file validation utilities
import { ALLOWED_FILE_TYPES } from '../constants'

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
  // Parse CSV for the upload preview only (row count + duration).
  // The server enforces min-duration and numeric validation on analyze.
  const lines = text
    .split('\n')
    .map(l => l.trim())
    .filter(l => l && !l.startsWith('#'))

  return {
    lines,
    duration: (lines.length / samplingRate).toFixed(1),
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
