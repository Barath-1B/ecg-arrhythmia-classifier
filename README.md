# Portable ECG Monitor with Arrhythmia Detector

> Upload a raw ECG recording, get a clinical-grade arrhythmia diagnosis in seconds.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![React](https://img.shields.io/badge/Frontend-React%2018-61dafb)
![Flask](https://img.shields.io/badge/Backend-Flask%203-lightgrey)

---

## Overview

This project is a full-stack medical screening tool that takes a raw ECG signal (CSV file) and
classifies it into one of five cardiac rhythms:

- **Normal Sinus Rhythm**
- **Atrial Fibrillation (AFib)**
- **Premature Ventricular Contraction (PVC)**
- **Bradycardia**
- **Tachycardia**

The pipeline runs Pan & Tompkins R-peak detection, extracts 11 clinical features, applies a
hybrid classifier (deterministic clinical rules first, then a Random Forest fallback), and returns
a JSON report with diagnosis, confidence score, severity, and recommended action.

A React web interface lets you upload a CSV, view the results with a probability bar chart, and
browse a history of past analyses.

---

## Features

- **Pan & Tompkins R-peak detection** — clinical gold-standard QRS detection algorithm
- **11 clinical features** — heart rate (mean/std), RR coefficient of variation, RR entropy,
  SDNN, RMSSD, pNN50, QRS duration (mean/std), P-wave ratio, ST elevation
- **Hybrid classifier** — deterministic decision rules handle unambiguous cases; Random Forest
  handles the rest
- **5-class detection** — Normal, AFib, PVC, Bradycardia, Tachycardia
- **Confidence + severity scoring** — every diagnosis includes a 0–1 confidence and a
  NONE / LOW / MODERATE / HIGH severity flag
- **Flask REST API** — `POST /api/analyze` accepts a CSV upload, returns JSON
- **React + Vite web UI** — upload form, results display, probability chart, patient history
- **Phase 5 clinical validation** — bootstrap 95% confidence intervals, FDA 510(k)-style report

---

## Architecture

```
Raw ECG CSV
    │
Phase 1 ── Pan-Tompkins R-peak detection + 11-feature extraction
    │          (src/ecg_phase1.py)
    │
Phase 2 ── Clinical decision rules
    │          AFib  : RR_CV > 0.25 AND RR_entropy > 0.75 AND P_ratio < 0.6
    │          Brady : HR < 50 AND RR_CV < 0.15 AND QRS < 120 ms
    │          Tachy : HR > 120 AND RR_CV < 0.20 AND QRS < 120 ms
    │          VT    : HR > 120 AND QRS > 130 ms
    │          (src/ecg_phase2.py)
    │
Phase 3 ── Random Forest (200 trees, grid-search tuned, stratified 5-fold CV)
    │          class_weight='balanced' | f1_weighted scoring
    │          (src/ecg_phase3.py  →  models/ecg_model.joblib)
    │
Phase 5 ── Bootstrap clinical validation + FDA-ready report
    │          (src/ecg_phase5_validation.py)
    │
ecg_analyzer.py  ←  production CLI (Phases 1-3 combined)
    │
Flask API        ←  /api/health  |  POST /api/analyze
    │                (backend/app.py)
    │
React UI         ←  upload → results → probability chart → history
                     (frontend/src/)
```

---

## Performance Metrics

Evaluated on 232 held-out test samples from the MIT-BIH Arrhythmia Database (Phase 5 bootstrap validation).

| Metric | Score |
|--------|-------|
| Overall Accuracy | **97.0%** |
| Macro Sensitivity | **90.3%** |
| Macro Specificity | **98.7%** |
| ROC-AUC | **0.9973** |

Per-class results (bootstrap 95% CI):

| Class | Sensitivity | Specificity | Precision | F1 | Support |
|-------|-------------|-------------|-----------|-----|---------|
| Normal | 100.0% (100–100%) | 100.0% (100–100%) | 100.0% | 1.000 | 42 |
| AFib | 55.6% (20–90%) | 100.0% (100–100%) | 100.0% | 0.714 | 9 |
| PVC | 98.6% (96.5–100%) | 94.3% (89–98.8%) | 96.6% | 0.976 | 144 |
| Bradycardia | 97.2% (90.3–100%) | 99.0% (97.4–100%) | 94.6% | 0.959 | 36 |
| Tachycardia | 100.0% (0–100%) | 100.0% (100–100%) | 100.0% | 1.000 | 1 |

> AFib sensitivity (55.6%) reflects limited training samples in MIT-BIH (~60 records).
> The wide 95% CI (20–90%) is due to small test support (9 samples). A larger dataset
> such as PTB-XL would improve AFib detection substantially.

**Clinical safety checks (all PASS):**
- False negative rate: 0% (target < 12%)
- False positive rate: 0% (target < 8%)
- Dangerous misclassifications (e.g. VT → Normal, AFib → Normal): 0

---

## Methodology: Why This Approach

### Pan & Tompkins for R-peak detection
Pan & Tompkins (1985) is the clinical standard for real-time QRS detection. It bandpass-filters
the signal (5–15 Hz), differentiates, squares, and applies moving-window integration to find
adaptive peaks. It is robust to baseline wander and muscle noise, runs in O(n) time, and does
not need a GPU — making it practical for embedded hardware (AD8232 module + microcontroller).

### Hybrid rule-based + Random Forest classifier
A pure ML model on the MIT-BIH dataset (~180 usable records per split) would overfit. Clinical
decision rules handle the unambiguous cases (extreme heart rates, wide QRS) with 100% confidence
and full interpretability. The Random Forest steps in only when the rules are inconclusive, giving
the best of both worlds: determinism where possible, learned generalisation where needed.

### MIT-BIH Arrhythmia Database
MIT-BIH is the most widely cited ECG benchmark in the literature (5000+ publications). It contains
48 half-hour two-lead recordings sampled at 360 Hz, annotated beat-by-beat by cardiologists — the
closest thing to a ground truth for arrhythmia detection research.

### Random Forest over deep learning
Random Forest was chosen for three reasons:
1. **Interpretability** — feature importance reveals which clinical features drive each diagnosis,
   which is essential for building trust in a medical screening tool.
2. **Dataset size** — deep learning models (CNNs, Transformers) need tens of thousands of labelled
   recordings to outperform classical ML; MIT-BIH is too small.
3. **Portability** — the serialised model (`ecg_model.joblib`, 1.5 MB) runs on any CPU; no GPU,
   no inference server, no cloud dependency.

---

## Suggested Improvements (Future Work)

| Idea | Benefit |
|------|---------|
| **ResNet-1D / Transformer on PTB-XL** (21,837 records) | Push sensitivity above 95% for all classes including AFib |
| **Real-time AD8232 hardware integration** | Stream live ECG via serial/USB; add WebSocket endpoint | 
| **12-lead ECG support** | Enable ST-segment, axis deviation, and bundle branch block detection |
| **ONNX / TensorFlow Lite export** | Deploy on ESP32 or Raspberry Pi Zero for true wearable use |
| **More arrhythmia classes** | Atrial Flutter, 2nd/3rd-degree AV block, LBBB/RBBB |
| **Automated signal quality scoring** | Reject noisy or lead-off recordings before classification |
| **Real database for history** | Replace `localStorage` with SQLite / PostgreSQL for longitudinal tracking |
| **Pytest suite** | Unit tests for feature extraction and edge-case classifier inputs |
| **Docker / docker-compose** | One-command startup for backend + frontend |
| **CI/CD with GitHub Actions** | Automated lint, test, and model benchmark on every push |

---

## Project Structure

```
ecg-arrhythmia-monitor/
├── src/
│   ├── ecg_phase1.py               # Data loading, Pan-Tompkins, feature extraction
│   ├── ecg_phase2.py               # Clinical decision-rule classifier
│   ├── ecg_phase3.py               # Random Forest training & validation
│   ├── ecg_phase5_validation.py    # Bootstrap clinical validation, FDA report
│   └── ecg_analyzer.py             # Production CLI (Phases 1-3 combined)
├── backend/
│   ├── app.py                      # Flask REST API
│   ├── config.py                   # Constants (sampling rate, model path, etc.)
│   └── requirements.txt            # Flask-only deps (see root requirements.txt for all)
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # React router
│   │   ├── components/             # Header, UploadForm, ResultsDisplay, MetricsGrid,
│   │   │                           #   ProbabilityChart, HistoryDashboard
│   │   ├── pages/                  # AnalysisPage, DashboardPage
│   │   ├── services/               # api.js (HTTP client), storage.js (history)
│   │   └── utils/                  # fileValidation.js
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models/
│   └── ecg_model.joblib            # Trained Random Forest (1.5 MB, included)
├── outputs/
│   ├── plots/                      # Example: confusion matrix, ROC curves, feature importance
│   └── reports/                    # Example: metrics JSON, clinical validation report
├── run_all.py                      # Master runner (Phases 1 → 2 → 3 → 5)
├── requirements.txt                # All Python dependencies
├── LICENSE
└── README.md
```

---

## Installation & Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Clone the repo

```bash
git clone https://github.com/Barath-1B/ecg-arrhythmia-classifier.git
cd ecg-arrhythmia-monitor
```

### 2. Set up Python environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Download the training data

The MIT-BIH database is not included in this repo (74 MB). Download it with one command:

```bash
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mit-bih-arrhythmia-database-1.0.0')"
```

Or download manually from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).

### 4. Run the full ML pipeline (optional — model already included)

```bash
python run_all.py
# or faster, reusing the cached dataset:
python run_all.py --skip-build
```

This runs all 4 phases and saves a new `models/ecg_model.joblib`.

### 5. Analyze a custom ECG

```bash
python src/ecg_analyzer.py path/to/ecg.csv --patient-id P001
```

The analyzer expects a single-column CSV of raw voltage samples (default 360 Hz). It prints a
JSON report and saves it to the `outputs/reports/` directory.

### 6. Start the backend API

```bash
cd backend
python app.py
# Server starts at http://127.0.0.1:5000
```

### 7. Start the frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## API Reference

### `GET /api/health`

Liveness check. Returns `{"status": "ok"}`.

### `POST /api/analyze`

Upload a raw ECG CSV for analysis.

**Request:** `multipart/form-data` with a `file` field (CSV, max 16 MB)

**Response:**

```json
{
  "diagnosis": "PVC",
  "confidence": 0.94,
  "severity": "MODERATE",
  "recommendation": "Frequent PVC pattern detected. Recommend Holter monitoring.",
  "warnings": ["High PVC burden"],
  "metrics": {
    "hr_mean": 72.4,
    "hr_std": 8.1,
    "rr_cv": 0.11,
    "sdnn": 48.3,
    "rmssd": 31.7,
    "qrs_dur_mean": 118.5,
    "st_elevation": 12.0
  },
  "ml_probabilities": {
    "Normal": 0.02,
    "AFib": 0.01,
    "PVC": 0.94,
    "Bradycardia": 0.02,
    "Tachycardia": 0.01
  },
  "disclaimer": "FOR SCREENING PURPOSES ONLY. Not a diagnostic tool."
}
```

---

## Dataset & Citation

Training data: **MIT-BIH Arrhythmia Database**

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng in Med and Biol*
> 20(3):45-50 (May-June 2001). (PMID: 11446209)

> Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet.
> *Circulation* 101(23):e215-e220. June 2000.

Available at: https://physionet.org/content/mitdb/1.0.0/

---

## Medical Disclaimer

> **IMPORTANT:** This software is for screening and educational purposes only. It is NOT a
> certified medical device and has NOT been approved by the FDA or any regulatory authority.
> Results must NOT be used for clinical diagnosis or treatment decisions without review by a
> qualified healthcare professional. The authors accept no liability for clinical outcomes arising
> from use of this software.

---

## License

MIT — see [LICENSE](LICENSE).
