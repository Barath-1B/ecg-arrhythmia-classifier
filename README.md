<div align="center">

# 🫀 Portable ECG Monitor & Arrhythmia Detector

### Upload a raw ECG recording, get a clinical-grade arrhythmia screen in seconds

[![CI](https://github.com/Barath-1B/ecg-arrhythmia-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Barath-1B/ecg-arrhythmia-classifier/actions/workflows/ci.yml)

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask_3-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![React](https://img.shields.io/badge/React_18-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)

*Pan–Tompkins R-peak detection · 11 clinical features · hybrid rule + Random Forest classifier · MIT-BIH*

</div>

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
- **67-test pytest suite + CI** — data-free tests (synthetic signals, mock model) run on every
  push alongside `ruff` and a frontend build

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
Phase 3 ── Random Forest, grid-search tuned (100 trees, max_depth 20)
    │          class_weight='balanced' | f1_weighted scoring
    │          trained on Normal/AFib/PVC only (rate classes are rule-owned)
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

Evaluated by **5-fold record-level cross-validation** over **4,946 windows** from the MIT-BIH
Arrhythmia Database. The split is grouped by patient record, so a patient's overlapping
windows never appear in both training and evaluation (leakage-free). Every window is scored
out-of-fold by a system that never saw its record. **The numbers below grade the hybrid
pipeline that actually ships** — Phase 2 rules first, then the Random Forest fallback — not a
bare model. They are copied verbatim from `outputs/reports/ecg_clinical_validation_report.json`
(regenerate with `python run_all.py`).

The classifier is split by design: the **ML model owns Normal / AFib / PVC** (all learnable,
multi-record); the **deterministic rules own Bradycardia / Tachycardia** (rate-defined). See
[Methodology](#methodology-why-this-approach) for why.

| Metric | Score | Target |
|--------|-------|--------|
| Overall accuracy (hybrid, 5 classes) | **79.1%** | — |
| Macro sensitivity (5 classes) | **39.7%** | ≥ 85% — not met |
| Macro specificity (5 classes) | **90.4%** | ≥ 90% — met |
| **ML sub-model ROC-AUC** (Normal/AFib/PVC) | **0.915** | ≥ 0.90 — **met** |

Per-class results (bootstrap 95% CI on out-of-fold predictions):

| Class | Owner | Sensitivity | Specificity | Precision | F1 | Support |
|-------|-------|-------------|-------------|-----------|-----|---------|
| Normal | ML | 91.7% (90.8–92.6%) | 63.9% (61.5–66.4%) | 86.3% | 0.889 | 3,526 |
| AFib | ML | 57.9% (53.5–62.4%) | 98.1% (97.6–98.4%) | 76.4% | 0.659 | 487 |
| PVC | ML | 49.0% (45.6–52.3%) | 91.9% (91.1–92.7%) | 54.1% | 0.514 | 809 |
| Bradycardia | rules | 0.0% | 97.9% (97.5–98.3%) | 0.0% | 0.000 | 119 |
| Tachycardia | rules | 0.0% | 100.0% | 0.0% | 0.000 | 5 |

> **Known limitations (read this).** This is an honest screening prototype, not a passing
> medical device.
> - **The ML sub-model is the strong part.** Restricting the Random Forest to the three
>   learnable classes lifted its ROC-AUC from 0.80 to **0.915** (now above target) and AFib
>   sensitivity from 45% to **58%**. PVC in 30-second windows stays hard (~49%): most missed
>   PVCs fall below the 15%-ectopy labelling threshold and read as Normal.
> - **Bradycardia and Tachycardia are near-zero, and that is the truthful number.** In
>   MIT-BIH these rhythms come from a *single* record (232, ~59 bpm — borderline, above the
>   `HR < 50` rule) and *three* records / 5 windows respectively. They are structurally
>   unlearnable under record-grouped CV, so they are handed to the rate rules — which, on
>   these specific borderline records, mostly do not fire. No amount of tuning fixes this on
>   this dataset; it needs a larger, balanced corpus (e.g. PTB-XL, 21,837 records).
> - **Macro sensitivity (39.7%) is low by construction**: it averages five classes, two of
>   which are the structural zeros above.

**Clinical safety checks (arrhythmia-vs-Normal), current run:**
- False-negative rate (arrhythmia read as Normal): **36.1%** (512/1420) — target < 12% — **FAIL**
- False-positive rate (Normal flagged): **8.3%** (293/3526) — target < 8% — **FAIL**
- Overall clinical-safety gate: **FAIL** — appropriate for a research prototype that is *not*
  FDA cleared. Do not read these numbers as clearance.

> **Note on earlier figures.** Prior versions of this README reported 87.5% accuracy / 61.8%
> macro sensitivity / 0.864 ROC-AUC over "1,546 windows." Those predated a rhythm-code
> labelling correction (`(B`/`(T` are ventricular bigeminy/trigeminy → PVC, not brady/tachy),
> which produced today's larger, Normal-dominated 4,946-window dataset. The old numbers were
> never regenerated and are unreproducible; the figures above are the honest, current ones.

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

**The two arrhythmia families are split on purpose.** Bradycardia and Tachycardia are defined by
heart *rate*, and in MIT-BIH they come from just one and three records respectively — too few to
learn under record-grouped cross-validation (any fold holding out that record trains on zero
examples, guaranteeing 0% recall). So they are owned by the deterministic rate rules (`HR < 50`
/ `HR > 120`), which are exactly the right tool for a rate threshold. The Random Forest trains
only on Normal / AFib / PVC, which are multi-record and genuinely learnable; freeing it from the
unlearnable classes is what raised its ROC-AUC to 0.915. The API/UI still expose all five classes
— the model's probability vector is expanded to five columns and the rules fill the rate slots.

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
3. **Portability** — the serialised model (`ecg_model.joblib`, ~3 MB) runs on any CPU; no GPU,
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
| **Docker / docker-compose** | One-command startup for backend + frontend |
| **Model benchmark in CI** | Extend the existing lint/test workflow to gate on validation metrics |

---

## Project Structure

```
ecg-arrhythmia-classifier/
├── src/
│   ├── ecg_phase1.py               # Data loading, Pan-Tompkins, feature extraction
│   ├── ecg_phase2.py               # Clinical decision-rule classifier
│   ├── ecg_phase3.py               # Random Forest training & validation
│   ├── ecg_phase5_validation.py    # Bootstrap clinical validation, FDA report
│   └── ecg_analyzer.py             # Production CLI (Phases 1-3 combined)
├── backend/
│   ├── app.py                      # Flask REST API
│   ├── config.py                   # Constants (sampling rate, model path, etc.)
│   └── requirements.txt            # -r ../requirements.txt (full deps; app.py needs them)
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
├── tests/                          # Pytest suite (data-free: synthetic signals + mock model)
│   ├── test_classify_ecg.py        # Phase 2 clinical decision rules
│   ├── test_predict_combined.py    # Rules ↔ ML merge and fallback logic
│   ├── test_diagnosis_mapping.py   # Rule-name → canonical-label mapping
│   ├── test_signal_processing.py   # R-peak detection, entropy
│   ├── test_signal_quality.py      # Signal-quality gating and loading
│   └── test_backend_api.py         # Flask endpoints
├── models/
│   └── ecg_model.joblib            # Trained Random Forest (~3 MB, 3-class, included)
├── outputs/
│   ├── plots/                      # Example: confusion matrix, ROC curves, feature importance
│   └── reports/                    # Example: metrics JSON, clinical validation report
├── .github/workflows/ci.yml        # CI: ruff + pytest + frontend build
├── run_all.py                      # Master runner (Phases 1 → 2 → 3 → 5)
├── conftest.py                     # Pytest bootstrap
├── pyproject.toml                  # Pytest + ruff config (puts src/ and backend/ on the path)
├── requirements.txt                # All Python dependencies
├── requirements-dev.txt            # pytest + ruff, on top of the root deps
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
cd ecg-arrhythmia-classifier
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

### 8. Run the tests

The test suite is data-free (synthetic signals + a mock model), so it needs
neither the MIT-BIH download nor a trained model on disk.

```bash
pip install -r requirements-dev.txt   # pytest + ruff, on top of the root deps
pytest                                 # whole suite, from the repo root
ruff check .                           # lint
pytest tests/test_classify_ecg.py      # a single file
```

`pyproject.toml` puts `src/` and `backend/` on the path, so tests import the
phase modules and the Flask app directly. CI (`.github/workflows/ci.yml`) runs
`ruff` + `pytest` and a frontend `npm run build` on every push and PR.

---

## API Reference

### `GET /api/health`

Liveness check. Reports whether the model loaded at startup:

```json
{ "status": "ok", "model_loaded": true, "timestamp": "2026-07-08T20:47:35.205639" }
```

### `POST /api/analyze`

Upload a raw ECG CSV for analysis.

**Request:** `multipart/form-data` with:

| Field | Required | Notes |
|-------|----------|-------|
| `file` | yes | Single-column CSV of raw voltage samples, max 16 MB |
| `patient_id` | no | Echoed back in the report |
| `sampling_rate` | no | One of `256` / `360` / `500` (default `360`); anything else is rejected with 400 |

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
  "disclaimer": "⚠️  FOR SCREENING PURPOSES ONLY. This device is NOT a diagnostic tool. Results must be reviewed by a qualified healthcare professional before any clinical decision is made. Do not use as a substitute for professional medical advice, diagnosis, or treatment."
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
