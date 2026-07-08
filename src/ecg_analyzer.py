# -*- coding: utf-8 -*-
"""
ECG Monitor - Phase 4: Production-Ready Analyzer
=================================================
Command-line tool that loads a raw ECG CSV, validates signal quality,
extracts 11 clinical features, and produces a structured JSON report
using both clinical decision rules (Phase 2) and the trained Random
Forest model (Phase 3).

Usage
-----
    python ecg_analyzer.py <ecg_file.csv> [options]

Options
-------
  --model     Path to trained model (default: ecg_model.joblib)
  --patient-id  Patient identifier string (default: UNKNOWN)
  --sr          Sampling rate in Hz (default: 360)
  --duration    Seconds to analyse, 0 = full recording (default: 0)
  --out-dir     Directory for output JSON reports (default: .)

Signal quality checks
---------------------
  - Recording must be >= 10 seconds
  - Signal must not be flat (std >= 0.01 mV)
  - At least 5 R-peaks must be detected

Error codes in JSON
-------------------
  "Recording too short"
  "Lead disconnected"
  "Could not detect heartbeats"
  "Result inconclusive"  (ML confidence < 0.60)
"""

import os
import sys
import json
import argparse
import warnings
import datetime
import numpy as np

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252 and cannot encode the disclaimer emoji,
# so every CLI run would die with UnicodeEncodeError (exit 1) even on success.
# Force stdout to UTF-8; redirected/detached streams keep their own encoding.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass  # ponytail: best-effort console tweak, not a data path

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from ecg_phase1 import (
    FS as DEFAULT_FS, FEATURE_KEYS, detect_r_peaks, extract_features,
    _bandpass_filter,
)
from ecg_phase2 import classify_ecg

CLASSES = ["Normal", "AFib", "PVC", "Bradycardia", "Tachycardia"]

# Phase 2 (rules) emits verbose clinical names; the model, ml_probabilities,
# API contract and frontend all use the 5 canonical class labels. Normalise the
# rule names so the returned `diagnosis` matches the ml_probabilities keys and
# the UI can highlight the winning bar. "Ventricular Tachycardia" (rules-only,
# no ML class) and "INCONCLUSIVE" have no canonical equivalent and pass through.
RULE_TO_CANONICAL = {
    "Normal Sinus Rhythm": "Normal",
    "Atrial Fibrillation": "AFib",
    "Sinus Tachycardia":   "Tachycardia",
    "Bradycardia":         "Bradycardia",
}


def _canonical_diagnosis(name: str) -> str:
    return RULE_TO_CANONICAL.get(name, name)

# Severity + recommendation per ML class label, used only on the ml_fallback path
# (rule-based diagnoses carry their own from Phase 2). Keyed by ML label, which
# differs from Phase 2's full diagnosis names — do not merge with Phase 2.
ML_SEV_REC = {
    "Normal"      : ("NONE",     "No immediate action required."),
    "AFib"        : ("MODERATE", "Consult a cardiologist for rate "
                                 "control and stroke risk assessment."),
    "PVC"         : ("LOW",      "Consult a physician; isolated PVCs "
                                 "are often benign but warrant review."),
    "Bradycardia" : ("MODERATE", "Consult a physician; may indicate "
                                 "sinus node dysfunction."),
    "Tachycardia" : ("LOW",      "Investigate underlying cause "
                                 "(fever, dehydration, anxiety)."),
}

DISCLAIMER = (
    "\u26a0\ufe0f  FOR SCREENING PURPOSES ONLY. This device is NOT a diagnostic tool. "
    "Results must be reviewed by a qualified healthcare professional before any "
    "clinical decision is made. Do not use as a substitute for professional "
    "medical advice, diagnosis, or treatment."
)

CONFIDENCE_THRESHOLD = 0.60
MIN_DURATION_S        = 10.0
FLAT_STD_THRESHOLD    = 0.01   # mV
MIN_R_PEAKS           = 5

# Per-feature medians of the training dataset (data/ecg_dataset_cache.npz,
# 4946 windows), in FEATURE_KEYS order. Used to impute individual NaN features
# before ML inference so a single unmeasurable feature (e.g. QRS/ST on a noisy
# beat) no longer disables the whole ML fallback. Values match the distribution
# the committed model was trained on. Regenerate if the dataset/model changes
# (recompute with np.nanmedian over the rebuilt cache).
FEATURE_MEDIANS = {
    "hr_mean":      74.611399,
    "hr_std":        8.036804,
    "rr_cv":         0.123219,
    "rr_entropy":    0.443045,
    "sdnn":        104.756540,
    "rmssd":       134.388757,
    "pnn50":        29.729730,
    "qrs_dur_mean": 115.325899,
    "qrs_dur_std":   4.391157,
    "p_wave_ratio":  0.700000,
    "st_elevation": -151.006790,
}
_FEATURE_MEDIAN_VEC = np.array([FEATURE_MEDIANS[k] for k in FEATURE_KEYS])


# =============================================================================
# 1. SIGNAL LOADING & QUALITY CHECKS
# =============================================================================

def load_ecg_csv(path: str, sr: int = DEFAULT_FS,
                 duration_s: float = 0.0) -> np.ndarray:
    """
    Load a single-column CSV file as a raw ECG signal.

    Parameters
    ----------
    path       : path to CSV file (one sample per row, mV)
    sr         : sampling rate (Hz)
    duration_s : seconds to load (0 = full file)

    Returns
    -------
    signal : 1-D float64 array
    """
    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.ndim > 1:
        data = data[:, 0]   # take first column if multi-column
    data = data.astype(np.float64)

    if duration_s > 0:
        n_samples = int(sr * duration_s)
        data = data[:n_samples]

    return data


def check_signal_quality(signal: np.ndarray, sr: int) -> str | None:
    """
    Validate signal quality.

    Returns None if OK, or an error message string if the signal fails.
    """
    duration = len(signal) / sr
    if duration < MIN_DURATION_S:
        return f"Recording too short ({duration:.1f} s < {MIN_DURATION_S} s)"

    # Reject NaN/Inf before the flat-line test: np.std of a signal containing NaN
    # is NaN, and `NaN < FLAT_STD_THRESHOLD` is False, so garbage would otherwise
    # sail through the quality gate as a valid recording.
    if not np.all(np.isfinite(signal)):
        return "Signal contains non-finite values (NaN/Inf)"

    if np.std(signal) < FLAT_STD_THRESHOLD:
        return "Lead disconnected (flat line detected)"

    return None   # OK


# =============================================================================
# 2. COMBINED PREDICTION (rules + ML)
# =============================================================================

def predict_combined(signal: np.ndarray, sr: int,
                     model=None) -> dict:
    """
    Run the full analysis pipeline on a signal.

    1. Detect R-peaks (Pan & Tompkins)
    2. Extract 11 features
    3. Apply clinical decision rules (Phase 2)
    4. If model is provided, also run ML prediction
    5. Merge: use rule-based result unless inconclusive, then fall back to ML

    Returns
    -------
    report dict (see module docstring for format)
    """
    # Compute the 5-15 Hz bandpass once and share it between R-peak detection and
    # feature extraction, which both need it (previously each recomputed it — the
    # heaviest DSP step, run twice per analysis).
    bp = _bandpass_filter(signal, sr)

    # R-peak detection
    r_peaks = detect_r_peaks(signal, sr, bp=bp)

    if len(r_peaks) < MIN_R_PEAKS:
        return _error_report(
            "Could not detect heartbeats",
            f"Only {len(r_peaks)} R-peak(s) detected (need >= {MIN_R_PEAKS})"
        )

    # Feature extraction
    features = extract_features(signal, r_peaks, sr, bp=bp)

    # Clinical rule-based diagnosis (Phase 2)
    rule_result = classify_ecg(features)

    # ML prediction (Phase 3 model)
    ml_result = None
    ml_probs  = {}

    if model is not None:
        vec = np.array([[features.get(k, np.nan) for k in FEATURE_KEYS]])
        # Impute individual NaN features with training-set medians so one
        # unmeasurable feature doesn't disable the whole ML fallback. If every
        # feature is NaN (degenerate signal) there's nothing to impute from —
        # skip ML rather than predict on an all-median vector.
        nan_mask = np.isnan(vec[0])
        if not np.all(nan_mask):
            if np.any(nan_mask):
                vec[0, nan_mask] = _FEATURE_MEDIAN_VEC[nan_mask]
            proba_raw = model.predict_proba(vec)[0]
            # Expand probabilities to full class range (model may be missing classes)
            proba = np.zeros(len(CLASSES))
            for col_idx, cls_idx in enumerate(model.classes_):
                proba[cls_idx] = proba_raw[col_idx]
            ml_class  = int(np.argmax(proba))
            ml_conf   = float(proba[ml_class])
            ml_probs  = {CLASSES[i]: round(float(proba[i]), 4)
                         for i in range(len(CLASSES))}
            ml_result = {
                "diagnosis"  : CLASSES[ml_class],
                "confidence" : round(ml_conf, 4),
            }

    # Merge: rule-based takes priority; fall back to ML if inconclusive
    final_diagnosis   = rule_result["diagnosis"]
    final_confidence  = rule_result["confidence"]
    final_severity    = rule_result["severity"]
    final_recommend   = rule_result["recommendation"]
    final_warnings    = list(rule_result["warnings"])
    method_used       = "clinical_rules"

    if final_diagnosis == "INCONCLUSIVE" and ml_result is not None:
        if ml_result["confidence"] >= CONFIDENCE_THRESHOLD:
            final_diagnosis  = ml_result["diagnosis"]
            final_confidence = ml_result["confidence"]
            method_used      = "ml_fallback"
            # Severity/recommendation for the ML-assigned label
            final_severity, final_recommend = ML_SEV_REC.get(
                final_diagnosis, ("LOW", "Consult a physician.")
            )
        else:
            final_warnings.append(
                f"ML confidence low ({ml_result['confidence']:.2f}); "
                "result inconclusive"
            )

    # Normalise rule diagnosis to the canonical class label (UI/API contract)
    final_diagnosis = _canonical_diagnosis(final_diagnosis)

    # Build full metrics block
    metrics = {k: (round(float(features.get(k, np.nan)), 4)
                   if not np.isnan(features.get(k, np.nan))
                   else None)
               for k in FEATURE_KEYS}
    metrics["r_peaks_detected"] = len(r_peaks)
    metrics["duration_s"]       = round(len(signal) / sr, 2)

    report = {
        "patient_id"    : "UNKNOWN",
        "timestamp"     : datetime.datetime.now().isoformat(),
        "diagnosis"     : final_diagnosis,
        "confidence"    : round(final_confidence, 4),
        "severity"      : final_severity,
        "recommendation": final_recommend,
        "warnings"      : final_warnings,
        "method"        : method_used,
        "metrics"       : metrics,
        "ml_probabilities": ml_probs,
        "disclaimer"    : DISCLAIMER,
        "error"         : None,
    }
    return report


def _error_report(error_msg: str, detail: str = "") -> dict:
    return {
        "patient_id"       : "UNKNOWN",
        "timestamp"        : datetime.datetime.now().isoformat(),
        "diagnosis"        : None,
        "confidence"       : None,
        "severity"         : None,
        "recommendation"   : None,
        "warnings"         : [detail] if detail else [],
        "method"           : None,
        "metrics"          : {},
        "ml_probabilities" : {},
        "disclaimer"       : DISCLAIMER,
        "error"            : error_msg,
    }


# =============================================================================
# 3. CLI ENTRY POINT
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Portable ECG Arrhythmia Analyser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ecg_file", help="Path to raw ECG CSV (single column, mV)")
    parser.add_argument("--model",      default=os.path.join(ROOT, "models", "ecg_model.joblib"),
                        help="Trained RF model (default: ecg_model.joblib)")
    parser.add_argument("--patient-id", default="UNKNOWN",
                        help="Patient identifier")
    parser.add_argument("--sr",         type=int,   default=DEFAULT_FS,
                        help=f"Sampling rate Hz (default: {DEFAULT_FS})")
    parser.add_argument("--duration",   type=float, default=0.0,
                        help="Seconds to analyse, 0=full (default: 0)")
    parser.add_argument("--out-dir",    default=".",
                        help="Output directory for JSON reports")

    args = parser.parse_args(argv)

    # ---- Load signal ----------------------------------------------------
    print(f"\n[1] Loading ECG from: {args.ecg_file}")
    try:
        signal = load_ecg_csv(args.ecg_file, sr=args.sr, duration_s=args.duration)
    except Exception as e:
        report = _error_report("File load error", str(e))
        _write_and_print(report, args)
        return 1

    print(f"    Loaded {len(signal)} samples at {args.sr} Hz "
          f"({len(signal)/args.sr:.1f} s)")

    # ---- Quality check --------------------------------------------------
    print("[2] Checking signal quality ...")
    err = check_signal_quality(signal, args.sr)
    if err:
        report = _error_report(err)
        _write_and_print(report, args)
        return 1
    print("    Signal quality OK")

    # ---- Load ML model (optional) ---------------------------------------
    model = None
    if os.path.isfile(args.model):
        import joblib
        model = joblib.load(args.model)
        print(f"[3] Model loaded: {args.model}")
    else:
        print(f"[3] Model not found at {args.model} -- using rules only")

    # ---- Predict --------------------------------------------------------
    print("[4] Running analysis ...")
    report = predict_combined(signal, args.sr, model=model)

    if report["error"]:
        _write_and_print(report, args)
        return 1

    # Set patient ID
    report["patient_id"] = args.patient_id

    # ---- Output ---------------------------------------------------------
    _write_and_print(report, args)
    return 0


def _write_and_print(report: dict, args) -> None:
    """Save report to JSON and print a summary."""
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid  = report.get("patient_id", "UNKNOWN")
    fname = f"ecg_report_{pid}_{ts}.json"
    out_path = os.path.join(args.out_dir, fname)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*58}")
    if report["error"]:
        print(f"  ERROR: {report['error']}")
        if report["warnings"]:
            for w in report["warnings"]:
                print(f"    [!] {w}")
    else:
        sev_icon = {"NONE":"[ ]","LOW":"[L]","MODERATE":"[M]","HIGH":"[H]"}.get(
            report["severity"], "   ")
        print(f"  {sev_icon} Diagnosis      : {report['diagnosis']}")
        print(f"      Confidence   : {report['confidence']*100:.1f}%")
        print(f"      Severity     : {report['severity']}")
        print(f"      Method       : {report['method']}")
        print(f"      Recommendation: {report['recommendation']}")
        if report["warnings"]:
            print(f"  Warnings:")
            for w in report["warnings"]:
                print(f"    [!] {w}")
        m = report.get("metrics", {})
        print(f"\n  Key metrics:")
        print(f"    HR      : {m.get('hr_mean'):.1f} bpm" if m.get("hr_mean") else "    HR      : N/A")
        print(f"    RR CV   : {m.get('rr_cv'):.3f}" if m.get("rr_cv") is not None else "    RR CV   : N/A")
        print(f"    SDNN    : {m.get('sdnn'):.1f} ms" if m.get("sdnn") is not None else "    SDNN    : N/A")
        print(f"    QRS dur : {m.get('qrs_dur_mean'):.1f} ms" if m.get("qrs_dur_mean") is not None else "    QRS dur : N/A")
        print(f"    R-peaks : {m.get('r_peaks_detected')}")
        if report["ml_probabilities"]:
            print(f"\n  ML class probabilities:")
            for cls, p in report["ml_probabilities"].items():
                bar = "#" * int(p * 20)
                print(f"    {cls:<16} {p:.3f}  {bar}")
    print(f"\n  {report['disclaimer']}")
    print(f"{'='*58}")
    print(f"\n  Report saved -> {out_path}")


# =============================================================================
if __name__ == "__main__":
    sys.exit(main())
