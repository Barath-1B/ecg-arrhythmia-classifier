# -*- coding: utf-8 -*-
"""
ECG Monitor - Phase 2: Clinical Classification Rules
=====================================================
Implements a hybrid rule-based + threshold classifier that maps the 11
extracted ECG features to a structured clinical diagnosis dict.

Priority order follows standard clinical triage:
  1. Atrial Fibrillation (AFib)
  2. Bradycardia
  3. Sinus Tachycardia
  4. Ventricular Tachycardia (VT)  -- highest danger
  5. Normal Sinus Rhythm
  6. Inconclusive

Warnings are always applied AFTER the primary diagnosis (additive).

Clinical thresholds are based on:
  - AHA/ACC 2019 arrhythmia guidelines
  - Goldberger AL, "Clinical Electrocardiography" (8th ed.)
"""

import numpy as np


# =============================================================================
# 1. CLINICAL THRESHOLDS  (all values have clinical references)
# =============================================================================

# Heart rate thresholds (bpm)
HR_BRADYCARDIA_HIGH = 50      # clinical bradycardia cutoff
HR_BRADYCARDIA_SEVERE = 40    # severe bradycardia -> HIGH severity
HR_TACHY_LOW        = 120     # sustained tachycardia threshold
HR_EXTREME_LOW      = 40      # extreme bradycardia warning
HR_EXTREME_HIGH     = 200     # extreme tachycardia warning
HR_NORMAL_LOW       = 60      # lower bound of normal sinus
HR_NORMAL_HIGH      = 100     # upper bound of normal sinus

# QRS duration thresholds (ms)
QRS_NORMAL_MAX       = 120    # >120 ms = wide complex (bundle branch block / VT)
QRS_VT_MIN           = 130    # VT typically > 120-130 ms wide complex

# HRV / RR variability thresholds
RR_CV_AFIB           = 0.25   # high variability -> AFib pattern
RR_CV_BRADY_MAX      = 0.15   # regular rhythm -> bradycardia
RR_CV_TACHY_MAX      = 0.20   # regular rhythm -> sinus tachycardia
RR_CV_NORMAL_MAX     = 0.12   # tight regularity for normal sinus

ENTROPY_AFIB_MIN     = 0.75   # high irregularity -> AFib
P_WAVE_AFIB_MAX      = 0.60   # few P-waves -> AFib
P_WAVE_NORMAL_MIN    = 0.90   # clear P-waves -> sinus rhythm

# Warning thresholds
ST_ELEV_THRESHOLD    = 200.0  # uV; may indicate myocardial infarction (MI)
SDNN_LOW_HRV         = 50.0   # ms; low SDNN associated with poor prognosis


# =============================================================================
# 2. HYBRID CLASSIFIER
# =============================================================================

def classify_ecg(features: dict) -> dict:
    """
    Rule-based hybrid classifier mapping ECG features to a clinical diagnosis.

    Applies clinical decision rules in strict priority order, then appends
    severity-adjusted warnings for extreme values.

    Parameters
    ----------
    features : dict
        Must contain the 11 keys produced by extract_features() in phase 1:
        hr_mean, hr_std, rr_cv, rr_entropy, sdnn, rmssd, pnn50,
        qrs_dur_mean, qrs_dur_std, p_wave_ratio, st_elevation

    Returns
    -------
    result : dict with keys:
        diagnosis      : str  - primary diagnosis name
        confidence     : float - estimated probability (0-1)
        severity       : str  - 'NONE' | 'LOW' | 'MODERATE' | 'HIGH'
        recommendation : str  - clinical action
        warnings       : list of str - additional red flags
    """
    # Unpack features (use NaN-safe defaults)
    hr      = float(features.get("hr_mean", np.nan))
    rr_cv   = float(features.get("rr_cv", np.nan))
    entropy = float(features.get("rr_entropy", np.nan))
    sdnn    = float(features.get("sdnn", np.nan))
    qrs     = float(features.get("qrs_dur_mean", np.nan))
    p_ratio = float(features.get("p_wave_ratio", np.nan))
    st_elev = float(features.get("st_elevation", 0.0))

    warnings = []

    # -----------------------------------------------------------------
    # ALWAYS-ON WARNINGS (checked regardless of primary diagnosis)
    # -----------------------------------------------------------------
    severity_override = None

    # Extreme heart rate
    if not np.isnan(hr):
        if hr < HR_EXTREME_LOW:
            warnings.append(f"Extreme bradycardia: HR={hr:.0f} bpm (< {HR_EXTREME_LOW})")
            severity_override = "HIGH"
        elif hr > HR_EXTREME_HIGH:
            warnings.append(f"Extreme tachycardia: HR={hr:.0f} bpm (> {HR_EXTREME_HIGH})")
            severity_override = "HIGH"

    # ST elevation -- possible MI
    if not np.isnan(st_elev) and st_elev > ST_ELEV_THRESHOLD:
        warnings.append(
            f"ST elevation: {st_elev:.0f} uV (> {ST_ELEV_THRESHOLD} uV) -- Possible MI"
        )
        severity_override = "HIGH"

    # Low HRV (associated with increased cardiac mortality)
    if not np.isnan(sdnn) and sdnn < SDNN_LOW_HRV:
        warnings.append(f"Low HRV: SDNN={sdnn:.1f} ms (< {SDNN_LOW_HRV} ms)")

    # -----------------------------------------------------------------
    # PRIMARY DIAGNOSIS  (evaluated in clinical priority order)
    # -----------------------------------------------------------------

    diagnosis      = "INCONCLUSIVE"
    confidence     = 0.65
    severity       = "LOW"
    recommendation = "Repeat recording; consult a physician if symptoms persist."

    # ---- Rule 1: Atrial Fibrillation --------------------------------
    # Hallmarks: irregular RR intervals (high CV & entropy), absent P-waves
    # Clinical threshold: CV > 0.25 AND entropy > 0.75 AND P-ratio < 0.60
    if (not any(np.isnan([rr_cv, entropy, p_ratio])) and
            rr_cv > RR_CV_AFIB and
            entropy > ENTROPY_AFIB_MIN and
            p_ratio < P_WAVE_AFIB_MAX):
        diagnosis      = "Atrial Fibrillation"
        confidence     = 0.92
        severity       = "MODERATE"
        recommendation = (
            "Consult a cardiologist. Rate control (beta-blockers) and "
            "anticoagulation assessment required. Monitor for stroke risk."
        )

    # ---- Rule 2: Bradycardia ----------------------------------------
    # Regular, slow rhythm; normal QRS (rules out complete heart block / VT)
    elif (not any(np.isnan([hr, rr_cv, qrs])) and
              hr < HR_BRADYCARDIA_HIGH and
              rr_cv < RR_CV_BRADY_MAX and
              qrs < QRS_NORMAL_MAX):
        diagnosis  = "Bradycardia"
        confidence = 0.88
        if hr < HR_BRADYCARDIA_SEVERE:
            severity       = "HIGH"
            recommendation = (
                "Seek urgent medical evaluation. HR below 40 bpm may require "
                "temporary pacing."
            )
        else:
            severity       = "MODERATE"
            recommendation = (
                "Consult a physician. May indicate sinus node dysfunction or "
                "drug effect. Avoid beta-blockers / calcium channel blockers."
            )

    # ---- Rule 3: Ventricular Tachycardia (checked BEFORE sinus tachy) -
    # Wide complex tachycardia: HR > 120 AND QRS > 130 ms
    # NOTE: Evaluated before sinus tachycardia to avoid dangerous misclass.
    elif (not any(np.isnan([hr, qrs])) and
              hr > HR_TACHY_LOW and
              qrs > QRS_VT_MIN):
        diagnosis      = "Ventricular Tachycardia"
        confidence     = 0.85
        severity       = "HIGH"
        recommendation = "SEEK IMMEDIATE MEDICAL ATTENTION. VT may degenerate to VF."

    # ---- Rule 4: Sinus Tachycardia ----------------------------------
    # Fast but regular, narrow-complex rhythm
    elif (not any(np.isnan([hr, rr_cv, qrs])) and
              hr > HR_TACHY_LOW and
              rr_cv < RR_CV_TACHY_MAX and
              qrs < QRS_NORMAL_MAX):
        diagnosis      = "Sinus Tachycardia"
        confidence     = 0.87
        severity       = "LOW"
        recommendation = (
            "Investigate underlying cause (fever, dehydration, anxiety, anemia, "
            "hyperthyroidism). Treat the cause; avoid stimulants."
        )

    # ---- Rule 5: Normal Sinus Rhythm --------------------------------
    # HR 60-100, low variability, narrow QRS, clear P-waves
    elif (not any(np.isnan([hr, rr_cv, qrs, p_ratio])) and
              HR_NORMAL_LOW <= hr <= HR_NORMAL_HIGH and
              rr_cv < RR_CV_NORMAL_MAX and
              qrs < QRS_NORMAL_MAX and
              p_ratio > P_WAVE_NORMAL_MIN):
        diagnosis      = "Normal Sinus Rhythm"
        confidence     = 0.95
        severity       = "NONE"
        recommendation = "No immediate action required. Routine follow-up as appropriate."

    # ---- Rule 6: Inconclusive (default fallback) --------------------
    else:
        diagnosis      = "INCONCLUSIVE"
        confidence     = 0.65
        severity       = "LOW"
        recommendation = (
            "Recording does not meet criteria for a specific diagnosis. "
            "Repeat with improved electrode contact and longer recording."
        )

    # Apply severity override from critical warnings
    if severity_override is not None:
        severity_levels = ["NONE", "LOW", "MODERATE", "HIGH"]
        if severity_levels.index(severity_override) > severity_levels.index(severity):
            severity = severity_override

    return {
        "diagnosis"      : diagnosis,
        "confidence"     : round(confidence, 4),
        "severity"       : severity,
        "recommendation" : recommendation,
        "warnings"       : warnings,
    }


# =============================================================================
# 3. PRETTY PRINTING
# =============================================================================

def print_diagnosis(result: dict, label: str = "") -> None:
    """Print a formatted clinical report for one diagnosis result."""
    sev_icons = {"NONE": "[  ] ", "LOW": "[L] ", "MODERATE": "[M] ", "HIGH": "[H] "}
    icon = sev_icons.get(result["severity"], "    ")

    print(f"\n  {'='*50}")
    if label:
        print(f"  Case: {label}")
    print(f"  {'='*50}")
    print(f"  {icon}Diagnosis     : {result['diagnosis']}")
    print(f"  Confidence      : {result['confidence']*100:.1f}%")
    print(f"  Severity        : {result['severity']}")
    print(f"  Recommendation  : {result['recommendation']}")
    if result["warnings"]:
        print(f"  Warnings:")
        for w in result["warnings"]:
            print(f"    [!] {w}")
    else:
        print(f"  Warnings        : none")
    print(f"  {'='*50}")


# =============================================================================
# 4. SYNTHETIC TEST CASES  (phase 2 validation)
# =============================================================================

SYNTHETIC_CASES = {
    "Normal": {
        "hr_mean"      : 72.0,
        "hr_std"       : 3.5,
        "rr_cv"        : 0.05,
        "rr_entropy"   : 0.40,
        "sdnn"         : 55.0,
        "rmssd"        : 45.0,
        "pnn50"        : 8.0,
        "qrs_dur_mean" : 90.0,
        "qrs_dur_std"  : 5.0,
        "p_wave_ratio" : 0.95,
        "st_elevation" : 50.0,
    },
    "AFib": {
        "hr_mean"      : 88.0,
        "hr_std"       : 22.0,
        "rr_cv"        : 0.34,
        "rr_entropy"   : 0.85,
        "sdnn"         : 180.0,
        "rmssd"        : 210.0,
        "pnn50"        : 55.0,
        "qrs_dur_mean" : 95.0,
        "qrs_dur_std"  : 8.0,
        "p_wave_ratio" : 0.30,
        "st_elevation" : 30.0,
    },
    "PVC (Inconclusive/wide QRS)": {
        "hr_mean"      : 75.0,
        "hr_std"       : 8.0,
        "rr_cv"        : 0.18,
        "rr_entropy"   : 0.55,
        "sdnn"         : 80.0,
        "rmssd"        : 70.0,
        "pnn50"        : 12.0,
        "qrs_dur_mean" : 145.0,
        "qrs_dur_std"  : 25.0,
        "p_wave_ratio" : 0.65,
        "st_elevation" : 40.0,
    },
    "Bradycardia": {
        "hr_mean"      : 44.0,
        "hr_std"       : 1.5,
        "rr_cv"        : 0.10,
        "rr_entropy"   : 0.30,
        "sdnn"         : 30.0,
        "rmssd"        : 25.0,
        "pnn50"        : 2.0,
        "qrs_dur_mean" : 88.0,
        "qrs_dur_std"  : 4.0,
        "p_wave_ratio" : 0.92,
        "st_elevation" : 20.0,
    },
    "Sinus Tachycardia": {
        "hr_mean"      : 135.0,
        "hr_std"       : 5.0,
        "rr_cv"        : 0.08,
        "rr_entropy"   : 0.35,
        "sdnn"         : 18.0,
        "rmssd"        : 15.0,
        "pnn50"        : 0.5,
        "qrs_dur_mean" : 92.0,
        "qrs_dur_std"  : 3.0,
        "p_wave_ratio" : 0.95,
        "st_elevation" : 30.0,
    },
    "Ventricular Tachycardia": {
        "hr_mean"      : 165.0,
        "hr_std"       : 4.0,
        "rr_cv"        : 0.05,
        "rr_entropy"   : 0.20,
        "sdnn"         : 10.0,
        "rmssd"        : 8.0,
        "pnn50"        : 0.0,
        "qrs_dur_mean" : 155.0,
        "qrs_dur_std"  : 10.0,
        "p_wave_ratio" : 0.25,
        "st_elevation" : 60.0,
    },
    "Extreme HR warning (HR=38)": {
        "hr_mean"      : 38.0,
        "hr_std"       : 2.0,
        "rr_cv"        : 0.08,
        "rr_entropy"   : 0.25,
        "sdnn"         : 28.0,
        "rmssd"        : 22.0,
        "pnn50"        : 1.0,
        "qrs_dur_mean" : 88.0,
        "qrs_dur_std"  : 3.0,
        "p_wave_ratio" : 0.91,
        "st_elevation" : 20.0,
    },
    "Possible MI (ST elevation)": {
        "hr_mean"      : 85.0,
        "hr_std"       : 4.0,
        "rr_cv"        : 0.06,
        "rr_entropy"   : 0.42,
        "sdnn"         : 45.0,
        "rmssd"        : 40.0,
        "pnn50"        : 5.0,
        "qrs_dur_mean" : 100.0,
        "qrs_dur_std"  : 6.0,
        "p_wave_ratio" : 0.93,
        "st_elevation" : 350.0,   # > 200 uV threshold -> MI warning
    },
    "Low HRV warning": {
        "hr_mean"      : 76.0,
        "hr_std"       : 2.0,
        "rr_cv"        : 0.06,
        "rr_entropy"   : 0.38,
        "sdnn"         : 30.0,    # < 50 ms -> low HRV warning
        "rmssd"        : 25.0,
        "pnn50"        : 3.0,
        "qrs_dur_mean" : 95.0,
        "qrs_dur_std"  : 4.0,
        "p_wave_ratio" : 0.91,
        "st_elevation" : 40.0,
    },
}

# Expected primary diagnoses for validation
EXPECTED_DIAGNOSES = {
    "Normal"                          : "Normal Sinus Rhythm",
    "AFib"                            : "Atrial Fibrillation",
    "PVC (Inconclusive/wide QRS)"     : "INCONCLUSIVE",
    "Bradycardia"                     : "Bradycardia",
    "Sinus Tachycardia"               : "Sinus Tachycardia",
    "Ventricular Tachycardia"         : "Ventricular Tachycardia",
    "Extreme HR warning (HR=38)"      : "Bradycardia",
    "Possible MI (ST elevation)"      : "Normal Sinus Rhythm",  # normal HR but ST warning
    "Low HRV warning"                 : "Normal Sinus Rhythm",  # NSR with low HRV warning
}


def run_phase2():
    """
    Full Phase 2 test: classify all synthetic test cases and print results.
    Also verifies that expected diagnoses match.
    """
    print("\n" + "="*60)
    print("  PHASE 2 -- Clinical Classification Rules")
    print("="*60)

    passed = 0
    failed = 0

    for label, features in SYNTHETIC_CASES.items():
        result = classify_ecg(features)
        print_diagnosis(result, label=label)

        expected = EXPECTED_DIAGNOSES.get(label)
        if expected is not None:
            if result["diagnosis"] == expected:
                print(f"  [PASS] Expected: '{expected}'")
                passed += 1
            else:
                print(f"  [FAIL] Expected: '{expected}', Got: '{result['diagnosis']}'")
                failed += 1

    print(f"\n  {'='*60}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed+failed} cases")

    # Verify specific clinical safety requirements
    print("\n  Clinical Safety Checks:")

    # VT must never be classified as Normal
    vt_result = classify_ecg(SYNTHETIC_CASES["Ventricular Tachycardia"])
    assert vt_result["diagnosis"] != "Normal Sinus Rhythm", \
        "SAFETY FAIL: VT classified as Normal!"
    print("  [PASS] VT is not misclassified as Normal Sinus Rhythm")

    # VT must be HIGH severity
    assert vt_result["severity"] == "HIGH", \
        f"VT should be HIGH severity, got {vt_result['severity']}"
    print("  [PASS] VT severity is HIGH")

    # Extreme HR must trigger severity override
    extreme_result = classify_ecg(SYNTHETIC_CASES["Extreme HR warning (HR=38)"])
    assert extreme_result["severity"] == "HIGH", \
        f"Extreme HR should override to HIGH, got {extreme_result['severity']}"
    print("  [PASS] Extreme HR (38 bpm) triggers HIGH severity")

    # MI warning must trigger
    mi_result = classify_ecg(SYNTHETIC_CASES["Possible MI (ST elevation)"])
    assert any("MI" in w or "Possible" in w for w in mi_result["warnings"]), \
        "SAFETY FAIL: ST elevation > 200 uV did not generate MI warning!"
    print("  [PASS] ST elevation > 200 uV generates 'Possible MI' warning")

    # Low HRV warning must trigger
    hrv_result = classify_ecg(SYNTHETIC_CASES["Low HRV warning"])
    assert any("HRV" in w or "SDNN" in w for w in hrv_result["warnings"]), \
        "SDNN < 50 ms did not generate Low HRV warning!"
    print("  [PASS] SDNN < 50 ms generates Low HRV warning")

    print(f"\n  All safety checks passed.")
    print("\n" + "="*60)
    print("  PHASE 2 COMPLETE")
    print("="*60)


# =============================================================================
if __name__ == "__main__":
    run_phase2()
