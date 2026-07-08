"""Phase 2 clinical rule engine (`classify_ecg`).

Drives the synthetic fixtures already defined in ecg_phase2 (previously only
exercised from a __main__ block) plus targeted tests for the load-bearing
behaviours: priority ordering (VT before sinus tachycardia), severity overrides,
and NaN-guard fall-through to INCONCLUSIVE.
"""

import numpy as np
import pytest

from ecg_phase2 import (
    classify_ecg,
    SYNTHETIC_CASES,
    EXPECTED_DIAGNOSES,
)


@pytest.mark.parametrize("case_name", list(EXPECTED_DIAGNOSES))
def test_synthetic_cases_match_expected_diagnosis(case_name):
    result = classify_ecg(SYNTHETIC_CASES[case_name])
    assert result["diagnosis"] == EXPECTED_DIAGNOSES[case_name]


@pytest.mark.parametrize("case_name", list(EXPECTED_DIAGNOSES))
def test_result_shape_is_complete(case_name):
    result = classify_ecg(SYNTHETIC_CASES[case_name])
    assert set(result) == {
        "diagnosis", "confidence", "severity", "recommendation", "warnings"
    }
    assert result["severity"] in {"NONE", "LOW", "MODERATE", "HIGH"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["warnings"], list)


def test_vt_takes_priority_over_sinus_tachycardia():
    # Wide-complex tachycardia must be labelled VT, never benign sinus tachy.
    vt = classify_ecg(SYNTHETIC_CASES["Ventricular Tachycardia"])
    assert vt["diagnosis"] == "Ventricular Tachycardia"
    assert vt["severity"] == "HIGH"


def test_narrow_complex_tachycardia_is_sinus_not_vt():
    # Same fast HR but narrow QRS -> sinus tachycardia, not VT.
    result = classify_ecg(SYNTHETIC_CASES["Sinus Tachycardia"])
    assert result["diagnosis"] == "Sinus Tachycardia"


def test_extreme_bradycardia_forces_high_severity():
    result = classify_ecg(SYNTHETIC_CASES["Extreme HR warning (HR=38)"])
    assert result["severity"] == "HIGH"
    assert any("bradycardia" in w.lower() for w in result["warnings"])


def test_st_elevation_raises_mi_warning_and_high_severity():
    result = classify_ecg(SYNTHETIC_CASES["Possible MI (ST elevation)"])
    # Underlying rhythm is normal, but the ST warning must override severity.
    assert result["diagnosis"] == "Normal Sinus Rhythm"
    assert result["severity"] == "HIGH"
    assert any("MI" in w for w in result["warnings"])


def test_st_elevation_nan_does_not_raise_mi_warning():
    # A4: unmeasurable ST is NaN (not 0.0). The MI guard must skip it, and a NaN
    # ST must never be reported as a >200uV elevation.
    case = dict(SYNTHETIC_CASES["Possible MI (ST elevation)"])
    case["st_elevation"] = np.nan
    result = classify_ecg(case)
    assert not any("MI" in w for w in result["warnings"])


def test_low_hrv_warning_present():
    result = classify_ecg(SYNTHETIC_CASES["Low HRV warning"])
    assert any("HRV" in w for w in result["warnings"])


def test_all_nan_features_are_inconclusive():
    nan_features = {k: np.nan for k in SYNTHETIC_CASES["Normal"]}
    result = classify_ecg(nan_features)
    assert result["diagnosis"] == "INCONCLUSIVE"
    # No numeric warnings should fire when every guard sees NaN.
    assert result["warnings"] == []
