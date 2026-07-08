"""The rules <-> ML merge in `ecg_analyzer.predict_combined`.

This is the highest-safety logic in the codebase and was previously untested.
We monkeypatch the three pipeline calls (detect_r_peaks, extract_features,
classify_ecg) in the ecg_analyzer namespace so the tests exercise the merge,
confidence gating, probability expansion, and NaN imputation deterministically —
no DSP, no real model, no MIT-BIH data.
"""

import numpy as np
import pytest

import ecg_analyzer
from ecg_analyzer import predict_combined, CONFIDENCE_THRESHOLD, CLASSES
from ecg_phase1 import FEATURE_KEYS


def _finite_features():
    """A full, finite feature dict (values are arbitrary but non-NaN)."""
    return {k: 1.0 for k in FEATURE_KEYS}


def _rule_result(diagnosis, severity="LOW", confidence=0.9):
    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "severity": severity,
        "recommendation": "rule recommendation",
        "warnings": [],
    }


@pytest.fixture
def patch_pipeline(monkeypatch):
    """Factory to stub the pipeline with a chosen feature dict + rule result."""
    def _apply(features, rule_result, n_peaks=10):
        monkeypatch.setattr(ecg_analyzer, "detect_r_peaks",
                            lambda sig, sr, bp=None: np.arange(n_peaks))
        monkeypatch.setattr(ecg_analyzer, "extract_features",
                            lambda sig, peaks, sr, bp=None: features)
        monkeypatch.setattr(ecg_analyzer, "classify_ecg",
                            lambda feats: rule_result)
    return _apply


SIGNAL = np.zeros(4000)  # length only feeds metrics; content is stubbed over


def test_too_few_rpeaks_returns_error(monkeypatch):
    monkeypatch.setattr(ecg_analyzer, "detect_r_peaks", lambda sig, sr, bp=None: np.arange(3))
    report = predict_combined(SIGNAL, 360, model=None)
    assert report["error"] == "Could not detect heartbeats"
    assert report["diagnosis"] is None


def test_conclusive_rule_wins_over_model(patch_pipeline, make_model):
    patch_pipeline(_finite_features(), _rule_result("Normal Sinus Rhythm"))
    model = make_model([0, 1, 2, 3, 4], [0.0, 0.0, 1.0, 0.0, 0.0])  # would say PVC
    report = predict_combined(SIGNAL, 360, model=model)
    assert report["method"] == "clinical_rules"
    assert report["diagnosis"] == "Normal"  # canonicalised, not "PVC"


def test_inconclusive_falls_back_to_ml_when_confident(patch_pipeline, make_model):
    patch_pipeline(_finite_features(), _rule_result("INCONCLUSIVE"))
    # argmax -> index 2 (PVC), confidence 0.70 >= threshold
    model = make_model([0, 1, 2, 3, 4], [0.1, 0.05, 0.70, 0.10, 0.05])
    report = predict_combined(SIGNAL, 360, model=model)
    assert report["method"] == "ml_fallback"
    assert report["diagnosis"] == "PVC"
    assert report["severity"] == "LOW"  # from ML_SEV_REC["PVC"]
    assert report["confidence"] >= CONFIDENCE_THRESHOLD


def test_inconclusive_stays_inconclusive_when_ml_low_confidence(patch_pipeline, make_model):
    patch_pipeline(_finite_features(), _rule_result("INCONCLUSIVE"))
    model = make_model([0, 1, 2, 3, 4], [0.3, 0.25, 0.2, 0.15, 0.1])  # max 0.30 < 0.60
    report = predict_combined(SIGNAL, 360, model=model)
    assert report["method"] == "clinical_rules"
    assert report["diagnosis"] == "INCONCLUSIVE"
    assert any("confidence low" in w.lower() for w in report["warnings"])


def test_probability_expansion_for_missing_classes(patch_pipeline, make_model):
    patch_pipeline(_finite_features(), _rule_result("INCONCLUSIVE"))
    # Model only knows classes 0, 2, 3 (missing AFib=1 and Tachycardia=4).
    model = make_model([0, 2, 3], [0.2, 0.7, 0.1])
    report = predict_combined(SIGNAL, 360, model=model)
    probs = report["ml_probabilities"]
    assert set(probs) == set(CLASSES)          # all 5 keys present
    assert probs["AFib"] == 0.0                # missing class -> 0
    assert probs["Tachycardia"] == 0.0
    assert probs["PVC"] == 0.7                 # class index 2 got 0.7
    assert report["diagnosis"] == "PVC"


def test_single_nan_feature_is_imputed_not_dropped(patch_pipeline):
    feats = _finite_features()
    feats["st_elevation"] = np.nan  # one unmeasurable feature

    captured = {}

    class RecordingModel:
        classes_ = np.array([0, 1, 2, 3, 4])

        def predict_proba(self, X):
            captured["X"] = np.asarray(X, dtype=float)
            return np.array([[0.1, 0.05, 0.70, 0.10, 0.05]])

    patch_pipeline(feats, _rule_result("INCONCLUSIVE"))
    report = predict_combined(SIGNAL, 360, model=RecordingModel())

    assert report["ml_probabilities"], "ML should still run after imputation"
    assert np.all(np.isfinite(captured["X"])), "NaN feature must be imputed before predict"
    # The imputed column should carry the committed training median.
    st_idx = FEATURE_KEYS.index("st_elevation")
    assert captured["X"][0, st_idx] == pytest.approx(
        ecg_analyzer.FEATURE_MEDIANS["st_elevation"]
    )


def test_all_nan_features_skip_ml(patch_pipeline, make_model):
    feats = {k: np.nan for k in FEATURE_KEYS}
    patch_pipeline(feats, _rule_result("INCONCLUSIVE"))
    model = make_model([0, 1, 2, 3, 4], [0.1, 0.05, 0.70, 0.10, 0.05])
    report = predict_combined(SIGNAL, 360, model=model)
    assert report["ml_probabilities"] == {}   # nothing to impute from -> ML skipped
    assert report["diagnosis"] == "INCONCLUSIVE"


def test_report_always_carries_disclaimer(patch_pipeline):
    patch_pipeline(_finite_features(), _rule_result("Normal Sinus Rhythm"))
    report = predict_combined(SIGNAL, 360, model=None)
    assert report["disclaimer"]
    assert "SCREENING" in report["disclaimer"].upper()
