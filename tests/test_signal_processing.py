"""Signal-processing sanity checks for Phase 1.

Includes an entropy PARITY guard: `_reference_sample_entropy` below is a verbatim
copy of the original O(N^2) `_sample_entropy`. The production function is asserted
equal to it within tolerance, so the Phase B vectorized/capped rewrite cannot
change the feature value the committed model was trained on.
"""

import numpy as np
import pytest

import ecg_phase1
from ecg_phase1 import detect_r_peaks, extract_features, FEATURE_KEYS, FS


# --------------------------------------------------------------------------
# Reference implementation (frozen copy of the original _sample_entropy).
# --------------------------------------------------------------------------
def _reference_sample_entropy(rr, m=2, r_tol=0.2):
    N = len(rr)
    if N < m + 2:
        return 0.0
    r = r_tol * np.std(rr, ddof=1)
    if r == 0:
        return 0.0

    def _phi(m_val):
        count = 0
        total = 0
        for i in range(N - m_val):
            template = rr[i:i + m_val]
            for j in range(N - m_val):
                if i == j:
                    continue
                if np.max(np.abs(template - rr[j:j + m_val])) <= r:
                    count += 1
            total += 1
        return count / total if total > 0 else 0

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m == 0:
        return 0.0
    raw = -np.log(phi_m1 / phi_m + 1e-10)
    return float(np.clip(raw / 3.0, 0.0, 1.0))


# Deterministic RR fixtures spanning regular, irregular, and degenerate cases.
_RR_REGULAR = np.array([800.0, 802.0, 799.0, 801.0, 800.0, 803.0, 798.0,
                        800.0, 801.0, 799.0, 800.0, 802.0])
_RR_IRREGULAR = np.array([600.0, 950.0, 720.0, 1100.0, 680.0, 890.0, 1200.0,
                          640.0, 980.0, 760.0, 1050.0, 700.0])
_RR_CONSTANT = np.array([800.0] * 12)  # zero std -> reference returns 0.0
_RR_SHORT = np.array([800.0, 810.0, 790.0])  # N < m+2 -> 0.0


@pytest.mark.parametrize("rr", [_RR_REGULAR, _RR_IRREGULAR, _RR_CONSTANT, _RR_SHORT])
def test_sample_entropy_matches_reference(rr):
    assert ecg_phase1._sample_entropy(rr) == pytest.approx(
        _reference_sample_entropy(rr), abs=1e-9
    )


def test_sample_entropy_in_unit_range():
    for rr in (_RR_REGULAR, _RR_IRREGULAR):
        val = ecg_phase1._sample_entropy(rr)
        assert 0.0 <= val <= 1.0


# --------------------------------------------------------------------------
# R-peak detection on a synthetic beat train.
# --------------------------------------------------------------------------
def _synthetic_ecg(n_beats=15, rr_s=0.8, fs=FS, amp=2.0):
    """Build a clean signal with sharp triangular QRS spikes at known spacing."""
    spacing = int(rr_s * fs)
    n = spacing * (n_beats + 1)
    sig = np.zeros(n)
    half = 4  # ~22 ms triangular spike at 360 Hz
    peak_locs = []
    for b in range(1, n_beats + 1):
        c = b * spacing
        for k in range(-half, half + 1):
            idx = c + k
            if 0 <= idx < n:
                sig[idx] = amp * (1.0 - abs(k) / (half + 1))
        peak_locs.append(c)
    return sig, np.array(peak_locs)


def test_detect_r_peaks_finds_expected_count():
    sig, true_locs = _synthetic_ecg(n_beats=15)
    peaks = detect_r_peaks(sig, FS)
    assert peaks.dtype.kind == "i"
    # Allow the detector to miss the first/last beat (windowing/threshold) or add
    # at most one from filter ringing — it must find roughly the right count on
    # this clean, well-separated train, not an order-of-magnitude different one.
    assert len(true_locs) - 2 <= len(peaks) <= len(true_locs) + 1


def test_detected_peaks_align_with_true_locations():
    sig, true_locs = _synthetic_ecg(n_beats=15)
    peaks = detect_r_peaks(sig, FS)
    tol = int(0.03 * FS)  # 30 ms
    for p in peaks:
        assert np.min(np.abs(true_locs - p)) <= tol


def test_refractory_period_enforced():
    sig, _ = _synthetic_ecg(n_beats=15)
    peaks = detect_r_peaks(sig, FS)
    if len(peaks) > 1:
        min_gap = int(FS * 200.0 / 1000.0)  # 200 ms refractory
        assert np.all(np.diff(peaks) >= min_gap)


# --------------------------------------------------------------------------
# Feature extraction edge cases.
# --------------------------------------------------------------------------
def test_extract_features_all_nan_when_too_few_peaks():
    sig = np.zeros(4000)
    feats = extract_features(sig, np.array([100]), FS)  # single peak -> <2 RR
    assert set(feats) == set(FEATURE_KEYS)
    assert all(np.isnan(v) for v in feats.values())


def test_extract_features_returns_finite_hr_on_clean_signal():
    sig, locs = _synthetic_ecg(n_beats=15)
    feats = extract_features(sig, locs, FS)
    assert not np.isnan(feats["hr_mean"])
    # 0.8 s RR -> ~75 bpm
    assert 60.0 <= feats["hr_mean"] <= 90.0
