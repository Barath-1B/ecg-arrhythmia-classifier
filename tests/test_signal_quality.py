"""Signal-quality gating and CSV loading in ecg_analyzer."""

import numpy as np
import pytest

from ecg_analyzer import (
    check_signal_quality,
    load_ecg_csv,
    MIN_DURATION_S,
    FLAT_STD_THRESHOLD,
)

SR = 360


def _good_signal(seconds=12.0):
    n = int(SR * seconds)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm-ish, clearly non-flat


def test_good_signal_passes():
    assert check_signal_quality(_good_signal(), SR) is None


def test_recording_too_short():
    short = _good_signal(seconds=MIN_DURATION_S - 1)
    msg = check_signal_quality(short, SR)
    assert msg is not None and "too short" in msg.lower()


def test_flat_line_detected():
    flat = np.ones(int(SR * 12))  # std == 0 < FLAT_STD_THRESHOLD
    msg = check_signal_quality(flat, SR)
    assert msg is not None and "flat" in msg.lower()


def test_near_flat_below_threshold_detected():
    n = int(SR * 12)
    tiny = np.zeros(n)
    tiny[::2] = FLAT_STD_THRESHOLD / 10.0  # std stays well under the threshold
    assert check_signal_quality(tiny, SR) is not None


def test_nan_signal_rejected_before_flat_check():
    # A3: a long-enough signal containing NaN must be rejected, not silently
    # accepted because np.std(...) is NaN and `NaN < threshold` is False.
    sig = _good_signal()
    sig[100] = np.nan
    msg = check_signal_quality(sig, SR)
    assert msg is not None and "non-finite" in msg.lower()


def test_inf_signal_rejected():
    sig = _good_signal()
    sig[200] = np.inf
    assert check_signal_quality(sig, SR) is not None


# ---- load_ecg_csv -------------------------------------------------------

def test_load_single_column(tmp_path):
    p = tmp_path / "ecg.csv"
    p.write_text("\n".join(str(v) for v in [0.1, 0.2, 0.3, 0.4]))
    sig = load_ecg_csv(str(p))
    assert sig.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_load_multicolumn_takes_first_column(tmp_path):
    p = tmp_path / "ecg.csv"
    p.write_text("0.1,9\n0.2,9\n0.3,9\n")
    sig = load_ecg_csv(str(p))
    assert sig.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_load_skips_comment_lines(tmp_path):
    p = tmp_path / "ecg.csv"
    p.write_text("# header comment\n0.1\n0.2\n")
    sig = load_ecg_csv(str(p))
    assert sig.tolist() == pytest.approx([0.1, 0.2])


def test_load_duration_truncation(tmp_path):
    p = tmp_path / "ecg.csv"
    p.write_text("\n".join(str(v) for v in range(1000)))
    sig = load_ecg_csv(str(p), sr=100, duration_s=2.0)  # keep 100*2 = 200 samples
    assert len(sig) == 200
