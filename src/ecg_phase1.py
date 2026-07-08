# -*- coding: utf-8 -*-
"""
ECG Monitor - Phase 1: Data Loading & Feature Extraction
=========================================================
Loads ECG data from the MIT-BIH Arrhythmia Database (PhysioNet),
implements Pan & Tompkins R-peak detection, and extracts 11 clinical
features per recording for downstream classification.

References:
  - Pan J, Tompkins WJ (1985). "A Real-Time QRS Detection Algorithm."
    IEEE Trans Biomed Eng 32(3):230-236.
  - PhysioNet / MIT-BIH Arrhythmia Database:
    https://physionet.org/content/mitdb/1.0.0/
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend - saves to file instead of pop-up
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")

# Path to the local MIT-BIH database
_HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(_HERE)
DB_PATH   = os.path.join(ROOT, "data", "mit-bih-arrhythmia-database-1.0.0",
                          "mit-bih-arrhythmia-database-1.0.0")
PLOTS_DIR = os.path.join(ROOT, "outputs", "plots")

# Sampling rate of the MIT-BIH database (fixed at 360 Hz)
FS = 360   # Hz

# Ordered feature names — the single source of truth for feature column order.
# LOAD-BEARING: the committed Random Forest (models/ecg_model.joblib) was trained
# on this exact order. Reorder only if you retrain.
FEATURE_KEYS = [
    "hr_mean", "hr_std", "rr_cv", "rr_entropy",
    "sdnn", "rmssd", "pnn50",
    "qrs_dur_mean", "qrs_dur_std", "p_wave_ratio", "st_elevation",
]


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_mitbih_record(record_id: int, *, duration_s: float = 30.0):
    """
    Load one MIT-BIH record from the local database.

    Parameters
    ----------
    record_id : int
        Record number (e.g. 100).
    duration_s : float
        How many seconds to load (default 30 s). Set to None for full record.

    Returns
    -------
    signal  : 1-D ndarray  - MLII channel, physical units (mV)
    ann     : wfdb.Annotation object
    fs      : int - sampling rate (360 Hz)
    """
    record_path = os.path.join(DB_PATH, str(record_id))
    sampto = int(FS * duration_s) if duration_s is not None else None

    record = wfdb.rdrecord(record_path, sampto=sampto)
    ann    = wfdb.rdann(record_path, "atr", sampto=sampto)

    # Select the MLII lead by NAME, not position. Most records have MLII on
    # channel 0, but e.g. record 114 has it on channel 1, and records 102/104
    # have no MLII at all (V5/V2) — for those we fall back to channel 0 and the
    # caller is responsible for knowing it's a substitute lead.
    signal = record.p_signal[:, _mlii_channel(record.sig_name)].astype(np.float64)

    return signal, ann, FS


def _mlii_channel(sig_names: list) -> int:
    """Index of the MLII lead, or 0 (substitute lead) if the record has none."""
    return sig_names.index("MLII") if "MLII" in sig_names else 0


def print_record_info(record_id: int, signal: np.ndarray,
                      ann, fs: int) -> None:
    """Print summary statistics for a loaded record."""
    duration = len(signal) / fs
    beat_mask = [sym not in ("+", "~", "|", "!") for sym in ann.symbol]
    n_beats   = sum(beat_mask)
    symbols   = sorted(set(ann.symbol))

    print(f"\n{'='*55}")
    print(f"  MIT-BIH Record {record_id}")
    print(f"{'='*55}")
    print(f"  Signal shape   : {signal.shape}")
    print(f"  Sampling rate  : {fs} Hz")
    print(f"  Duration       : {duration:.1f} s")
    print(f"  Beats detected : {n_beats}")
    print(f"  Beat symbols   : {symbols}")
    print(f"  mV range       : [{signal.min():.3f}, {signal.max():.3f}]")
    print(f"{'='*55}\n")


# =============================================================================
# 2. PAN & TOMPKINS R-PEAK DETECTION
# =============================================================================

def _bandpass_filter(signal: np.ndarray, fs: int,
                     low: float = 5.0, high: float = 15.0) -> np.ndarray:
    """
    5-15 Hz Butterworth bandpass - isolates QRS energy.
    (Pan & Tompkins 1985, step 1)
    """
    nyq  = fs / 2.0
    b, a = butter(2, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def _differentiate(signal: np.ndarray) -> np.ndarray:
    """
    5-point derivative approximation (Pan & Tompkins 1985, step 2).
    Emphasises the steep slopes of the QRS complex.
    """
    d = np.zeros_like(signal)
    # Classic 5-point central difference for derivative estimation
    d[2:-2] = (
        -signal[:-4] - 2.0 * signal[1:-3] + 2.0 * signal[3:-1] + signal[4:]
    ) / 8.0
    return d


def _moving_window_integration(signal: np.ndarray, fs: int,
                                window_ms: float = 60.0) -> np.ndarray:
    """
    Moving-window integrator (Pan & Tompkins 1985, step 4).
    window_ms: window width in milliseconds.
    (~150 ms in original paper; 60 ms used here for sharper localisation.)
    """
    w = max(1, int(fs * window_ms / 1000.0))
    kernel = np.ones(w) / w
    return np.convolve(signal, kernel, mode="same")


def detect_r_peaks(signal: np.ndarray, fs: int,
                   refractory_ms: float = 200.0,
                   bp: np.ndarray = None) -> np.ndarray:
    """
    Pan & Tompkins (1985) R-peak detector.

    Steps
    -----
    1. 5-15 Hz bandpass filter
    2. 5-point derivative
    3. Squaring (all positive, emphasises large slopes)
    4. Moving-window integration (~60 ms)
    5. Adaptive threshold = mean + 2*std of integrated signal
    6. find_peaks above threshold
    7. Enforce refractory period (remove peaks < 200 ms apart)

    Parameters
    ----------
    signal        : raw ECG, 1-D array
    fs            : sampling rate (Hz)
    refractory_ms : minimum inter-peak interval in ms (default 200 ms = 300 bpm max)

    Returns
    -------
    r_peaks : 1-D int array of R-peak sample indices

    `bp` is an optional precomputed 5-15 Hz bandpass of `signal`; pass it to avoid
    recomputing the same filter that feature extraction also needs (see
    extract_features). When None it is computed here as before.
    """
    # Step 1 - bandpass (reuse a precomputed one if provided)
    if bp is None:
        bp = _bandpass_filter(signal, fs)

    # Step 2 - derivative
    deriv = _differentiate(bp)

    # Step 3 - squaring (makes all values positive; emphasises large slopes)
    squared = deriv ** 2

    # Step 4 - moving-window integration
    mwi = _moving_window_integration(squared, fs, window_ms=60.0)

    # Step 5 - adaptive threshold
    threshold = mwi.mean() + 2.0 * mwi.std()

    # Step 6 - find local maxima above threshold
    # min_distance enforces refractory period at coarse stage
    min_dist = max(1, int(fs * refractory_ms / 1000.0))
    peaks, _ = find_peaks(mwi, height=threshold, distance=min_dist)

    # Step 7 - refine to true R-peak position in original signal
    # Search +/-20 ms around each MWI peak for the true maximum
    search_rad = int(fs * 0.020)
    refined = []
    for p in peaks:
        lo = max(0, p - search_rad)
        hi = min(len(signal), p + search_rad + 1)
        refined.append(lo + int(np.argmax(signal[lo:hi])))

    r_peaks = np.array(sorted(set(refined)), dtype=int)

    # Final refractory filter on refined peaks
    if len(r_peaks) > 1:
        keep = [r_peaks[0]]
        for pk in r_peaks[1:]:
            if pk - keep[-1] >= min_dist:
                keep.append(pk)
        r_peaks = np.array(keep, dtype=int)

    return r_peaks


def plot_ecg_with_rpeaks(signal: np.ndarray, r_peaks: np.ndarray,
                          fs: int, record_id,
                          save_path: str = None) -> None:
    """
    Plot the ECG signal with R-peaks overlaid as red dots.
    Saves to save_path (or auto-names the file).
    """
    t = np.arange(len(signal)) / fs

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, signal, color="steelblue", linewidth=0.8, label="ECG (MLII)")
    ax.scatter(r_peaks / fs, signal[r_peaks], color="red", s=40,
               zorder=5, label=f"R-peaks (n={len(r_peaks)})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(f"MIT-BIH Record {record_id} -- ECG with R-peaks")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = save_path or os.path.join(PLOTS_DIR, f"ecg_record{record_id}_rpeaks.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Saved -> {path}")


# =============================================================================
# 3. FEATURE EXTRACTION  (11 features per recording)
# =============================================================================

def _compute_rr_intervals(r_peaks: np.ndarray, fs: int) -> np.ndarray:
    """Convert R-peak indices to RR intervals in milliseconds."""
    if len(r_peaks) < 2:
        return np.array([])
    return np.diff(r_peaks).astype(float) * 1000.0 / fs   # ms


# Cap on the number of RR intervals fed to the entropy estimate. Sample entropy
# is inherently O(N^2); this bounds worst-case cost on a long upload (the DoS
# vector on /api/analyze) while never triggering for the short windows the model
# was trained on. When exceeded, the most-recent intervals are used.
SAMPLE_ENTROPY_MAX_INTERVALS = 1000


def _sample_entropy(rr: np.ndarray, m: int = 2, r_tol: float = 0.2) -> float:
    """
    Approximate sample entropy of an RR interval sequence.

    A high value indicates irregularity (e.g. AFib); a low value indicates
    monotonous rhythm (e.g. sinus tachycardia / bradycardia).

    Vectorized O(N^2) implementation (numpy pairwise Chebyshev distance). It is
    numerically identical to the original nested-loop version but avoids the
    per-pair Python overhead; the interval count is capped for a bounded cost on
    long signals. See tests/test_signal_processing.py for the parity guard.

    Parameters
    ----------
    rr    : array of RR intervals (ms)
    m     : template length (default 2)
    r_tol : tolerance as fraction of std (default 0.2)

    Returns
    -------
    entropy normalised to [0, 1]
    """
    rr = np.asarray(rr, dtype=float)
    if len(rr) > SAMPLE_ENTROPY_MAX_INTERVALS:
        rr = rr[-SAMPLE_ENTROPY_MAX_INTERVALS:]

    N = len(rr)
    if N < m + 2:
        return 0.0
    r = r_tol * np.std(rr, ddof=1)
    if r == 0:
        return 0.0

    def _phi(m_val):
        # Templates of length m_val: row i is rr[i:i+m_val]. Pairwise Chebyshev
        # distance D[i,j] = max_k |T[i,k]-T[j,k]|, built column-by-column so only
        # (M, M) memory is used (m_val is 2 or 3). Matches the reference exactly:
        # count ordered pairs (i != j) with D <= r, averaged over the M templates.
        M = N - m_val
        if M <= 0:
            return 0.0
        idx = np.arange(M)[:, None] + np.arange(m_val)[None, :]
        T = rr[idx]                                   # (M, m_val)
        D = np.zeros((M, M))
        for k in range(m_val):
            col = T[:, k]
            np.maximum(D, np.abs(col[:, None] - col[None, :]), out=D)
        count = int((D <= r).sum()) - M               # drop the M self-matches
        return count / M

    phi_m   = _phi(m)
    phi_m1  = _phi(m + 1)

    if phi_m == 0:
        return 0.0
    raw = -np.log(phi_m1 / phi_m + 1e-10)
    # Normalise to [0, 1] via scaling (max ~3 nats for highly irregular signals)
    return float(np.clip(raw / 3.0, 0.0, 1.0))


# QRS width calibration knobs. The QRS envelope is the 5-15 Hz bandpass signal,
# squared, then smoothed with a moving-window integrator (Pan-Tompkins style) so
# it forms one clean hump per beat. Onset/offset are where that hump falls below
# QRS_WIDTH_FRACTION of the beat's local peak. Tuned so known-normal MIT-BIH 100
# reads ~90-100 ms. Physical signals need this tuning — keep both knobs.
#   raise fraction / shrink smooth_ms -> narrower measured QRS
#   lower fraction / grow  smooth_ms  -> wider measured QRS
QRS_WIDTH_FRACTION = 0.30
QRS_SMOOTH_MS      = 60.0


def _estimate_qrs_duration(signal: np.ndarray, r_peaks: np.ndarray,
                            fs: int, frac: float = QRS_WIDTH_FRACTION,
                            smooth_ms: float = QRS_SMOOTH_MS,
                            bp: np.ndarray = None) -> tuple:
    """
    Estimate QRS duration (ms) per beat as onset-to-offset width.

    Method: build a smooth per-beat energy hump (bandpass -> square -> moving
    window integration), find its peak near each R-peak, then walk outward until
    the hump drops below `frac` of that peak. The contiguous onset->offset span
    is the QRS width. Walking from the peak measures the single QRS deflection,
    not scattered neighbouring energy — the old squared-signal-at-10%-over-the-
    whole-window method overread width (every record came out >=115 ms).

    Returns
    -------
    mean_ms, std_ms : float, float
    """
    if bp is None:
        bp = _bandpass_filter(signal, fs)
    squared  = bp ** 2
    env      = _moving_window_integration(squared, fs, window_ms=smooth_ms)
    durations = []
    search_s  = int(fs * 0.14)          # +/-140 ms search window
    local_s   = max(1, int(fs * 0.03))  # +/-30 ms to locate the true peak

    for pk in r_peaks:
        lo = max(0, pk - search_s)
        hi = min(len(env), pk + search_s + 1)
        seg = env[lo:hi]
        center = pk - lo
        # Locate the envelope peak within +/-30 ms of the R-peak
        c0 = max(0, center - local_s)
        c1 = min(len(seg), center + local_s + 1)
        if c1 <= c0:
            continue
        cmax = c0 + int(np.argmax(seg[c0:c1]))
        peak_val = seg[cmax]
        if peak_val <= 0:
            continue
        thr = frac * peak_val

        # Walk left/right from the peak to the onset/offset crossings
        left = cmax
        while left > 0 and seg[left] >= thr:
            left -= 1
        right = cmax
        while right < len(seg) - 1 and seg[right] >= thr:
            right += 1

        width_ms = (right - left) / fs * 1000.0
        if 40 <= width_ms <= 200:       # physiological sanity bound
            durations.append(width_ms)

    if len(durations) == 0:
        return np.nan, np.nan
    return float(np.mean(durations)), float(np.std(durations))


def _estimate_p_wave_ratio(signal: np.ndarray, r_peaks: np.ndarray,
                            fs: int) -> float:
    """
    Estimate the fraction of beats that have a detectable P-wave.

    P-wave appears 100-200 ms before the R-peak. We count a P-wave
    present if there is a local maximum in that pre-R window with
    amplitude > 10% of the QRS peak.

    A low ratio (< 0.6) suggests AFib where P-waves are absent/disorganised.
    """
    count_with_p = 0
    valid_beats  = 0

    for pk in r_peaks:
        p_start = pk - int(0.20 * fs)   # 200 ms before R
        p_end   = pk - int(0.10 * fs)   # 100 ms before R
        if p_start < 0:
            continue
        valid_beats += 1
        p_seg   = signal[p_start:p_end]
        qrs_amp = abs(signal[pk])
        if qrs_amp == 0:
            continue
        # Use absolute value: P-waves can be positive or negative depending on lead
        # Threshold: 25% of QRS amplitude (clinical standard: P-waves are 20-30% of R amplitude)
        if np.max(np.abs(p_seg)) > 0.25 * qrs_amp:
            count_with_p += 1

    if valid_beats == 0:
        return np.nan   # signal measurement failure; filtered downstream
    return count_with_p / valid_beats


def _estimate_st_elevation(signal: np.ndarray, r_peaks: np.ndarray,
                             fs: int) -> float:
    """
    Estimate mean ST-segment deviation in microvolts (uV).

    The ST segment is measured 60-100 ms after the R-peak relative to
    the isoelectric baseline (PR segment, 200-160 ms before R-peak).

    MIT-BIH signal is in mV; multiply by 1000 to get uV.
    ST elevation > 200 uV may indicate myocardial infarction (MI).
    """
    st_deviations = []

    for pk in r_peaks:
        # Baseline: 200-160 ms before R (PR segment)
        bl_start = pk - int(0.20 * fs)
        bl_end   = pk - int(0.16 * fs)
        # ST measurement: 60-100 ms after R (J-point + 40 ms)
        st_start = pk + int(0.06 * fs)
        st_end   = pk + int(0.10 * fs)

        if bl_start < 0 or st_end > len(signal):
            continue

        baseline_mv = signal[bl_start:bl_end].mean()
        st_mv       = signal[st_start:st_end].mean()
        st_deviations.append((st_mv - baseline_mv) * 1000.0)   # mV -> uV

    if len(st_deviations) == 0:
        return np.nan   # measurement failure — do NOT report as isoelectric (0.0),
                        # which would mask a real ST-elevation MI warning downstream
    return float(np.mean(st_deviations))


def extract_features(signal: np.ndarray, r_peaks: np.ndarray,
                     fs: int, bp: np.ndarray = None) -> dict:
    """
    Extract 11 clinical ECG features from a signal and its R-peak indices.

    Parameters
    ----------
    signal  : 1-D ndarray, ECG in mV
    r_peaks : 1-D int array, R-peak sample indices
    fs      : sampling rate (Hz)

    Returns
    -------
    features : dict with keys:
        hr_mean       - mean heart rate (bpm)
        hr_std        - std of instantaneous heart rate (bpm)
        rr_cv         - coefficient of variation of RR intervals (0-1)
        rr_entropy    - sample entropy of RR intervals (0-1, high = irregular)
        sdnn          - std dev of RR intervals (ms)  [HRV time-domain]
        rmssd         - root-mean-square of successive RR diffs (ms)
        pnn50         - fraction of successive diffs > 50 ms (%)
        qrs_dur_mean  - mean QRS duration (ms)
        qrs_dur_std   - std  QRS duration (ms)
        p_wave_ratio  - fraction of beats with detectable P-wave (0-1)
        st_elevation  - mean ST deviation (uV)
    """
    rr = _compute_rr_intervals(r_peaks, fs)

    nan_features = {k: np.nan for k in FEATURE_KEYS}

    if len(rr) < 2:
        return nan_features

    # Heart-rate features
    hr_inst = 60_000.0 / rr          # instantaneous bpm for each RR interval
    # Rate from the MEDIAN RR interval, not mean(60000/RR). The arithmetic mean
    # of instantaneous rates is biased high (Jensen's inequality: 1/RR is convex),
    # which pushed borderline-slow records into the Normal rate band. The median
    # RR gives an unbiased central rate. hr_std stays the spread of instantaneous
    # rates as a variability measure.
    hr_mean = float(60_000.0 / np.median(rr))
    hr_std  = float(np.std(hr_inst, ddof=1))

    # RR variability features
    rr_mean    = float(np.mean(rr))
    rr_cv      = float(np.std(rr, ddof=1) / rr_mean) if rr_mean > 0 else 0.0
    rr_entropy = _sample_entropy(rr)

    # SDNN: gold-standard HRV measure (< 50 ms = low HRV, > 100 ms = high HRV)
    sdnn = float(np.std(rr, ddof=1))   # ms

    # RMSSD: reflects parasympathetic activity (short-term HRV)
    succ_diff = np.diff(rr)
    rmssd     = float(np.sqrt(np.mean(succ_diff ** 2))) if len(succ_diff) else np.nan

    # pNN50: percentage of successive RR differences > 50 ms
    pnn50 = float(np.mean(np.abs(succ_diff) > 50.0) * 100.0) if len(succ_diff) else np.nan

    # Morphological features (reuse a precomputed bandpass if the caller passed one)
    qrs_mean, qrs_std = _estimate_qrs_duration(signal, r_peaks, fs, bp=bp)
    p_ratio           = _estimate_p_wave_ratio(signal, r_peaks, fs)
    st_elev           = _estimate_st_elevation(signal, r_peaks, fs)

    return {
        "hr_mean"      : hr_mean,
        "hr_std"       : hr_std,
        "rr_cv"        : rr_cv,
        "rr_entropy"   : rr_entropy,
        "sdnn"         : sdnn,
        "rmssd"        : rmssd,
        "pnn50"        : pnn50,
        "qrs_dur_mean" : qrs_mean,
        "qrs_dur_std"  : qrs_std,
        "p_wave_ratio" : p_ratio,
        "st_elevation" : st_elev,
    }


# =============================================================================
# 4. VALIDATION & TESTING
# =============================================================================

# Expected normal ranges for sanity-checking features
FEATURE_RANGES = {
    "hr_mean"      : (20,  250),    # bpm
    "hr_std"       : (0,   60),     # bpm
    "rr_cv"        : (0,   1.0),
    "rr_entropy"   : (0,   1.0),
    "sdnn"         : (0,   500),    # ms
    "rmssd"        : (0,   500),    # ms
    "pnn50"        : (0,   100),    # %
    "qrs_dur_mean" : (40,  250),    # ms
    "qrs_dur_std"  : (0,   100),    # ms
    "p_wave_ratio" : (0,   1.0),
    "st_elevation" : (-1000, 1000), # uV
}


def validate_features(features: dict, record_id) -> None:
    """
    Print feature values and flag any NaN or out-of-bounds values.
    """
    print(f"\n  Features for record {record_id}:")
    print(f"  {'Feature':<18} {'Value':>12}  Status")
    print(f"  {'-'*48}")
    all_ok = True
    for name, val in features.items():
        lo, hi = FEATURE_RANGES.get(name, (-np.inf, np.inf))
        if val is None or (isinstance(val, float) and np.isnan(val)):
            status = "WARN: NaN"
            all_ok = False
        elif not (lo <= val <= hi):
            status = f"WARN: OOB [{lo}, {hi}]"
            all_ok = False
        else:
            status = "OK"
        unit_map = {
            "hr_mean":"bpm","hr_std":"bpm","sdnn":"ms","rmssd":"ms",
            "pnn50":"%","qrs_dur_mean":"ms","qrs_dur_std":"ms","st_elevation":"uV",
        }
        unit = unit_map.get(name, "")
        print(f"  {name:<18} {val:>10.3f}  {unit:<4}  {status}")
    print(f"  {'-'*48}")
    print(f"  -> {'All features valid' if all_ok else 'ISSUES FOUND -- check above'}")


def run_phase1():
    """
    Full Phase 1 test:
      1. Load record 100; print info; detect R-peaks; plot ECG.
      2. Extract and validate features for records 100, 103, 105, 114, 213.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  PHASE 1 -- Data Loading & Feature Extraction")
    print("="*60)

    # 1a. Load record 100
    print("\n[1] Loading MIT-BIH record 100 ...")
    signal, ann, fs = load_mitbih_record(100, duration_s=30.0)
    print_record_info(100, signal, ann, fs)

    # 1b. Pan & Tompkins R-peak detection
    print("[2] Running Pan & Tompkins R-peak detector ...")
    r_peaks = detect_r_peaks(signal, fs)
    n_annotated = sum(1 for s in ann.symbol if s not in ('+','~','|','!'))
    print(f"  Detected   : {len(r_peaks)} R-peaks")
    print(f"  Annotated  : {n_annotated} beats")
    # Raw count ratio only — NOT beat-to-beat matched sensitivity/PPV. A ratio
    # near 1.0 means similar counts, not that the same beats were found.
    ratio = len(r_peaks) / n_annotated if n_annotated else float("nan")
    print(f"  Count ratio (detected/annotated, not beat-matched): {ratio:.2f}")

    assert len(r_peaks) >= 30, (
        f"Expected >=30 R-peaks in 30 s, got {len(r_peaks)}. "
        "Check bandpass or threshold."
    )
    print(f"  [PASS] >=30 beats detected  ({len(r_peaks)} found)")

    # 1c. Plot ECG with R-peaks
    plot_ecg_with_rpeaks(signal, r_peaks, fs, record_id=100,
                         save_path=os.path.join(PLOTS_DIR, "ecg_record100_rpeaks.png"))

    # 1d. Feature extraction on 5 records
    test_records = [100, 103, 105, 114, 213]
    print(f"\n[3] Extracting features from records: {test_records}")

    all_features = {}
    for rid in test_records:
        print(f"\n  -- Record {rid} --")
        sig, ann_r, fs_r = load_mitbih_record(rid, duration_s=30.0)
        peaks             = detect_r_peaks(sig, fs_r)
        print(f"  Detected {len(peaks)} R-peaks")
        feats             = extract_features(sig, peaks, fs_r)
        validate_features(feats, rid)
        all_features[rid] = feats

    # 1e. Summary plot: features across records
    _plot_feature_summary(all_features, test_records)

    print("\n" + "="*60)
    print("  PHASE 1 COMPLETE")
    print("="*60)

    return all_features


def _plot_feature_summary(all_features: dict, record_ids: list) -> None:
    """Bar chart of all 11 features across the 5 test records."""
    feat_names = list(next(iter(all_features.values())).keys())

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

    for i, fname in enumerate(feat_names):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        vals = [all_features[rid].get(fname, np.nan) for rid in record_ids]
        colors = ["steelblue" if not np.isnan(v) else "red" for v in vals]
        ax.bar([str(r) for r in record_ids], vals, color=colors,
               edgecolor="k", linewidth=0.5)
        ax.set_title(fname, fontsize=9, fontweight="bold")
        ax.set_xlabel("Record", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    # hide any unused subplot slots
    total_slots = 4 * 3
    for j in range(len(feat_names), total_slots):
        fig.add_subplot(gs[j // 3, j % 3]).set_visible(False)

    fig.suptitle("Phase 1 -- Feature Summary Across 5 MIT-BIH Records",
                 fontsize=12, fontweight="bold")
    path = os.path.join(PLOTS_DIR, "ecg_phase1_features.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  [Plot] Feature summary saved -> {path}")


# =============================================================================
if __name__ == "__main__":
    run_phase1()
