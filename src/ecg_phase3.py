# -*- coding: utf-8 -*-
"""
ECG Monitor - Phase 3: Random Forest Training & Validation
===========================================================
Builds a labelled dataset from the MIT-BIH database using the feature
extractor from Phase 1, then trains and evaluates a Random Forest
classifier for 5-class arrhythmia detection.

Class labels
------------
  0  Normal Sinus Rhythm
  1  Atrial Fibrillation
  2  Premature Ventricular Contractions (PVC)
  3  Bradycardia
  4  Sinus Tachycardia

Targets (from FDA 510(k) requirements)
---------------------------------------
  Accuracy   >= 85 %
  Sensitivity >= 85 %  (macro-averaged recall)
  Specificity >= 90 %
  ROC-AUC    >= 0.90
"""

import os
import sys
import warnings
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit, GridSearchCV, train_test_split
)
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.preprocessing import label_binarize
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(_HERE)
DB_PATH     = os.path.join(ROOT, "data", "mit-bih-arrhythmia-database-1.0.0",
                            "mit-bih-arrhythmia-database-1.0.0")
MODEL_PATH  = os.path.join(ROOT, "models", "ecg_model.joblib")
PLOTS_DIR   = os.path.join(ROOT, "outputs", "plots")
REPORTS_DIR = os.path.join(ROOT, "outputs", "reports")

# Add src/ to path so we can import Phase 1/2 helpers
sys.path.insert(0, _HERE)
from ecg_phase1 import (
    FS, FEATURE_KEYS, load_mitbih_record, detect_r_peaks, extract_features,
    _mlii_channel,
)
from ecg_phase2 import classify_ecg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES      = ["Normal", "AFib", "PVC", "Bradycardia", "Tachycardia"]
N_CLASSES    = len(CLASSES)
# Classes the Random Forest is trained to predict. Bradycardia (class 3) and
# Tachycardia (class 4) are rate-defined and, in MIT-BIH, come from a single
# record (232) and 3 records / 5 windows respectively — unlearnable under
# record-grouped CV. They are owned by the deterministic Phase 2 rules instead
# (HR<50 / HR>120), which run before the ML fallback in predict_combined(). The
# model therefore trains only on Normal/AFib/PVC; predict_proba is still expanded
# back to all N_CLASSES columns (3/4 stay 0 from ML, filled by the rules).
ML_CLASSES   = (0, 1, 2)   # Normal, AFib, PVC
WINDOW_S     = 30.0   # seconds per feature window
STEP_S       = 15.0   # stride (50 % overlap -> more samples)
RANDOM_STATE = 42

# MIT-BIH records available in the local database (48 records: 100–124, 200–234)
ALL_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119,
    121, 122, 123, 124,
    200, 201, 202, 203, 205, 207, 208, 209, 210,
    212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
    228, 230, 231, 232, 233, 234,
]


# =============================================================================
# 1. ANNOTATION-BASED LABELLING
# =============================================================================

# Beat symbols that count as PVC (ventricular ectopic) events. PVC is a
# BEAT-level phenomenon annotated per-beat by the cardiologists, so it is
# labelled from beat symbols — this is genuine ground truth, not circular.
PVC_SYMS    = {"V", "r", "E", "F"}
# Symbols that are not real beats (rhythm/noise markers)
NON_BEAT    = {"+", "~", "|", "!", "[", "]", "x", "(", ")", "p", "t", "u",
               "`", "'", "^", "Q", "?"}

# MIT-BIH rhythm annotations (aux_note at '+' markers) -> our 5 class indices.
# This is the actual clinical ground truth. Only rhythms that map cleanly to a
# target class are kept; everything else (AFL, VT, VFL, paced, nodal, AV block,
# pre-excitation, atrial bigeminy, ...) makes the window ambiguous -> skipped.
# NOTE: MIT-BIH '(B' = ventricular BIGEMINY and '(T' = ventricular TRIGEMINY —
# these are PVC rhythms, NOT brady/tachy. Bradycardia is '(SBR', tachycardia is
# '(SVTA'. (This corrects a common misreading of the rhythm codes.)
RHYTHM_TO_CLASS = {
    "(N":    0,   # normal sinus rhythm            -> Normal
    "(AFIB": 1,   # atrial fibrillation            -> AFib
    "(B":    2,   # ventricular bigeminy           -> PVC
    "(T":    2,   # ventricular trigeminy          -> PVC
    "(SBR":  3,   # sinus bradycardia              -> Bradycardia
    "(SVTA": 4,   # supraventricular tachyarrhythmia -> Tachycardia
}

# A window must be dominated by one rhythm for its rhythm label to be trusted.
RHYTHM_DOMINANCE = 0.80
# Fraction of ventricular-ectopic beats above which a window is labelled PVC.
PVC_FRACTION = 0.15


def label_window(beat_symbols: list, dom_rhythm, dom_frac: float) -> int | None:
    """
    Assign one of the 5 class labels (or None to skip) to a 30-second window,
    using the cardiologist annotations as ground truth.

    Ground-truth sources (NOT the model's own features -- avoids circularity):
      - PVC (2): fraction of ventricular-ectopic BEAT symbols >= PVC_FRACTION.
      - Normal/AFib/Brady/Tachy: the dominant RHYTHM annotation (aux_note),
        via RHYTHM_TO_CLASS, when it covers >= RHYTHM_DOMINANCE of the beats.

    PVC beats override only Normal/PVC/unlabelled rhythm windows, so frequent
    isolated PVCs in sinus rhythm are caught, without stealing AFib/Brady/Tachy
    windows that happen to contain a few ectopics.

    Returns None (skip) for short, mixed-rhythm, or unmapped windows.
    """
    beats   = [s for s in beat_symbols if s not in NON_BEAT]
    n_beats = len(beats)
    if n_beats < 5:          # too few beats to characterise
        return None

    pvc_frac = sum(1 for s in beats if s in PVC_SYMS) / n_beats

    # PVC from beat annotations (ground truth), but don't override a clearly
    # non-Normal atrial/rate rhythm.
    if pvc_frac >= PVC_FRACTION and dom_rhythm in (None, 0, 2):
        return 2

    # Otherwise defer to the annotated rhythm, if one clearly dominates.
    if dom_rhythm is None or dom_frac < RHYTHM_DOMINANCE:
        return None
    return dom_rhythm


def _rhythm_events(ann) -> list:
    """
    Extract (sample, class_or_None) rhythm-change events from an annotation.

    Rhythm changes are '+' beat markers whose aux_note holds a code like '(AFIB.
    Codes outside RHYTHM_TO_CLASS map to None (an explicit 'unknown rhythm from
    here' marker, so a window in an unmapped rhythm is skipped rather than
    inheriting the previous rhythm).
    """
    events = []
    aux = getattr(ann, "aux_note", None)
    if aux is None:
        return events
    for i in range(len(ann.sample)):
        note = (aux[i] or "").replace("\x00", "").strip()
        if note.startswith("("):
            events.append((int(ann.sample[i]), RHYTHM_TO_CLASS.get(note, None)))
    return events


def _dominant_rhythm(events: list, beat_samples: list):
    """
    Given sorted rhythm-change events and the beat sample indices in a window,
    return (dominant_class_or_None, dominant_fraction).

    Each beat inherits the rhythm of the most recent event at/before it; the
    dominant class is the most common across the window's beats.
    """
    if not beat_samples:
        return None, 0.0
    counts = {}
    for s in beat_samples:
        cls = None
        for ev_sample, ev_cls in events:
            if ev_sample <= s:
                cls = ev_cls
            else:
                break
        counts[cls] = counts.get(cls, 0) + 1
    dom_cls = max(counts, key=counts.get)
    return dom_cls, counts[dom_cls] / len(beat_samples)


# =============================================================================
# 2. DATASET BUILDER
# =============================================================================

def build_dataset(verbose: bool = True) -> tuple:
    """
    Iterate over all MIT-BIH records, extract features from sliding windows,
    and label each window.

    Returns
    -------
    X : ndarray, shape (n_samples, 11)
    y : ndarray, shape (n_samples,)
    record_ids : ndarray, shape (n_samples,)  [for record-level stratification]
    """
    import wfdb

    X_rows      = []
    y_labels    = []
    record_ids  = []

    for rec_id in ALL_RECORDS:
        rec_path = os.path.join(DB_PATH, str(rec_id))

        # Check the record exists
        if not os.path.isfile(rec_path + ".hea"):
            continue

        try:
            record = wfdb.rdrecord(rec_path)
            ann    = wfdb.rdann(rec_path, "atr")
        except Exception as e:
            if verbose:
                print(f"  [SKIP] Record {rec_id}: {e}")
            continue

        # Use the MLII lead by name for a consistent morphology across records.
        # Records without MLII (102, 104: V5/V2) are skipped so the training set
        # is single-lead-consistent.
        if "MLII" not in record.sig_name:
            if verbose:
                print(f"  [SKIP] Record {rec_id}: no MLII lead "
                      f"({record.sig_name})")
            continue
        sig_full = record.p_signal[:, _mlii_channel(record.sig_name)].astype(np.float64)
        n_total  = len(sig_full)
        duration = n_total / FS

        if duration < WINDOW_S:
            continue   # recording too short

        # Ground-truth rhythm-change events for this record (carried forward).
        rhythm_events = _rhythm_events(ann)

        # Sliding windows
        win_samples  = int(WINDOW_S * FS)
        step_samples = int(STEP_S   * FS)

        for start in range(0, n_total - win_samples + 1, step_samples):
            end    = start + win_samples
            window = sig_full[start:end]

            # Annotations that fall within this window
            idxs      = [i for i in range(len(ann.sample))
                         if start <= ann.sample[i] < end]
            w_symbols = [ann.symbol[i] for i in idxs]
            beat_samples = [int(ann.sample[i]) for i in idxs
                            if ann.symbol[i] not in NON_BEAT]

            # Detect R-peaks and extract features
            try:
                r_peaks = detect_r_peaks(window, FS)
                feats   = extract_features(window, r_peaks, FS)
            except Exception:
                continue

            # Skip windows with NaN in critical features
            if any(np.isnan(feats.get(k, np.nan))
                   for k in ["hr_mean", "rr_cv", "rr_entropy"]):
                continue

            # Label from cardiologist annotations (rhythm + PVC beats)
            dom_rhythm, dom_frac = _dominant_rhythm(rhythm_events, beat_samples)
            label = label_window(w_symbols, dom_rhythm, dom_frac)
            if label is None:
                continue

            row = [feats.get(k, np.nan) for k in FEATURE_KEYS]
            # Replace remaining NaN with column medians later; skip NaN rows here
            if any(np.isnan(v) for v in row):
                continue

            X_rows.append(row)
            y_labels.append(label)
            record_ids.append(rec_id)  # track source record for stratification

        if verbose:
            class_counts = {c: y_labels.count(i)
                            for i, c in enumerate(CLASSES)}
            print(f"  Record {rec_id:4d}: total windows so far = {len(X_rows)}")

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_labels, dtype=int)
    record_ids = np.array(record_ids, dtype=int)

    if verbose:
        print(f"\n  Dataset shape  : {X.shape}")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    Class {u} ({CLASSES[u]:<15}): {c} samples")

    return X, y, record_ids


# =============================================================================
# 3. HYPERPARAMETER SEARCH + TRAINING
# =============================================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray,
                verbose: bool = True) -> RandomForestClassifier:
    """
    Grid-search over RF hyperparameters using stratified 5-fold CV,
    optimising for weighted F1.

    Returns the best fitted estimator.
    """
    param_grid = {
        "n_estimators"    : [50, 100, 200],
        "max_depth"       : [10, 15, 20],
        "min_samples_split": [5, 10, 15],
    }

    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if verbose:
        print("\n  Running GridSearchCV (stratified 5-fold, scoring=f1_weighted) ...")
        t0 = time.time()

    gs = GridSearchCV(
        base_rf,
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    if verbose:
        elapsed = time.time() - t0
        print(f"  Grid search done in {elapsed:.1f} s")
        print(f"  Best params  : {gs.best_params_}")
        print(f"  Best CV F1   : {gs.best_score_:.4f}")

    return gs.best_estimator_


# =============================================================================
# 4. EVALUATION
# =============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   save_plots: bool = True) -> dict:
    """
    Compute per-class and aggregate metrics; generate ROC curves.

    Returns a results dict with all metrics.
    """
    y_pred  = model.predict(X_test)
    _raw_proba = model.predict_proba(X_test)   # (n_samples, len(model.classes_))
    # Expand to full N_CLASSES columns (some classes may be absent from training data)
    y_proba = np.zeros((_raw_proba.shape[0], N_CLASSES))
    for col_idx, cls in enumerate(model.classes_):
        y_proba[:, cls] = _raw_proba[:, col_idx]

    # ---- Accuracy --------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)

    # ---- Confusion matrix ------------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=list(range(N_CLASSES)))

    # ---- Per-class sensitivity, specificity, precision, F1 ---------------
    per_class = {}
    for c in range(N_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = (2 * sens * prec / (sens + prec)
                if (sens + prec) > 0 else 0.0)
        per_class[CLASSES[c]] = {
            "sensitivity": sens,
            "specificity": spec,
            "precision"  : prec,
            "f1"         : f1,
            "support"    : int(cm[c, :].sum()),
        }

    macro_sens = np.mean([v["sensitivity"] for v in per_class.values()])
    macro_spec = np.mean([v["specificity"] for v in per_class.values()])

    # ---- ROC-AUC (one-vs-rest) -------------------------------------------
    y_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
    present_classes = np.unique(y_test)
    if len(present_classes) >= 2:
        try:
            roc_auc = roc_auc_score(
                y_bin[:, present_classes],
                y_proba[:, present_classes],
                average="macro",
                multi_class="ovr",
            )
        except Exception:
            roc_auc = float("nan")
    else:
        roc_auc = float("nan")

    # ---- Feature importance ----------------------------------------------
    importances = dict(zip(FEATURE_KEYS, model.feature_importances_))

    # ---- Print report ----------------------------------------------------
    print(f"\n  {'='*62}")
    print(f"  EVALUATION REPORT")
    print(f"  {'='*62}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  Macro sens  : {macro_sens*100:.2f}%")
    print(f"  Macro spec  : {macro_spec*100:.2f}%")
    print(f"  ROC-AUC     : {roc_auc:.4f}")

    print(f"\n  {'Class':<22} {'Sens':>6} {'Spec':>6} {'Prec':>6} "
          f"{'F1':>6} {'Support':>8}")
    print(f"  {'-'*58}")
    for name, m in per_class.items():
        print(f"  {name:<22} {m['sensitivity']*100:6.1f} "
              f"{m['specificity']*100:6.1f} {m['precision']*100:6.1f} "
              f"{m['f1']*100:6.1f} {m['support']:8d}")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "  {:>12}".format("") + "".join(
        f"  {c[:6]:>6}" for c in CLASSES
    )
    print(header)
    for i, row_name in enumerate(CLASSES):
        row_str = f"  {row_name[:12]:>12}" + "".join(
            f"  {cm[i,j]:6d}" for j in range(N_CLASSES)
        )
        print(row_str)

    print(f"\n  Feature importances (top 5):")
    sorted_fi = sorted(importances.items(), key=lambda x: -x[1])
    for fname, imp in sorted_fi[:5]:
        bar = "=" * int(imp * 40)
        print(f"    {fname:<18} {imp:.4f}  {bar}")

    # ---- Performance targets check ---------------------------------------
    print(f"\n  {'='*62}")
    print(f"  PERFORMANCE TARGETS")
    print(f"  {'='*62}")
    _check("Accuracy   >= 85 %", acc * 100,   85.0)
    _check("Macro Sens >= 85 %", macro_sens * 100, 85.0)
    _check("Macro Spec >= 90 %", macro_spec * 100, 90.0)
    _check("ROC-AUC    >= 0.90", roc_auc,     0.90)

    # ---- Plots -----------------------------------------------------------
    if save_plots:
        _plot_confusion_matrix(cm)
        _plot_roc_curves(y_bin, y_proba)
        _plot_feature_importance(importances)

    return {
        "accuracy"   : acc,
        "macro_sens" : macro_sens,
        "macro_spec" : macro_spec,
        "roc_auc"    : roc_auc,
        "per_class"  : per_class,
        "confusion_matrix": cm.tolist(),
        "feature_importances": importances,
    }


def _check(label: str, value: float, threshold: float) -> None:
    status = "PASS" if value >= threshold else "FAIL"
    print(f"  [{status}] {label:<28}  got {value:.4f}")


def _plot_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ecg_phase3_confusion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  [Plot] Confusion matrix saved -> {path}")


def _plot_roc_curves(y_bin: np.ndarray, y_proba: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["steelblue", "darkorange", "forestgreen", "crimson", "purple"]
    for c in range(N_CLASSES):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
        try:
            auc_val = roc_auc_score(y_bin[:, c], y_proba[:, c])
        except Exception:
            auc_val = float("nan")
        ax.plot(fpr, tpr, color=colors[c], lw=1.8,
                label=f"{CLASSES[c]} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves (one-vs-rest)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ecg_phase3_roc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] ROC curves saved -> {path}")


def _plot_feature_importance(importances: dict) -> None:
    names = list(importances.keys())
    vals  = list(importances.values())
    order = np.argsort(vals)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([names[i] for i in order], [vals[i] for i in order],
           color="steelblue", edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Importance", fontsize=11)
    ax.set_title("Random Forest Feature Importance",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels([names[i] for i in order],
                       rotation=35, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ecg_phase3_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Feature importance saved -> {path}")


# =============================================================================
# 5. MAIN
# =============================================================================

def run_phase3(skip_build: bool = False,
               dataset_cache: str = "ecg_dataset_cache.npz") -> None:
    """
    Full Phase 3 pipeline:
      1. Build labelled dataset from MIT-BIH (or load cached)
      2. Train / Val / Test split  (70 / 15 / 15)
      3. GridSearchCV hyperparameter tuning on train
      4. Evaluate on validation set
      5. Final evaluation on held-out test set
      6. Save model to ecg_model.joblib
    """
    os.makedirs(PLOTS_DIR,   exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

    print("\n" + "="*65)
    print("  PHASE 3 -- Random Forest Training & Validation")
    print("="*65)

    cache_path = os.path.join(ROOT, "data", dataset_cache)

    # ---- Step 1: dataset ------------------------------------------------
    if skip_build and os.path.isfile(cache_path):
        print(f"\n[1] Loading cached dataset from {cache_path} ...")
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
        record_ids = data.get("record_ids", None)  # backward compatibility
        print(f"  Loaded X={X.shape}, y={y.shape}")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    Class {u} ({CLASSES[u]:<15}): {c} samples")
    else:
        print("\n[1] Building dataset from MIT-BIH records ...")
        X, y, record_ids = build_dataset(verbose=True)
        np.savez(cache_path, X=X, y=y, record_ids=record_ids)
        print(f"  Dataset cached -> {cache_path}")

    assert len(X) >= 50, (
        f"Too few samples ({len(X)}) — check DB_PATH: {DB_PATH}"
    )

    # ---- Restrict ML to the learnable classes ---------------------------
    # The full dataset (all 5 classes) stays cached for Phase 5's hybrid
    # evaluation. The RF, however, only trains/evaluates on Normal/AFib/PVC;
    # Bradycardia/Tachycardia are owned by the Phase 2 rules (see ML_CLASSES).
    ml_mask = np.isin(y, ML_CLASSES)
    X_ml, y_ml, record_ids_ml = X[ml_mask], y[ml_mask], record_ids[ml_mask]
    print(f"\n  ML training set (classes {ML_CLASSES}): {len(X_ml)} / {len(X)} windows "
          f"({int(ml_mask.sum())} kept; {int((~ml_mask).sum())} rate-class windows "
          f"reserved for the rules)")

    # ---- Step 2: splits -------------------------------------------------
    print("\n[2] Splitting ML dataset at RECORD level (70 train / 15 val / 15 test) ...")

    # CRITICAL: Split at record level, not window level, to prevent data leakage
    # (adjacent windows from same record must not appear in both train and test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss.split(X_ml, y_ml, groups=record_ids_ml))

    X_trainval, X_test = X_ml[trainval_idx], X_ml[test_idx]
    y_trainval, y_test = y_ml[trainval_idx], y_ml[test_idx]
    record_ids_trainval = record_ids_ml[trainval_idx]

    # Then split train/val from remaining (15 % of trainval ≈ 17.6% of total)
    val_frac = 0.15 / 0.85
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups=record_ids_trainval))

    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
    record_ids_train = record_ids_trainval[train_idx]

    print(f"  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)}   samples")
    print(f"  Test  : {len(X_test)}  samples")

    # ---- Step 3: train --------------------------------------------------
    print("\n[3] Training Random Forest ...")
    model = train_model(X_train, y_train, verbose=True)

    # ---- Step 4: validation set -----------------------------------------
    print("\n[4] Validation-set evaluation:")
    val_results = evaluate_model(model, X_val, y_val, save_plots=False)

    # ---- Step 5: test set -----------------------------------------------
    print("\n[5] Held-out test-set evaluation (final):")
    test_results = evaluate_model(model, X_test, y_test, save_plots=True)

    # ---- Step 6: refit on ALL ML data and save the deployed model -------
    # The split above exists only for honest evaluation. The DEPLOYED model is
    # refit on the full ML dataset (train+val+test, Normal/AFib/PVC) using the
    # tuned hyperparameters, so it uses every learnable window. Rate classes are
    # handled by the rules. Phase 5 estimates the hybrid's generalisation via
    # record-level cross-validation.
    deployed = clone(model).fit(X_ml, y_ml)
    joblib.dump(deployed, MODEL_PATH)
    print(f"\n[6] Deployed model refit on full ML dataset ({len(X_ml)} samples, "
          f"classes {ML_CLASSES}) -> {MODEL_PATH}")

    # ---- Save metrics JSON ----------------------------------------------
    metrics = {
        "_note": (
            "Development metrics from a SINGLE record-level hold-out split. "
            "These are NOT the reported performance figures — a small/imbalanced "
            "test split can make them optimistic. The reported, leakage-free "
            "numbers come from Phase 5 record-level cross-validation "
            "(outputs/reports/ecg_clinical_validation_report.*)."
        ),
        "validation": {k: (float(v) if isinstance(v, (np.floating, float))
                           else v)
                       for k, v in val_results.items()
                       if k not in ("confusion_matrix", "per_class",
                                    "feature_importances")},
        "test": {k: (float(v) if isinstance(v, (np.floating, float))
                     else v)
                 for k, v in test_results.items()
                 if k not in ("confusion_matrix", "per_class",
                              "feature_importances")},
    }
    metrics_path = os.path.join(REPORTS_DIR, "ecg_phase3_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved -> {metrics_path}")

    # ---- Safety assertion: VT must not be classified as Normal ----------
    # Checks the actual deployed decision path, not the bare RF in isolation.
    # ecg_analyzer.predict_combined() always runs Phase 2's rules first and
    # only falls back to this model when rules are INCONCLUSIVE -- so the
    # real safety guarantee is "rules catch it, OR (if they don't) the model
    # doesn't call it Normal". Testing the raw model alone against a vector
    # rules already catch deterministically is a false alarm, not a real gap.
    print("\n[7] Clinical safety assertion ...")
    vt_features = {
        "hr_mean": 165.0, "hr_std": 4.0, "rr_cv": 0.05, "rr_entropy": 0.20,
        "sdnn": 10.0, "rmssd": 8.0, "pnn50": 0.0,
        "qrs_dur_mean": 155.0, "qrs_dur_std": 10.0,
        "p_wave_ratio": 0.25, "st_elevation": 60.0,
    }
    rule_result = classify_ecg(vt_features)
    if rule_result["diagnosis"] == "INCONCLUSIVE":
        vt_vec = np.array([[vt_features[k] for k in FEATURE_KEYS]])
        vt_pred = deployed.predict(vt_vec)[0]
        assert vt_pred != 0, (
            f"SAFETY FAIL: rules were INCONCLUSIVE on VT features and the "
            f"ML fallback predicted Normal (class 0)! "
            f"Predicted class: {vt_pred} ({CLASSES[vt_pred]})"
        )
        print(f"  [PASS] Rules INCONCLUSIVE; ML fallback not Normal "
              f"(predicted: {CLASSES[vt_pred]})")
    else:
        assert rule_result["diagnosis"] != "Normal Sinus Rhythm", (
            f"SAFETY FAIL: VT features classified as Normal Sinus Rhythm "
            f"by clinical rules!"
        )
        print(f"  [PASS] VT features caught by clinical rules "
              f"(diagnosis: {rule_result['diagnosis']}) -- ML never consulted")

    print("\n" + "="*65)
    print("  PHASE 3 COMPLETE")
    print("="*65)


# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ECG Phase 3: Train RF model")
    parser.add_argument("--skip-build", action="store_true",
                        help="Load cached dataset instead of rebuilding")
    args = parser.parse_args()
    run_phase3(skip_build=args.skip_build)
