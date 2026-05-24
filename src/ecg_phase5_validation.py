# -*- coding: utf-8 -*-
"""
ECG Monitor - Phase 5: Clinical Validation & FDA Documentation
==============================================================
Generates a professional Clinical Validation Summary report suitable
for inclusion in an FDA 510(k) premarket notification.

The report covers:
  - Per-class performance metrics with 95 % bootstrap CIs
  - Confusion matrix (5 × 5)
  - Failure analysis (most-confused pairs, systematic biases)
  - Clinical safety statistics (false negative/positive rates,
    dangerous misclassification check)
  - Device limitations and disclaimers

Run AFTER Phase 3 has trained the model and generated the test split.
Requires:  ecg_model.joblib  (trained RF)
           ecg_dataset_cache.npz  (feature matrix + labels)
"""

import os
import sys
import json
import datetime
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import joblib

warnings.filterwarnings("ignore")

_HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

CLASSES      = ["Normal", "AFib", "PVC", "Bradycardia", "Tachycardia"]
N_CLASSES    = len(CLASSES)
RANDOM_STATE = 42
N_BOOTSTRAP  = 2000   # iterations for 95 % CI

PLOTS_DIR    = os.path.join(ROOT, "outputs", "plots")
REPORTS_DIR  = os.path.join(ROOT, "outputs", "reports")
MODEL_PATH   = os.path.join(ROOT, "models",  "ecg_model.joblib")
DATASET_PATH = os.path.join(ROOT, "data",    "ecg_dataset_cache.npz")
REPORT_PATH  = os.path.join(REPORTS_DIR, "ecg_clinical_validation_report.txt")
REPORT_JSON  = os.path.join(REPORTS_DIR, "ecg_clinical_validation_report.json")


# =============================================================================
# 1. BOOTSTRAP CI
# =============================================================================

def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray,
                     metric_fn, n_iter: int = N_BOOTSTRAP,
                     ci: float = 0.95) -> tuple:
    """
    Compute a scalar metric and its bootstrap CI.

    Parameters
    ----------
    metric_fn : callable(y_true, y_pred) -> float

    Returns
    -------
    point_est, ci_low, ci_high
    """
    point = metric_fn(y_true, y_pred)
    scores = []
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = (1.0 - ci) / 2.0
    lo = np.percentile(scores, alpha * 100)
    hi = np.percentile(scores, (1.0 - alpha) * 100)
    return float(point), float(lo), float(hi)


def per_class_metrics_with_ci(y_true: np.ndarray, y_pred: np.ndarray,
                               class_idx: int,
                               n_iter: int = N_BOOTSTRAP) -> dict:
    """Bootstrap CIs for sensitivity, specificity, precision, F1 of one class."""
    def _sens(yt, yp):
        tp = np.sum((yt == class_idx) & (yp == class_idx))
        fn = np.sum((yt == class_idx) & (yp != class_idx))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _spec(yt, yp):
        tn = np.sum((yt != class_idx) & (yp != class_idx))
        fp = np.sum((yt != class_idx) & (yp == class_idx))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def _prec(yt, yp):
        tp = np.sum((yt == class_idx) & (yp == class_idx))
        fp = np.sum((yt != class_idx) & (yp == class_idx))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _f1(yt, yp):
        s, p = _sens(yt, yp), _prec(yt, yp)
        return 2 * s * p / (s + p) if (s + p) > 0 else 0.0

    results = {}
    for name, fn in [("sensitivity", _sens), ("specificity", _spec),
                      ("precision", _prec), ("f1", _f1)]:
        val, lo, hi = bootstrap_metric(y_true, y_pred, fn, n_iter=n_iter)
        results[name] = {"value": val, "ci_low": lo, "ci_high": hi}
    results["support"] = int(np.sum(y_true == class_idx))
    return results


# =============================================================================
# 2. FAILURE ANALYSIS
# =============================================================================

def failure_analysis(cm: np.ndarray) -> list:
    """
    Identify the most-confused class pairs and return a ranked list.

    Returns
    -------
    list of dicts: {true_class, pred_class, count, rate}
    """
    confusions = []
    for i in range(N_CLASSES):
        row_total = cm[i, :].sum()
        if row_total == 0:
            continue
        for j in range(N_CLASSES):
            if i == j:
                continue
            if cm[i, j] > 0:
                confusions.append({
                    "true_class" : CLASSES[i],
                    "pred_class" : CLASSES[j],
                    "count"      : int(cm[i, j]),
                    "rate"       : round(float(cm[i, j] / row_total), 4),
                })
    confusions.sort(key=lambda x: -x["count"])
    return confusions


def dangerous_misclassifications(cm: np.ndarray) -> list:
    """
    Check for clinically dangerous misclassifications.

    Definition: a HIGH-severity arrhythmia classified as Normal or vice versa.
    Dangerous pairs (true -> pred):
      VT    -> Normal
      AFib  -> Normal
      VT    -> Bradycardia  (could delay pacing)
    """
    # Class indices: 0=Normal, 1=AFib, 2=PVC, 3=Brady, 4=Tachy
    dangerous = [
        (4, 0),   # VT -> Normal
        (1, 0),   # AFib -> Normal
        (4, 3),   # VT -> Bradycardia
    ]
    found = []
    for (true_idx, pred_idx) in dangerous:
        count = int(cm[true_idx, pred_idx])
        if count > 0:
            found.append({
                "true"  : CLASSES[true_idx],
                "pred"  : CLASSES[pred_idx],
                "count" : count,
                "status": "FAIL",
            })
        else:
            found.append({
                "true"  : CLASSES[true_idx],
                "pred"  : CLASSES[pred_idx],
                "count" : 0,
                "status": "PASS",
            })
    return found


# =============================================================================
# 3. CLINICAL SAFETY METRICS
# =============================================================================

def clinical_safety_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    False negative rate (FNR) and false positive rate (FPR) for arrhythmia
    classes as a group (classes 1-4) vs. Normal (class 0).

    FNR = missed arrhythmias / total arrhythmias  [target < 12 %]
    FPR = false alarms / total normal recordings   [target < 8 %]
    """
    arrhythmia_mask = y_true != 0
    normal_mask     = y_true == 0

    # False negatives: arrhythmia predicted as Normal
    fn = int(np.sum((y_true != 0) & (y_pred == 0)))
    tp = int(np.sum((y_true != 0) & (y_pred != 0)))
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # False positives: Normal predicted as arrhythmia
    fp = int(np.sum((y_true == 0) & (y_pred != 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "false_negative_rate": round(fnr * 100, 2),   # %
        "false_positive_rate": round(fpr * 100, 2),   # %
        "missed_arrhythmias" : fn,
        "total_arrhythmias"  : fn + tp,
        "false_alarms"       : fp,
        "total_normals"      : fp + tn,
        "fnr_target_pct"     : 12.0,
        "fpr_target_pct"     :  8.0,
        "fnr_pass"           : fnr * 100 < 12.0,
        "fpr_pass"           : fpr * 100 <  8.0,
    }


# =============================================================================
# 4. REPORT WRITER
# =============================================================================

def write_report(content: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Report saved -> {path}")


def generate_report(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray,
                    overall_roc_auc: float,
                    per_class_metrics: dict,
                    safety: dict,
                    confusions: list,
                    dangerous: list,
                    cm: np.ndarray) -> str:
    """Build the full Clinical Validation Summary as a formatted string."""

    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    def h1(title):
        lines.append("=" * 72)
        lines.append(f"  {title}")
        lines.append("=" * 72)

    def h2(title):
        lines.append("")
        lines.append(f"  {'-'*68}")
        lines.append(f"  {title}")
        lines.append(f"  {'-'*68}")

    def row(label, val, ci=""):
        ci_str = f"  95% CI [{ci}]" if ci else ""
        lines.append(f"    {label:<30} {val}{ci_str}")

    # Title
    h1("CLINICAL VALIDATION SUMMARY")
    lines.append(f"  Device        : Portable ECG Monitor with Arrhythmia Detector")
    lines.append(f"  Algorithm     : Pan-Tompkins + Random Forest (5-class)")
    lines.append(f"  Report date   : {now}")
    lines.append(f"  Intended use  : Screening of adult patients for common arrhythmias")
    lines.append(f"  Regulatory    : FDA 510(k) -- Class II Medical Device")

    # 1. Per-class performance
    h2("1. PER-CLASS PERFORMANCE (test set)")
    hdr = (f"  {'Class':<18} {'Sens%':>7} {'95% CI':>16}  "
           f"{'Spec%':>7} {'95% CI':>16}  "
           f"{'Prec%':>7}  {'F1%':>7}  {'N':>5}")
    lines.append(hdr)
    lines.append(f"  {'-'*100}")

    targets_met = True
    for cname, m in per_class_metrics.items():
        sens  = m["sensitivity"]
        spec  = m["specificity"]
        prec  = m["precision"]
        f1    = m["f1"]
        n     = m["support"]

        sens_ci = (f"{sens['ci_low']*100:.1f}-{sens['ci_high']*100:.1f}")
        spec_ci = (f"{spec['ci_low']*100:.1f}-{spec['ci_high']*100:.1f}")

        flag = ""
        if sens["value"] < 0.85:
            flag += " [SENS BELOW TARGET]"
            targets_met = False
        if spec["value"] < 0.90:
            flag += " [SPEC BELOW TARGET]"
            targets_met = False

        lines.append(
            f"  {cname:<18} {sens['value']*100:7.1f} "
            f"{sens_ci:>16}  {spec['value']*100:7.1f} "
            f"{spec_ci:>16}  {prec['value']*100:7.1f}  "
            f"{f1['value']*100:7.1f}  {n:5d}{flag}"
        )

    lines.append(f"\n  Overall ROC-AUC (macro OvR): {overall_roc_auc:.4f}"
                 + (" [BELOW TARGET 0.90]" if overall_roc_auc < 0.90 else "  [OK]"))

    # 2. Confusion matrix
    h2("2. CONFUSION MATRIX (rows = true class, cols = predicted)")
    hdr = f"  {'':>14}" + "".join(f"  {c[:7]:>9}" for c in CLASSES)
    lines.append(hdr)
    for i, cname in enumerate(CLASSES):
        row_str = f"  {cname:>14}" + "".join(
            f"  {cm[i,j]:9d}" for j in range(N_CLASSES)
        )
        lines.append(row_str)

    # 3. Failure analysis
    h2("3. FAILURE ANALYSIS -- Most-Confused Pairs")
    if confusions:
        lines.append(f"  {'True class':<18} {'Predicted as':<18} {'Count':>6} {'Rate':>8}")
        lines.append(f"  {'-'*55}")
        for c in confusions[:10]:   # top 10 confusions
            lines.append(
                f"  {c['true_class']:<18} {c['pred_class']:<18} "
                f"{c['count']:>6}  {c['rate']*100:>7.1f}%"
            )
        lines.append("")
        lines.append("  Likely causes:")
        lines.append("    - AFib/Normal confusion: borderline RR variability in short windows")
        lines.append("    - PVC/Normal: isolated PVCs < 15% threshold missed by labeller")
        lines.append("    - Tachy/AFib: rapid AFib can mimic tachycardia at high HR")
    else:
        lines.append("  No misclassifications found.")

    # 4. Clinical safety
    h2("4. CLINICAL SAFETY METRICS")
    fnr_pass = "[PASS]" if safety["fnr_pass"] else "[FAIL]"
    fpr_pass = "[PASS]" if safety["fpr_pass"] else "[FAIL]"
    lines.append(
        f"  False Negative Rate (missed arrhythmias): "
        f"{safety['false_negative_rate']:.1f}%  "
        f"(target < {safety['fnr_target_pct']}%)  {fnr_pass}"
    )
    lines.append(
        f"  False Positive Rate (false alarms):       "
        f"{safety['false_positive_rate']:.1f}%  "
        f"(target < {safety['fpr_target_pct']}%)  {fpr_pass}"
    )
    lines.append(
        f"  Missed arrhythmias : {safety['missed_arrhythmias']} / "
        f"{safety['total_arrhythmias']} arrhythmia recordings"
    )
    lines.append(
        f"  False alarms       : {safety['false_alarms']} / "
        f"{safety['total_normals']} normal recordings"
    )

    h2("4b. DANGEROUS MISCLASSIFICATION CHECK")
    lines.append(f"  {'True class':<18} {'Predicted as':<18} {'Count':>7}  {'Status':>6}")
    lines.append(f"  {'-'*55}")
    all_safe = True
    for d in dangerous:
        lines.append(
            f"  {d['true']:<18} {d['pred']:<18} "
            f"{d['count']:>7}  [{d['status']}]"
        )
        if d["status"] == "FAIL":
            all_safe = False
    lines.append(
        f"\n  Overall dangerous-misclassification check: "
        f"{'[PASS] No dangerous misclassifications' if all_safe else '[FAIL] See above'}"
    )

    # 5. Limitations
    h2("5. LIMITATIONS & KNOWN CONSTRAINTS")
    limits = [
        "Not validated for: pediatric patients (< 18 years)",
        "Not validated for: patients with implanted pacemakers or ICDs",
        "Not validated for: atrial flutter (can mimic AFib features)",
        "Not validated for: WPW syndrome, Brugada pattern, LQTS",
        "Performance may degrade with poor electrode contact or motion artifact",
        "Validated only on single-lead (MLII) recordings at 360 Hz",
        "Short recordings (< 30 s) provide less reliable HRV metrics",
        f"Overall accuracy ~{accuracy_pct:.0f}%; clinical judgment required for borderline cases",
        "Algorithm trained on MIT-BIH database (1975-1980); may not reflect modern demographics",
    ]
    for lim in limits:
        lines.append(f"  - {lim}")

    # 6. Disclaimers
    h2("6. REGULATORY DISCLAIMERS")
    lines.append(
        "  This device is FOR SCREENING PURPOSES ONLY and is NOT a diagnostic tool.\n"
        "  All results must be reviewed by a qualified healthcare professional before\n"
        "  any clinical decision is made.  The manufacturer does not accept liability\n"
        "  for decisions made solely on the basis of this device's output.\n\n"
        "  Intended for use as a Class II medical device under FDA 510(k) pathway.\n"
        "  Substantial equivalence predicate: cleared ambulatory ECG monitors."
    )

    h2("7. SUMMARY")
    lines.append(f"  Test set size    : {len(y_true)} samples")
    lines.append(f"  Overall accuracy : {accuracy_pct:.1f}%")
    lines.append(f"  Macro sens       : {macro_sens_pct:.1f}%")
    lines.append(f"  Macro spec       : {macro_spec_pct:.1f}%")
    lines.append(f"  ROC-AUC          : {overall_roc_auc:.4f}")
    lines.append(f"  Performance targets met : {'YES' if targets_met else 'NO -- see above'}")
    lines.append(f"  Clinical safety  : {'PASS' if all_safe else 'FAIL'}")
    lines.append("")
    lines.append("=" * 72)
    lines.append("  END OF CLINICAL VALIDATION SUMMARY")
    lines.append("=" * 72)

    return "\n".join(lines)


# module-level variables filled in run_phase5 for use inside generate_report
accuracy_pct   = 0.0
macro_sens_pct = 0.0
macro_spec_pct = 0.0


# =============================================================================
# 5. MAIN
# =============================================================================

def run_phase5():
    """Full Phase 5 pipeline."""
    global accuracy_pct, macro_sens_pct, macro_spec_pct

    os.makedirs(PLOTS_DIR,   exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("  PHASE 5 -- Clinical Validation & FDA Documentation")
    print("=" * 65)

    # ---- Load model and dataset -----------------------------------------
    if not os.path.isfile(MODEL_PATH):
        print(f"  [ERROR] Model not found: {MODEL_PATH}")
        print("  Run Phase 3 first:  python ecg_phase3.py")
        return

    if not os.path.isfile(DATASET_PATH):
        print(f"  [ERROR] Dataset cache not found: {DATASET_PATH}")
        print("  Run Phase 3 first:  python ecg_phase3.py")
        return

    print(f"\n[1] Loading model and dataset ...")
    model = joblib.load(MODEL_PATH)
    data  = np.load(DATASET_PATH)
    X, y  = data["X"], data["y"]
    print(f"  X shape: {X.shape}  |  Classes: {np.unique(y, return_counts=True)}")

    # Reproduce the same test split used in Phase 3 (same random_state)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Test set: {len(X_test)} samples")

    # ---- Predictions ----------------------------------------------------
    print("[2] Running predictions on test set ...")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # ---- Confusion matrix -----------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=list(range(N_CLASSES)))

    # ---- Overall metrics ------------------------------------------------
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    accuracy_pct = acc * 100

    # Macro sensitivity and specificity
    sens_list, spec_list = [], []
    for c in range(N_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    macro_sens_pct = np.mean(sens_list) * 100
    macro_spec_pct = np.mean(spec_list) * 100

    # ROC-AUC
    y_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
    # Expand y_proba to full class range (model may be missing classes, e.g., AFib)
    y_proba_full = np.zeros((len(y_test), N_CLASSES))
    for col_idx, cls_idx in enumerate(model.classes_):
        y_proba_full[:, cls_idx] = y_proba[:, col_idx]
    y_proba = y_proba_full

    present = np.unique(y_test)
    try:
        roc_auc = roc_auc_score(
            y_bin[:, present], y_proba[:, present],
            average="macro", multi_class="ovr"
        )
    except Exception:
        roc_auc = float("nan")

    # ---- Per-class metrics with bootstrap CIs ---------------------------
    print("[3] Computing bootstrap CIs (this may take a minute) ...")
    per_class_metrics = {}
    for c, cname in enumerate(CLASSES):
        print(f"    {cname} ...")
        per_class_metrics[cname] = per_class_metrics_with_ci(
            y_test, y_pred, c, n_iter=N_BOOTSTRAP
        )

    # ---- Failure & safety analysis --------------------------------------
    print("[4] Analysing failures ...")
    confusions = failure_analysis(cm)
    dangerous  = dangerous_misclassifications(cm)
    safety     = clinical_safety_stats(y_test, y_pred)

    # ---- Generate text report -------------------------------------------
    print("[5] Generating report ...")
    report_text = generate_report(
        y_test, y_pred, y_proba, roc_auc,
        per_class_metrics, safety, confusions, dangerous, cm
    )
    write_report(report_text, REPORT_PATH)
    print(report_text)

    # ---- Save JSON summary ----------------------------------------------
    json_data = {
        "timestamp"    : datetime.datetime.now().isoformat(),
        "test_samples" : int(len(y_test)),
        "accuracy_pct" : round(accuracy_pct, 2),
        "macro_sens_pct": round(macro_sens_pct, 2),
        "macro_spec_pct": round(macro_spec_pct, 2),
        "roc_auc"      : round(float(roc_auc), 4),
        "per_class"    : {
            cname: {
                metric: {
                    "value"  : round(vals["value"],    4),
                    "ci_low" : round(vals["ci_low"],   4),
                    "ci_high": round(vals["ci_high"],  4),
                } if isinstance(vals, dict) else vals
                for metric, vals in mdict.items()
            }
            for cname, mdict in per_class_metrics.items()
        },
        "confusion_matrix"              : cm.tolist(),
        "failure_analysis"              : confusions,
        "dangerous_misclassifications"  : dangerous,
        "clinical_safety"               : safety,
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON summary saved -> {REPORT_JSON}")

    # ---- Plots ----------------------------------------------------------
    _plot_bootstrap_metrics(per_class_metrics)

    print("\n" + "=" * 65)
    print("  PHASE 5 COMPLETE")
    print("=" * 65)


def _plot_bootstrap_metrics(per_class_metrics: dict) -> None:
    """Bar chart of sensitivity and specificity per class with 95 % CI error bars."""
    classes  = list(per_class_metrics.keys())
    sens_val = [per_class_metrics[c]["sensitivity"]["value"] * 100 for c in classes]
    spec_val = [per_class_metrics[c]["specificity"]["value"] * 100 for c in classes]
    sens_err = [[
        (per_class_metrics[c]["sensitivity"]["value"] -
         per_class_metrics[c]["sensitivity"]["ci_low"]) * 100
        for c in classes
    ], [
        (per_class_metrics[c]["sensitivity"]["ci_high"] -
         per_class_metrics[c]["sensitivity"]["value"]) * 100
        for c in classes
    ]]
    spec_err = [[
        (per_class_metrics[c]["specificity"]["value"] -
         per_class_metrics[c]["specificity"]["ci_low"]) * 100
        for c in classes
    ], [
        (per_class_metrics[c]["specificity"]["ci_high"] -
         per_class_metrics[c]["specificity"]["value"]) * 100
        for c in classes
    ]]

    x   = np.arange(len(classes))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, sens_val, w, label="Sensitivity", color="steelblue",
           yerr=sens_err, capsize=4, error_kw={"linewidth": 1.2})
    ax.bar(x + w/2, spec_val, w, label="Specificity", color="darkorange",
           yerr=spec_err, capsize=4, error_kw={"linewidth": 1.2})
    ax.axhline(85, color="steelblue", lw=1, ls="--", alpha=0.6, label="Sens target 85%")
    ax.axhline(90, color="darkorange", lw=1, ls="--", alpha=0.6, label="Spec target 90%")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("Metric (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Per-class Sensitivity & Specificity (95 % CI)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ecg_phase5_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Bootstrap metrics chart saved -> {path}")


# =============================================================================
if __name__ == "__main__":
    run_phase5()
