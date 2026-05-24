# -*- coding: utf-8 -*-
"""
ECG Arrhythmia Monitor - Master Runner
=======================================
Executes the full pipeline in order:

  Phase 1 -- Data loading, R-peak detection, feature extraction
  Phase 2 -- Clinical decision-rule classifier (synthetic test cases)
  Phase 3 -- Random Forest training & validation
  Phase 5 -- Clinical validation & FDA-ready documentation

Usage
-----
    python run_all.py                 # full pipeline (build dataset from scratch)
    python run_all.py --skip-build    # reuse cached dataset (faster if Phase 3 ran before)
    python run_all.py --phase 1       # run a single phase
    python run_all.py --phase 3 --skip-build

After Phase 3 completes, a trained model (ecg_model.joblib) is saved.
You can then run the production analyser on any ECG CSV:

    python ecg_analyzer.py <your_ecg.csv> --patient-id P001

Output files generated
-----------------------
  ecg_record100_rpeaks.png         -- Phase 1 ECG plot
  ecg_phase1_features.png          -- Phase 1 feature summary
  ecg_dataset_cache.npz            -- Feature matrix (cached for Phase 3 re-runs)
  ecg_model.joblib                 -- Trained Random Forest
  ecg_phase3_confusion.png         -- Confusion matrix
  ecg_phase3_roc.png               -- ROC curves
  ecg_phase3_importance.png        -- Feature importance
  ecg_phase3_metrics.json          -- Val + test metrics
  ecg_clinical_validation_report.txt  -- Phase 5 FDA report
  ecg_clinical_validation_report.json -- Phase 5 JSON summary
  ecg_phase5_metrics.png           -- Bootstrap CI bar chart
"""

import argparse
import sys
import os
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))


def run_phase(n: int, skip_build: bool = False):
    print(f"\n{'#'*68}")
    print(f"#  PHASE {n}")
    print(f"{'#'*68}")
    t0 = time.time()

    if n == 1:
        from ecg_phase1 import run_phase1
        run_phase1()

    elif n == 2:
        from ecg_phase2 import run_phase2
        run_phase2()

    elif n == 3:
        from ecg_phase3 import run_phase3
        run_phase3(skip_build=skip_build)

    elif n == 5:
        from ecg_phase5_validation import run_phase5
        run_phase5()

    else:
        print(f"  Unknown phase: {n}")
        return

    elapsed = time.time() - t0
    print(f"\n  Phase {n} finished in {elapsed:.1f} s")


def main():
    parser = argparse.ArgumentParser(description="ECG Monitor -- master runner")
    parser.add_argument("--phase",       type=int, default=0,
                        help="Run a single phase (1/2/3/5). Default: all.")
    parser.add_argument("--skip-build",  action="store_true",
                        help="Phase 3: load cached dataset instead of rebuilding")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.phase:
        run_phase(args.phase, skip_build=args.skip_build)
    else:
        for phase in [1, 2, 3, 5]:
            run_phase(phase, skip_build=args.skip_build)

    print(f"\n{'='*68}")
    print("  All requested phases complete.")
    print("  Model: models/ecg_model.joblib")
    print("  Run the analyzer:")
    print("    python src/ecg_analyzer.py <ecg_file.csv> --patient-id P001")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
