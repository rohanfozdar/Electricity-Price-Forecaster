"""
Calibrate the v2 enhanced classifier.

Why: the raw XGBoost output for enhanced_v2 is squeezed into a narrow
band (test-set probabilities min=0.358, max=0.655, median=0.477). With
that distribution, no fixed threshold gives a useful recall/precision
tradeoff: the classifier flips from 100% recall at 0.4 to 0% recall at
0.6 over a tiny range. Calibration spreads the probabilities back across
[0,1] so threshold sweeps actually behave like sweeps.

Approach: fit `CalibratedClassifierCV(estimator=trained_xgb, cv='prefit')`
on the held-out 2023 validation set. We try two methods:
    - Platt scaling (sigmoid)  - assumes logistic post-fit, parametric
    - Isotonic regression       - non-parametric, more flexible but
                                  needs more validation data

Outputs:
    models/artifacts/calibrated_classifier_platt.pkl
    models/artifacts/calibrated_classifier_isotonic.pkl
    models/artifacts/calibrated_classifier.pkl  (the better of the two)
    models/artifacts/calibration_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

from utils.config import FEATURES_DIR
from models.utils import (
    TARGET_CLASS,
    chronological_split,
    select_enhanced_features,
    classification_report,
)

MODELS_DIR = Path("models/artifacts")


def _load_v2_data():
    df = pd.read_parquet(FEATURES_DIR / "feature_matrix_engineered_v2.parquet")
    df = df.dropna(subset=[TARGET_CLASS])
    return df


def _proba_distribution(p: np.ndarray) -> dict:
    qs = np.quantile(p, [0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(qs[0]),
        "q25": float(qs[1]),
        "median": float(qs[2]),
        "q75": float(qs[3]),
        "max": float(qs[4]),
        "frac_below_0.1": float((p < 0.1).mean()),
        "frac_above_0.9": float((p > 0.9).mean()),
    }


def _threshold_sweep(y_true, p, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    rows = []
    for thr in thresholds:
        y_pred = (p >= thr).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        rows.append({
            "threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision,
        })
    return rows


def calibrate():
    print("\n" + "=" * 70)
    print("  PROBABILITY CALIBRATION ON v2 ENHANCED CLASSIFIER")
    print("=" * 70)

    df = _load_v2_data()
    train, val, test = chronological_split(df)
    features = select_enhanced_features(df)

    X_train = train[features].ffill().fillna(0); y_train = train[TARGET_CLASS].values
    X_val   = val[features].ffill().fillna(0);   y_val   = val[TARGET_CLASS].values
    X_test  = test[features].ffill().fillna(0);  y_test  = test[TARGET_CLASS].values

    # Reload the already-trained v2 enhanced classifier
    booster = xgb.XGBClassifier()
    booster.load_model(str(MODELS_DIR / "enhanced_v2_classifier.json"))

    # Sanity: the loaded model should reproduce the test probabilities
    raw_proba_test = booster.predict_proba(X_test)[:, 1]
    raw_proba_val  = booster.predict_proba(X_val)[:, 1]

    print(f"\n  Raw (uncalibrated) test-set distribution:")
    raw_dist = _proba_distribution(raw_proba_test)
    for k, v in raw_dist.items():
        print(f"    {k:18s} {v:.4f}")

    raw_brier = brier_score_loss(y_test, raw_proba_test)
    raw_pr = average_precision_score(y_test, raw_proba_test)
    raw_roc = roc_auc_score(y_test, raw_proba_test)
    print(f"\n  Raw test PR-AUC:  {raw_pr:.4f}")
    print(f"  Raw test ROC-AUC: {raw_roc:.4f}")
    print(f"  Raw test Brier:   {raw_brier:.4f}")

    # CalibratedClassifierCV needs sklearn-style fit; cv='prefit' uses our
    # held-out validation set as the calibration set without re-training.
    results = {"raw": {
        "distribution": raw_dist,
        "pr_auc": float(raw_pr),
        "roc_auc": float(raw_roc),
        "brier": float(raw_brier),
        "threshold_sweep": _threshold_sweep(y_test, raw_proba_test),
    }}

    methods = {}
    for method in ("sigmoid", "isotonic"):
        print(f"\n  --- Fitting {method} calibrator on validation set ---")
        cal = CalibratedClassifierCV(estimator=booster, method=method, cv="prefit")
        cal.fit(X_val, y_val)

        cal_proba_test = cal.predict_proba(X_test)[:, 1]
        dist = _proba_distribution(cal_proba_test)
        pr  = average_precision_score(y_test, cal_proba_test)
        roc = roc_auc_score(y_test, cal_proba_test)
        brier = brier_score_loss(y_test, cal_proba_test)

        print(f"\n  {method} test-set distribution:")
        for k, v in dist.items():
            print(f"    {k:18s} {v:.4f}")
        print(f"\n  {method} test PR-AUC:  {pr:.4f}")
        print(f"  {method} test ROC-AUC: {roc:.4f}")
        print(f"  {method} test Brier:   {brier:.4f}")

        sweep = _threshold_sweep(y_test, cal_proba_test)
        print(f"\n  Threshold sweep ({method}):")
        print(f"    {'thr':>5}  {'recall':>8}  {'precision':>10}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
        for r in sweep:
            print(f"    {r['threshold']:>5.2f}  {r['recall']:>8.2%}  {r['precision']:>10.2%}  "
                  f"{r['tp']:>5}  {r['fp']:>5}  {r['fn']:>5}")

        methods[method] = {
            "distribution": dist,
            "pr_auc": float(pr),
            "roc_auc": float(roc),
            "brier": float(brier),
            "threshold_sweep": sweep,
        }
        out_pkl = MODELS_DIR / f"calibrated_classifier_{'platt' if method=='sigmoid' else 'isotonic'}.pkl"
        joblib.dump(cal, out_pkl)
        print(f"\n  Saved {out_pkl}")

    results.update({
        "platt": methods["sigmoid"],
        "isotonic": methods["isotonic"],
    })

    # Pick the calibrator with the lowest Brier (best probability calibration)
    best = min(("platt", methods["sigmoid"]["brier"]),
               ("isotonic", methods["isotonic"]["brier"]),
               key=lambda x: x[1])
    best_name = best[0]
    print(f"\n  ==> Best calibrator (lowest Brier): {best_name}")
    src = MODELS_DIR / f"calibrated_classifier_{best_name}.pkl"
    dst = MODELS_DIR / "calibrated_classifier.pkl"
    import shutil
    shutil.copy(src, dst)
    print(f"  Saved best to {dst}")
    results["best"] = best_name

    with open(MODELS_DIR / "calibration_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Calibration report saved to {MODELS_DIR / 'calibration_report.json'}")

    # Side-by-side distribution summary
    print("\n" + "=" * 70)
    print("  DISTRIBUTION COMPARISON (test-set predicted probabilities)")
    print("=" * 70)
    print(f"  {'method':<12} {'min':>7} {'Q25':>7} {'median':>8} {'Q75':>7} {'max':>7}  {'<0.1':>6}  {'>0.9':>6}")
    for label, key in [("uncalibrated", "raw"), ("Platt", "platt"), ("isotonic", "isotonic")]:
        d = results[key]["distribution"]
        print(f"  {label:<12} {d['min']:>7.3f} {d['q25']:>7.3f} {d['median']:>8.3f} "
              f"{d['q75']:>7.3f} {d['max']:>7.3f}  {d['frac_below_0.1']:>6.2%}  {d['frac_above_0.9']:>6.2%}")

    return results


if __name__ == "__main__":
    calibrate()
