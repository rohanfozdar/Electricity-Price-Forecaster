"""
Benchmark: v2 enhanced model vs ERCOT day-ahead market on the 2024 test set.

Two head-to-heads:

(A) PRICE FORECAST: v2 enhanced regressor vs DAM clearing price for HB_HUBAVG.
    Compare overall RMSE/MAE plus stress-only and normal-only breakdowns.
    The headline question: even if overall RMSE is similar, does the model
    win specifically during stress hours where forecast accuracy matters most?

(B) SPIKE DETECTION: calibrated v2 classifier vs naive "DAM > $150" detector.
    Did the calibrated probability >0.3 alert beat just watching the DAM?

Inputs:
    data/features/feature_matrix_engineered_v2.parquet
    models/artifacts/enhanced_v2_regressor.json
    models/artifacts/calibrated_classifier.pkl

Output:
    evaluation/dam_benchmark.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from utils.config import FEATURES_DIR
from models.utils import (
    TARGET_REG,
    chronological_split,
    select_enhanced_features,
)

STRESS_PRICE = 100.0   # $/MWh defining "stress hour" for breakdown
SPIKE_PRICE = 200.0    # $/MWh defining "actual spike"
DAM_SPIKE_THRESHOLD = 150.0
CAL_SPIKE_THRESHOLD = 0.3


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "n": int(len(y_true)),
    }


def _spike_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_spikes = y_true > SPIKE_PRICE
    pred_spikes = y_pred > SPIKE_PRICE
    tp = int((true_spikes & pred_spikes).sum())
    fp = int((~true_spikes & pred_spikes).sum())
    fn = int((true_spikes & ~pred_spikes).sum())
    tn = int((~true_spikes & ~pred_spikes).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * rec * prec / max(rec + prec, 1e-9)
    return {"recall": rec, "precision": prec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _binary_spike_metrics(y_true_spike: np.ndarray, y_pred_spike: np.ndarray) -> dict:
    """Spike-classification metrics where both arrays are 0/1 labels."""
    tp = int((y_true_spike & y_pred_spike).sum())
    fp = int((~y_true_spike & y_pred_spike).sum())
    fn = int((y_true_spike & ~y_pred_spike).sum())
    tn = int((~y_true_spike & ~y_pred_spike).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * rec * prec / max(rec + prec, 1e-9)
    return {"recall": rec, "precision": prec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _pick(a: float, b: float, lower_better: bool = True) -> str:
    if a == b:
        return "tie"
    if lower_better:
        return "Enhanced" if a < b else "DAM"
    return "Enhanced" if a > b else "DAM"


def _print_head_to_head(label: str, model: dict, dam: dict, lower_better=True):
    if label in ("RMSE", "MAE", "Stress MAE", "Normal MAE"):
        winner = _pick(model["mae" if "MAE" in label else "rmse"],
                       dam["mae" if "MAE" in label else "rmse"], lower_better=True)
    else:
        winner = ""
    print(f"  {label:<28} {model:<15.2f if not isinstance(model, dict) else ''} ...")  # not used


def benchmark_vs_dam():
    print("\n" + "=" * 75)
    print("  BENCHMARK A: v2 ENHANCED REGRESSOR vs ERCOT DAY-AHEAD MARKET")
    print("=" * 75)

    matrix_path = FEATURES_DIR / "feature_matrix_engineered_v2.parquet"
    df = pd.read_parquet(matrix_path)
    if "da_HB_HUBAVG" not in df.columns:
        raise RuntimeError("da_HB_HUBAVG missing from v2 matrix - cannot compare DAM")

    df = df.dropna(subset=[TARGET_REG, "da_HB_HUBAVG"])
    _, _, test = chronological_split(df)

    reg_path = Path("models/artifacts/enhanced_v2_regressor.json")
    if not reg_path.exists():
        raise RuntimeError(f"Missing {reg_path} - run models/train_all_v2.py first")
    reg = xgb.XGBRegressor()
    reg.load_model(str(reg_path))

    features = select_enhanced_features(df)
    X_test = test[features].ffill().fillna(0)
    y_test = test[TARGET_REG].values

    model_pred = reg.predict(X_test)
    dam_pred = test["da_HB_HUBAVG"].values

    overall_model = _metrics(y_test, model_pred)
    overall_dam   = _metrics(y_test, dam_pred)
    overall_model.update(_spike_metrics(y_test, model_pred))
    overall_dam.update(_spike_metrics(y_test, dam_pred))

    # Stress / normal breakdown
    stress_mask = y_test > STRESS_PRICE
    normal_mask = ~stress_mask
    print(f"\n  Test rows: {len(y_test):,}  "
          f"(stress={stress_mask.sum():,}  normal={normal_mask.sum():,})")

    stress_model = _metrics(y_test[stress_mask], model_pred[stress_mask])
    stress_dam   = _metrics(y_test[stress_mask], dam_pred[stress_mask])
    normal_model = _metrics(y_test[normal_mask], model_pred[normal_mask])
    normal_dam   = _metrics(y_test[normal_mask], dam_pred[normal_mask])

    print(f"\n  {'Metric':<28}  {'Enhanced':>11}  {'DAM':>11}  {'Winner':<10}")
    print("  " + "-" * 65)
    rows = [
        ("Overall RMSE ($/MWh)",     overall_model["rmse"], overall_dam["rmse"], True),
        ("Overall MAE  ($/MWh)",     overall_model["mae"],  overall_dam["mae"],  True),
        ("Stress-hours MAE  (>$100)", stress_model["mae"],  stress_dam["mae"],   True),
        ("Normal-hours MAE  (<$100)", normal_model["mae"],  normal_dam["mae"],   True),
        ("Spike recall   (>$200)",   overall_model["recall"]    * 100, overall_dam["recall"]    * 100, False),
        ("Spike precision (>$200)",  overall_model["precision"] * 100, overall_dam["precision"] * 100, False),
        ("Spike F1       (>$200)",   overall_model["f1"]        * 100, overall_dam["f1"]        * 100, False),
    ]
    for label, m, d, low in rows:
        winner = _pick(m, d, lower_better=low)
        suffix = "%" if "recall" in label.lower() or "precision" in label.lower() or "f1" in label.lower() else ""
        print(f"  {label:<28}  {m:>10.2f}{suffix}  {d:>10.2f}{suffix}  {winner:<10}")

    print(f"\n  Stress hours = actual price > ${STRESS_PRICE:.0f}, "
          f"Spike = actual price > ${SPIKE_PRICE:.0f}")

    # Benchmark B: spike detection
    print("\n" + "=" * 75)
    print("  BENCHMARK B: CALIBRATED v2 CLASSIFIER vs NAIVE 'DAM > $150' DETECTOR")
    print("=" * 75)

    cal = joblib.load("models/artifacts/calibrated_classifier.pkl")
    cal_features = list(cal.feature_names_in_) if hasattr(cal, "feature_names_in_") \
                   else features
    X_test_cal = test[cal_features].ffill().fillna(0)
    cal_proba = cal.predict_proba(X_test_cal)[:, 1]

    # Use the future_spike_24h target (the classifier's actual training target)
    if "future_spike_24h" not in test.columns:
        raise RuntimeError("future_spike_24h missing from test - cannot benchmark classifier")
    y_test_spike = test["future_spike_24h"].fillna(0).astype(bool).values

    cal_pred  = cal_proba > CAL_SPIKE_THRESHOLD
    dam_pred_spike = (test["da_HB_HUBAVG"].values > DAM_SPIKE_THRESHOLD)

    cal_m = _binary_spike_metrics(y_test_spike, cal_pred)
    dam_m = _binary_spike_metrics(y_test_spike, dam_pred_spike)

    print(f"\n  Target: future_spike_24h (any actual price >${SPIKE_PRICE:.0f} in next 24h)")
    print(f"  Calibrated detector: prob > {CAL_SPIKE_THRESHOLD}")
    print(f"  DAM detector:        da_HB_HUBAVG > ${DAM_SPIKE_THRESHOLD:.0f}")
    print(f"\n  {'Metric':<14}  {'Calibrated':>11}  {'DAM>$150':>11}  {'Winner':<10}")
    print("  " + "-" * 50)
    for label, key in [("Recall", "recall"), ("Precision", "precision"), ("F1", "f1")]:
        c = cal_m[key] * 100
        d = dam_m[key] * 100
        w = _pick(c, d, lower_better=False)
        # Map "Enhanced" -> "Calibrated" for clarity
        w = "Calibrated" if w == "Enhanced" else w
        print(f"  {label:<14}  {c:>10.2f}%  {d:>10.2f}%  {w:<10}")
    print(f"\n  Calibrated confusion: TP={cal_m['tp']} FP={cal_m['fp']} "
          f"FN={cal_m['fn']} TN={cal_m['tn']}")
    print(f"  DAM>$150  confusion: TP={dam_m['tp']} FP={dam_m['fp']} "
          f"FN={dam_m['fn']} TN={dam_m['tn']}")

    out = {
        "regression": {
            "overall": {"enhanced": overall_model, "dam": overall_dam},
            "stress_hours": {"enhanced": stress_model, "dam": stress_dam},
            "normal_hours": {"enhanced": normal_model, "dam": normal_dam},
            "stress_threshold": STRESS_PRICE,
            "spike_threshold": SPIKE_PRICE,
        },
        "spike_detection": {
            "calibrated_threshold": CAL_SPIKE_THRESHOLD,
            "dam_threshold": DAM_SPIKE_THRESHOLD,
            "calibrated": cal_m,
            "dam_naive": dam_m,
        },
    }
    Path("evaluation").mkdir(exist_ok=True)
    with open("evaluation/dam_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to evaluation/dam_benchmark.json\n")
    return out


if __name__ == "__main__":
    benchmark_vs_dam()
