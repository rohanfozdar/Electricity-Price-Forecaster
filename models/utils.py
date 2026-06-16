"""
Shared model utilities - feature groups, chronological splits, metrics.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/val/test split:
        train: 2016-2022 (~7 years)
        val:   2023
        test:  2024

    Critical: no random shuffling. Time-series data leaks badly if you
    shuffle because future information contaminates the training set.
    """
    train = df[df.index < "2023-01-01"]
    val = df[(df.index >= "2023-01-01") & (df.index < "2024-01-01")]
    test = df[df.index >= "2024-01-01"]
    print(f"  Split boundaries:")
    print(f"    Train: {train.index.min()} → {train.index.max()} ({len(train):,} rows)")
    print(f"    Val:   {val.index.min()} → {val.index.max()} ({len(val):,} rows)")
    print(f"    Test:  {test.index.min()} → {test.index.max()} ({len(test):,} rows)")
    return train, val, test


# Feature groups --------------------------------------------------------------

TARGET_REG = "HB_HUBAVG"
TARGET_CLASS = "future_spike_24h"

# Columns that are targets or would leak future info - never used as features
LEAK_COLS = {
    "HB_HUBAVG", "HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST",
    "price_spike_flag",
    "future_spike_24h", "future_price_max_24h",
}

# Baseline feature family: weather + lagged price + load + temporal
# (mirrors published electricity price models - no sentiment, no trends)
BASELINE_PREFIXES = (
    "temperature_", "relative_humidity_", "cloud_cover",
    "wind_speed_", "wind_gusts_", "precipitation",
    "temp_", "wind_",
    "hubavg_lag", "hubavg_roll",
    "load_", "da_",
    "henry_hub", "storage_bcf", "storage_wow",
    "reactor_", "reactors_offline",
    "stress_cold", "stress_freeze", "stress_heat", "stress_extreme",
    "stress_low_wind", "stress_gas", "stress_reactor",
    "hour", "day_of_week", "month", "is_weekend", "is_peak",
)

WEATHER_SUFFIXES = (
    "_temperature_2m",
    "_relative_humidity_2m",
    "_cloud_cover",
    "_wind_speed_10m",
    "_wind_gusts_10m",
    "_precipitation",
)

# Enhanced family: baseline + sentiment + trends
ENHANCED_EXTRA_PREFIXES = (
    "gdelt_", "stress_score",
)


def select_baseline_features(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in LEAK_COLS:
            continue
        if c.startswith(BASELINE_PREFIXES) or c.endswith(WEATHER_SUFFIXES):
            cols.append(c)
    return sorted(cols)


def select_enhanced_features(df: pd.DataFrame) -> list[str]:
    baseline = select_baseline_features(df)
    extras = []
    for c in df.columns:
        if c in LEAK_COLS or c in baseline:
            continue
        if c.startswith(BASELINE_PREFIXES + ENHANCED_EXTRA_PREFIXES):
            extras.append(c)
    return sorted(set(baseline + extras))


# Metrics --------------------------------------------------------------------

def regression_report(y_true, y_pred, name: str = "") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Spike-specific metrics
    true_spikes = y_true > 200
    pred_spikes = y_pred > 200
    spike_recall = (
        (true_spikes & pred_spikes).sum() / max(true_spikes.sum(), 1)
    )
    spike_precision = (
        (true_spikes & pred_spikes).sum() / max(pred_spikes.sum(), 1)
    )
    print(f"\n--- Regression report: {name} ---")
    print(f"  MAE:             ${mae:.2f}/MWh")
    print(f"  RMSE:            ${rmse:.2f}/MWh")
    print(f"  Spike recall:    {spike_recall:.2%}")
    print(f"  Spike precision: {spike_precision:.2%}")
    print("---------------------------------\n")
    return {
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "spike_recall": spike_recall,
        "spike_precision": spike_precision,
    }


def apply_feature_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p / binary transforms to known-skewed feature families.
    Mirrors the transforms validated in notebooks/11_residual_diagnostics.ipynb.
    Pure: returns a new DataFrame, does not mutate the input.
    Columns not matching any filter pass through unchanged.
    """
    _log1p = lambda x: np.log1p(x.clip(lower=0))
    _binary = lambda x: (x > 0).astype(float)

    _transforms = [
        (lambda f: f.startswith("hubavg_lag"),                              _log1p),
        (lambda f: f.startswith("hubavg_roll"),                             _log1p),
        (lambda f: f.endswith("_wind_speed_10m"),                           _log1p),
        (lambda f: f.endswith("_wind_gusts_10m"),                           _log1p),
        (lambda f: f.startswith("da_"),                                     _log1p),
        (lambda f: f.endswith("_precipitation"),                            _log1p),
        (lambda f: f.startswith(("henry_hub", "storage_bcf", "storage_wow")), _log1p),
    ]

    out = df.copy()
    for col in df.columns:
        for predicate, transform in _transforms:
            if predicate(col):
                out[col] = transform(df[col])
                break
    return out


def classification_report(y_true, y_pred_proba, name: str = "",
                          threshold: float = 0.5) -> dict:
    ap = average_precision_score(y_true, y_pred_proba)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = float("nan")
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    print(f"\n--- Classification report: {name} ---")
    print(f"  PR-AUC:    {ap:.3f}")
    print(f"  ROC-AUC:   {auc:.3f}")
    print(f"  Recall:    {recall:.2%}  (at threshold {threshold})")
    print(f"  Precision: {precision:.2%}")
    print(f"  Confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("-------------------------------------\n")
    return {
        "name": name, "pr_auc": ap, "roc_auc": auc,
        "recall": recall, "precision": precision,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
