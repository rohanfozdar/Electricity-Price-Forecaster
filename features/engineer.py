"""
Feature engineering.

Takes the raw feature matrix (one row per hour, one column per raw signal)
and produces the engineered matrix used for model training.

Engineered features fall into four families:

1. Lagged prices - previous 1h, 3h, 6h, 24h, 168h (1 week) values of HB_HUBAVG.
2. Rolling stats  - 24h and 168h rolling mean and std of price, load, temp.
3. Stress indicators - binary flags for known danger conditions.
4. Temporal - hour of day, day of week, month, is_weekend.

IMPORTANT: Lagged price features are computed strictly from PAST values
to avoid look-ahead leakage. The target variable is HB_HUBAVG at time t,
but every feature with 'lag' or 'rolling' in its name uses only data from
time < t.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import FEATURES_DIR, SPIKE_THRESHOLD_MWH
from utils.helpers import save_parquet


def engineer_features(matrix: pd.DataFrame) -> pd.DataFrame:
    df = matrix.copy()
    print("\n=== Engineering Features ===\n")
    print(f"  Starting shape: {df.shape}")

    # --- 1. Lagged target values ---
    for lag in [1, 12, 24, 48, 168, 720]:
        df[f"hubavg_lag_{lag}h"] = df["HB_HUBAVG"].shift(lag)

    # --- 2. Rolling stats on target ---
    for window in [24, 168]:
        df[f"hubavg_rollmean_{window}h"] = df["HB_HUBAVG"].shift(1).rolling(window).mean()
        df[f"hubavg_rollstd_{window}h"] = df["HB_HUBAVG"].shift(1).rolling(window).std()

    # --- 3. Load features ---
    if "load_actual_mw" in df.columns:
        df["load_lag_1h"] = df["load_actual_mw"].shift(1)
        df["load_rollmean_24h"] = df["load_actual_mw"].shift(1).rolling(24).mean()

    # --- 4. Weather-based features ---
    temp_cols = [c for c in df.columns if "temperature_2m" in c]
    wind_cols = [c for c in df.columns if "wind_speed_10m" in c]

    if temp_cols:
        df["temp_min_across_zones"] = df[temp_cols].min(axis=1)
        df["temp_max_across_zones"] = df[temp_cols].max(axis=1)
        df["temp_range_across_zones"] = df["temp_max_across_zones"] - df["temp_min_across_zones"]
    if wind_cols:
        df["wind_mean_across_zones"] = df[wind_cols].mean(axis=1)
        df["wind_min_across_zones"] = df[wind_cols].min(axis=1)

    # --- 5. Stress indicators (binary flags) ---
    if "temp_min_across_zones" in df.columns:
        df["stress_cold_snap"] = (df["temp_min_across_zones"] < 20).astype(int)
        df["stress_freeze"] = (df["temp_min_across_zones"] < 32).astype(int)
    if "temp_max_across_zones" in df.columns:
        df["stress_heat_wave"] = (df["temp_max_across_zones"] > 90).astype(int)
        df["stress_extreme_heat"] = (df["temp_max_across_zones"] > 105).astype(int)
    if "wind_min_across_zones" in df.columns:
        df["stress_low_wind"] = (df["wind_min_across_zones"] < 5).astype(int)
    if "henry_hub_price" in df.columns:
        gas_p90 = df["henry_hub_price"].quantile(0.90)
        df["stress_gas_spike"] = (df["henry_hub_price"] > gas_p90).astype(int)
    if "reactors_offline_pct_sum" in df.columns:
        df["stress_reactor_outage"] = (df["reactors_offline_pct_sum"] > 25).astype(int)

    # Combined stress score - how many concurrent stress conditions
    stress_cols = [c for c in df.columns if c.startswith("stress_")]
    if stress_cols:
        df["stress_score"] = df[stress_cols].sum(axis=1)

    # --- 6. Temporal features ---
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["is_peak_hours"] = ((df["hour"] >= 14) & (df["hour"] <= 20)).astype(int)

    # --- 7. Sentiment lag (sentiment 24h ahead of price - the whole thesis) ---
    for col in ["gdelt_tone", "gdelt_article_volume", "gdelt_norm"]:
        if col in df.columns:
            df[f"{col}_lag_24h"] = df[col].shift(24)
            df[f"{col}_lag_48h"] = df[col].shift(48)

    # --- 8. Target: future price spike in next 24h (for the classifier) ---
    # This looks forward, which is fine - it's the target, not a feature.
    df["future_spike_24h"] = (
        df["HB_HUBAVG"].rolling(24).max().shift(-24) > SPIKE_THRESHOLD_MWH
    ).astype(int)
    df["future_price_max_24h"] = df["HB_HUBAVG"].rolling(24).max().shift(-24)

    # Drop rows where we don't have enough history for lag features
    df = df.dropna(subset=["hubavg_lag_720h", "hubavg_rollmean_168h"])

    print(f"  Final shape:    {df.shape}")
    print(f"  Spike rate:     {df['price_spike_flag'].mean():.2%}")
    print(f"  Future spike rate (24h window): {df['future_spike_24h'].mean():.2%}")

    out_path = FEATURES_DIR / "feature_matrix_engineered.parquet"
    save_parquet(df, out_path)
    print(f"  Saved: {out_path}\n")
    return df


if __name__ == "__main__":
    raw_matrix_path = FEATURES_DIR / "feature_matrix_raw.parquet"
    if not raw_matrix_path.exists():
        raise RuntimeError("Run features/build_matrix.py first")
    raw = pd.read_parquet(raw_matrix_path)
    engineer_features(raw)
