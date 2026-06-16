"""
Feature engineering, v2.

Operates on `feature_matrix_raw_v2.parquet` produced by build_matrix_v2.py.

Key difference from v1: sentiment lag/change features are NO LONGER computed
here at hourly resolution. They were computed correctly at native daily/weekly
resolution inside build_matrix_v2.py, then merged onto the hourly index.
This eliminates the forward-fill dilution that made hourly sentiment lags
identical to the same-day sentiment value for 23 out of every 24 rows.

Hourly-only engineering still happens here:
    - HB_HUBAVG lag and rolling stats (1h..168h)
    - Load lag and rolling stats
    - Weather cross-zone aggregates and stress flags
    - Combined stress score
    - Temporal calendar features
    - Forward-target columns for the classifier
"""
## ADDS THE DERIVATIVE FEAUTURES

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import FEATURES_DIR, SPIKE_THRESHOLD_MWH
from utils.helpers import save_parquet


def engineer_features_v2(matrix: pd.DataFrame) -> pd.DataFrame:
    df = matrix.copy()
    print("\n=== Engineering Features v2 ===\n")
    print(f"  Starting shape: {df.shape}")

    # --- 1. Lagged target values (hourly resolution - genuine intra-day signal) ---
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

    # --- 4. Weather cross-zone aggregates ---
    temp_cols = [c for c in df.columns if "temperature_2m" in c]
    wind_cols = [c for c in df.columns if "wind_speed_10m" in c]
    if temp_cols:
        df["temp_min_across_zones"] = df[temp_cols].min(axis=1)
        df["temp_max_across_zones"] = df[temp_cols].max(axis=1)
        df["temp_range_across_zones"] = df["temp_max_across_zones"] - df["temp_min_across_zones"]
    if wind_cols:
        df["wind_mean_across_zones"] = df[wind_cols].mean(axis=1)
        df["wind_min_across_zones"] = df[wind_cols].min(axis=1)

    # --- 5. Stress indicators ---
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

    stress_cols = [c for c in df.columns if c.startswith("stress_")]
    if stress_cols:
        df["stress_score"] = df[stress_cols].sum(axis=1)

    # --- 6. Temporal features ---
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["is_peak_hours"] = ((df["hour"] >= 14) & (df["hour"] <= 20)).astype(int)

    # --- 7. NO sentiment lag block here (handled in build_matrix_v2 at daily) ---

    # --- 8. Target columns ---
    df["future_spike_24h"] = (
        df["HB_HUBAVG"].rolling(24).max().shift(-24) > SPIKE_THRESHOLD_MWH
    ).astype(int)
    df["future_price_max_24h"] = df["HB_HUBAVG"].rolling(24).max().shift(-24)

    # Drop rows with insufficient lag history
    df = df.dropna(subset=["hubavg_lag_720h", "hubavg_rollmean_168h"])

    print(f"  Final shape:    {df.shape}")
    print(f"  Spike rate:     {df['price_spike_flag'].mean():.2%}")
    print(f"  Future spike rate (24h window): {df['future_spike_24h'].mean():.2%}")

    out_path = FEATURES_DIR / "feature_matrix_engineered_v2.parquet"
    save_parquet(df, out_path)
    print(f"  Saved: {out_path}\n")
    return df


if __name__ == "__main__":
    raw_path = FEATURES_DIR / "feature_matrix_raw_v2.parquet"
    if not raw_path.exists():
        raise RuntimeError("Run features/build_matrix_v2.py first")
    raw = pd.read_parquet(raw_path)
    engineer_features_v2(raw)
