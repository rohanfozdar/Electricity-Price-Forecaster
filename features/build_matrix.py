"""
Feature matrix builder.

Merges every raw data source into a single hourly-indexed feature matrix
ready for feature engineering and model training.

Frequency strategy:
- Hourly index is the common denominator.
- 15-minute ERCOT real-time prices are resampled to hourly using MAX
  (preserves spike signals instead of averaging them away).
- Daily features (gas, storage, NRC, GDELT) are forward-filled across
  the 24 hourly slots of each day.
- Weekly features (Google Trends, gas storage) are forward-filled across
  all hours of the week.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.config import RAW_DIR, FEATURES_DIR, SPIKE_THRESHOLD_MWH
from utils.helpers import save_parquet


def _load_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [skip] {path.name} - file not found")
        return None
    df = pd.read_parquet(path)
    print(f"  [load] {path.name}: {len(df):,} rows, {len(df.columns)} cols")
    return df


def _to_hourly_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has a UTC hourly-compatible datetime index."""
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def build_feature_matrix(raw_dir: Path = RAW_DIR,
                         out_dir: Path = FEATURES_DIR) -> pd.DataFrame:
    print("\n=== Building Feature Matrix ===\n")

    # --- 1. Target variable: ERCOT real-time prices, resampled hourly with MAX ---
    rt = _load_if_exists(raw_dir / "ercot_rt_prices.parquet")
    if rt is None:
        raise RuntimeError("ERCOT RT prices required - run that pipeline first")
    rt = _to_hourly_utc(rt)
    # Resample 15-min -> hourly, taking the max so spike events are preserved
    price_cols = [c for c in rt.columns if c.startswith("HB_")]
    rt_hourly = rt[price_cols].resample("1h").max()
    rt_hourly["price_spike_flag"] = rt_hourly["HB_HUBAVG"] > SPIKE_THRESHOLD_MWH

    matrix = rt_hourly.copy()
    print(f"  Base matrix shape after RT prices: {matrix.shape}")

    # --- 2. Day-ahead prices (benchmark, also a feature) ---
    da = _load_if_exists(raw_dir / "ercot_da_prices.parquet")
    if da is not None:
        da = _to_hourly_utc(da)
        matrix = matrix.merge(da, how="left", left_index=True, right_index=True)
        print(f"  After DA merge: {matrix.shape}")

    # --- 3. Weather (hourly) ---
    weather = _load_if_exists(raw_dir / "weather.parquet")
    if weather is not None:
        weather = _to_hourly_utc(weather)
        matrix = matrix.merge(weather, how="left", left_index=True, right_index=True)
        print(f"  After weather merge: {matrix.shape}")

    # --- 4. ERCOT load (hourly) ---
    load = _load_if_exists(raw_dir / "ercot_load.parquet")
    if load is not None:
        load = _to_hourly_utc(load)
        matrix = matrix.merge(load, how="left", left_index=True, right_index=True)
        print(f"  After load merge: {matrix.shape}")

    # --- 5. EIA gas (daily -> forward-fill) ---
    gas = _load_if_exists(raw_dir / "eia_gas.parquet")
    if gas is not None:
        gas = _to_hourly_utc(gas)
        gas_hourly = gas.resample("1h").ffill()
        matrix = matrix.merge(gas_hourly, how="left", left_index=True, right_index=True)
        matrix["henry_hub_price"] = matrix["henry_hub_price"].ffill()
        print(f"  After gas merge: {matrix.shape}")

    # --- 6. EIA storage (weekly -> forward-fill) ---
    storage = _load_if_exists(raw_dir / "eia_storage.parquet")
    if storage is not None:
        storage = _to_hourly_utc(storage)
        storage_hourly = storage.resample("1h").ffill()
        matrix = matrix.merge(storage_hourly, how="left", left_index=True, right_index=True)
        for col in ["storage_bcf", "storage_wow_change"]:
            if col in matrix.columns:
                matrix[col] = matrix[col].ffill()
        print(f"  After storage merge: {matrix.shape}")

    # --- 7. NRC reactors (daily) ---
    nrc = _load_if_exists(raw_dir / "nrc_reactors.parquet")
    if nrc is not None:
        nrc = _to_hourly_utc(nrc)
        nrc_hourly = nrc.resample("1h").ffill()
        matrix = matrix.merge(nrc_hourly, how="left", left_index=True, right_index=True)
        for col in nrc.columns:
            if col in matrix.columns:
                matrix[col] = matrix[col].ffill()
        print(f"  After NRC merge: {matrix.shape}")

    # --- 8. GDELT (daily) ---
    gdelt = _load_if_exists(raw_dir / "gdelt.parquet")
    if gdelt is not None:
        gdelt = _to_hourly_utc(gdelt)
        gdelt_hourly = gdelt.resample("1h").ffill()
        matrix = matrix.merge(gdelt_hourly, how="left", left_index=True, right_index=True)
        for col in gdelt.columns:
            if col in matrix.columns:
                matrix[col] = matrix[col].ffill()
        print(f"  After GDELT merge: {matrix.shape}")

    matrix = matrix.sort_index()

    out_path = out_dir / "feature_matrix_raw.parquet"
    save_parquet(matrix, out_path)
    print(f"\n  Feature matrix saved: {out_path}")
    print(f"  Final shape: {matrix.shape}")
    print(f"  Date range:  {matrix.index.min()} → {matrix.index.max()}")
    print(f"  Spike rate:  {matrix['price_spike_flag'].mean():.2%}")
    return matrix


if __name__ == "__main__":
    build_feature_matrix()
