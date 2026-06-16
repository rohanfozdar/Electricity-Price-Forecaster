"""
Feature matrix builder, v2 - dual-resolution merging.

Problem with v1:
    Daily/weekly features (GDELT, gas, NRC, Trends, storage) were resampled
    to hourly via forward-fill. Then engineer.py computed lags at hourly
    resolution. Result: gdelt_tone_lag_24h was identical to gdelt_tone for
    all 24 hours of the same day - the lag carried no new information for
    23 of every 24 rows, and the model couldn't learn the lead-lag dynamic.

Fix in v2:
    Compute lag, change, and rolling features AT THE NATURAL FREQUENCY of
    each source (daily for GDELT/gas/NRC, weekly for Trends/storage).
    Only after those engineered columns exist do we map them onto the hourly
    index via DATE/WEEK join. Each hour of a given day still shares the
    same daily values, but those values now correctly encode genuine
    day-over-day or week-over-week dynamics.

Output: data/features/feature_matrix_raw_v2.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _engineer_gdelt_daily(gdelt: pd.DataFrame) -> pd.DataFrame:
    """Compute change-based GDELT features at native daily frequency."""
    g = gdelt.copy()
    # Pure daily lags
    g["gdelt_tone_lag_1d"] = g["gdelt_tone"].shift(1)
    g["gdelt_tone_lag_2d"] = g["gdelt_tone"].shift(2)
    g["gdelt_tone_lag_3d"] = g["gdelt_tone"].shift(3)
    g["gdelt_volume_lag_1d"] = g["gdelt_article_volume"].shift(1)
    g["gdelt_volume_lag_2d"] = g["gdelt_article_volume"].shift(2)

    # Day-over-day changes (the real signal we want)
    g["gdelt_tone_change_1d"] = g["gdelt_tone"] - g["gdelt_tone"].shift(1)
    g["gdelt_tone_change_3d"] = g["gdelt_tone"] - g["gdelt_tone"].shift(3)
    g["gdelt_volume_change_1d"] = (
        g["gdelt_article_volume"] - g["gdelt_article_volume"].shift(1)
    )
    g["gdelt_volume_change_3d"] = (
        g["gdelt_article_volume"] - g["gdelt_article_volume"].shift(3)
    )

    # Volume z-score vs trailing 30-day baseline
    vol_mean = g["gdelt_article_volume"].shift(1).rolling(30, min_periods=10).mean()
    vol_std = g["gdelt_article_volume"].shift(1).rolling(30, min_periods=10).std()
    g["gdelt_volume_zscore_30d"] = (g["gdelt_article_volume"] - vol_mean) / vol_std.replace(0, np.nan)

    tone_mean = g["gdelt_tone"].shift(1).rolling(30, min_periods=10).mean()
    tone_std = g["gdelt_tone"].shift(1).rolling(30, min_periods=10).std()
    g["gdelt_tone_zscore_30d"] = (g["gdelt_tone"] - tone_mean) / tone_std.replace(0, np.nan)

    # Drop helper col we don't need downstream
    g = g.drop(columns=[c for c in ["norm"] if c in g.columns])
    return g


def _engineer_gas_daily(gas: pd.DataFrame) -> pd.DataFrame:
    g = gas.copy()
    g["henry_hub_lag_1d"] = g["henry_hub_price"].shift(1)
    g["henry_hub_change_1d"] = g["henry_hub_price"] - g["henry_hub_price"].shift(1)
    g["henry_hub_change_7d"] = g["henry_hub_price"] - g["henry_hub_price"].shift(7)
    return g


def _engineer_nrc_daily(nrc: pd.DataFrame) -> pd.DataFrame:
    n = nrc.copy()
    # Day-over-day change in total offline capacity
    n["reactors_offline_change_1d"] = (
        n["reactors_offline_pct_sum"] - n["reactors_offline_pct_sum"].shift(1)
    )
    return n


def _engineer_storage_weekly(storage: pd.DataFrame) -> pd.DataFrame:
    s = storage.copy()
    s["storage_bcf_lag_1w"] = s["storage_bcf"].shift(1)
    return s


def _merge_daily_to_hourly(matrix: pd.DataFrame, daily: pd.DataFrame,
                           label: str) -> pd.DataFrame:
    """Join an hourly-indexed matrix with a daily-indexed DataFrame.

    Strategy: forward-fill the daily df by 1 day boundaries onto hourly,
    but the lag/change/zscore columns inside `daily` were already computed
    at native daily resolution, so this ffill spreads correctly-computed
    daily values across each day's 24 hourly slots without creating fake
    intra-day dynamics.
    """
    # daily index is at midnight UTC; reindex onto hourly index via ffill
    # Use merge_asof with direction='backward' so each hour gets the most
    # recent daily row at-or-before its timestamp.
    matrix = matrix.sort_index()
    daily = daily.sort_index()
    merged = pd.merge_asof(
        matrix,
        daily,
        left_index=True,
        right_index=True,
        direction="backward",
    )
    print(f"  After {label} merge: {merged.shape}")
    return merged


def build_feature_matrix_v2(raw_dir: Path = RAW_DIR,
                            out_dir: Path = FEATURES_DIR) -> pd.DataFrame:
    print("\n=== Building Feature Matrix v2 (dual-resolution) ===\n")

    # --- HOURLY BASE: ERCOT RT prices ---
    rt = _load_if_exists(raw_dir / "ercot_rt_prices.parquet")
    if rt is None:
        raise RuntimeError("ERCOT RT prices required - run that pipeline first")
    rt = _to_utc(rt)
    price_cols = [c for c in rt.columns if c.startswith("HB_")]
    rt_hourly = rt[price_cols].resample("1h").max()
    rt_hourly["price_spike_flag"] = rt_hourly["HB_HUBAVG"] > SPIKE_THRESHOLD_MWH
    matrix = rt_hourly.copy()
    print(f"  Base matrix shape after RT prices: {matrix.shape}")

    # --- HOURLY: Day-ahead prices ---
    da = _load_if_exists(raw_dir / "ercot_da_prices.parquet")
    if da is not None:
        da = _to_utc(da)
        matrix = matrix.merge(da, how="left", left_index=True, right_index=True)
        print(f"  After DA merge: {matrix.shape}")

    # --- HOURLY: Weather ---
    weather = _load_if_exists(raw_dir / "weather.parquet")
    if weather is not None:
        weather = _to_utc(weather)
        matrix = matrix.merge(weather, how="left", left_index=True, right_index=True)
        print(f"  After weather merge: {matrix.shape}")

    # --- HOURLY: ERCOT load ---
    load = _load_if_exists(raw_dir / "ercot_load.parquet")
    if load is not None:
        load = _to_utc(load)
        matrix = matrix.merge(load, how="left", left_index=True, right_index=True)
        print(f"  After load merge: {matrix.shape}")

    # --- DAILY: GDELT (engineer at native frequency, then merge) ---
    gdelt = _load_if_exists(raw_dir / "gdelt.parquet")
    if gdelt is not None:
        gdelt = _to_utc(gdelt)
        gdelt = _engineer_gdelt_daily(gdelt)
        matrix = _merge_daily_to_hourly(matrix, gdelt, "GDELT (daily)")

    # --- DAILY: EIA gas (typically daily, may have gaps) ---
    gas = _load_if_exists(raw_dir / "eia_gas.parquet")
    if gas is not None:
        gas = _to_utc(gas)
        # gas may have non-business days missing; reindex to daily and ffill
        # only at the daily level (so lag/change reflect real prior-trading-day)
        gas = gas[~gas.index.duplicated(keep="first")].sort_index()
        gas_daily = gas.resample("1D").ffill()
        gas_daily = _engineer_gas_daily(gas_daily)
        matrix = _merge_daily_to_hourly(matrix, gas_daily, "EIA gas (daily)")

    # --- DAILY: NRC reactors ---
    nrc = _load_if_exists(raw_dir / "nrc_reactors.parquet")
    if nrc is not None:
        nrc = _to_utc(nrc)
        nrc = _engineer_nrc_daily(nrc)
        matrix = _merge_daily_to_hourly(matrix, nrc, "NRC reactors (daily)")

    # --- WEEKLY: EIA storage ---
    storage = _load_if_exists(raw_dir / "eia_storage.parquet")
    if storage is not None:
        storage = _to_utc(storage)
        storage = _engineer_storage_weekly(storage)
        matrix = _merge_daily_to_hourly(matrix, storage, "EIA storage (weekly)")

    matrix = matrix.sort_index()

    out_path = out_dir / "feature_matrix_raw_v2.parquet"
    save_parquet(matrix, out_path)
    print(f"\n  Feature matrix v2 saved: {out_path}")
    print(f"  Final shape: {matrix.shape}")
    print(f"  Date range:  {matrix.index.min()} -> {matrix.index.max()}")
    print(f"  Spike rate:  {matrix['price_spike_flag'].mean():.2%}")
    return matrix


if __name__ == "__main__":
    build_feature_matrix_v2()
