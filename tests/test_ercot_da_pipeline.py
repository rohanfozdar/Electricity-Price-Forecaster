"""Tests for the ERCOT DA prices pipeline output."""

from __future__ import annotations

import pandas as pd

from utils.config import RAW_DIR

PARQUET_PATH = RAW_DIR / "ercot_da_prices.parquet"
EXPECTED_COLS = [
    "da_HB_HOUSTON",
    "da_HB_NORTH",
    "da_HB_SOUTH",
    "da_HB_WEST",
    "da_HB_HUBAVG",
]


def test_parquet_exists_and_has_rows():
    df = pd.read_parquet(PARQUET_PATH)
    assert len(df) > 0, "Parquet file has no rows"


def test_has_da_prefixed_columns():
    df = pd.read_parquet(PARQUET_PATH)
    for col in EXPECTED_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_datetime_index_monotonic():
    df = pd.read_parquet(PARQUET_PATH)
    assert df.index.is_monotonic_increasing, "Datetime index is not monotonically increasing"


def test_hubavg_null_rate():
    df = pd.read_parquet(PARQUET_PATH)
    null_rate = df["da_HB_HUBAVG"].isna().mean()
    assert null_rate <= 0.05, f"da_HB_HUBAVG null rate is {null_rate:.1%}, exceeds 5%"


def test_hourly_frequency():
    df = pd.read_parquet(PARQUET_PATH)
    deltas = df.index.to_series().diff().dropna()
    # All time deltas should be exactly 1 hour
    expected = pd.Timedelta(hours=1)
    pct_hourly = (deltas == expected).mean()
    assert pct_hourly >= 0.95, (
        f"Only {pct_hourly:.1%} of intervals are 1-hour; expected >=95%"
    )
