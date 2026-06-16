"""Tests for the ERCOT RT prices pipeline output."""

from __future__ import annotations

import pandas as pd

from utils.config import RAW_DIR

PARQUET_PATH = RAW_DIR / "ercot_rt_prices.parquet"
EXPECTED_HUBS = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HUBAVG"]


def test_parquet_exists_and_has_rows():
    df = pd.read_parquet(PARQUET_PATH)
    assert len(df) > 0, "Parquet file has no rows"


def test_has_hub_columns():
    df = pd.read_parquet(PARQUET_PATH)
    for hub in EXPECTED_HUBS:
        assert hub in df.columns, f"Missing column: {hub}"


def test_datetime_index_monotonic():
    df = pd.read_parquet(PARQUET_PATH)
    assert df.index.is_monotonic_increasing, "Datetime index is not monotonically increasing"


def test_hubavg_null_rate():
    df = pd.read_parquet(PARQUET_PATH)
    null_rate = df["HB_HUBAVG"].isna().mean()
    assert null_rate <= 0.05, f"HB_HUBAVG null rate is {null_rate:.1%}, exceeds 5%"
