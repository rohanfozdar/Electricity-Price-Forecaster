"""Latency rules for the corrected conservative information set."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.pit_latency import (
    MANIFEST_COLUMNS,
    PRIMARY_PRICE_DELAY_HOURS,
    align_chronos,
    availability_manifest,
    context_hash,
    interval_available,
    latest_complete_hour,
)
from src.features.pit_sets import (
    assert_chronological,
    day_ahead_features,
    fundamental_frames,
    previous_hour_intervals,
    price_history,
    train_median_impute,
)

ROOT = Path(__file__).resolve().parents[1]
UTC = "UTC"


def _hours(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="h", tz=UTC)


def test_positive_delay_excludes_the_interval_that_ends_at_issue_time():
    issue = pd.Timestamp("2024-06-01 02:00", tz=UTC)
    ending = issue - pd.Timedelta(minutes=15)
    assert not interval_available(ending, issue, pd.Timedelta(hours=1))
    assert not interval_available(ending, issue, pd.Timedelta(minutes=1))
    assert interval_available(ending, issue, pd.Timedelta(0))
    start = issue - pd.Timedelta(hours=4)
    rt_idx = pd.date_range(start, issue + pd.Timedelta(minutes=15), freq="15min", tz=UTC)
    rt = pd.Series(np.arange(len(rt_idx), dtype=float), index=rt_idx)
    out = previous_hour_intervals(rt, pd.DatetimeIndex([issue]), delay=pd.Timedelta(hours=1))
    visible = latest_complete_hour(issue, 1)
    assert visible == issue - pd.Timedelta(hours=2)
    assert out["rt15_last"].iloc[0] == rt.loc[visible + pd.Timedelta(minutes=45)]
    assert out["rt15_last"].iloc[0] != rt.loc[ending]
    for minutes in (0, 15, 30, 45):
        stamp = visible + pd.Timedelta(minutes=minutes)
        assert interval_available(stamp, issue, pd.Timedelta(hours=1))


def test_delay_moves_lags_and_rolling_windows_off_the_boundary_hour():
    idx = _hours(48)
    y = pd.Series(np.arange(48, dtype=float), index=idx)
    out = price_history(y, idx, delay=pd.Timedelta(hours=1))
    t = idx[30]
    assert out.loc[t, "hubavg_lag_1h"] == y.loc[t - pd.Timedelta(hours=2)]
    assert out.loc[t, "hubavg_lag_24h"] == y.loc[t - pd.Timedelta(hours=25)]
    window_end = t - pd.Timedelta(hours=2)
    expect = y.loc[window_end - pd.Timedelta(hours=23):window_end].mean()
    assert out.loc[t, "hubavg_rollmean_24h"] == pytest.approx(expect)
    bumped = y.copy()
    bumped.loc[t - pd.Timedelta(hours=1)] = 1e9
    changed = price_history(bumped, idx, delay=pd.Timedelta(hours=1))
    assert changed.loc[t, "hubavg_lag_1h"] == out.loc[t, "hubavg_lag_1h"]
    assert changed.loc[t, "hubavg_rollmean_24h"] == pytest.approx(out.loc[t, "hubavg_rollmean_24h"])


def test_dst_utc_hours_keep_four_intervals_and_the_delay_still_binds():
    for day in ("2024-03-10", "2024-11-03"):
        start = pd.Timestamp(day, tz=UTC)
        rt_idx = pd.date_range(start, periods=24 * 4, freq="15min", tz=UTC)
        counts = pd.DatetimeIndex(rt_idx).floor("h").value_counts()
        assert (counts == 4).all()
        rt = pd.Series(np.arange(len(rt_idx), dtype=float), index=rt_idx)
        issue = start + pd.Timedelta(hours=8)
        out = previous_hour_intervals(rt, pd.DatetimeIndex([issue]), delay=pd.Timedelta(hours=1))
        forbidden = issue - pd.Timedelta(minutes=15)
        assert out["rt15_last"].iloc[0] != rt.loc[forbidden]


def test_day_ahead_gate_does_not_pull_a_future_operating_hour():
    before = pd.Timestamp("2024-07-10 18:00", tz=UTC)
    later = pd.Timestamp("2024-07-11 12:00", tz=UTC)
    da = pd.DataFrame({"da_HB_HUBAVG": [123.0, 456.0]}, index=[before, later])
    y = pd.Series([10.0, 10.0], index=da.index)
    out = day_ahead_features(y, da)
    assert out.loc[before, "da_HB_HUBAVG"] == pytest.approx(123.0)
    assert out.loc[before, "da_HB_HUBAVG"] != pytest.approx(456.0)
    assert out.loc[later, "da_HB_HUBAVG"] == pytest.approx(456.0)


def test_storage_week_is_not_visible_on_the_week_ending_friday():
    friday = pd.Timestamp("2024-03-01", tz=UTC)
    storage = pd.DataFrame({"storage_bcf": [100.0], "storage_wow_change": [1.0]}, index=[friday])
    gas = pd.DataFrame({"henry_hub_price": [3.0]}, index=[pd.Timestamp("2024-01-01", tz=UTC)])
    nrc = pd.DataFrame({"reactors_offline_pct_sum": [1.0]}, index=[pd.Timestamp("2024-01-01", tz=UTC)])
    early = friday + pd.Timedelta(hours=12)
    released = friday + pd.Timedelta(days=7, hours=18)
    out = fundamental_frames(gas, storage, nrc, pd.DatetimeIndex([early, released]))["storage"]
    assert np.isnan(out.loc[early, "storage_bcf"])
    assert out.loc[released, "storage_bcf"] == pytest.approx(100.0)


def test_imputer_uses_training_rows_only():
    idx = _hours(10, "2019-01-01")
    train = pd.DataFrame({"hubavg_lag_1h": [1, 2, 3, 4, 5, 6]}, index=idx[:6])
    later = pd.DataFrame({"hubavg_lag_1h": [np.nan, 999.0, np.nan]}, index=idx[7:])
    filled, med = train_median_impute(train, [train, later], ["hubavg_lag_1h"])
    assert med["hubavg_lag_1h"] == pytest.approx(3.5)
    assert filled[1]["hubavg_lag_1h"].iloc[0] == pytest.approx(3.5)
    assert filled[1]["hubavg_lag_1h"].iloc[-1] == pytest.approx(3.5)
    assert filled[1]["hubavg_lag_1h"].iloc[1] == pytest.approx(999.0)
    assert_chronological(
        pd.DataFrame(index=idx[:6]),
        pd.DataFrame(index=idx[6:7]),
        pd.DataFrame(index=idx[7:]),
    )


def test_chronos_context_ends_at_the_last_legal_hour_and_ignores_later_prices():
    idx = _hours(8)
    cache = pd.DataFrame({
        "chronos_pred": np.arange(8, dtype=float),
        "chronos_q10": np.arange(8, dtype=float),
        "chronos_q90": np.arange(8, dtype=float) + 2,
        "chronos_mean": np.arange(8, dtype=float),
        "chronos_p_gt200": np.zeros(8),
    }, index=idx)
    aligned, latest = align_chronos(cache, idx, delay_hours=1)
    issue = idx[5]
    assert aligned.loc[issue, "chronos_pred"] == cache.loc[issue - pd.Timedelta(hours=1), "chronos_pred"]
    assert latest.loc[issue] == issue - pd.Timedelta(hours=2)
    assert latest.loc[issue] + pd.Timedelta(hours=1) + pd.Timedelta(hours=1) == issue
    prices = pd.Series(np.arange(8, dtype=float), index=idx)
    base = context_hash(prices.loc[:latest.loc[issue]].to_numpy())
    changed = prices.copy()
    changed.loc[issue] = 999.0
    assert context_hash(changed.loc[:latest.loc[issue]].to_numpy()) == base
    changed.loc[latest.loc[issue]] = 999.0
    assert context_hash(changed.loc[:latest.loc[issue]].to_numpy()) != base


def test_manifest_covers_the_conservative_columns_and_rejects_a_future_label():
    manifest = availability_manifest(PRIMARY_PRICE_DELAY_HOURS, PRIMARY_PRICE_DELAY_HOURS)
    assert list(manifest.columns) == list(MANIFEST_COLUMNS)
    names = set(manifest["feature_name"])
    saved = json.loads((ROOT / "runs" / "pit_ablation" / "matrices" / "conservative_columns.json").read_text())
    assert set(saved).issubset(names)
    assert "chronos_context_hourly_maximum" in names
    assert "chronos_context_latest_input_timestamp" in names
    assert set(manifest["verification_status"]).issubset({
        "verified historical availability",
        "documented rule with uncertain historical applicability",
        "assumed",
        "unverified",
        "known leak",
    })
    price = manifest.loc[manifest["feature_name"] == "hubavg_lag_1h"].iloc[0]
    assert price["verification_status"] == "assumed"
    assert "none measured" in price["verified_publication_delay"]
