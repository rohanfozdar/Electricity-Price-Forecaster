"""Point-in-time feature rules for the one-hour-ahead ablation."""

import numpy as np
import pandas as pd
import pytest

from src.data.asof import available_at
from src.features.pit_sets import (
    RULES,
    asof_frame,
    calendar_central,
    conservative_columns,
    feature_subsets,
    day_ahead_features,
    gas_spike_flag,
    hourly_interval_counts,
    lagged_weather,
    previous_hour_intervals,
    price_history,
    train_median_impute,
    assert_chronological,
)


def _hours(n=48, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def test_price_features_exclude_the_target_hour_and_use_the_clock():
    idx = _hours(48)
    y = pd.Series(np.arange(48, dtype=float), index=idx)
    base = price_history(y, idx)
    bumped = y.copy()
    bumped.iloc[-1] = 999.0
    changed = price_history(bumped, idx)
    pd.testing.assert_frame_equal(base.iloc[[-1]], changed.iloc[[-1]])
    bumped.iloc[-2] = 999.0
    changed = price_history(bumped, idx)
    assert changed["hubavg_lag_1h"].iloc[-1] == 999.0
    # Trailing 24-hour mean at the last hour is the mean of the previous 24 hours.
    expect = y.iloc[-25:-1].mean()
    assert base["hubavg_rollmean_24h"].iloc[-1] == pytest.approx(expect)
    assert base["hubavg_lag_1h"].iloc[-1] == y.iloc[-2]


def test_clock_lag_does_not_cross_a_missing_hour():
    idx = _hours(4)
    y = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    y = y.drop(idx[1])
    out = price_history(y, idx)
    assert out["hubavg_lag_1h"].iloc[1] == 1.0
    assert np.isnan(out["hubavg_lag_1h"].iloc[2])
    assert out["hubavg_lag_1h"].iloc[3] == 3.0


def test_previous_hour_intervals_exclude_the_target_hour():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rt_idx = pd.date_range(start, periods=8, freq="15min", tz="UTC")
    rt = pd.Series([1.0, 2.0, 3.0, 10.0, 50.0, 60.0, 70.0, 80.0], index=rt_idx)
    hourly = pd.DatetimeIndex([start + pd.Timedelta(hours=1)])
    out = previous_hour_intervals(rt, hourly)
    assert out["rt15_last"].iloc[0] == 10.0
    assert out["rt15_max"].iloc[0] == 10.0
    assert list(out.iloc[0][["rt15_min", "rt15_mean"]]) == [1.0, 4.0]


def test_day_ahead_spread_ignores_the_current_realtime_price():
    idx = _hours(48)
    y = pd.Series(np.arange(48, dtype=float), index=idx)
    da = pd.DataFrame({"da_HB_HUBAVG": np.full(48, 20.0)}, index=idx)
    base = day_ahead_features(y, da)
    bumped = y.copy()
    bumped.iloc[-1] = 999.0
    changed = day_ahead_features(bumped, da)
    assert base["da_shape_spread_24h"].iloc[-1] == pytest.approx(changed["da_shape_spread_24h"].iloc[-1])
    earlier = y.copy()
    earlier.iloc[-2] = 999.0
    assert base["da_shape_spread_24h"].iloc[-1] != pytest.approx(
        day_ahead_features(earlier, da)["da_shape_spread_24h"].iloc[-1])


def test_same_day_day_ahead_max_uses_only_that_published_day():
    idx = _hours(48, "2024-07-10 06:00")  # 2024-07-10 01:00 Central through the next day
    da = pd.DataFrame({"da_HB_HUBAVG": np.arange(48, dtype=float)}, index=idx)
    y = pd.Series(np.zeros(48), index=idx)
    out = day_ahead_features(y, da)
    local_day = idx.tz_convert("US/Central").floor("D")
    first = np.flatnonzero(local_day == local_day[0])[-1]
    assert out["da_shape_day_max"].iloc[0] == pytest.approx(da["da_HB_HUBAVG"].iloc[: first + 1].max())
    nxt = da.copy()
    nxt.iloc[-1, 0] = 500.0
    assert out["da_shape_day_max"].iloc[0] == pytest.approx(day_ahead_features(y, nxt)["da_shape_day_max"].iloc[0])


def test_gas_release_changes_when_the_origin_moves_across_it():
    gas = pd.DataFrame(
        {"henry_hub_price": [10.0, 80.0]},
        index=pd.to_datetime(["2024-03-04", "2024-03-11"], utc=True),
    )
    release = available_at(pd.DatetimeIndex(gas.index[:1]), RULES["eia_gas"])[0]
    before = pd.DatetimeIndex([release - pd.Timedelta(hours=1)])
    after = pd.DatetimeIndex([release])
    got_before = asof_frame(gas, before, RULES["eia_gas"])["henry_hub_price"].iloc[0]
    got_after = asof_frame(gas, after, RULES["eia_gas"])["henry_hub_price"].iloc[0]
    assert np.isnan(got_before)
    assert got_after == 10.0
    later = asof_frame(gas, pd.DatetimeIndex([release + pd.Timedelta(hours=1)]), RULES["eia_gas"])
    assert later["henry_hub_price"].iloc[0] == 10.0


def test_storage_is_not_visible_on_the_week_ending_friday():
    fri = pd.Timestamp("2024-03-01", tz="UTC")
    storage = pd.DataFrame({"storage_bcf": [100.0]}, index=pd.DatetimeIndex([fri]))
    release = available_at(pd.DatetimeIndex([fri]), RULES["eia_storage"])[0]
    assert release > fri + pd.Timedelta(days=6)
    hidden = asof_frame(storage, pd.DatetimeIndex([fri + pd.Timedelta(days=6)]), RULES["eia_storage"])
    shown = asof_frame(storage, pd.DatetimeIndex([release]), RULES["eia_storage"])
    assert np.isnan(hidden["storage_bcf"].iloc[0])
    assert shown["storage_bcf"].iloc[0] == 100.0


def test_gas_threshold_ignores_later_prices():
    idx = _hours(6)
    price = pd.Series([1, 2, 3, 100, 100, 100], index=idx, dtype=float)
    flag, p90 = gas_spike_flag(price, idx[:3])
    assert p90 <= 3
    assert flag.iloc[-1] == 1.0
    # A higher later price does not move the percentile.
    price2 = price.copy()
    price2.iloc[-1] = 1e6
    _, p90_b = gas_spike_flag(price2, idx[:3])
    assert p90_b == p90


def test_imputer_uses_training_rows_only():
    idx = _hours(5)
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0]}, index=idx[:3])
    test = pd.DataFrame({"a": [1000.0, np.nan]}, index=idx[3:])
    filled, med = train_median_impute(train, [train, test], ["a"])
    assert med["a"] == pytest.approx(2.0)
    assert filled[0]["a"].iloc[1] == pytest.approx(2.0)
    assert filled[1]["a"].iloc[0] == pytest.approx(1000.0)
    assert filled[1]["a"].iloc[1] == pytest.approx(2.0)


def test_splits_must_be_chronological():
    idx = _hours(9)
    frame = pd.DataFrame({"y": np.arange(9)}, index=idx)
    assert_chronological(frame.iloc[:3], frame.iloc[3:6], frame.iloc[6:])
    with pytest.raises(AssertionError):
        assert_chronological(frame.iloc[:4], frame.iloc[3:6], frame.iloc[6:])


def test_conservative_set_drops_weather_gas_and_nrc_and_keeps_storage():
    columns = [
        "hubavg_lag_1h", "rt15_max", "da_HB_HUBAVG", "da_shape_ramp",
        "load_lag_1h", "load_actual_mw", "Houston_temperature_2m", "temp_max_across_zones",
        "stress_heat_wave", "henry_hub_price", "stress_gas_spike", "storage_bcf",
        "reactor_south_texas_1_pct", "stress_reactor_outage", "chronos_pred",
        "chronos_p_gt200", "hour_local", "miss_weather", "miss_gas", "miss_load",
        "miss_storage", "stress_score", "gfs_Houston_t_d1",
    ]
    kept = conservative_columns(columns)
    assert "storage_bcf" in kept and "miss_storage" in kept
    assert "chronos_pred" in kept and "load_lag_1h" in kept and "hour_local" in kept
    for banned in (
        "Houston_temperature_2m", "temp_max_across_zones", "stress_heat_wave",
        "henry_hub_price", "stress_gas_spike", "reactor_south_texas_1_pct",
        "stress_reactor_outage", "miss_weather", "miss_gas", "stress_score",
        "gfs_Houston_t_d1", "load_actual_mw",
    ):
        assert banned not in kept


def test_missing_weather_does_not_become_a_zero_stress_flag():
    idx = _hours(3)
    weather = pd.DataFrame({
        "Houston_temperature_2m": [np.nan, 10.0, 100.0],
        "Houston_wind_speed_10m": [np.nan, 1.0, 20.0],
    }, index=idx)
    out = lagged_weather(weather, idx, hours=1)
    assert np.isnan(out["stress_cold_snap"].iloc[1])
    assert np.isnan(out["Houston_temperature_2m"].iloc[1])
    assert out["Houston_temperature_2m"].iloc[2] == pytest.approx(10.0)


def test_calendar_peak_is_central_weekday_evening():
    # 2024-07-10 22:00 UTC is 17:00 CDT, a Wednesday.
    idx = pd.DatetimeIndex([pd.Timestamp("2024-07-10 22:00", tz="UTC")])
    out = calendar_central(idx)
    assert out["hour_local"].iloc[0] == 17
    assert out["is_peak_local"].iloc[0] == 1.0
    # 14:00 UTC is 09:00 CDT, which the UTC feature would have marked as peak.
    morning = calendar_central(pd.DatetimeIndex([pd.Timestamp("2024-07-10 14:00", tz="UTC")]))
    assert morning["is_peak_local"].iloc[0] == 0.0


def test_subsets_isolate_chronos_columns():
    cols = ["hubavg_lag_1h", "chronos_pred", "load_lag_1h", "chronos_spread"]
    parts = feature_subsets(cols)
    assert parts["xgb_exog"] == ["hubavg_lag_1h", "load_lag_1h"]
    assert parts["xgb_chronos"] == ["chronos_pred", "chronos_spread"]
    assert parts["xgb_both"] == cols


def test_dst_days_keep_four_intervals_per_utc_hour():
    spring = pd.date_range("2024-03-10", "2024-03-11", freq="15min", tz="US/Central", inclusive="left")
    fall = pd.date_range("2024-11-03", "2024-11-04", freq="15min", tz="US/Central", inclusive="left")
    assert len(spring) == 23 * 4
    assert len(fall) == 25 * 4
    for local in (spring, fall):
        counts = hourly_interval_counts(local.tz_convert("UTC"))
        assert (counts == 4).all()


def test_dst_transition_hours_are_counted_when_an_interval_is_missing():
    spring = pd.date_range("2024-03-10", "2024-03-11", freq="15min", tz="US/Central", inclusive="left")
    utc = spring.tz_convert("UTC")
    dropped = utc.delete(0)
    counts = hourly_interval_counts(dropped)
    assert (counts == 3).any()
    assert (counts == 4).sum() == len(counts) - 1
