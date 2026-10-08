"""Protocol decisions and information-set helpers for the honest MAE run."""

import numpy as np
import pandas as pd
import pytest

from models.chronos_hourly_features import context_bounds
from models.honest_mae import (
    clock_lag_frame,
    day_ahead_shape,
    leakage_test_pair,
    preferred_leakage,
    previous_hour_intervals,
)


def test_context_stops_before_the_target_hour():
    assert context_bounds(10, 4, 20) == (6, 10)
    with pytest.raises(AssertionError):
        context_bounds(3, 4, 20)
    with pytest.raises(AssertionError):
        context_bounds(20, 4, 20)


def test_clock_lag_is_one_hour_not_one_row():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    gap = idx.delete(1)
    df = pd.DataFrame({"a": [1, 3, 4]}, index=gap)
    out = clock_lag_frame(df, ["a"], 1)
    assert np.isnan(out["a"].iloc[0])
    assert np.isnan(out["a"].iloc[1])
    assert out["a"].iloc[2] == 3.0


def test_previous_hour_intervals_exclude_the_target_hour():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rt_idx = pd.date_range(start, periods=8, freq="15min", tz="UTC")
    rt = pd.Series([1.0, 2.0, 3.0, 10.0, 50.0, 60.0, 70.0, 80.0], index=rt_idx)
    hourly = pd.DatetimeIndex([start + pd.Timedelta(hours=1)])
    out = previous_hour_intervals(rt, hourly)
    assert out["rt15_last"].iloc[0] == 10.0
    assert out["rt15_max"].iloc[0] == 10.0
    assert out["rt15_min"].iloc[0] == 1.0
    assert out["rt15_mean"].iloc[0] == 4.0


def test_day_ahead_spread_does_not_use_the_current_realtime_price():
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    raw = pd.DataFrame({
        "HB_HUBAVG": np.arange(48, dtype=float),
        "da_HB_HUBAVG": np.full(48, 20.0),
    }, index=idx)
    base = day_ahead_shape(raw)
    bumped = raw.copy()
    bumped.iloc[-1, bumped.columns.get_loc("HB_HUBAVG")] = 999.0
    changed = day_ahead_shape(bumped)
    assert base["da_shape_spread_24h"].iloc[-1] == changed["da_shape_spread_24h"].iloc[-1]
    assert base["da_shape_ramp"].iloc[-1] == changed["da_shape_ramp"].iloc[-1]
    assert base["da_shape_day_max"].iloc[-1] == changed["da_shape_day_max"].iloc[-1]
    earlier = raw.copy()
    earlier.iloc[-2, earlier.columns.get_loc("HB_HUBAVG")] = 999.0
    assert base["da_shape_spread_24h"].iloc[-1] != day_ahead_shape(earlier)["da_shape_spread_24h"].iloc[-1]


def test_leakage_pair_is_fixed_before_the_test_year():
    rows = [
        {"name": "unshifted", "mae": 32.0},
        {"name": "weather_lag_1", "mae": 31.0},
        {"name": "chronos_lag_1", "mae": 1.0},
        {"name": "plus_features", "mae": 0.5},
    ]
    assert preferred_leakage(rows) == "weather_lag_1"
    assert leakage_test_pair("weather_lag_1") == ["weather_lag_1", "unshifted"]
    assert leakage_test_pair("unshifted") == ["unshifted", "weather_lag_1"]


def test_saved_xgboost_predicts_the_early_stopped_trees(tmp_path):
    import xgboost as xgb

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = x[:, 0] + rng.normal(scale=0.1, size=200)
    model = xgb.XGBRegressor(
        n_estimators=30, max_depth=2, learning_rate=0.1, tree_method="hist",
        device="cpu", early_stopping_rounds=5, eval_metric="mae", random_state=0)
    model.fit(x[:150], y[:150], eval_set=[(x[150:], y[150:])], verbose=False)
    model.save_model(tmp_path / "m.json")
    loaded = xgb.XGBRegressor()
    loaded.load_model(tmp_path / "m.json")
    end = int(model.best_iteration) + 1
    got = loaded.predict(x[:10], iteration_range=(0, end))
    expect = model.predict(x[:10], iteration_range=(0, end))
    assert np.allclose(got, expect)
