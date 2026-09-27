"""Point-in-time guarantees: no feature at origin t can change when data after its
publication time is altered."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_yaml
from src.data.asof import asof_join, available_at, dam_known_until
from src.features import build

RULES = load_yaml("configs/feature_availability.yaml")["sources"]
UTC = "UTC"


def _hourly(n=24 * 60, start="2023-01-01", seed=0):
    idx = pd.date_range(start, periods=n, freq="h", tz=UTC)
    return pd.Series(np.random.default_rng(seed).gamma(2, 20, n), index=idx)


def test_asof_join_never_uses_unpublished_rows():
    idx = pd.date_range("2023-01-06", periods=30, freq="7D", tz=UTC)       # Fridays
    s = pd.Series(np.arange(30.0), index=idx, name="v")
    origins = pd.date_range("2023-01-01", "2023-07-01", freq="6h", tz=UTC)
    out = asof_join(s, origins, RULES["eia_storage"])
    avail = available_at(idx, RULES["eia_storage"])
    for t, v in zip(origins, out["v"]):
        if np.isnan(v):
            assert (avail > t).all()
        else:
            assert avail[int(v)] <= t
            later = avail[int(v) + 1] if int(v) + 1 < len(avail) else None
            assert later is None or later > t


def test_storage_release_is_after_the_thursday_report():
    fri = pd.DatetimeIndex([pd.Timestamp("2024-03-01", tz=UTC)])
    a = available_at(fri, RULES["eia_storage"])[0]
    assert a >= pd.Timestamp("2024-03-07 15:30", tz=UTC)                  # Thursday 10:30 ET


def test_gdelt_day_not_visible_until_next_utc_day():
    day = pd.DatetimeIndex([pd.Timestamp("2023-05-10", tz=UTC)])
    a = available_at(day, RULES["gdelt"])[0]
    assert a > pd.Timestamp("2023-05-10 23:59", tz=UTC)


def test_eia_spot_weekly_release():
    mon = pd.DatetimeIndex([pd.Timestamp("2024-03-04", tz=UTC)])        # Monday trade date
    a = available_at(mon, RULES["eia_gas"])[0]
    assert a.dayofweek == 2 and a >= mon[0] + pd.Timedelta(days=5)


def test_dam_publication_boundary():
    # 2024-07-10 13:00 CDT = 18:00 UTC: tomorrow's DAM not yet published
    before = pd.DatetimeIndex([pd.Timestamp("2024-07-10 18:00", tz=UTC)])
    after = pd.DatetimeIndex([pd.Timestamp("2024-07-10 19:00", tz=UTC)])  # 14:00 CDT
    end_before = dam_known_until(before, RULES["ercot_da_prices"]["publish_local"])[0]
    end_after = dam_known_until(after, RULES["ercot_da_prices"]["publish_local"])[0]
    assert end_before == pd.Timestamp("2024-07-11 05:00", tz=UTC)        # end of Jul 10 local
    assert end_after == pd.Timestamp("2024-07-12 05:00", tz=UTC)         # end of Jul 11 local
    hour = pd.DatetimeIndex([pd.Timestamp("2024-07-11 12:00", tz=UTC)])
    assert available_at(hour, RULES["ercot_da_prices"])[0] == pd.Timestamp("2024-07-10 19:00", tz=UTC)


@pytest.mark.parametrize("family", ["price", "load", "weather", "day_ahead"])
def test_features_invariant_to_future_perturbation(family):
    y = _hourly()
    da = _hourly(seed=1)
    load = _hourly(seed=2) * 500
    wx = pd.DataFrame({f"{z}_{v}": _hourly(seed=3 + i) for i, (z, v) in enumerate(
        [(z, v) for z in ("Houston", "West (Midland)") for v in ("temperature_2m", "wind_speed_10m")])})
    origins = pd.date_range("2023-01-20", "2023-02-20", freq="6h", tz=UTC)
    pub = RULES["ercot_da_prices"]["publish_local"]

    def compute(y_, da_, load_, wx_):
        if family == "price":
            return build.price_history(y_, origins)
        if family == "load":
            return build.load_features(load_, origins)
        if family == "weather":
            return build.weather_obs(wx_, origins, 1)
        return pd.concat([build.day_ahead_common(y_, da_, origins, pub),
                          build.day_ahead_window(da_, origins, 72, pub)], axis=1)

    base = compute(y, da, load, wx)
    rng = np.random.default_rng(9)
    for t in origins[::7]:
        known_end = dam_known_until(pd.DatetimeIndex([t]), pub)[0]
        y2, da2, l2, w2 = y.copy(), da.copy(), load.copy(), wx.copy()
        y2[y2.index >= t] = rng.normal(1e4, 1, (y2.index >= t).sum())
        l2[l2.index >= t] = -1.0
        w2.loc[w2.index >= t] = 999.0
        da2[da2.index >= known_end] = 5e3
        pert = compute(y2, da2, l2, w2)
        pd.testing.assert_series_equal(base.loc[t], pert.loc[t], check_names=False)


def test_forecast_cube_uses_only_issued_runs():
    k = np.arange(144)
    leads = build.lead_day_for_offset(k, 6)
    issue_time_offsets = k - 24 * leads          # hours relative to origin at which the run was valid-24N
    assert (issue_time_offsets <= -6).all()
    assert leads.max() <= 7
