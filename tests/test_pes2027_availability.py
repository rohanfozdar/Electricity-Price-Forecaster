"""Future-perturbation invariance for every gated source, on the real archived inputs.

For an issue time T and delay d, values that are not yet usable at T are replaced with
noise. Every feature at T must be unchanged; the target at T must change, which shows
that the perturbation reached the data. Frames are built on a contiguous window around
T because some day-ahead features aggregate over the rows of the index.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "ercot_rt_prices.parquet"
CACHE = ROOT / "runs" / "pit_verified" / "chronos" / "delay_1h.parquet"
pytestmark = pytest.mark.skipif(not (DATA.exists() and CACHE.exists()), reason="archived inputs not present")

ISSUES = pd.DatetimeIndex(["2024-03-10 08:00", "2024-06-15 12:00", "2024-11-03 07:00", "2024-08-20 22:00",
                           "2023-02-03 05:00"], tz="UTC")


def _window(t):
    return pd.date_range(t - pd.Timedelta(hours=48), t + pd.Timedelta(hours=24), freq="h", tz="UTC")


@pytest.fixture(scope="module")
def sources():
    from models.run_pit_ablation import load_sources
    from models.run_pit_verified import study_sources

    s = study_sources(load_sources())
    s["cache"] = pd.read_parquet(CACHE)
    return s


def _row(s, t, delay, rt=None, da=None, load=None, storage=None):
    from src.features.pit_latency import build_delayed_frame

    frame, _ = build_delayed_frame(
        _window(t), s["rt"]["HB_HUBAVG"] if rt is None else rt, s["da"] if da is None else da,
        s["load"] if load is None else load, s["storage"] if storage is None else storage,
        s["gas"], s["nrc"], s["cache"], price_delay_hours=delay, load_delay_hours=delay,
        compute_hash=False, chronos_delay_hours=0)
    return frame.loc[[t]]


def _noise(values, rng):
    return values * rng.uniform(5, 50, size=values.shape) + rng.uniform(100, 9000, size=values.shape)


def _same_features(a, b):
    pd.testing.assert_frame_equal(a.drop(columns=["HB_HUBAVG"]), b.drop(columns=["HB_HUBAVG"]), check_exact=True)


@pytest.mark.parametrize("delay", [1, 6])
def test_realtime_prices_after_cutoff_do_not_move_features(sources, delay):
    rng = np.random.default_rng(delay)
    for t in ISSUES:
        rt = sources["rt"]["HB_HUBAVG"].copy()
        late = rt.index >= t - pd.Timedelta(hours=delay)
        rt[late] = _noise(rt[late].to_numpy(), rng)
        base, moved = _row(sources, t, delay), _row(sources, t, delay, rt=rt)
        _same_features(base, moved)
        assert base["HB_HUBAVG"].iloc[0] != moved["HB_HUBAVG"].iloc[0]


@pytest.mark.parametrize("delay", [1, 6])
def test_load_after_cutoff_does_not_move_features(sources, delay):
    rng = np.random.default_rng(10 + delay)
    for t in ISSUES:
        load = sources["load"].copy()
        late = load.index >= t - pd.Timedelta(hours=delay)
        load[late] = _noise(load[late].to_numpy(), rng)
        _same_features(_row(sources, t, delay), _row(sources, t, delay, load=load))


def test_day_ahead_rows_published_after_issue_do_not_move_features(sources):
    from src.data.asof import available_at
    from src.features.pit_sets import RULES

    rng = np.random.default_rng(3)
    avail = available_at(sources["da"].index, RULES["ercot_da_prices"])
    for t in ISSUES:
        da = sources["da"].copy()
        late = np.asarray(avail > t)
        assert late.any()
        da.loc[late] = _noise(da.loc[late].to_numpy(float), rng)
        _same_features(_row(sources, t, 1), _row(sources, t, 1, da=da))


def test_storage_rows_released_after_issue_do_not_move_features(sources):
    from src.data.asof import available_at
    from src.features.pit_sets import RULES

    rng = np.random.default_rng(4)
    avail = available_at(sources["storage"].index, RULES["eia_storage"])
    for t in ISSUES:
        st = sources["storage"].astype(float).copy()
        late = np.asarray(avail > t)
        st.loc[late] = _noise(st.loc[late].to_numpy(float), rng)
        _same_features(_row(sources, t, 1), _row(sources, t, 1, storage=st))


@pytest.mark.parametrize("delay", [1, 6])
def test_new_e1_and_e3_blocks_ignore_data_after_cutoff(sources, delay):
    from models.pes2027.common import hourly_max
    from models.pes2027.features_extra import lear_design, representation
    from src.data.asof import available_at
    from src.features.pit_sets import RULES

    rng = np.random.default_rng(20 + delay)
    avail = available_at(sources["da"].index, RULES["ercot_da_prices"])
    for t in ISSUES:
        rt = sources["rt"]["HB_HUBAVG"].copy()
        late = rt.index >= t - pd.Timedelta(hours=delay)
        rt[late] = _noise(rt[late].to_numpy(), rng)
        da = sources["da"].copy()
        da.loc[np.asarray(avail > t)] = 1e6
        h0, h1 = hourly_max(sources["rt"]["HB_HUBAVG"]), hourly_max(rt)
        pd.testing.assert_frame_equal(representation("R3", h0, _window(t), delay).loc[[t]],
                                      representation("R3", h1, _window(t), delay).loc[[t]], check_exact=True)
        pd.testing.assert_frame_equal(lear_design(h0, sources["da"], _window(t), delay).loc[[t]],
                                      lear_design(h1, da, _window(t), delay).loc[[t]], check_exact=True)


def test_dense_lags_agree_with_original_price_history(sources):
    from models.pes2027.common import hourly_max
    from models.pes2027.features_extra import dense_lags
    from src.features.pit_sets import price_history

    h = hourly_max(sources["rt"]["HB_HUBAVG"])
    ours = dense_lags(h, ISSUES, 1, lags=(1, 12, 24, 48, 168))
    orig = price_history(h, ISSUES, delay=pd.Timedelta(hours=1))
    for k in (1, 12, 24, 48, 168):
        np.testing.assert_array_equal(ours[f"hubavg_lag_{k}h"].to_numpy(), orig[f"hubavg_lag_{k}h"].to_numpy())


def test_chronos_context_ignores_prices_after_cutoff(sources):
    from models.chronos_delayed_cache import CONTEXT, _contexts
    from models.pes2027.common import hourly_max

    t = ISSUES[1]
    windows = []
    for bump in (0.0, 1000.0):
        rt = sources["rt"]["HB_HUBAVG"].copy()
        rt[rt.index >= t - pd.Timedelta(hours=1)] += bump
        grid, values, positions, end_pos = _contexts(hourly_max(rt), 1)
        p = int(np.flatnonzero(grid == t)[0])
        assert grid[end_pos[p]] == t - pd.Timedelta(hours=2)
        windows.append(values[end_pos[p] - CONTEXT + 1:end_pos[p] + 1])
    np.testing.assert_array_equal(windows[0], windows[1])
