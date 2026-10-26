"""Incomplete hours, the frozen index, split disjointness, and paired origins."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "pit_verified" / "predictions"
DATA = ROOT / "data" / "raw" / "ercot_rt_prices.parquet"
need_data = pytest.mark.skipif(not (DATA.exists() and PRED.exists()), reason="archived inputs not present")


def test_hourly_maximum_keeps_incomplete_hours_and_counts_intervals():
    from models.pes2027.common import hourly_max, interval_counts

    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:30", "2024-01-01 00:45",
                            "2024-01-01 01:00"], tz="UTC")
    rt = pd.Series([5.0, 9.0, 7.0, 1.0, 3.0], index=idx)
    h, c = hourly_max(rt), interval_counts(rt)
    assert h.tolist() == [9.0, 3.0]
    assert c.tolist() == [4, 1]


def test_calendar_is_market_local_across_dst():
    from src.features.pit_sets import calendar_central

    spring = pd.date_range(pd.Timestamp("2024-03-10", tz="US/Central"), periods=23, freq="h").tz_convert("UTC")
    fall = pd.date_range(pd.Timestamp("2024-11-03", tz="US/Central"), periods=25, freq="h").tz_convert("UTC")
    assert 2 not in calendar_central(spring)["hour_local"].tolist()
    assert calendar_central(fall)["hour_local"].tolist().count(1) == 2


@need_data
def test_frozen_index_and_incomplete_hours():
    from models.pes2027.common import FROZEN_HASH, index_hash, interval_counts, raw_rt

    test = pd.read_parquet(PRED / "delay_1h__max__asinh__xgb_exog__seed0.parquet").index
    assert len(test) == 8767 and index_hash(test) == FROZEN_HASH
    counts = interval_counts(raw_rt()["HB_HUBAVG"])
    short = counts.reindex(test).fillna(0)
    assert short[short < 4].index.tolist() == [pd.Timestamp("2024-12-31 06:00", tz="UTC")]
    calendar = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    absent = calendar.difference(test)
    assert len(absent) == 17 and absent.min() == pd.Timestamp("2024-12-31 07:00", tz="UTC")


@need_data
def test_paired_models_share_origins_and_targets_in_every_saved_scenario():
    prefixes = sorted({p.name.split("__xgb_")[0] for p in PRED.glob("*__xgb_both__seed*.parquet")})
    assert prefixes
    for prefix in prefixes:
        seeds = (42,) if prefix.startswith("crisis2021") else (0, 1, 2, 3, 42)
        frames = [pd.read_parquet(PRED / f"{prefix}__xgb_{m}__seed{s}.parquet") for m in ("both", "exog") for s in seeds]
        for f in frames[1:]:
            assert f.index.equals(frames[0].index), prefix
            np.testing.assert_array_equal(f["y_true"].to_numpy(), frames[0]["y_true"].to_numpy())


@need_data
def test_primary_splits_are_disjoint_and_ordered():
    import json

    split = json.loads((ROOT / "runs" / "pit_verified" / "splits" / "delay_1h.json").read_text())
    assert (split["train_rows"], split["val_rows"], split["test_rows"]) == (34850, 8760, 8767)
    rerun = ROOT / "runs" / "pes2027" / "E0-REPRO" / "rerun" / "rerun_summary.json"
    if rerun.exists():
        check = json.loads(rerun.read_text())["split_check"]
        assert check["train_max_lt_val_min"] and check["val_max_lt_test_min"]


@need_data
def test_saved_per_origin_seed_average_is_mean_of_dollar_forecasts():
    path = ROOT / "runs" / "pes2027" / "E0-REPRO" / "primary_per_origin.parquet"
    if not path.exists():
        pytest.skip("E0 not run")
    po = pd.read_parquet(path)
    for m in ("exog", "both"):
        cols = [f"pred_xgb_{m}_seed{s}" for s in (0, 1, 2, 3, 42)]
        np.testing.assert_allclose(po[f"pred_xgb_{m}_seed_avg"], po[cols].mean(axis=1), rtol=0, atol=1e-9)
