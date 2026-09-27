import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, load_yaml
from src.experiment import resolve_threshold, split_masks

CFG = load_yaml("configs/experiment_early_warning.yaml")
ORIGINS = pd.date_range("2016-02-01", "2024-12-31", freq="6h", tz="UTC")


@pytest.mark.parametrize("fold", CFG["folds"], ids=lambda f: f["name"])
@pytest.mark.parametrize("h", CFG["target"]["horizons_hours"])
def test_purged_splits_do_not_share_label_hours(fold, h):
    tr, va, te = split_masks(ORIGINS, h, fold, None)
    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()
    H = pd.Timedelta(hours=h)
    v0 = pd.Timestamp(fold["val"][0], tz="UTC")
    s0 = pd.Timestamp(fold["test"][0], tz="UTC")
    s1 = pd.Timestamp(fold["test"][1], tz="UTC")
    assert ORIGINS[tr].max() + H <= v0             # last train label hour < validation start
    assert ORIGINS[va].min() >= v0 and ORIGINS[va].max() + H <= s0
    assert ORIGINS[te].min() >= s0 and ORIGINS[te].max() + H <= s1


def test_period_start_restricts_train_and_val_only():
    fold = CFG["folds"][1]
    start = pd.Timestamp("2021-03-30", tz="UTC")
    tr, va, te = split_masks(ORIGINS, 48, fold, start)
    assert ORIGINS[tr].min() >= start and te.sum() > 0


def test_train_percentile_threshold_uses_train_hours_only():
    idx = pd.date_range("2021-01-01", "2023-12-31", freq="h", tz="UTC")
    y = pd.Series(np.where(idx.year == 2023, 1e4, 10.0), index=idx)
    assert resolve_threshold("train_p99", y, pd.Timestamp("2023-01-01", tz="UTC")) == 10.0


def test_chronos_context_ends_before_origin():
    spec = importlib.util.spec_from_file_location("rcc", ROOT / "scripts" / "run_chronos_cache.py")
    rcc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rcc)
    idx = pd.date_range("2020-01-01", periods=2000, freq="h", tz="UTC")
    y = pd.Series(np.arange(2000, dtype=float), index=idx)
    origins = rcc.origin_grid(idx, 6, "2020-01-10T00:00Z")
    ctx = rcc.contexts(y, origins, 512)
    pos = idx.get_indexer(origins)
    assert np.array_equal(ctx[:, -1], pos - 1)          # last context value is hour t-1
    assert np.isnan(rcc.contexts(y, origins[:1], 1024)[0, :100]).all()   # before series start: NaN, not wrapped
