import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.evaluation.metrics import average_precision
from src.evaluation.statistical_tests import diebold_mariano, holm, paired_bootstrap
from src.features.targets import window_labels
from src.models.chronos_features import LEVELS, ChronosCache, features, step_exceedance
from src.models.xgboost_alert import f1_threshold


def _cache(tmp_path, origins, kind="quantile"):
    n = len(origins)
    if kind == "quantile":
        base = np.arange(n, dtype=np.float32)[:, None, None]           # value encodes origin position
        vals = base + np.arange(144, dtype=np.float32)[None, :, None] * 0 + LEVELS[None, None, :].astype(np.float32)
    else:
        vals = np.repeat(np.arange(n, dtype=np.float32)[:, None, None], 5, axis=1).repeat(144, axis=2)
    p = tmp_path / f"c_{kind}.npz"
    np.savez(p, origins_ns=origins.as_unit("ns").asi8, values=vals, levels=LEVELS.astype(np.float32),
             meta=json.dumps({"kind": kind, "checkpoint": "test"}))
    return ChronosCache(p)


def _series(origins):
    idx = pd.date_range(origins[0] - pd.Timedelta(hours=600), origins[-1] + pd.Timedelta(hours=200), freq="h")
    return pd.Series(10.0, index=idx), pd.Series(10.0, index=idx)


def test_chronos_features_align_to_their_own_origin(tmp_path):
    origins = pd.date_range("2023-03-01", periods=40, freq="6h", tz="UTC")
    cache = _cache(tmp_path, origins)
    y, da = _series(origins)
    shuffled = origins[np.random.default_rng(0).permutation(len(origins))]
    z = window_labels(y, origins, 24, 200.0)
    f = features(cache, shuffled, 24, 200.0, y, da, z, "14:00", 6)
    expected = cache.origins.get_indexer(shuffled) + 0.5                # median level = position + 0.5
    assert np.allclose(f["ch_step1_q50"].to_numpy(), expected)


def test_recent_skill_uses_resolved_forecast(tmp_path):
    origins = pd.date_range("2023-03-01", periods=40, freq="6h", tz="UTC")
    cache = _cache(tmp_path, origins, kind="samples")
    y, da = _series(origins)
    z = window_labels(y, origins, 24, 200.0)
    f = features(cache, origins, 24, 5.0, y, da, z, "14:00", 6)
    assert f["ch_recent_brier"].iloc[:4].isna().all()                   # t-24h not in grid yet
    assert f["ch_recent_brier"].iloc[4:].notna().all()


def test_step_exceedance_monotone_and_consistent():
    q = np.sort(np.random.default_rng(0).gamma(2, 30, (50, 9)), axis=1)
    thr = np.linspace(-50, 600, 60)
    p = np.stack([step_exceedance(q, t) for t in thr], axis=1)
    assert (np.diff(p, axis=1) <= 1e-12).all()
    at_median = [step_exceedance(q[i:i + 1], q[i, 4] + 1e-9)[0] for i in range(len(q))]
    assert np.allclose(at_median, 0.5, atol=1e-6)


def test_average_precision_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500), 2)
    assert np.isclose(average_precision(y, p), average_precision_score(y, p))


def test_f1_threshold_on_separable_data():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert f1_threshold(p, np.array([0, 0, 1, 1])) == 0.8


def test_statistics_sanity():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    p = rng.random(400)
    b = paired_bootstrap(average_precision, y, p, p, [200, 200], 28, 200)
    assert b["diff"] == 0 and b["ci_low"] == 0 and b["ci_high"] == 0
    d = diebold_mariano(rng.random(300), rng.random(300) + 0.5, 4)
    assert d["stat"] < 0 and d["p_value"] < 0.01
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
