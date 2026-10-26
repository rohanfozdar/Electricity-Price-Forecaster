"""Paired inference and metric definitions used by the PES 2027 package."""

import math

import numpy as np
import pandas as pd
import pytest

from models.pes2027.indep_stats import (
    bartlett_lrv, dm_test, holm_adjust, moving_block_draw, paired_block_bootstrap,
    paired_summary, point_metrics, segment_lengths,
)
from src.evaluation.statistical_tests import block_indices, diebold_mariano, holm, paired_bootstrap


def _ar1(n, phi=0.6, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_bartlett_lrv_matches_direct_sum():
    d = _ar1(2000)
    lag = 24
    n = len(d)
    x = d - d.mean()
    direct = x @ x / n + 2 * sum((1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n for k in range(1, lag + 1))
    assert bartlett_lrv(d, lag) == pytest.approx(direct, rel=1e-10)


def test_bartlett_lrv_matches_statsmodels_hac_of_the_mean():
    sm = pytest.importorskip("statsmodels.api")
    d = _ar1(3000, seed=3)
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 24, "use_correction": False})
    assert res.bse[0] ** 2 == pytest.approx(bartlett_lrv(d, 24) / len(d), rel=1e-8)


def test_dm_original_convention_equals_repository_function():
    a, b = np.abs(_ar1(5000, seed=1)), np.abs(_ar1(5000, seed=2))
    mine = dm_test(a, b, lag=24, horizon=25)
    theirs = diebold_mariano(a, b, 24)
    assert mine["stat"] == pytest.approx(theirs["stat"], rel=1e-10)
    assert mine["p"] == pytest.approx(theirs["p_value"], rel=1e-8)


def test_dm_hln_factor_for_one_step_horizon():
    a, b = np.abs(_ar1(1000, seed=4)), np.abs(_ar1(1000, seed=5))
    res = dm_test(a, b, lag=0, horizon=1)
    d = a - b
    raw = d.mean() / math.sqrt(np.var(d) / len(d))
    assert res["stat"] == pytest.approx(raw * math.sqrt((len(d) - 1) / len(d)), rel=1e-10)


def test_dm_sign_negative_when_first_loss_smaller():
    rng = np.random.default_rng(0)
    b = rng.uniform(1, 2, 500)
    a = b - 0.5 + rng.normal(0, 0.05, 500)
    assert dm_test(a, b, lag=5)["stat"] < 0


@pytest.mark.parametrize("sizes,block", [([8767], 24), ([30, 5, 100], 24), ([7], 24), ([48, 49], 12)])
def test_moving_block_draw_consumes_rng_like_the_original(sizes, block):
    r1, r2 = np.random.default_rng(11), np.random.default_rng(11)
    for _ in range(5):
        assert np.array_equal(moving_block_draw(sizes, block, r1), block_indices(sizes, block, r2))


def test_moving_block_draw_stays_inside_each_segment_and_has_full_length():
    rng = np.random.default_rng(0)
    sizes = [30, 5, 100]
    idx = moving_block_draw(sizes, 24, rng)
    assert len(idx) == sum(sizes)
    assert idx[:30].max() < 30 and idx[30:35].min() >= 30 and idx[30:35].max() < 35 and idx[35:].min() >= 35


def test_bootstrap_matches_original_on_paired_absolute_losses():
    rng = np.random.default_rng(1)
    y = rng.gamma(2, 20, 2000)
    p1, p2 = y + rng.normal(0, 5, 2000), y + rng.normal(0.3, 5, 2000)
    sizes = [2000]
    mine = paired_block_bootstrap(np.abs(p1 - y), np.abs(p2 - y), sizes, 24, 500, seed=0)
    theirs = paired_bootstrap(lambda yy, pp: float(np.mean(np.abs(pp - yy))), y, p1, p2, sizes, 24, 500, seed=0)
    assert mine["mean_diff"] == pytest.approx(theirs["diff"], abs=1e-12)
    assert mine["ci_low"] == pytest.approx(theirs["ci_low"], abs=1e-9)
    assert mine["ci_high"] == pytest.approx(theirs["ci_high"], abs=1e-9)
    assert mine["p_interval_inversion"] == pytest.approx(theirs["p_value"], abs=1e-12)


def test_bootstrap_p_value_is_interval_inversion():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 3000)
    b = a + 0.2 + rng.normal(0, 0.5, 3000)
    res = paired_block_bootstrap(a, b, [3000], 24, 1000, seed=0)
    excludes = res["ci_high"] < 0 or res["ci_low"] > 0
    assert excludes == (res["p_interval_inversion"] < 0.05)
    assert res["p_interval_inversion"] >= 1 / 1000


def test_bootstrap_constant_differential_is_degenerate():
    res = paired_block_bootstrap(np.full(500, 3.0), np.full(500, 1.0), [500], 24, 200, seed=0)
    assert res["ci_low"] == res["ci_high"] == res["mean_diff"] == 2.0


def test_segment_lengths_split_on_gaps_over_ninety_minutes():
    idx = pd.DatetimeIndex(list(pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"))
                           + list(pd.date_range("2024-01-01 07:00", periods=3, freq="h", tz="UTC")))
    assert segment_lengths(idx) == [5, 3]


def test_holm_matches_repository_and_known_case():
    p = [0.01, 0.04, 0.03, 0.2]
    assert holm_adjust(p) == pytest.approx(holm(p))
    assert holm_adjust(p) == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_point_metric_conventions():
    y = np.array([10.0, 200.0, 250.0, 500.0])
    pred = np.array([12.0, 201.0, 150.0, 600.0])
    m = point_metrics(y, pred)
    assert m["bias_pred_minus_y"] == pytest.approx(np.mean(pred - y))
    assert m["n_spikes"] == 2  # 200 is not a spike: the threshold is strict
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["spike_precision"] == 0.5 and m["spike_recall"] == 0.5


def test_seed_average_is_on_the_dollar_scale_and_differs_from_mean_of_maes():
    y = np.array([0.0, 100.0])
    seeds = np.array([[10.0, 50.0], [-10.0, 150.0]])
    mae_of_mean = np.mean(np.abs(seeds.mean(axis=0) - y))
    mean_of_mae = np.mean([np.mean(np.abs(s - y)) for s in seeds])
    assert mae_of_mean == 0.0 and mean_of_mae == 30.0
    decoded_mean = np.sinh(np.mean(np.arcsinh(seeds), axis=0))
    assert not np.allclose(decoded_mean, seeds.mean(axis=0))


def test_paired_summary_reports_both_dm_conventions():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=1000, freq="h", tz="UTC")
    y = rng.gamma(2, 10, 1000)
    s = paired_summary(y, y + rng.normal(0, 3, 1000), y + rng.normal(0, 3, 1000), idx, n_boot=200)
    assert {"dm_p_hln_h1", "dm_p_hln_h_lag_plus_1", "p_boot_interval_inversion", "p_boot_centered"} <= set(s)
    assert s["dm_lag"] == 24
