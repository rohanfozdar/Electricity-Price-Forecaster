"""Metrics and paired inference written independently of src/evaluation and models/run_*.

The moving-block resampler consumes the random generator in the same order as
src.evaluation.statistical_tests.block_indices, so saved intervals can be
reproduced exactly. The arithmetic is otherwise separate: the bootstrap works on
the loss differential directly and the HAC variance is computed by FFT.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

SPIKE_PRICE = 200.0


def point_metrics(y, pred, spike: float = SPIKE_PRICE) -> dict:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if y.shape != pred.shape or not (np.isfinite(y).all() and np.isfinite(pred).all()):
        raise ValueError("targets and forecasts must be finite and the same length")
    err = pred - y
    abs_err = np.abs(err)
    true_s = y > spike
    pred_s = pred > spike
    tp = int(np.sum(true_s & pred_s))
    fp = int(np.sum(~true_s & pred_s))
    fn = int(np.sum(true_s & ~pred_s))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    if tp + fp and tp + fn:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    else:
        f1 = float("nan")
    stress = y > 100.0
    return {
        "n": int(len(y)),
        "mae": float(abs_err.mean()),
        "rmse": float(math.sqrt(np.mean(err * err))),
        "medae": float(np.median(abs_err)),
        "bias_pred_minus_y": float(err.mean()),
        "n_spikes": int(true_s.sum()),
        "n_pred_spikes": int(pred_s.sum()),
        "tp": tp, "fp": fp, "fn": fn,
        "spike_precision": float(precision),
        "spike_recall": float(recall),
        "spike_f1": float(f1),
        "spike_mae": float(abs_err[true_s].mean()) if true_s.any() else float("nan"),
        "nonspike_mae": float(abs_err[~true_s].mean()) if (~true_s).any() else float("nan"),
        "stress_mae_y_gt_100": float(abs_err[stress].mean()) if stress.any() else float("nan"),
        "normal_mae_y_le_100": float(abs_err[~stress].mean()) if (~stress).any() else float("nan"),
        "pred_max": float(pred.max()),
        "y_max": float(y.max()),
    }


def segment_lengths(index) -> list[int]:
    """Lengths of runs whose consecutive spacing is at most 1.5 hours."""
    ns = np.asarray(index.asi8, dtype=np.int64)
    if len(ns) == 0:
        return []
    breaks = np.flatnonzero(np.diff(ns) > int(1.5 * 3600 * 1e9))
    edges = np.concatenate([[0], breaks + 1, [len(ns)]])
    return [int(b - a) for a, b in zip(edges[:-1], edges[1:])]


def moving_block_draw(sizes: list[int], block: int, rng: np.random.Generator) -> np.ndarray:
    """Non-circular moving blocks inside each segment, truncated to the segment length.

    Block starts are uniform on [0, n - block]. When a segment is shorter than a
    block, the start is 0 and positions past the end are clamped to the last row.
    """
    pieces, offset = [], 0
    for n in sizes:
        k = -(-n // block)
        upper = n - block + 1 if n - block + 1 > 0 else 1
        starts = rng.integers(0, upper, size=k)
        pos = np.add.outer(starts, np.arange(block)).reshape(-1)[:n]
        pieces.append(offset + np.where(pos > n - 1, n - 1, pos))
        offset += n
    return np.concatenate(pieces)


def paired_block_bootstrap(loss_a, loss_b, sizes: list[int], block: int = 24,
                           n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile interval for mean(loss_a - loss_b) from shared block resamples.

    `p_interval_inversion` is 2 * min(P*(d <= 0), P*(d >= 0)), floored at 1/B. It is
    the smallest two-sided level at which the percentile interval excludes zero.
    `p_centered` is the share of centered draws at least as far from zero as the
    observed mean. It is reported as a supplementary convention only.
    """
    d = np.asarray(loss_a, np.float64) - np.asarray(loss_b, np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = d[moving_block_draw(sizes, block, rng)].mean()
    obs = float(d.mean())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    p_inv = 2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0)))
    p_inv = min(1.0, max(p_inv, 1.0 / n_boot))
    centered = draws - draws.mean()
    p_cen = float(np.mean(np.abs(centered) >= abs(obs)))
    return {
        "mean_diff": obs, "ci_low": float(lo), "ci_high": float(hi),
        "p_interval_inversion": float(p_inv), "p_centered": max(p_cen, 1.0 / n_boot),
        "n_boot": int(n_boot), "block_hours": int(block), "seed": int(seed),
        "boot_sd": float(draws.std(ddof=1)),
        "draws": draws,
    }


def bartlett_lrv(d: np.ndarray, lag: int) -> float:
    """Bartlett-weighted long-run variance; autocovariances divided by n, via FFT."""
    d = np.asarray(d, np.float64)
    n = len(d)
    x = d - d.mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, size)
    acov = np.fft.irfft(f * np.conj(f), size)[: lag + 1] / n
    w = 1.0 - np.arange(lag + 1) / (lag + 1.0)
    return float(acov[0] + 2.0 * np.sum(w[1:] * acov[1:]))


def dm_test(loss_a, loss_b, lag: int = 24, horizon: int = 1) -> dict:
    """Diebold-Mariano with Bartlett HAC (bandwidth `lag`) and the HLN factor for `horizon`.

    Reference distribution t(n-1). Negative statistic means loss_a is smaller.
    """
    d = np.asarray(loss_a, np.float64) - np.asarray(loss_b, np.float64)
    n = len(d)
    lrv = bartlett_lrv(d, lag)
    if lrv <= 0:
        return {"stat": float("nan"), "p": float("nan"), "lrv": lrv}
    raw = d.mean() / math.sqrt(lrv / n)
    h = horizon
    factor = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = raw * factor
    p = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return {"stat": float(stat), "p": float(p), "stat_unadjusted": float(raw),
            "p_normal_unadjusted": float(2.0 * stats.norm.sf(abs(raw))),
            "lrv": lrv, "lag": int(lag), "hln_horizon": int(h), "n": int(n)}


def holm_adjust(pvalues) -> list[float]:
    p = np.asarray(pvalues, np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_summary(y, pred_a, pred_b, index, block: int = 24, n_boot: int = 2000,
                   seed: int = 0, dm_lag: int | None = None) -> dict:
    """Everything reported for one paired comparison of absolute losses (a minus b)."""
    y = np.asarray(y, np.float64)
    la = np.abs(np.asarray(pred_a, np.float64) - y)
    lb = np.abs(np.asarray(pred_b, np.float64) - y)
    sizes = segment_lengths(index)
    boot = paired_block_bootstrap(la, lb, sizes, block, n_boot, seed)
    lag = block if dm_lag is None else dm_lag
    dm1 = dm_test(la, lb, lag=lag, horizon=1)
    dm_orig = dm_test(la, lb, lag=lag, horizon=lag + 1)
    return {
        "n": int(len(y)), "segments": sizes,
        "mae_a": float(la.mean()), "mae_b": float(lb.mean()),
        "gap_a_minus_b": boot["mean_diff"],
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "p_boot_interval_inversion": boot["p_interval_inversion"],
        "p_boot_centered": boot["p_centered"],
        "boot_sd": boot["boot_sd"],
        "block_hours": block, "n_boot": n_boot, "boot_seed": seed,
        "dm_lag": lag,
        "dm_stat_hln_h1": dm1["stat"], "dm_p_hln_h1": dm1["p"],
        "dm_stat_hln_h_lag_plus_1": dm_orig["stat"], "dm_p_hln_h_lag_plus_1": dm_orig["p"],
        "interval_excludes_zero": bool(boot["ci_high"] < 0 or boot["ci_low"] > 0),
    }
