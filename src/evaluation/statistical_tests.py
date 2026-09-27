"""Paired moving-block bootstrap, Diebold-Mariano with HLN correction, Holm adjustment."""

from __future__ import annotations

import numpy as np
from scipy import stats


def block_indices(group_sizes: list[int], block: int, rng: np.random.Generator) -> np.ndarray:
    """One moving-block resample, drawn separately inside each contiguous group."""
    out, offset = [], 0
    for n in group_sizes:
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out.append(offset + np.minimum(idx, n - 1))
        offset += n
    return np.concatenate(out)


def paired_bootstrap(metric, y: np.ndarray, p1: np.ndarray, p2: np.ndarray, group_sizes: list[int],
                     block: int, n_boot: int, seed: int = 0) -> dict:
    """Bootstrap distribution of metric(y, p1) - metric(y, p2) on shared resamples."""
    rng = np.random.default_rng(seed)
    obs = metric(y, p1) - metric(y, p2)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = block_indices(group_sizes, block, rng)
        diffs[b] = metric(y[i], p1[i]) - metric(y[i], p2[i])
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, max(p_two, 1.0 / len(diffs)))), "n_boot": int(len(diffs))}


def bootstrap_ci(metric, y: np.ndarray, p: np.ndarray, group_sizes: list[int], block: int,
                 n_boot: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.array([metric(y[i], p[i]) for i in (block_indices(group_sizes, block, rng) for _ in range(n_boot))])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def diebold_mariano(loss1: np.ndarray, loss2: np.ndarray, lag: int) -> dict:
    """DM test of equal expected loss; Bartlett HAC variance; Harvey-Leybourne-Newbold correction.

    Negative statistic means model 1 has lower loss.
    """
    d = np.asarray(loss1, float) - np.asarray(loss2, float)
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma0 = np.dot(dc, dc) / n
    var = gamma0
    for k in range(1, lag + 1):
        g = np.dot(dc[k:], dc[:-k]) / n
        var += 2 * (1 - k / (lag + 1)) * g
    if var <= 0:
        return {"stat": float("nan"), "p_value": float("nan"), "mean_diff": float(dbar)}
    dm = dbar / np.sqrt(var / n)
    h = lag + 1
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"stat": float(stat), "p_value": float(p), "mean_diff": float(dbar)}


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
