"""Chronos meta-features from a forecast cache written by scripts/run_chronos_cache.py.

The forecast issued at t covers hours t, t+1, ..., t+143 (step k = hour t+k), from a
context that ends at hour t-1. Features for horizon H use steps 0..H-1 only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.asof import dam_known_until
from src.features.targets import window_matrix

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


class ChronosCache:
    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=False)
        self.path = Path(path)
        self.meta = json.loads(str(z["meta"]))
        self.kind = self.meta["kind"]
        self.origins = pd.DatetimeIndex(pd.to_datetime(z["origins_ns"], unit="ns", utc=True))
        v = z["values"].astype(np.float64)
        if self.kind == "samples":
            self.samples = v                                    # (n, S, H)
            self.quantiles = np.moveaxis(np.quantile(v, LEVELS, axis=1), 0, -1)   # (n, H, 9)
        else:
            self.samples = None
            self.quantiles = np.sort(v, axis=-1)                # guard against crossing
        self.horizon = self.quantiles.shape[1]

    def align(self, origins: pd.DatetimeIndex) -> np.ndarray:
        pos = self.origins.get_indexer(origins)
        return pos


def step_exceedance(q: np.ndarray, threshold: float) -> np.ndarray:
    """P(y_step > threshold) from 9 quantiles per step, with exponential tails past q10/q90."""
    below = (q < threshold).sum(axis=-1)                        # quantiles strictly below thr
    q10, q50, q90 = q[..., 0], q[..., 4], q[..., 8]
    j = np.clip(below, 1, len(LEVELS) - 1)
    lo = np.take_along_axis(q, (j - 1)[..., None], axis=-1)[..., 0]
    hi = np.take_along_axis(q, j[..., None], axis=-1)[..., 0]
    frac = np.where(hi > lo, (threshold - lo) / np.where(hi > lo, hi - lo, 1), 0.5)
    cdf_mid = LEVELS[j - 1] + np.clip(frac, 0, 1) * (LEVELS[j] - LEVELS[j - 1])
    upper = 0.1 * np.exp(-(threshold - q90) / np.maximum(q90 - q50, 1.0))
    lower = 1 - 0.1 * np.exp(-(q10 - threshold) / np.maximum(q50 - q10, 1.0))
    p = np.where(below == 0, lower, np.where(below == len(LEVELS), upper, 1 - cdf_mid))
    return np.clip(p, 0.0, 1.0)


def features(cache: ChronosCache, origins: pd.DatetimeIndex, horizon: int, threshold: float,
             y: pd.Series, da: pd.Series, labels_prev: pd.Series, publish_local: str,
             every_hours: int) -> pd.DataFrame:
    """Per-origin Chronos features for one horizon.

    labels_prev: window labels z(t', H) indexed by origin; the feature at t uses the
    forecast and outcome of t' = t - H, whose window [t-H, t-1] has resolved by t.
    """
    if horizon > cache.horizon:
        raise ValueError(f"cache covers {cache.horizon} h < {horizon} h")
    pos = cache.align(origins)
    ok = pos >= 0
    q = np.full((len(origins), horizon, len(LEVELS)), np.nan)
    q[ok] = cache.quantiles[pos[ok], :horizon, :]
    q10, q50, q90 = q[..., 0], q[..., 4], q[..., 8]
    p_step = step_exceedance(q, threshold)
    p_step[~ok] = np.nan

    f = pd.DataFrame(index=origins)
    f["ch_q50_max"] = q50.max(axis=1)
    f["ch_q90_max"] = q90.max(axis=1)
    f["ch_q50_mean"] = q50.mean(axis=1)
    f["ch_spread_mean"] = (q90 - q10).mean(axis=1)
    f["ch_pstep_max"] = p_step.max(axis=1)
    f["ch_pwin_indep"] = 1 - np.prod(1 - np.clip(p_step, 0, 0.999), axis=1)
    f["ch_step1_q50"] = q50[:, 0]

    known_end = dam_known_until(origins, publish_local)
    n_known = np.clip(((known_end - origins) / pd.Timedelta(hours=1)).to_numpy(), 0, horizon).astype(int)
    da_ext = da.reindex(pd.date_range(da.index.min(), da.index.max() + 400 * pd.Timedelta(hours=1), freq="h"))
    dw = window_matrix(da_ext, origins, horizon)
    dw = np.where(np.arange(horizon)[None, :] < n_known[:, None], dw, np.nan)
    with np.errstate(all="ignore"):
        diff = np.where(np.isnan(dw), np.nan, q50 - dw)
        f["ch_minus_da"] = np.where(np.isnan(diff).all(axis=1), np.nan, np.nanmean(diff, axis=1))

    if cache.kind == "samples":
        smp = np.full((len(origins), cache.samples.shape[1], horizon), np.nan)
        smp[ok] = cache.samples[pos[ok], :, :horizon]
        f["ch_pathmax_p"] = np.where(ok, (smp.max(axis=2) > threshold).mean(axis=1), np.nan)

    # recent skill: forecast issued at t-H, whose window [t-H, t-1] is complete at t
    prev = origins - pd.Timedelta(hours=horizon)
    prev_feat = pd.Series(f["ch_pstep_max"].to_numpy(), index=origins).reindex(prev).to_numpy()
    prev_q50max = pd.Series(f["ch_q50_max"].to_numpy(), index=origins).reindex(prev).to_numpy()
    z_prev = labels_prev.reindex(prev).to_numpy()
    f["ch_recent_brier"] = (prev_feat - z_prev) ** 2
    realized_prev = window_matrix(y, origins, horizon, offset=-horizon)
    with np.errstate(all="ignore"):
        rmax = np.where(np.isnan(realized_prev).any(axis=1), np.nan, np.nanmax(realized_prev, axis=1))
    f["ch_recent_logerr"] = np.log1p(np.clip(rmax, 0, None)) - np.log1p(np.clip(prev_q50max, 0, None))
    f.loc[~ok, :] = np.nan
    return f


def chronos_only_score(f: pd.DataFrame) -> pd.Series:
    """Raw Chronos exceedance score. T5 caches use the exact path-max probability,
    Bolt caches the max per-step probability; ties broken by q90 path max."""
    base = f["ch_pathmax_p"] if "ch_pathmax_p" in f else f["ch_pstep_max"]
    tie = np.clip(f["ch_q90_max"].to_numpy() / 10000.0, 0, 1) * 1e-3
    return (base + tie).rename("score")
