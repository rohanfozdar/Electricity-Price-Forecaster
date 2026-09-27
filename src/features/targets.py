"""Issue-time grid, window labels, and exceedance events.

Convention: the hourly series is indexed by hour beginning (UTC). An origin t may use
y[h] for h <= t-1. The label window for horizon H is y[t], ..., y[t+H-1], i.e. the
real-time interval (t, t+H].
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def origin_grid(y: pd.Series, every_hours: int, start: str) -> pd.DatetimeIndex:
    first = max(pd.Timestamp(start), y.index.min())
    first = first.ceil(f"{every_hours}h")
    return pd.date_range(first, y.index.max(), freq=f"{every_hours}h", name="origin")


def window_matrix(y: pd.Series, origins: pd.DatetimeIndex, length: int, offset: int = 0) -> np.ndarray:
    """(n_origins, length) array of y[t+offset+k], k = 0..length-1; NaN outside the series."""
    pos = y.index.get_indexer(origins)
    if (pos < 0).any():
        raise ValueError("every origin must lie on the hourly index of y")
    vals = np.append(y.to_numpy(dtype=float), np.nan)
    idx = pos[:, None] + offset + np.arange(length)[None, :]
    idx = np.where((idx >= 0) & (idx < len(y)), idx, len(y))
    return vals[idx]


def window_max(y: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """max(y[t..t+H-1]); NaN if any hour in the window is missing or beyond the data."""
    w = window_matrix(y, origins, horizon)
    complete = ~np.isnan(w).any(axis=1)
    out = np.where(complete, np.nanmax(np.where(np.isnan(w), -np.inf, w), axis=1), np.nan)
    return pd.Series(out, index=origins, name=f"winmax_{horizon}h")


def window_labels(y: pd.Series, origins: pd.DatetimeIndex, horizon: int, threshold: float) -> pd.Series:
    m = window_max(y, origins, horizon)
    z = (m > threshold).astype(float).where(m.notna())
    return z.rename(f"z_{horizon}h")


def extract_events(y: pd.Series, threshold: float, gap_hours: int) -> pd.DataFrame:
    """Merge exceedance hours separated by at most `gap_hours` non-exceedance hours."""
    hot = y.index[(y > threshold).to_numpy()]
    if len(hot) == 0:
        return pd.DataFrame(columns=["onset", "end", "n_hours", "peak"])
    step = np.asarray((hot[1:] - hot[:-1]) / pd.Timedelta(hours=1))
    breaks = np.where(step > gap_hours + 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(hot) - 1]
    rows = []
    for s, e in zip(starts, ends):
        seg = y.loc[hot[s]:hot[e]]
        rows.append({"onset": hot[s], "end": hot[e], "n_hours": int((seg > threshold).sum()),
                     "peak": float(seg.max())})
    return pd.DataFrame(rows)
