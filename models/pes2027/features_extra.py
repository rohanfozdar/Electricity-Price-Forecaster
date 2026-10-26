"""Extra feature blocks for E1 (historical coverage) and E3 (out-of-fold LEAR-style forecast).

Every block is a function of hourly prices that end at t - (1 + d), the same endpoint as
the original price history and the Chronos context.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.features.pit_sets import calendar_central, clock_shift, day_ahead_features

ASINH_CLIP = 10.0
R1_LAGS = tuple(range(1, 25)) + (48, 72, 96, 120, 144, 168, 336, 504)
WINDOWS = (72, 512)
LEAR_LAGS = tuple(range(1, 25)) + (48, 168)
LEAR_ALPHAS = tuple(float(a) for a in np.geomspace(1e-6, 10, 15))


def _grid(hourly: pd.Series) -> pd.Series:
    hourly = hourly.sort_index().astype(float)
    grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", tz=hourly.index.tz)
    return hourly.reindex(grid)


def dense_lags(hourly: pd.Series, index: pd.DatetimeIndex, delay_hours: int, lags=R1_LAGS) -> pd.DataFrame:
    """hubavg_lag_{k}h = hourly maximum at t - (k + d). Lag 1 is the latest legal hour."""
    out = pd.DataFrame(index=index)
    for k in lags:
        out[f"hubavg_lag_{k}h"] = clock_shift(hourly.sort_index().astype(float), k + delay_hours, index).to_numpy()
    return out


def window_summaries(hourly: pd.Series, index: pd.DatetimeIndex, delay_hours: int,
                     windows=WINDOWS) -> pd.DataFrame:
    """Summaries of the w clock hours ending at t - (1 + d). Missing hours are skipped.

    A summary is missing when fewer than half of the window's hours are observed.
    """
    series = _grid(hourly)
    above = series.gt(200.0).astype(float).where(series.notna())
    out = pd.DataFrame(index=index)
    for w in windows:
        roll = series.rolling(w, min_periods=w // 2)
        stats = {
            "mean": roll.mean(), "std": roll.std(), "min": roll.min(), "max": roll.max(),
            "q10": roll.quantile(0.10), "q50": roll.quantile(0.50), "q90": roll.quantile(0.90),
            "frac_gt200": above.rolling(w, min_periods=w // 2).mean(),
        }
        for name, values in stats.items():
            out[f"hubavg_win{w}_{name}"] = clock_shift(values, 1 + delay_hours, index).to_numpy()
    return out


def representation(name: str, hourly: pd.Series, index: pd.DatetimeIndex, delay_hours: int) -> pd.DataFrame:
    """Columns added to the original 37 baseline features for R1, R2, or R3."""
    if name == "R1":
        return dense_lags(hourly, index, delay_hours)
    if name == "R2":
        return window_summaries(hourly, index, delay_hours)
    if name == "R3":
        return pd.concat([dense_lags(hourly, index, delay_hours), window_summaries(hourly, index, delay_hours)], axis=1)
    raise ValueError(name)


def lear_design(hourly: pd.Series, da: pd.DataFrame, index: pd.DatetimeIndex, delay_hours: int = 1) -> pd.DataFrame:
    """LEAR-style regressors without load: delayed price lags, gated day-ahead hub price, Central dummies."""
    hourly = hourly.sort_index().astype(float)
    frame = pd.DataFrame(index=index)
    for k in LEAR_LAGS:
        frame[f"p_lag_{k}h"] = clock_shift(hourly, k + delay_hours, index).to_numpy()
    gated = day_ahead_features(hourly, da, price_delay=pd.Timedelta(hours=delay_hours))
    frame["da_HB_HUBAVG"] = gated["da_HB_HUBAVG"].reindex(index).to_numpy()
    cal = calendar_central(index)
    for col, levels in (("hour_local", range(24)), ("day_of_week_local", range(7)), ("month_local", range(1, 13))):
        values = cal[col].astype(int).to_numpy()
        for level in levels:
            frame[f"cal_{col}_{level}"] = (values == level).astype(float)
    return frame


def _prepare(train: pd.DataFrame, parts: list[pd.DataFrame]):
    price_cols = [c for c in train.columns if c.startswith("p_lag_") or c == "da_HB_HUBAVG"]

    def tf(part):
        part = part.copy()
        part[price_cols] = np.arcsinh(part[price_cols].astype(float))
        return part

    tr = tf(train)
    med = tr.median(numeric_only=True)
    tr = tr.fillna(med)
    mu = tr.mean()
    sd = tr.std().replace(0, 1.0).fillna(1.0)
    scale = lambda p: ((tf(p).fillna(med) - mu) / sd).to_numpy(float)
    return scale(train), [scale(p) for p in parts]


def _lasso(x, y, alpha):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso

    model = Lasso(alpha=alpha, max_iter=100000, random_state=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def decode(z) -> np.ndarray:
    return np.sinh(np.clip(np.asarray(z, float), -ASINH_CLIP, ASINH_CLIP))


def oof_lear(hourly: pd.Series, da: pd.DataFrame, years, delay_hours: int = 1, alphas=LEAR_ALPHAS,
             index: pd.DatetimeIndex | None = None) -> tuple[pd.Series, list[dict]]:
    """Chronological out-of-fold LEAR-style forecasts.

    For target year Y the model is fit on rows before Jan 1 of Y. The penalty is the
    converged value with the lowest MAE on year Y-1 when fit on rows before Jan 1 of Y-1.
    A row of year Y therefore never sees a label from year Y or later.
    """
    hourly = hourly.sort_index().astype(float)
    if index is None:
        index = hourly.index
    design = lear_design(hourly, da, index, delay_hours)
    y = hourly.reindex(index)
    keep = y.notna()
    design, y = design.loc[keep], y.loc[keep]
    out = pd.Series(np.nan, index=design.index, name="lear_oof_pred")
    folds = []
    for year in years:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        prev = pd.Timestamp(f"{year - 1}-01-01", tz="UTC")
        inner_tr = design.index < prev
        inner_va = (design.index >= prev) & (design.index < start)
        outer_tr = design.index < start
        target = (design.index >= start) & (design.index < end)
        if not target.any():
            continue
        x_in, (x_va,) = _prepare(design.loc[inner_tr], [design.loc[inner_va]])
        z_in = np.arcsinh(y.loc[inner_tr].to_numpy(float))
        path = []
        for a in alphas:
            m, ok = _lasso(x_in, z_in, a)
            path.append({"alpha": a, "converged": ok,
                         "val_mae": float(np.mean(np.abs(decode(m.predict(x_va)) - y.loc[inner_va].to_numpy(float))))})
        eligible = [p for p in path if p["converged"]] or path
        best = min(eligible, key=lambda p: p["val_mae"])
        x_tr, (x_te,) = _prepare(design.loc[outer_tr], [design.loc[target]])
        m, ok = _lasso(x_tr, np.arcsinh(y.loc[outer_tr].to_numpy(float)), best["alpha"])
        out.loc[target] = decode(m.predict(x_te))
        folds.append({
            "year": year, "alpha": best["alpha"], "inner_val_mae": best["val_mae"],
            "alpha_at_grid_edge": best["alpha"] in (min(alphas), max(alphas)),
            "final_fit_converged": ok, "n_inner_train": int(inner_tr.sum()), "n_inner_val": int(inner_va.sum()),
            "n_train": int(outer_tr.sum()), "n_target": int(target.sum()),
            "train_end": str(design.index[outer_tr].max()), "n_nonzero_coef": int(np.sum(m.coef_ != 0)),
            "path": path,
        })
    return out, folds
