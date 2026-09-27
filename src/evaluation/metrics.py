"""Probabilistic and operating-point metrics for binary alerts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Step-wise AP (same definition as sklearn), vectorized for bootstrap use."""
    y = np.asarray(y, dtype=float)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    distinct = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]
    tp = np.cumsum(ys)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(err / len(y))


def reliability(y, p, bins: int = 10) -> list[dict]:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    return [{"p_mean": float(p[c].mean()), "y_rate": float(y[c].mean()), "n": int(len(c))}
            for c in np.array_split(order, bins) if len(c)]


def operating_point(y, alert, hours_per_origin: float) -> dict:
    y, a = np.asarray(y).astype(int), np.asarray(alert).astype(int)
    tp = int(((a == 1) & (y == 1)).sum())
    fp = int(((a == 1) & (y == 0)).sum())
    fn = int(((a == 0) & (y == 1)).sum())
    days = len(y) * hours_per_origin / 24
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "alerts_per_30d": float(a.sum() / days * 30) if days else float("nan")}


def summary(y, p, alert, hours_per_origin: float) -> dict:
    y = np.asarray(y)
    out = {"n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
           "average_precision": average_precision(y, np.asarray(p)),
           "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
           "brier": brier(y, p), "log_loss": log_loss(y, p), "ece": ece(y, p)}
    out.update(operating_point(y, alert, hours_per_origin))
    return out
