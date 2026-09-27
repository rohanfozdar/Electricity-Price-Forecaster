"""Baselines. Each returns calibrated probabilities for val and test plus a val-chosen threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.models.xgboost_alert import Platt, f1_threshold


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class ScoreResult:
    p_val: np.ndarray
    p_test: np.ndarray
    threshold: float
    info: dict


def score_model(s_val: np.ndarray, s_test: np.ndarray, y_val: np.ndarray, transform=_signed_log,
                fill: float | None = None) -> ScoreResult:
    """Monotone heuristic score -> Platt calibration on validation -> F1 threshold on validation."""
    if fill is None:
        fill = np.nanmin(s_val) if np.isfinite(s_val).any() else 0.0
    sv, st = np.where(np.isfinite(s_val), s_val, fill), np.where(np.isfinite(s_test), s_test, fill)
    pl = Platt.fit(sv, y_val, transform=transform)
    pv, pt = pl.apply(sv, transform=transform), pl.apply(st, transform=transform)
    return ScoreResult(pv, pt, f1_threshold(pv, y_val), {"platt_a": pl.a, "platt_b": pl.b})


def prevalence(y_train: np.ndarray, n_val: int, n_test: int, y_val: np.ndarray) -> ScoreResult:
    p = float(np.mean(y_train))
    pv, pt = np.full(n_val, p), np.full(n_test, p)
    return ScoreResult(pv, pt, f1_threshold(pv, y_val), {"prevalence": p})


class LogisticStructured:
    """L2 logistic regression with train-median imputation, missing indicators and scaling."""

    def __init__(self, C: float):
        self.C = C

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogisticStructured":
        self.cols = list(X.columns)
        self.med = X.median()
        self.miss_cols = [c for c in self.cols if X[c].isna().any()]
        Z = self._design(X, fit=True)
        self.lr = LogisticRegression(C=self.C, max_iter=2000)
        self.lr.fit(Z, y)
        return self

    def _design(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        Xi = X[self.cols].fillna(self.med).to_numpy(dtype=float)
        Xi = _signed_log(Xi)
        if fit:
            self.mu, self.sd = Xi.mean(axis=0), Xi.std(axis=0)
            self.sd[self.sd == 0] = 1.0
        Xi = (Xi - self.mu) / self.sd
        M = X[self.miss_cols].isna().to_numpy(dtype=float) if self.miss_cols else np.empty((len(X), 0))
        return np.hstack([Xi, M])

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self._design(X))[:, 1]


def logistic_structured(Xtr, ytr, Xva, yva, Xte, C_grid) -> ScoreResult:
    best = None
    for C in C_grid:
        m = LogisticStructured(C).fit(Xtr, ytr)
        pv = m.predict_raw(Xva)
        ap = average_precision_score(yva, pv) if yva.sum() else -1
        if best is None or ap > best[0]:
            best = (ap, C, m, pv)
    ap, C, m, pv_raw = best
    pl = Platt.fit(pv_raw, yva)
    pv, pt = pl.apply(pv_raw), pl.apply(m.predict_raw(Xte))
    return ScoreResult(pv, pt, f1_threshold(pv, yva), {"C": C, "val_ap_raw": float(ap)})
