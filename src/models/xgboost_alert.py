"""XGBoost alert classifier: validation-only selection, Platt calibration, F1 threshold."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class Platt:
    """Sigmoid calibration on a 1-D score, fit on validation predictions."""
    a: float = 1.0
    b: float = 0.0
    identity: bool = False

    @classmethod
    def fit(cls, score: np.ndarray, y: np.ndarray, transform=_logit) -> "Platt":
        y = np.asarray(y)
        if y.min() == y.max():
            return cls(identity=True)
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(transform(score).reshape(-1, 1), y)
        return cls(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))

    def apply(self, score: np.ndarray, transform=_logit) -> np.ndarray:
        if self.identity:
            return np.clip(score, 0, 1)
        return 1 / (1 + np.exp(-(self.a * transform(score) + self.b)))


def f1_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximizing F1 on (p, y); alerts are p >= threshold."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float(np.inf)
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    last_of_tie = np.r_[ps[1:] != ps[:-1], True]
    f1 = 2 * tp / (k + ys.sum())
    f1 = np.where(last_of_tie, f1, -1)
    i = int(np.argmax(f1))
    return float(ps[i])


@dataclass
class FittedAlertModel:
    name: str
    features: list[str]
    params: dict
    val_ap: float
    best_iteration: int
    booster: xgb.XGBClassifier | None = None
    platt: Platt = field(default_factory=Platt)
    threshold: float = 0.5
    importance: pd.Series | None = None

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.platt.apply(self.predict_raw(X))


def grid_configs(grid: dict, n_pos: int, n_neg: int) -> list[dict]:
    spw = [1.0 if v == 1 else float(np.sqrt(n_neg / max(n_pos, 1))) for v in grid["scale_pos_weight"]]
    keys = ["max_depth", "min_child_weight", "scale_pos_weight"]
    vals = [grid["max_depth"], grid["min_child_weight"], spw]
    return [dict(zip(keys, v)) for v in itertools.product(*vals)]


def fit_alert_model(name: str, Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray,
                    features: list[str], cfg: dict, seed: int) -> FittedAlertModel:
    fixed = dict(cfg["fixed"])
    n_est = fixed.pop("n_estimators")
    es = fixed.pop("early_stopping_rounds")
    best = None
    for params in grid_configs(cfg["grid"], int(ytr.sum()), int(len(ytr) - ytr.sum())):
        clf = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=es, random_state=seed,
                                n_jobs=-1, objective="binary:logistic", **fixed, **params)
        clf.fit(Xtr[features], ytr, eval_set=[(Xva[features], yva)], verbose=False)
        p = clf.predict_proba(Xva[features])[:, 1]
        ap = average_precision_score(yva, p) if yva.sum() > 0 else float("nan")
        if best is None or (np.nan_to_num(ap, nan=-1) > np.nan_to_num(best[0], nan=-1)):
            best = (ap, params, clf, p)
    ap, params, clf, p_val = best
    platt = Platt.fit(p_val, yva)
    thr = f1_threshold(platt.apply(p_val), yva)
    imp = pd.Series(clf.get_booster().get_score(importance_type="gain")).reindex(features).fillna(0.0)
    return FittedAlertModel(name=name, features=features, params=params, val_ap=float(ap),
                            best_iteration=int(clf.best_iteration), booster=clf, platt=platt,
                            threshold=thr, importance=imp.sort_values(ascending=False))
