"""Experiment orchestration: dataset assembly, purged folds, model zoo, predictions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_yaml
from src.data import sources
from src.features import build
from src.features.targets import origin_grid, window_labels, window_matrix
from src.models import baselines
from src.models.chronos_features import ChronosCache, chronos_only_score
from src.models.chronos_features import features as chronos_feats
from src.models.xgboost_alert import Platt, f1_threshold, fit_alert_model

warnings.filterwarnings("ignore", category=RuntimeWarning)
H1 = pd.Timedelta(hours=1)

FAMILIES_COMMON = ("price_history", "calendar", "day_ahead", "load", "weather_obs", "fundamentals", "news")


class Dataset:
    """All issue-time features and labels for one origin grid."""

    def __init__(self, cfg: dict, weather_lag_hours: int = 1, raw_dir: Path = sources.RAW):
        self.cfg = cfg
        self.avail = load_yaml(cfg["data"]["availability"])["sources"]
        self.publish_local = self.avail["ercot_da_prices"]["publish_local"]
        self.every = int(cfg["origins"]["every_hours"])
        self.y = sources.hourly_price(raw_dir)
        self.da = sources.day_ahead(raw_dir)
        self.origins = origin_grid(self.y, self.every, cfg["origins"]["start"])
        o = self.origins
        wx = sources.weather(raw_dir)
        self.common = {
            "price_history": build.price_history(self.y, o),
            "calendar": build.calendar(o, cfg["data"]["timezone_calendar"]),
            "day_ahead": build.day_ahead_common(self.y, self.da, o, self.publish_local),
            "load": build.load_features(sources.load(raw_dir), o),
            "weather_obs": build.weather_obs(wx, o, weather_lag_hours),
            "fundamentals": build.fundamentals(sources.henry_hub(raw_dir), sources.storage(raw_dir),
                                               sources.nrc(raw_dir), o, self.avail),
            "news": build.news(sources.gdelt(raw_dir), o, self.avail["gdelt"]),
        }
        max_h = max(cfg["target"]["horizons_hours"])
        self.oracle_cube = build.realized_temperature_cube(wx, o, max_h)
        gfs = sources.gfs_forecasts(raw_dir)
        rule = self.avail["weather_forecast_gfs"]
        self.fc_cube = None if gfs is None else build.forecast_temperature_cube(
            gfs, o, max_h, int(pd.Timedelta(rule["run_delay"]) / H1), int(rule["max_lead_days"]))
        self._cache: dict = {}

    def horizon_family(self, name: str, h: int) -> pd.DataFrame:
        key = (name, h)
        if key not in self._cache:
            if name == "day_ahead":
                self._cache[key] = build.day_ahead_window(self.da, self.origins, h, self.publish_local)
            elif name == "weather_forecast":
                if self.fc_cube is None:
                    raise RuntimeError("GFS forecasts missing: run scripts/fetch_gfs_forecasts.py")
                self._cache[key] = build.window_temperature_features(self.fc_cube, self.origins, h, "wxfc")
            elif name == "weather_oracle":
                self._cache[key] = build.window_temperature_features(self.oracle_cube, self.origins, h, "wxor")
            else:
                self._cache[key] = pd.DataFrame(index=self.origins)
        return self._cache[key]

    def features(self, families: list[str], h: int) -> pd.DataFrame:
        parts = []
        for fam in families:
            if fam in self.common:
                parts.append(self.common[fam])
            parts.append(self.horizon_family(fam, h))
        return pd.concat(parts, axis=1)

    def labels(self, h: int, threshold: float) -> pd.Series:
        key = ("z", h, threshold)
        if key not in self._cache:
            self._cache[key] = window_labels(self.y, self.origins, h, threshold)
        return self._cache[key]

    def chronos(self, cache: ChronosCache, h: int, threshold: float) -> pd.DataFrame:
        key = ("ch", str(cache.path), h, threshold)
        if key not in self._cache:
            self._cache[key] = chronos_feats(cache, self.origins, h, threshold, self.y, self.da,
                                             self.labels(h, threshold), self.publish_local, self.every)
        return self._cache[key]

    def seasonal_naive_score(self, h: int) -> np.ndarray:
        w = window_matrix(self.y, self.origins, h, offset=-168)
        return np.where(np.isnan(w).all(axis=1), np.nan, np.nanmax(np.where(np.isnan(w), -np.inf, w), axis=1))


def resolve_threshold(thr, y: pd.Series, train_end: pd.Timestamp) -> float:
    if thr == "train_p99":
        return float(y[y.index < train_end].quantile(0.99))
    return float(thr)


def split_masks(origins: pd.DatetimeIndex, h: int, fold: dict, period_start: pd.Timestamp | None):
    t = pd.Series(origins, index=origins)
    end = t + h * H1
    tr_end = pd.Timestamp(fold["train_end"], tz="UTC")
    v0, v1 = (pd.Timestamp(x, tz="UTC") for x in fold["val"])
    s0, s1 = (pd.Timestamp(x, tz="UTC") for x in fold["test"])
    start_ok = (t >= period_start) if period_start is not None else pd.Series(True, index=origins)
    train = start_ok & (end <= tr_end)
    val = start_ok & (t >= v0) & (end <= v1)
    test = (t >= s0) & (end <= s1)
    return train.to_numpy(), val.to_numpy(), test.to_numpy()


@dataclass
class RunSpec:
    experiment: str
    folds: list[str]
    feature_sets: dict
    models: list[str]
    threshold: object = 200.0
    horizons: list[int] = field(default_factory=lambda: [24, 48, 72, 96, 120, 144])
    period_start: str | None = None
    seed: int = 0
    tag: str = "primary"


def model_families(model: str, feature_sets: dict) -> tuple[list[str], bool]:
    """Feature families and whether Chronos features are added."""
    table = {
        "XGB_price": ("price", False), "XGB_structured": ("structured", False),
        "XGB_structured_news": ("structured_news", False),
        "Chronos_XGB_price": ("price", True), "Chronos_XGB_structured": ("structured", True),
        "Chronos_XGB_full": ("structured_news", True),
        "XGB_structured_news_wxfc": ("structured_news_wxfc", False),
        "Chronos_XGB_full_wxfc": ("structured_news_wxfc", True),
        "XGB_structured_news_oracle": ("structured_news_oracle", False),
        "Chronos_XGB_full_oracle": ("structured_news_oracle", True),
    }
    fs, ch = table[model]
    return list(feature_sets[fs]), ch


def run(ds: Dataset, spec: RunSpec, chronos: ChronosCache | None, log=print) -> tuple[pd.DataFrame, list[dict]]:
    cfg = ds.cfg
    folds = {f["name"]: f for f in cfg["folds"]}
    xcfg, lcfg = cfg["xgboost"], cfg["logistic"]
    period_start = pd.Timestamp(spec.period_start, tz="UTC") if spec.period_start else None
    preds, infos = [], []
    o = ds.origins
    for h in spec.horizons:
        for fname in spec.folds:
            fold = folds[fname]
            thr = resolve_threshold(spec.threshold, ds.y, pd.Timestamp(fold["train_end"], tz="UTC"))
            z = ds.labels(h, thr).to_numpy()
            tr, va, te = split_masks(o, h, fold, period_start)
            has = ~np.isnan(z)
            tr, va, te = tr & has, va & has, te & has
            ytr, yva, yte = z[tr].astype(int), z[va].astype(int), z[te].astype(int)
            ch = ds.chronos(chronos, h, thr) if chronos is not None else None
            base_meta = {"experiment": spec.experiment, "tag": spec.tag, "fold": fname, "horizon": h,
                         "threshold": thr, "seed": spec.seed}
            log(f"  [{spec.tag}] H={h:>3} {fname}: train {tr.sum():,} ({ytr.sum()} pos)  "
                f"val {va.sum():,} ({yva.sum()} pos)  test {te.sum():,} ({yte.sum()} pos)")
            cal_val: dict[str, np.ndarray] = {}
            cal_test: dict[str, np.ndarray] = {}

            def emit(model, p_val, p_test, thr_alert, info):
                cal_val[model], cal_test[model] = p_val, p_test
                for split, mask, p, yy in (("val", va, p_val, yva), ("test", te, p_test, yte)):
                    preds.append(pd.DataFrame({**base_meta, "model": model, "split": split,
                                               "origin": o[mask], "z": yy, "p": p,
                                               "alert": (p >= thr_alert).astype(int)}))
                infos.append({**base_meta, "model": model, "alert_threshold": float(thr_alert), **info})

            for model in spec.models:
                if model == "prevalence":
                    r = baselines.prevalence(ytr, va.sum(), te.sum(), yva)
                    emit(model, r.p_val, r.p_test, r.threshold, r.info)
                elif model == "persistence_max24h":
                    s = ds.common["price_history"]["p_max_24h"].to_numpy()
                    r = baselines.score_model(s[va], s[te], yva)
                    emit(model, r.p_val, r.p_test, r.threshold, r.info)
                elif model == "seasonal_naive_168h":
                    s = ds.seasonal_naive_score(h)
                    r = baselines.score_model(s[va], s[te], yva)
                    emit(model, r.p_val, r.p_test, r.threshold, r.info)
                elif model == "day_ahead_max":
                    dw = ds.horizon_family("day_ahead", h)["daw_max"].to_numpy()
                    s = np.where(np.isnan(dw), ds.common["day_ahead"]["da_lastday_max"].to_numpy(), dw)
                    r = baselines.score_model(s[va], s[te], yva)
                    emit(model, r.p_val, r.p_test, r.threshold, r.info)
                elif model == "logistic_structured":
                    X = ds.features(list(spec.feature_sets["structured"]), h)
                    r = baselines.logistic_structured(X[tr], ytr, X[va], yva, X[te], lcfg["C_grid"])
                    emit(model, r.p_val, r.p_test, r.threshold, r.info)
                elif model == "Chronos_only":
                    s = chronos_only_score(ch).to_numpy()
                    r = baselines.score_model(s[va], s[te], yva,
                                              transform=lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6))),
                                              fill=0.0)
                    emit(model, r.p_val, r.p_test, r.threshold, r.info)
                elif model == "Blend_full":
                    a, b = cal_val.get("Chronos_only"), cal_val.get("XGB_structured_news")
                    if a is None or b is None:
                        raise RuntimeError("Blend_full needs Chronos_only and XGB_structured_news earlier in the list")
                    from src.evaluation.metrics import log_loss
                    ws = np.round(np.arange(0, 1.01, 0.1), 2)
                    losses = [log_loss(yva, w * a + (1 - w) * b) for w in ws]
                    w = float(ws[int(np.argmin(losses))])
                    pv = w * a + (1 - w) * b
                    pt = w * cal_test["Chronos_only"] + (1 - w) * cal_test["XGB_structured_news"]
                    emit(model, pv, pt, f1_threshold(pv, yva), {"blend_weight_chronos": w})
                else:
                    fams, with_ch = model_families(model, spec.feature_sets)
                    X = ds.features(fams, h)
                    if with_ch:
                        if ch is None:
                            raise RuntimeError(f"{model} needs a Chronos cache")
                        X = pd.concat([X, ch], axis=1)
                    feats = list(X.columns)
                    m = fit_alert_model(model, X[tr], ytr, X[va], yva, feats, xcfg, spec.seed)
                    pv, pt = m.predict(X[va]), m.predict(X[te])
                    top = m.importance.head(15)
                    emit(model, pv, pt, m.threshold,
                         {"params": m.params, "val_ap_raw": m.val_ap, "best_iteration": m.best_iteration,
                          "n_features": len(feats), "platt_a": m.platt.a, "platt_b": m.platt.b,
                          "importance_top15": {k: float(v) for k, v in top.items()},
                          "chronos_gain_share": float(m.importance[[c for c in feats if c.startswith("ch_")]].sum()
                                                      / max(m.importance.sum(), 1e-12)) if with_ch else 0.0})
    return (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()), infos
