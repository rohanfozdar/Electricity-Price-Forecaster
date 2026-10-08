"""
Validation-only improvements to the 24h spike alert and its stress-event lead times.

The paper protocol (run_benchmark.py paper-pipeline) picks the calibrator by the
lowest 2024 Brier score and alerts at a fixed 0.3. Here every choice is made on 2023:

    calibrator  Platt or isotonic, by out-of-fold 2023 Brier. The calibrators are fit
                on 2023, so their in-sample Brier would always favour isotonic; the
                folds are 7-day blocks of 2023 assigned round-robin to 5 folds.
    threshold   the cut on the out-of-fold calibrated 2023 probabilities that
                maximizes F1 (alert when probability > threshold)
    classifier  the paper's untransformed features, optionally with log1p transforms
                and Chronos features, under the paper's booster settings or a variant;
                plus zero-shot Chronos-Bolt (its own P(spike in the next 24h)) as a
                reference

Every candidate is appended to runs/benchmark/alert_search_log.csv with 2023 scores
only. The candidate with the highest out-of-fold 2023 F1 is scored on 2024 once,
over 5 seeds, and saved as runs/benchmark/alert_24h_improved and lead_times_improved.
Event-level detection (src/evaluation/events.py) is reported for the paper protocol
and the improved alert.

Usage (from the repository root):
    python models/improve_alerts.py search
    python models/improve_alerts.py finalize
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluation.backtest as backtest
from utils.config import FEATURES_DIR
from models.run_benchmark import (
    ALERT_THRESHOLD,
    PAPER,
    RUNS_DIR,
    SPIKE_PRICE,
    alert_metrics,
    load_matrix,
    provenance,
    save_run,
)
from models.utils import TARGET_CLASS, TARGET_REG, chronological_split, select_enhanced_features
from src.evaluation.events import event_metrics
from src.features.targets import extract_events

SEARCH_LOG = RUNS_DIR / "alert_search_log.csv"
FINAL_SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
HORIZON = 24
EVENT_MERGE_GAP_HOURS = 6  # configs/experiment_early_warning.yaml

CHRONOS_DIR = FEATURES_DIR / "chronos_hourly"
WINDOW_COLUMNS = ["chronos24_q50_max", "chronos24_q50_mean", "chronos24_q90_max",
                  "chronos24_p_max", "chronos24_p_any"]
CHRONOS_FEATURES = {
    "t5_1h": (FEATURES_DIR / "chronos_pred.parquet",
              ["chronos_pred", "chronos_q10", "chronos_q90", "chronos_mean", "chronos_p_gt200"]),
    "bolt_small_24h": (CHRONOS_DIR / "bolt-small_positional_ctx512_h24.parquet", WINDOW_COLUMNS),
    "bolt_base_24h": (CHRONOS_DIR / "bolt-base_positional_ctx512_h24.parquet", WINDOW_COLUMNS),
}

BASE_PARAMS = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, objective="binary:logistic", eval_metric="aucpr",
                   tree_method="hist", early_stopping_rounds=30, n_jobs=16)
PARAMS = {
    "paper": {},
    "logloss": {"eval_metric": "logloss"},
    "regularized": {"n_estimators": 2000, "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 10,
                    "reg_lambda": 5.0, "early_stopping_rounds": 100},
    "unweighted": {"scale_pos_weight": 1.0, "eval_metric": "logloss"},
}

_CACHE: dict = {}


# --- Data ----------------------------------------------------------------

def alert_frame(transform: bool, chronos: tuple[str, ...]):
    """Feature matrix with Chronos columns joined, its feature list, and the 3 splits."""
    key = (transform, chronos)
    if key not in _CACHE:
        df = load_matrix(transform=transform)
        features = select_enhanced_features(df)
        for name in chronos:
            path, cols = CHRONOS_FEATURES[name]
            extra = pd.read_parquet(path)[cols].add_prefix(f"{name}__")
            df = df.join(extra, how="left")
            features = features + list(extra.columns)
        df = df.dropna(subset=[TARGET_CLASS])
        _CACHE[key] = (df, features, chronological_split(df))
    return _CACHE[key]


def week_folds(index: pd.DatetimeIndex) -> np.ndarray:
    weeks = np.asarray((index - index[0]) // pd.Timedelta(days=7), dtype=int)
    return weeks % N_FOLDS


# --- Scores --------------------------------------------------------------

class ColumnScore:
    """Zero-shot reference: one feature column used as the alert score."""

    classes_ = np.array([0, 1])
    best_iteration = None

    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.clip(X[self.column].to_numpy(dtype=float), 0.0, 1.0)
        return np.column_stack([1 - p, p])


def fit_scorer(cfg: dict, seed: int):
    df, features, (tr, vl, te) = alert_frame(cfg["transform"], tuple(cfg["chronos"]))
    X = {s: d[features].ffill().fillna(0) for s, d in (("train", tr), ("val", vl), ("test", te))}
    if cfg["model"] == "chronos_zero_shot":
        return ColumnScore(f"{cfg['chronos'][0]}__chronos24_p_any"), X, (tr, vl, te), features

    import xgboost as xgb

    y_tr = tr[TARGET_CLASS]
    params = {**BASE_PARAMS, "scale_pos_weight": (y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
              "random_state": seed, **PARAMS[cfg["params"]]}
    clf = xgb.XGBClassifier(**params)
    clf.fit(X["train"], y_tr, eval_set=[(X["val"], vl[TARGET_CLASS])], verbose=False)
    return clf, X, (tr, vl, te), features


def calibrator(scorer, method: str, X, y):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    if isinstance(scorer, ColumnScore):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        s = scorer.predict_proba(X)[:, 1]
        model = (IsotonicRegression(out_of_bounds="clip").fit(s, y) if method == "isotonic"
                 else LogisticRegression().fit(s.reshape(-1, 1), y))
        return lambda Xn: (model.predict(scorer.predict_proba(Xn)[:, 1]) if method == "isotonic"
                           else model.predict_proba(scorer.predict_proba(Xn)[:, 1].reshape(-1, 1))[:, 1])
    cal = CalibratedClassifierCV(FrozenEstimator(scorer), method="sigmoid" if method == "platt" else method).fit(X, y)
    return lambda Xn: cal.predict_proba(Xn)[:, 1]


def out_of_fold(scorer, method: str, X_val: pd.DataFrame, y_val: np.ndarray) -> np.ndarray:
    folds = week_folds(X_val.index)
    out = np.empty(len(y_val))
    for k in range(N_FOLDS):
        held = folds == k
        out[held] = calibrator(scorer, method, X_val[~held], y_val[~held])(X_val[held])
    return out


def choose_on_validation(scorer, X_val: pd.DataFrame, y_val: np.ndarray) -> dict:
    """Calibrator by out-of-fold Brier, then the F1-maximizing threshold."""
    oof = {m: out_of_fold(scorer, m, X_val, y_val) for m in ("platt", "isotonic")}
    brier = {m: float(brier_score_loss(y_val, p)) for m, p in oof.items()}
    method = min(brier, key=brier.get)
    f1 = [alert_metrics(y_val, oof[method], t)["f1"] for t in THRESHOLDS]
    threshold = float(THRESHOLDS[int(np.argmax(f1))])
    return {"calibrator": method, "threshold": threshold, "oof_brier": brier, "oof_proba": oof[method],
            "val": alert_metrics(y_val, oof[method], threshold),
            "val_at_0.3": alert_metrics(y_val, oof[method], ALERT_THRESHOLD)}


# --- Lead times and event-level detection --------------------------------

def stress_event_leads(full: pd.DataFrame, threshold: float) -> list[dict]:
    """evaluation/backtest.py's per-event summary at a given threshold."""
    saved = backtest.ALERT_THRESHOLD
    backtest.ALERT_THRESHOLD = threshold
    try:
        return [backtest._summarize_event(full, *ev) for ev in backtest.STRESS_EVENTS]
    finally:
        backtest.ALERT_THRESHOLD = saved


def event_level(proba: pd.Series, labels: pd.Series, price: pd.Series, threshold: float,
                year: int) -> dict:
    """Event recall, lead, and false-alert episodes within one calendar year.

    Row t's label window is hours t+1..t+24, so the row is passed to
    event_detection as origin t+1 (whose window is [origin, origin+23]); its lead
    is then onset - (t+1).
    """
    lo, hi = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    in_year = (proba.index >= lo) & (proba.index < hi)
    origins = proba.index[in_year] + pd.Timedelta(hours=1)
    events = extract_events(price, SPIKE_PRICE, EVENT_MERGE_GAP_HOURS)
    z = labels.reindex(proba.index).fillna(0).to_numpy()[in_year]
    m = event_metrics(events, origins, (proba[in_year] > threshold).to_numpy(), z,
                      HORIZON, 1, (origins[0], origins[-1] + pd.Timedelta(hours=1)))
    m["threshold"] = threshold
    return m


# --- Search --------------------------------------------------------------

def log_result(cfg: dict, choice: dict, raw_val: np.ndarray, y_val: np.ndarray, best) -> dict:
    v = choice["val"]
    row = {"config": json.dumps(cfg, sort_keys=True), "best_iteration": best,
           "calibrator": choice["calibrator"], "threshold": choice["threshold"],
           "oof_brier_platt": choice["oof_brier"]["platt"],
           "oof_brier_isotonic": choice["oof_brier"]["isotonic"],
           "val_recall": v["recall"], "val_precision": v["precision"], "val_f1": v["f1"],
           "val_pr_auc": float(average_precision_score(y_val, raw_val)),
           "val_f1_at_0.3": choice["val_at_0.3"]["f1"],
           "logged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    SEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(SEARCH_LOG, mode="a", header=not SEARCH_LOG.exists(), index=False)
    print(f"  val F1 {v['f1']:.3f} (R {v['recall']:.3f} P {v['precision']:.3f}) thr {choice['threshold']:.2f} "
          f"{choice['calibrator']:<8} PR-AUC {row['val_pr_auc']:.3f}  {row['config']}", flush=True)
    return row


def evaluate(cfg: dict) -> dict | None:
    if SEARCH_LOG.exists() and json.dumps(cfg, sort_keys=True) in set(pd.read_csv(SEARCH_LOG)["config"]):
        return None
    scorer, X, (tr, vl, te), _ = fit_scorer(cfg, seed=42)
    y_val = vl[TARGET_CLASS].to_numpy().astype(int)
    choice = choose_on_validation(scorer, X["val"], y_val)
    return log_result(cfg, choice, scorer.predict_proba(X["val"])[:, 1], y_val,
                      getattr(scorer, "best_iteration", None))


CHRONOS_SETS = [(), ("t5_1h",), ("bolt_small_24h",), ("bolt_base_24h",), ("t5_1h", "bolt_small_24h")]


def run_search() -> None:
    for name in ("bolt_small_24h", "bolt_base_24h"):
        evaluate({"model": "chronos_zero_shot", "transform": False, "chronos": [name], "params": None})
    for transform, chronos, params in itertools.product([False, True], CHRONOS_SETS, PARAMS):
        evaluate({"model": "xgb", "transform": transform, "chronos": list(chronos), "params": params})

    log = pd.read_csv(SEARCH_LOG).sort_values(["val_f1", "val_pr_auc"], ascending=False)
    top = log.iloc[0]
    sel = {"config": json.loads(top["config"]),
           "selection_rule": "highest out-of-fold 2023 F1 at the 2023-chosen calibrator and threshold",
           "val": {k: float(top[k]) for k in ("val_recall", "val_precision", "val_f1", "val_pr_auc",
                                              "threshold")},
           "calibrator": top["calibrator"], "n_configs_considered": int(len(log))}
    (RUNS_DIR / "selected_alert.json").write_text(json.dumps(sel, indent=2))
    print(f"\n  Selected: {json.dumps(sel, indent=2)}")


# --- Finalize ------------------------------------------------------------

ALERT_KEYS = ("threshold", "recall", "precision", "f1", "pr_auc", "brier")


def finalize() -> None:
    sel = json.loads((RUNS_DIR / "selected_alert.json").read_text())
    cfg = sel["config"]
    df, features, _ = alert_frame(cfg["transform"], tuple(cfg["chronos"]))
    X_full = df[features].ffill().fillna(0)
    price = load_matrix(transform=False)[TARGET_REG]
    labels_full = df[TARGET_CLASS].astype(int)

    per_seed, frames, leads, events_seed, levels = [], {}, {}, {}, []
    for seed in FINAL_SEEDS:
        scorer, X, (tr, vl, te), _ = fit_scorer(cfg, seed)
        y_val = vl[TARGET_CLASS].to_numpy().astype(int)
        y_test = te[TARGET_CLASS].fillna(0).to_numpy().astype(int)
        choice = choose_on_validation(scorer, X["val"], y_val)
        cal = calibrator(scorer, choice["calibrator"], X["val"], y_val)
        p_test = cal(X["test"])
        m = alert_metrics(y_test, p_test, choice["threshold"])
        m.update(calibrator=choice["calibrator"], best_iteration=getattr(scorer, "best_iteration", None))
        per_seed.append(m)
        print(f"  seed {seed}: {choice['calibrator']} thr {choice['threshold']:.2f}  test recall "
              f"{m['recall']:.2%} precision {m['precision']:.2%} F1 {m['f1']:.2%}", flush=True)

        frames.setdefault("val", pd.DataFrame({"split": "val", "y_true": y_val}, index=vl.index))
        frames.setdefault("test", pd.DataFrame({"split": "test", "y_true": y_test}, index=te.index))
        frames["val"][f"proba_oof_seed{seed}"] = choice["oof_proba"]
        frames["test"][f"proba_seed{seed}"] = p_test

        full = pd.DataFrame({"HB_HUBAVG": price.reindex(df.index), "spike_probability_calibrated": cal(X_full)},
                            index=df.index)
        events = stress_event_leads(full, choice["threshold"])
        events_seed[seed] = events
        for ev in events:
            leads.setdefault(ev["event"], []).append(ev.get("lead_time_hours"))
        proba_full = full["spike_probability_calibrated"]
        levels.append({"val": event_level(proba_full, labels_full, price, choice["threshold"], 2023),
                       "test": event_level(proba_full, labels_full, price, choice["threshold"], 2024)})
        if seed == FINAL_SEEDS[0]:
            full.to_parquet(RUNS_DIR / "_lead_times_improved_seed0.parquet")

    first = FINAL_SEEDS[0]
    frames["val"]["proba"] = frames["val"][f"proba_oof_seed{first}"]
    frames["test"]["proba"] = frames["test"][f"proba_seed{first}"]
    test_mean = {k: float(np.mean([m[k] for m in per_seed])) for k in ALERT_KEYS}
    test_std = {k: float(np.std([m[k] for m in per_seed])) for k in ALERT_KEYS}
    save_run("alert_24h_improved", pd.concat([frames["val"], frames["test"]]),
             {"test": test_mean, "test_std": test_std, "test_seeds": per_seed},
             {"config": cfg, "selection": sel, "seeds": FINAL_SEEDS,
              "calibrator_choice": "lowest out-of-fold 2023 Brier (7-day blocks, 5 folds)",
              "threshold_choice": "maximizes out-of-fold 2023 F1",
              "test_metrics_are": "mean over seeds; proba is the first seed",
              "val_proba_is": "out-of-fold calibrated 2023 probability"})
    print(f"  [alert_24h_improved] test recall {test_mean['recall']:.2%}  precision "
          f"{test_mean['precision']:.2%}  F1 {test_mean['f1']:.2%}  (paper {PAPER['alert_24h']})")

    paper_full = pd.read_parquet(RUNS_DIR / "lead_times_paper" / "predictions.parquet")
    paper_proba = paper_full["spike_probability_calibrated"]
    paper_levels = {"val": event_level(paper_proba, labels_full, price, ALERT_THRESHOLD, 2023),
                    "test": event_level(paper_proba, labels_full, price, ALERT_THRESHOLD, 2024)}

    out = RUNS_DIR / "lead_times_improved"
    out.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "_lead_times_improved_seed0.parquet").replace(out / "predictions.parquet")
    events = events_seed[first]
    for ev in events:
        vals = [v for v in leads[ev["event"]] if v is not None]
        ev["lead_time_hours_by_seed"] = leads[ev["event"]]
        ev["lead_time_range"] = [min(vals), max(vals)] if vals else None
    record = {"run": "lead_times_improved", "events": events,
              "alert_threshold": per_seed[0]["threshold"], "calibrator": per_seed[0]["calibrator"],
              "events_are": f"seed {first}; lead_time_hours_by_seed covers seeds {FINAL_SEEDS}",
              "event_level": {"improved_by_seed": levels, "paper_protocol": paper_levels,
                              "definition": "src/evaluation/events.py, 24h horizon, events merged across "
                                            f"gaps of up to {EVENT_MERGE_GAP_HOURS}h, origin = row time + 1h"},
              "provenance": provenance()}
    (out / "metrics.json").write_text(json.dumps(record, indent=2, default=str))

    for ev in events:
        print(f"  {ev['event']:<28} lead {ev.get('lead_time_hours')!s:>6}  seeds {ev['lead_time_hours_by_seed']}"
              f"  (paper {PAPER['lead_times'].get(ev['event'], '-')})")
    for label, lv in (("paper protocol", paper_levels), ("improved seed 0", levels[0])):
        for split in ("val", "test"):
            e = lv[split]
            print(f"  {label:<16} {split:<4} events {e['detected_events']}/{e['n_events']}  "
                  f"median lead {e['median_lead_hours']:.0f}h  episodes {e['alert_episodes']}  "
                  f"episode precision {e['event_precision']:.2f}  false/30d {e['false_alert_episodes_per_30d']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["search", "finalize"])
    args = ap.parse_args()
    run_search() if args.command == "search" else finalize()


if __name__ == "__main__":
    main()
