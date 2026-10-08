"""
Frozen benchmark harness for the paper's headline results.

Every run scores the same 2024 test timestamps (asserted: 8,767 hours with
114 hours above $200/MWh), writes its prediction vector and metrics under
runs/benchmark/<run>/, and is compared with the paper's reported value in
runs/benchmark/leaderboard.md.

Subcommands:
    xgb             XGBoost with and without GDELT (notebooks/01_xgboost.ipynb settings)
    paper-pipeline  Untransformed train_all_v2 regressor and classifier, calibration,
                    day-ahead comparison, 24h alerts, and stress-event lead times
    chronos         Zero-shot Chronos, read from the backfill checkpoint
    chronos-seeds   Zero-shot Chronos re-run on the test year with other sampling seeds
    hybrid          Record the faithful Chronos-XGBoost run of models/train_chronos_hybrid.py
    leaderboard     Write runs/benchmark/leaderboard.md

Run from the repository root:
    python models/run_benchmark.py xgb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import FEATURES_DIR
from models.utils import (
    TARGET_CLASS,
    TARGET_REG,
    apply_feature_transforms,
    chronological_split,
    select_enhanced_features,
)

RUNS_DIR = Path("runs/benchmark")
V2_MATRIX = FEATURES_DIR / "feature_matrix_engineered_v2.parquet"
CHRONOS_PRED_PATH = FEATURES_DIR / "chronos_pred.parquet"
HYBRID_PREDICTIONS = Path("models/artifacts/chronos_hybrid_predictions.parquet")

TEST_ROWS = 8767
TEST_SPIKES = 114
SPIKE_PRICE = 200.0
STRESS_PRICE = 100.0
ALERT_THRESHOLD = 0.3

# Values as printed in the paper. Lead times are the per-event values behind
# the paper's "18 to 143 hours" (evaluation/backtest_results.json, 2026-05-04).
PAPER = {
    "chronos_xgb": {"mae": 12.13, "rmse": 93.98, "spike_recall": 0.6140, "spike_precision": 0.5043},
    "chronos": {"mae": 14.72, "rmse": 115.59, "spike_recall": 0.4211, "spike_precision": 0.6076},
    "xgb": {"mae": 24.43, "rmse": 112.46, "spike_recall": 0.5702, "spike_precision": 0.2510},
    "xgb_gdelt": {"mae": 22.21, "rmse": 108.88, "spike_recall": 0.5439,
                  "spike_precision": 0.3024, "spike_pr_auc": 0.422},
    "xgb_no_gdelt": {"mae": 24.64, "rmse": 114.20, "spike_recall": 0.5789,
                     "spike_precision": 0.2481, "spike_pr_auc": 0.400},
    "enhanced_vs_dam": {"mae": 19.56, "rmse": 114.15, "stress_mae": 265.36, "spike_recall": 0.5175,
                        "spike_precision": 0.3533, "spike_f1": 0.4199},
    "dam": {"mae": 17.51, "rmse": 112.85, "stress_mae": 273.09, "spike_recall": 0.2895,
            "spike_precision": 0.4853, "spike_f1": 0.3626},
    "alert_24h": {"recall": 0.3463, "precision": 0.2422, "f1": 0.2850},
    "lead_times": {
        "Jan 2018 cold snap": 18.0, "Aug 2019 heatwave": 47.0, "Winter Storm Uri": 48.0,
        "June 2022 heat stress": 25.0, "Aug 2023 heatwave": 48.0,
        "Jan 2024 cold snap": 83.0, "Aug 2024 heat": 143.0,
    },
}

# run name -> paper row it is compared against in the leaderboard
REGRESSION_ROWS = [
    ("chronos_xgb_faithful", "chronos_xgb", "Chronos-XGBoost, faithful"),
    ("chronos_xgb_improved", "chronos_xgb", "Chronos-XGBoost, improved (validation-selected)"),
    ("chronos_xgb_alternative", "chronos_xgb", "Alternative architecture (validation-selected)"),
    ("chronos_zero_shot", "chronos", "Chronos zero-shot"),
    ("xgb_gdelt_nb01", "xgb", "XGBoost (consolidated row)"),
    ("xgb_gdelt_nb01", "xgb_gdelt", "XGBoost with GDELT (ablation)"),
    ("xgb_no_gdelt_nb01", "xgb_no_gdelt", "XGBoost without GDELT (ablation)"),
    ("enhanced_v2_paper", "enhanced_vs_dam", "Enhanced model vs day-ahead"),
    ("dam", "dam", "Day-ahead price"),
]


# --- Shared pieces --------------------------------------------------------

def load_matrix(transform: bool) -> pd.DataFrame:
    df = pd.read_parquet(V2_MATRIX)
    return apply_feature_transforms(df) if transform else df


def frozen_split(df: pd.DataFrame, target: str = TARGET_REG):
    train, val, test = chronological_split(df.dropna(subset=[target]))
    assert_test_index(test)
    return train, val, test


def assert_test_index(test: pd.DataFrame) -> None:
    n_spikes = int((test[TARGET_REG] > SPIKE_PRICE).sum())
    if len(test) != TEST_ROWS or n_spikes != TEST_SPIKES:
        raise AssertionError(
            f"Test set drifted: {len(test)} rows / {n_spikes} spikes, "
            f"expected {TEST_ROWS} / {TEST_SPIKES}")


def index_hash(index: pd.DatetimeIndex) -> str:
    return hashlib.sha256(np.asarray(index.asi8).tobytes()).hexdigest()[:16]


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, and spike scores with the same definitions as
    models.utils.regression_report and evaluation/benchmark_dam.py."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    true_s, pred_s = y_true > SPIKE_PRICE, y_pred > SPIKE_PRICE
    tp = int((true_s & pred_s).sum())
    fp = int((~true_s & pred_s).sum())
    fn = int((true_s & ~pred_s).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    stress = y_true > STRESS_PRICE
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "spike_recall": recall,
        "spike_precision": precision,
        "spike_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "spike_pr_auc": float(average_precision_score(true_s, y_pred)) if true_s.any() else float("nan"),
        "tp": tp, "fp": fp, "fn": fn,
        "stress_mae": float(np.mean(np.abs(err[stress]))) if stress.any() else float("nan"),
        "normal_mae": float(np.mean(np.abs(err[~stress]))),
    }


def alert_metrics(y_true, proba, threshold: float = ALERT_THRESHOLD) -> dict:
    """Binary alert scores, with `proba > threshold` as in evaluation/benchmark_dam.py."""
    y_true = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba, dtype=float)
    pred = proba > threshold
    tp = int((y_true & pred).sum())
    fp = int((~y_true & pred).sum())
    fn = int((y_true & ~pred).sum())
    tn = int((~y_true & ~pred).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "threshold": threshold, "recall": recall, "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def provenance() -> dict:
    def _git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except Exception:
            return None

    versions = {}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "torch", "chronos", "lightgbm", "catboost"):
        if mod in sys.modules:
            versions[mod] = getattr(sys.modules[mod], "__version__", "?")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0],
        "versions": versions,
    }


def save_run(name: str, predictions: pd.DataFrame, metrics: dict, meta: dict | None = None) -> dict:
    """Write predictions.parquet and metrics.json for one run.

    `predictions` is indexed by timestamp and has a `split` column; the test rows
    must be exactly the frozen test index.
    """
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    test_idx = predictions.index[predictions["split"] == "test"]
    pred_path = out / "predictions.parquet"
    predictions.to_parquet(pred_path)
    record = {
        "run": name,
        "test_index_hash": index_hash(test_idx),
        "test_rows": int(len(test_idx)),
        "predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest()[:16],
        "metrics": metrics,
        "meta": meta or {},
        "provenance": provenance(),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(record, f, indent=2, default=float)
    test_m = metrics.get("test", {})
    if "mae" in test_m:
        print(f"  [{name}] test MAE {test_m['mae']:.2f}  RMSE {test_m['rmse']:.2f}  "
              f"recall {test_m['spike_recall']:.2%}  precision {test_m['spike_precision']:.2%}")
    return record


def prediction_frame(parts: dict[str, tuple[pd.Index, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    frames = [pd.DataFrame({"split": split, "y_true": y, "y_pred": p}, index=idx)
              for split, (idx, y, p) in parts.items()]
    return pd.concat(frames)


# --- XGBoost with and without GDELT (notebook 01) -----------------------

def _xgb_regressor(**overrides):
    import xgboost as xgb

    params = dict(n_estimators=700, max_depth=7, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist",
                  early_stopping_rounds=30, eval_metric="rmse", n_jobs=-1, random_state=42)
    params.update(overrides)
    return xgb.XGBRegressor(**params)


def run_xgb() -> None:
    mat = load_matrix(transform=True)
    features = select_enhanced_features(mat)
    train, val, test = frozen_split(mat)
    no_gdelt = [f for f in features if not f.startswith("gdelt_")]
    for name, feats in (("xgb_gdelt_nb01", features), ("xgb_no_gdelt_nb01", no_gdelt)):
        X = {s: d[feats].ffill().fillna(0) for s, d in (("train", train), ("val", val), ("test", test))}
        model = _xgb_regressor()
        model.fit(X["train"], train[TARGET_REG], eval_set=[(X["val"], val[TARGET_REG])], verbose=False)
        preds = {s: model.predict(X[s]) for s in ("val", "test")}
        save_run(name,
                 prediction_frame({s: (d.index, d[TARGET_REG].values, preds[s])
                                   for s, d in (("val", val), ("test", test))}),
                 {s: regression_metrics(d[TARGET_REG].values, preds[s])
                  for s, d in (("val", val), ("test", test))},
                 {"source": "notebooks/01_xgboost.ipynb cells 3 and 10-11", "n_features": len(feats),
                  "transforms": True, "best_iteration": int(model.best_iteration),
                  "params": "700 trees, depth 7, lr 0.05, early stopping 30 on val RMSE"})


# --- Paper-era pipeline: day-ahead table, 24h alerts, lead times ---------

def fit_alert_classifier(df: pd.DataFrame, features: list[str], seed: int = 42,
                         extra_params: dict | None = None):
    """train_all_v2._train_classifier, then both calibrators of calibrate_classifier.py.

    Returns the booster, {"platt": cal, "isotonic": cal}, and each calibrator's
    test Brier score (the paper picked the lowest).
    """
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    tr, vl, te = chronological_split(df.dropna(subset=[TARGET_CLASS]))
    X = {s: d[features].ffill().fillna(0) for s, d in (("train", tr), ("val", vl), ("test", te))}
    y_tr = tr[TARGET_CLASS]
    params = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, objective="binary:logistic", eval_metric="aucpr",
                  scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                  tree_method="hist", early_stopping_rounds=30, n_jobs=-1, random_state=seed)
    params.update(extra_params or {})
    clf = xgb.XGBClassifier(**params)
    clf.fit(X["train"], y_tr, eval_set=[(X["val"], vl[TARGET_CLASS])], verbose=False)

    calibrators = {}
    for label, method in (("platt", "sigmoid"), ("isotonic", "isotonic")):
        cal = CalibratedClassifierCV(FrozenEstimator(clf), method=method)
        cal.fit(X["val"], vl[TARGET_CLASS].values)
        calibrators[label] = cal
    test_brier = {k: float(brier_score_loss(te[TARGET_CLASS].values, c.predict_proba(X["test"])[:, 1]))
                  for k, c in calibrators.items()}
    return clf, calibrators, test_brier


def run_paper_pipeline() -> None:
    """Re-run the untransformed train_all_v2 + calibrate_classifier pipeline.

    evaluation/dam_benchmark.json and backtest_results.json (2026-05-04) were
    produced before log1p transforms were added to the scripts (commit a8a4a57),
    so this path reads the raw matrix, as benchmark_dam.py and backtest.py do.
    The calibrator is chosen by lowest *test* Brier and the alert threshold is the
    fixed 0.3, exactly as in the paper; both are re-chosen on validation in
    models/improve_alerts.py.
    """
    import xgboost as xgb

    from evaluation.backtest import STRESS_EVENTS, _summarize_event

    df = load_matrix(transform=False)
    features = select_enhanced_features(df)

    # Regressor (train_all_v2._train_regressor) on every row with a target; the
    # day-ahead comparison then scores rows that also have a day-ahead price.
    train, val, test = frozen_split(df)
    val, test = (d.dropna(subset=["da_HB_HUBAVG"]) for d in (val, test))
    assert_test_index(test)
    X = {s: d[features].ffill().fillna(0) for s, d in (("train", train), ("val", val), ("test", test))}
    reg = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist",
                           early_stopping_rounds=25, eval_metric="rmse", n_jobs=-1, random_state=42)
    reg.fit(X["train"], train[TARGET_REG], eval_set=[(X["val"], val[TARGET_REG])], verbose=False)
    for name, pred_fn in (("enhanced_v2_paper", lambda s: reg.predict(X[s])),
                          ("dam", lambda s: {"val": val, "test": test}[s]["da_HB_HUBAVG"].values)):
        preds = {s: pred_fn(s) for s in ("val", "test")}
        save_run(name,
                 prediction_frame({s: (d.index, d[TARGET_REG].values, preds[s])
                                   for s, d in (("val", val), ("test", test))}),
                 {s: regression_metrics(d[TARGET_REG].values, preds[s])
                  for s, d in (("val", val), ("test", test))},
                 {"source": "models/train_all_v2.py at 609b629 (no transforms) + evaluation/benchmark_dam.py",
                  "n_features": len(features), "transforms": False,
                  "best_iteration": int(reg.best_iteration) if name != "dam" else None})

    clf, calibrators, test_brier = fit_alert_classifier(df, features)
    best = min(test_brier, key=test_brier.get)
    cal = calibrators[best]

    # Alert scores on the day-ahead benchmark's test rows (benchmark_dam.py part B).
    proba = {s: cal.predict_proba(X[s])[:, 1] for s in ("val", "test")}
    labels = {s: d[TARGET_CLASS].fillna(0).astype(bool).values for s, d in (("val", val), ("test", test))}
    alert_pred = pd.concat([
        pd.DataFrame({"split": s, "y_true": labels[s], "proba": proba[s]}, index=d.index)
        for s, d in (("val", val), ("test", test))])
    save_run("alert_24h_paper", alert_pred,
             {s: alert_metrics(labels[s], proba[s]) for s in ("val", "test")},
             {"source": "train_all_v2._train_classifier + calibrate_classifier.py + benchmark_dam.py",
              "calibrator": best, "calibrator_choice": "lowest test Brier (paper protocol)",
              "test_brier": test_brier, "threshold_choice": "fixed 0.3 (paper protocol)",
              "best_iteration": int(clf.best_iteration)})

    # Stress-event lead times (evaluation/backtest.py): score the whole matrix at once.
    full = df.copy()
    full["spike_probability_calibrated"] = cal.predict_proba(full[features].ffill().fillna(0))[:, 1]
    events = [_summarize_event(full, *ev) for ev in STRESS_EVENTS]
    out = RUNS_DIR / "lead_times_paper"
    out.mkdir(parents=True, exist_ok=True)
    full[["HB_HUBAVG", "spike_probability_calibrated"]].to_parquet(out / "predictions.parquet")
    with open(out / "metrics.json", "w") as f:
        json.dump({"run": "lead_times_paper", "events": events, "alert_threshold": ALERT_THRESHOLD,
                   "calibrator": best, "provenance": provenance()}, f, indent=2, default=str)
    for ev in events:
        lead = ev.get("lead_time_hours")
        print(f"  {ev['event']:<28} lead {lead if lead is not None else '-':>6}  "
              f"(paper {PAPER['lead_times'].get(ev['event'], '-')})")


# --- Chronos zero-shot ---------------------------------------------------

def run_chronos() -> None:
    """Zero-shot Chronos scored from the faithful backfill checkpoint.

    The backfill uses the same context (the prior 512 rows of HB_HUBAVG), model,
    and median as notebooks/07_chronos_and_hybrid.ipynb cell 5.
    """
    mat = load_matrix(transform=False)
    _, val, test = frozen_split(mat)
    cp = pd.read_parquet(CHRONOS_PRED_PATH)["chronos_pred"]
    missing = test.index.difference(cp.index)
    if len(missing):
        raise SystemExit(f"{len(missing)} test hours have no Chronos prediction; finish the backfill.")
    preds = {"val": cp.reindex(val.index).values, "test": cp.reindex(test.index).values}
    save_run("chronos_zero_shot",
             prediction_frame({s: (d.index, d[TARGET_REG].values, preds[s])
                               for s, d in (("val", val), ("test", test))}),
             {s: regression_metrics(d[TARGET_REG].values, preds[s])
              for s, d in (("val", val), ("test", test))},
             {"source": "data/features/chronos_pred.parquet (models/train_chronos_hybrid.py backfill)",
              "model": "amazon/chronos-t5-small", "context": 512, "num_samples": 20, "seed": 0})


def run_chronos_seeds(seeds: list[int], batch_size: int = 64) -> None:
    """Independent test-year Chronos runs with notebook 07's batch size, to show sampling noise."""
    import torch
    from chronos import ChronosPipeline

    mat = load_matrix(transform=False)
    _, _, test = frozen_split(mat)
    series = mat.dropna(subset=[TARGET_REG]).sort_index()[TARGET_REG].astype("float32")
    positions = series.index.get_indexer(test.index)
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cuda",
                                           torch_dtype=torch.bfloat16)
    for seed in seeds:
        torch.manual_seed(seed)
        out = []
        for start in range(0, len(positions), batch_size):
            ctx = [torch.tensor(series.iloc[p - 512:p].to_numpy()) for p in positions[start:start + batch_size]]
            with torch.inference_mode():
                samples = pipe.predict(ctx, prediction_length=1)
            out.extend(samples[:, :, 0].median(dim=1).values.float().cpu().numpy().tolist())
        pred = np.asarray(out)
        save_run(f"chronos_zero_shot_seed{seed}",
                 prediction_frame({"test": (test.index, test[TARGET_REG].values, pred)}),
                 {"test": regression_metrics(test[TARGET_REG].values, pred)},
                 {"source": "notebooks/07_chronos_and_hybrid.ipynb cell 5 settings", "seed": seed,
                  "batch_size": batch_size})


# --- Faithful hybrid -----------------------------------------------------

def hybrid_frame(chronos_path: Path = CHRONOS_PRED_PATH, columns=("chronos_pred",)):
    """The transformed v2 matrix restricted to hours with a Chronos forecast, as in
    train_chronos_hybrid.train_hybrid, with the requested Chronos columns appended."""
    mat = load_matrix(transform=True)
    features = select_enhanced_features(mat)
    cp = pd.read_parquet(chronos_path)
    aug = mat.loc[mat.index.intersection(cp.index)].copy()
    for col in columns:
        aug[col] = cp[col].reindex(aug.index).values
    return aug, features

def run_hybrid() -> None:
    if not HYBRID_PREDICTIONS.exists():
        raise SystemExit(f"Missing {HYBRID_PREDICTIONS}; run models/train_chronos_hybrid.py first.")
    preds = pd.read_parquet(HYBRID_PREDICTIONS)
    assert_test_index(preds[preds["split"] == "test"].rename(columns={"y_true": TARGET_REG}))
    report = json.loads(Path("models/artifacts/chronos_hybrid_report.json").read_text())
    save_run("chronos_xgb_faithful", preds,
             {s: regression_metrics(g["y_true"].values, g["y_pred"].values)
              for s, g in preds.groupby("split")},
             {"source": "models/train_chronos_hybrid.py (notebook 07 hybrid cells)",
              "n_features": report.get("n_features"),
              "chronos_pred_importance_rank": report.get("chronos_pred_importance_rank")})


# --- Seed sweep ----------------------------------------------------------

SWEEP_KEYS = ("mae", "rmse", "spike_recall", "spike_precision", "spike_pr_auc")


def run_seed_sweep(seeds: list[int]) -> None:
    """Retrain each faithful configuration with other XGBoost seeds.

    XGBoost row and column subsampling draws from the C++ standard library's
    distributions, which differ between macOS (libc++) and Linux (libstdc++), so
    a Mac-trained model cannot be matched bit for bit here. The sweep shows how
    far the reported numbers sit from this platform's run-to-run spread.
    """
    rows = []
    mat = load_matrix(transform=True)
    feats = select_enhanced_features(mat)
    no_gdelt = [f for f in feats if not f.startswith("gdelt_")]
    splits_t = frozen_split(mat)
    raw = load_matrix(transform=False)
    feats_raw = select_enhanced_features(raw)
    splits_raw = frozen_split(raw)
    splits_h = None
    if CHRONOS_PRED_PATH.exists():
        aug, _ = hybrid_frame()
        splits_h = frozen_split(aug)

    def reg_row(seed, run, model, features, splits):
        tr, va, te = splits
        X = {s: d[features].ffill().fillna(0) for s, d in (("train", tr), ("val", va), ("test", te))}
        model.fit(X["train"], tr[TARGET_REG], eval_set=[(X["val"], va[TARGET_REG])], verbose=False)
        m = regression_metrics(te[TARGET_REG].values, model.predict(X["test"]))
        rows.append({"seed": seed, "run": run, "best_iteration": int(model.best_iteration),
                     **{k: m[k] for k in SWEEP_KEYS}})

    for seed in seeds:
        reg_row(seed, "xgb_gdelt_nb01", _xgb_regressor(random_state=seed), feats, splits_t)
        reg_row(seed, "xgb_no_gdelt_nb01", _xgb_regressor(random_state=seed), no_gdelt, splits_t)
        reg_row(seed, "enhanced_v2_paper",
                _xgb_regressor(n_estimators=500, max_depth=6, early_stopping_rounds=25, random_state=seed),
                feats_raw, splits_raw)
        if splits_h is not None:
            reg_row(seed, "chronos_xgb_faithful",
                    _xgb_regressor(n_estimators=500, max_depth=6, random_state=seed),
                    feats + ["chronos_pred"], splits_h)
        clf, cals, brier = fit_alert_classifier(raw, feats_raw, seed=seed)
        te = splits_raw[2]
        best = min(brier, key=brier.get)
        m = alert_metrics(te[TARGET_CLASS].fillna(0).astype(bool).values,
                          cals[best].predict_proba(te[feats_raw].ffill().fillna(0))[:, 1])
        rows.append({"seed": seed, "run": "alert_24h_paper", "best_iteration": int(clf.best_iteration),
                     "recall": m["recall"], "precision": m["precision"], "f1": m["f1"], "pr_auc": m["pr_auc"]})
        print(f"  seed {seed} done", flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(RUNS_DIR / "seed_sweep.csv", index=False)
    print(out.groupby("run").agg(["min", "median", "max"]).T.to_string())


# --- Leaderboard ---------------------------------------------------------

def _load(name: str) -> dict | None:
    path = RUNS_DIR / name / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(v, pct: bool) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{100 * v:.2f}%" if pct else f"{v:.2f}"


HONEST_RUNS = [
    ("persistence", "Persistence y[t-1]"),
    ("seasonal_naive_24", "Seasonal naive y[t-24]"),
    ("seasonal_naive_168", "Seasonal naive y[t-168]"),
    ("weather_lag_1", "Weather lagged 1 hour"),
    ("weather_lag_120", "Weather lagged 120 hours"),
    ("gas_train_p90", "Train-only gas percentile"),
    ("gdelt_lag_24", "GDELT lagged one day"),
    ("unshifted", "Unshifted improved config, seed 42"),
    ("honest_tree", "Honest feature set, 2022-chosen window"),
    ("lear", "LEAR"),
    ("ensemble", "2023-chosen combiner"),
]


def _honest_lines() -> list[str]:
    present = [(label, run, _load(run)) for run, label in HONEST_RUNS if _load(run)]
    if not present:
        return []
    lines = ["", "## Honest MAE (settings chosen before the 2024 score)", "",
             "| Result | Run | MAE | RMSE | Normal MAE | Stress MAE | Spike recall | Spike precision |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, run, rec in present:
        m = rec["metrics"]["test"]
        lines.append(
            f"| {label} | `{run}` | {_fmt(m.get('mae'), False)} | {_fmt(m.get('rmse'), False)} | "
            f"{_fmt(m.get('normal_mae'), False)} | {_fmt(m.get('stress_mae'), False)} | "
            f"{_fmt(m.get('spike_recall'), True)} | {_fmt(m.get('spike_precision'), True)} |")
    checks = RUNS_DIR / "honest_mae_checks.json"
    if checks.exists():
        dm = json.loads(checks.read_text()).get("diebold_mariano", [])
        kept = [r for r in dm if r["model"] in ("honest_tree", "lear", "ensemble")
                and r["baseline"] in ("dam", "persistence")]
        if kept:
            lines += ["", "Diebold-Mariano on absolute error, lag 24. "
                      "A negative statistic means the model beats the baseline.", "",
                      "| Model | Baseline | Mean loss gap | Stat | p |",
                      "| --- | --- | ---: | ---: | ---: |"]
            for r in kept:
                lines.append(f"| `{r['model']}` | `{r['baseline']}` | {r['mean_diff']:.2f} | "
                             f"{r['stat']:.2f} | {r['p_value']:.3f} |")
    return lines


def write_leaderboard() -> Path:
    lines = ["# Paper vs reproduced (2024 test year)", "",
             f"Test index: {TEST_ROWS} hours, {TEST_SPIKES} spike hours above ${SPIKE_PRICE:.0f}/MWh.",
             "Each cell is paper / reproduced.", "",
             "| Result | Run | MAE | RMSE | Spike recall | Spike precision | Spike PR-AUC |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    hashes = set()
    for run, key, label in REGRESSION_ROWS:
        rec = _load(run)
        if rec is None:
            continue
        hashes.add(rec["test_index_hash"])
        m, p = rec["metrics"]["test"], PAPER[key]
        cells = [f"{_fmt(p.get(k), pct)} / {_fmt(m.get(k), pct)}"
                 for k, pct in (("mae", False), ("rmse", False), ("spike_recall", True),
                                ("spike_precision", True), ("spike_pr_auc", False))]
        lines.append(f"| {label} | `{run}` | " + " | ".join(cells) + " |")

    for run, _, label in REGRESSION_ROWS:
        rec = _load(run)
        if rec is None or "test_std" not in rec["metrics"]:
            continue
        m, sd, n = rec["metrics"]["test"], rec["metrics"]["test_std"], len(rec["metrics"]["test_seeds"])
        lines += ["", f"`{run}` is the mean over {n} seeds, chosen on 2023 only: MAE {m['mae']:.2f} ± "
                      f"{sd['mae']:.2f}, RMSE {m['rmse']:.2f} ± {sd['rmse']:.2f}, recall "
                      f"{100 * m['spike_recall']:.1f}% ± {100 * sd['spike_recall']:.1f}, precision "
                      f"{100 * m['spike_precision']:.1f}% ± {100 * sd['spike_precision']:.1f}."]

    seeds = sorted(RUNS_DIR.glob("chronos_zero_shot_seed*/metrics.json"))
    if seeds:
        maes = [json.loads(s.read_text())["metrics"]["test"]["mae"] for s in seeds]
        lines += ["", f"Chronos zero-shot sampling noise over {len(maes)} extra seeds: "
                      f"MAE {np.mean(maes):.2f} (min {min(maes):.2f}, max {max(maes):.2f})."]

    sweep_path = RUNS_DIR / "seed_sweep.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        sweep_rows = [("chronos_xgb_faithful", "chronos_xgb"), ("xgb_gdelt_nb01", "xgb_gdelt"),
                      ("xgb_gdelt_nb01", "xgb"), ("xgb_no_gdelt_nb01", "xgb_no_gdelt"),
                      ("enhanced_v2_paper", "enhanced_vs_dam")]
        lines += ["", f"## XGBoost seed sweep ({sweep['seed'].nunique()} seeds on this platform)", "",
                  "Test-year ranges over XGBoost seeds; the faithful single runs above use seed 42.", "",
                  "| Run | Paper row | Paper MAE | MAE min / median / max | Paper inside MAE range? "
                  "| Recall min / max | Precision min / max |",
                  "| --- | --- | ---: | ---: | --- | ---: | ---: |"]
        for run, key in sweep_rows:
            s = sweep[sweep["run"] == run]
            if s.empty:
                continue
            p = PAPER[key]["mae"]
            inside = "yes" if s["mae"].min() <= p <= s["mae"].max() else "no"
            lines.append(
                f"| `{run}` | {key} | {p:.2f} | {s['mae'].min():.2f} / {s['mae'].median():.2f} / "
                f"{s['mae'].max():.2f} | {inside} | {100 * s['spike_recall'].min():.1f}% / "
                f"{100 * s['spike_recall'].max():.1f}% | {100 * s['spike_precision'].min():.1f}% / "
                f"{100 * s['spike_precision'].max():.1f}% |")
        a = sweep[sweep["run"] == "alert_24h_paper"]
        if not a.empty:
            lines.append(f"\n24h alert over seeds (paper protocol): recall {100 * a['recall'].min():.1f}–"
                         f"{100 * a['recall'].max():.1f}%, precision {100 * a['precision'].min():.1f}–"
                         f"{100 * a['precision'].max():.1f}%, F1 {100 * a['f1'].min():.1f}–"
                         f"{100 * a['f1'].max():.1f}% (paper 34.63% / 24.22% / 28.50%).")

    dam = {r: _load(r) for r in ("enhanced_v2_paper", "dam")}
    if all(dam.values()):
        lines += ["", "## Day-ahead comparison", "",
                  "| Metric | Enhanced (paper / ours) | Day-ahead (paper / ours) |", "| --- | ---: | ---: |"]
        for k, label, pct in (("mae", "Overall MAE", False), ("rmse", "Overall RMSE", False),
                              ("stress_mae", "Stress MAE (y > 100)", False),
                              ("spike_recall", "Spike recall", True),
                              ("spike_precision", "Spike precision", True), ("spike_f1", "Spike F1", True)):
            e, d = dam["enhanced_v2_paper"]["metrics"]["test"], dam["dam"]["metrics"]["test"]
            lines.append(f"| {label} | {_fmt(PAPER['enhanced_vs_dam'][k], pct)} / {_fmt(e[k], pct)} | "
                         f"{_fmt(PAPER['dam'][k], pct)} / {_fmt(d[k], pct)} |")

    alert_runs = [("alert_24h_paper", "Paper protocol (test-chosen calibrator, threshold 0.3)"),
                  ("alert_24h_improved", "Validation-chosen calibrator and threshold")]
    alert_recs = [(lbl, _load(r)) for r, lbl in alert_runs if _load(r)]
    if alert_recs:
        p = PAPER["alert_24h"]
        lines += ["", "## 24h alert classifier (test year)", "",
                  "| Setting | Threshold | Recall | Precision | F1 | PR-AUC |", "| --- | ---: | ---: | ---: | ---: | ---: |",
                  f"| Paper | 0.3 | {_fmt(p['recall'], True)} | {_fmt(p['precision'], True)} | "
                  f"{_fmt(p['f1'], True)} | - |"]
        for lbl, rec in alert_recs:
            m = rec["metrics"]["test"]
            lines.append(f"| {lbl} | {m['threshold']:.2f} | {_fmt(m['recall'], True)} | "
                         f"{_fmt(m['precision'], True)} | {_fmt(m['f1'], True)} | {m['pr_auc']:.3f} |")
        improved = _load("alert_24h_improved")
        if improved and "test_std" in improved["metrics"]:
            sd, n = improved["metrics"]["test_std"], len(improved["metrics"]["test_seeds"])
            lines.append(f"\nThe validation-chosen row is the mean over {n} classifier seeds (F1 ± "
                         f"{100 * sd['f1']:.1f} points); each seed re-chooses its calibrator and threshold "
                         f"on 2023. Configuration: `{json.dumps(improved['meta']['config'], sort_keys=True)}`.")

    lead_runs = [("lead_times_paper", "paper protocol"), ("lead_times_improved", "validation-chosen")]
    lead_recs = [(lbl, _load(r)) for r, lbl in lead_runs if _load(r)]
    if lead_recs:
        lines += ["", "## Stress-event lead times (hours from first alert to first spike)", "",
                  "| Event | Split | Paper | " + " | ".join(lbl for lbl, _ in lead_recs) + " |",
                  "| --- | --- | ---: | " + " | ".join("---:" for _ in lead_recs) + " |"]
        for i, ev in enumerate(lead_recs[0][1]["events"]):
            cells = []
            for _, rec in lead_recs:
                e = rec["events"][i]
                if not e.get("actual_spike_occurred"):
                    cells.append("no spike")
                else:
                    cell = (_fmt(e.get("lead_time_hours"), False) if e.get("lead_time_hours") is not None
                            else "missed")
                    if e.get("lead_time_range"):
                        lo, hi = e["lead_time_range"]
                        cell += f" (seeds {lo:.0f}–{hi:.0f})"
                    cells.append(cell)
            paper = PAPER["lead_times"].get(ev["event"])
            lines.append(f"| {ev['event']} | {ev['split']} | {_fmt(paper, False) if paper else '-'} | "
                         + " | ".join(cells) + " |")

    improved_leads = _load("lead_times_improved")
    if improved_leads and "event_level" in improved_leads:
        lv = improved_leads["event_level"]
        lines += ["", "## Event-level detection (24h horizon, all spike events in the year)", "",
                  lv["definition"] + ". Lead is hours from the alert's information time to onset.", "",
                  "| Alert | Year | Threshold | Events detected | Median lead (h) | Alert episodes "
                  "| Episode precision | False episodes per 30 days |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        rows = [("Paper protocol", lv["paper_protocol"])] + \
               [(f"Validation-chosen, seed {i}", s) for i, s in enumerate(lv["improved_by_seed"][:1])]
        for lbl, by_split in rows:
            for split, year in (("val", 2023), ("test", 2024)):
                e = by_split[split]
                lines.append(f"| {lbl} | {year} | {e['threshold']:.2f} | {e['detected_events']} / {e['n_events']} | "
                             f"{_fmt(e['median_lead_hours'], False)} | {e['alert_episodes']} | "
                             f"{_fmt(e['event_precision'], True)} | {e['false_alert_episodes_per_30d']:.2f} |")

    lines += _honest_lines()
    if len(hashes) > 1:
        lines += ["", f"WARNING: runs disagree on the test index ({sorted(hashes)})."]
    elif hashes:
        lines += ["", f"All regression runs share test index hash `{hashes.pop()}`."]
    path = RUNS_DIR / "leaderboard.md"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["xgb", "paper-pipeline", "chronos", "chronos-seeds",
                                        "hybrid", "seed-sweep", "leaderboard", "naive"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="chronos-seeds default: 1 2 3; seed-sweep default: 0..9")
    args = ap.parse_args()
    {
        "xgb": run_xgb,
        "paper-pipeline": run_paper_pipeline,
        "chronos": run_chronos,
        "chronos-seeds": lambda: run_chronos_seeds(args.seeds or [1, 2, 3]),
        "seed-sweep": lambda: run_seed_sweep(args.seeds or list(range(10))),
        "hybrid": run_hybrid,
        "leaderboard": write_leaderboard,
        "naive": _run_naive,
    }[args.command]()


def _run_naive() -> None:
    from models.honest_mae import run_baselines
    run_baselines()


if __name__ == "__main__":
    main()
