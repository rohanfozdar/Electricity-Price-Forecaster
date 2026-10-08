"""
Validation-only search over Chronos-XGBoost variants, plus the alternative
architectures tried when the documented hybrid cannot reach the paper's scores.

Protocol
    Fit on 2016-2022, early-stop and rank on 2023. A configuration is eligible
    only if its 2023 spike F1 is at least the faithful hybrid's; among eligible
    configurations the lowest 2023 MAE wins. Every configuration tried is
    appended to runs/benchmark/search_log.csv with validation scores only. The
    2024 test year is scored once per finalist, over 5 seeds.

Subcommands
    search        Phase 2: loss, asinh target, residual target, Chronos quantile
                  features, gap-aware context, then a hyperparameter grid
    alternatives  Chronos-Bolt inputs, LightGBM and CatBoost second stages,
                  day-ahead residual, Chronos blend, spike-gated two-stage model
    finalize      Score a selected configuration on 2024 over 5 seeds
                  (--which improved | alternative)

Run from the repository root:
    python models/improve_hybrid.py search
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import FEATURES_DIR
from models.run_benchmark import (
    RUNS_DIR,
    SPIKE_PRICE,
    hybrid_frame,
    load_matrix,
    frozen_split,
    regression_metrics,
    save_run,
)
from models.utils import TARGET_REG

XGB_DEVICE = os.environ.get("ERCOT_XGB_DEVICE", "cuda")
CPU_THREADS = int(os.environ.get("ERCOT_CPU_THREADS", "16"))  # shared host; -1 oversubscribes
SEARCH_LOG = RUNS_DIR / "search_log.csv"
FINAL_SEEDS = [0, 1, 2, 3, 4]

CHRONOS_SOURCES = {
    "t5_positional": FEATURES_DIR / "chronos_pred.parquet",
    "t5_gap": FEATURES_DIR / "chronos_hourly" / "t5-small_gap-aware_ctx512.parquet",
    "bolt_small_positional": FEATURES_DIR / "chronos_hourly" / "bolt-small_positional_ctx512.parquet",
    "bolt_small_gap": FEATURES_DIR / "chronos_hourly" / "bolt-small_gap-aware_ctx512.parquet",
    "bolt_base_positional": FEATURES_DIR / "chronos_hourly" / "bolt-base_positional_ctx512.parquet",
    "bolt_base_gap": FEATURES_DIR / "chronos_hourly" / "bolt-base_gap-aware_ctx512.parquet",
}
CHRONOS_COLS = ["chronos_pred", "chronos_q10", "chronos_q90", "chronos_mean", "chronos_p_gt200"]
FEATSETS = {
    "median": ["chronos_pred"],
    "stats": CHRONOS_COLS + ["chronos_spread"],
}
ASINH_CLIP = 10.0  # sinh(10) is about $11,000/MWh, above ERCOT's offer cap

TARGETS = {
    "raw": (lambda d: d[TARGET_REG].values, lambda d, p: p),
    "asinh": (lambda d: np.arcsinh(d[TARGET_REG].values),
              lambda d, p: np.sinh(np.clip(p, -ASINH_CLIP, ASINH_CLIP))),
    "resid_chronos": (lambda d: d[TARGET_REG].values - d["chronos_pred"].values,
                      lambda d, p: d["chronos_pred"].values + p),
    "resid_da": (lambda d: d[TARGET_REG].values - d["da_raw"].values,
                 lambda d, p: d["da_raw"].values + p),
}

DEFAULTS = {"learner": "xgb", "depth": 6, "lr": 0.05, "mcw": 1}

_CACHE: dict = {}


# --- Data ----------------------------------------------------------------

def splits_for(source: str):
    if source not in _CACHE:
        if "da_raw" not in _CACHE:
            _CACHE["da_raw"] = load_matrix(transform=False)["da_HB_HUBAVG"]
        aug, base = hybrid_frame(CHRONOS_SOURCES[source], CHRONOS_COLS)
        aug["chronos_spread"] = aug["chronos_q90"] - aug["chronos_q10"]
        aug["da_raw"] = _CACHE["da_raw"].reindex(aug.index).values
        _CACHE[source] = (frozen_split(aug), base)
    return _CACHE[source]


def faithful_val_f1() -> float:
    rec = json.loads((RUNS_DIR / "chronos_xgb_faithful" / "metrics.json").read_text())
    return float(rec["metrics"]["val"]["spike_f1"])


# --- Learners ------------------------------------------------------------

def make_model(cfg: dict, seed: int):
    learner, obj = cfg["learner"], cfg["objective"]
    if learner == "xgb":
        import xgboost as xgb

        extra = {"huber_slope": 20.0} if obj == "huber" else {}
        return xgb.XGBRegressor(
            n_estimators=2000, max_depth=cfg["depth"], learning_rate=cfg["lr"],
            min_child_weight=cfg["mcw"], subsample=0.8, colsample_bytree=0.8,
            objective={"sq": "reg:squarederror", "abs": "reg:absoluteerror",
                       "huber": "reg:pseudohubererror"}[obj],
            tree_method="hist", device=XGB_DEVICE, early_stopping_rounds=50,
            eval_metric="mae", random_state=seed, **extra)
    if learner == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=cfg["lr"], num_leaves=2 ** cfg["depth"] - 1,
            min_child_samples=max(cfg["mcw"], 20), subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, objective={"sq": "regression", "abs": "regression_l1",
                                             "huber": "huber"}[obj],
            alpha=20.0, random_state=seed, n_jobs=CPU_THREADS, verbose=-1)
    if learner == "cat":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=3000, learning_rate=cfg["lr"], depth=cfg["depth"],
            loss_function={"sq": "RMSE", "abs": "MAE", "huber": "Huber:delta=20"}[obj],
            eval_metric="MAE", random_seed=seed, od_type="Iter", od_wait=50,
            thread_count=CPU_THREADS, allow_writing_files=False, verbose=False)
    raise ValueError(learner)


def fit_predict(cfg: dict, seed: int = 42, predict=("val",)):
    """Fit one configuration; return $/MWh predictions for the requested splits."""
    (tr, va, te), base = splits_for(cfg["source"])
    feats = base + FEATSETS[cfg["featset"]]
    encode, decode = TARGETS[cfg["target"]]
    X = {s: d[feats].ffill().fillna(0) for s, d in (("train", tr), ("val", va), ("test", te))}
    model = make_model(cfg, seed)
    if cfg["learner"] == "xgb":
        model.fit(X["train"], encode(tr), eval_set=[(X["val"], encode(va))], verbose=False)
        best = int(model.best_iteration)
    elif cfg["learner"] == "lgbm":
        import lightgbm as lgb

        model.fit(X["train"], encode(tr), eval_set=[(X["val"], encode(va))], eval_metric="l1",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        best = int(model.best_iteration_)
    else:
        model.fit(X["train"], encode(tr), eval_set=(X["val"], encode(va)), use_best_model=True)
        best = int(model.get_best_iteration())
    parts = {"val": va, "test": te}
    return {s: decode(parts[s], model.predict(X[s])) for s in predict}, best, parts


def fit_spike_classifier(cfg: dict, seed: int = 42):
    """Same-hour P(y > 200) on the hybrid features, for the gated two-stage model."""
    import xgboost as xgb

    (tr, va, te), base = splits_for(cfg["source"])
    feats = base + FEATSETS[cfg["featset"]]
    X = {s: d[feats].ffill().fillna(0) for s, d in (("train", tr), ("val", va), ("test", te))}
    y = {s: (d[TARGET_REG].values > SPIKE_PRICE).astype(int) for s, d in (("train", tr), ("val", va))}
    clf = xgb.XGBClassifier(
        n_estimators=2000, max_depth=cfg["depth"], learning_rate=cfg["lr"], subsample=0.8,
        colsample_bytree=0.8, objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=(y["train"] == 0).sum() / max(y["train"].sum(), 1),
        tree_method="hist", device=XGB_DEVICE, early_stopping_rounds=50, random_state=seed)
    clf.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=False)
    return {s: clf.predict_proba(X[s])[:, 1] for s in ("val", "test")}


def predict_config(cfg: dict, seed: int, predict=("val",)):
    """Predictions for any configuration kind: plain, blend, or gated."""
    kind = cfg.get("kind", "plain")
    if kind == "plain":
        preds, best, parts = fit_predict(cfg, seed, predict)
        return preds, best, parts
    if kind == "blend":
        preds, best, parts = fit_predict(cfg["component"], seed, predict)
        w = cfg["w"]
        return {s: w * parts[s]["chronos_pred"].values + (1 - w) * preds[s] for s in predict}, best, parts
    if kind == "gated":
        normal, best, parts = fit_predict(cfg["normal"], seed, predict)
        spike, _, _ = fit_predict(cfg["spike"], seed, predict)
        proba = fit_spike_classifier(cfg["spike"], seed)
        return {s: np.where(proba[s] > cfg["tau"], spike[s], normal[s]) for s in predict}, best, parts
    raise ValueError(kind)


# --- Search log ----------------------------------------------------------

def logged(stage: str, cfg: dict) -> pd.Series | None:
    """The existing log row for (stage, cfg), so an interrupted search resumes."""
    if not SEARCH_LOG.exists():
        return None
    log = pd.read_csv(SEARCH_LOG)
    hit = log[(log["stage"] == stage) & (log["config"] == json.dumps(cfg, sort_keys=True))]
    return hit.iloc[0] if len(hit) else None


def log_result(stage: str, cfg: dict, val_metrics: dict, best_iteration: int | None) -> dict:
    if logged(stage, cfg) is not None:
        return logged(stage, cfg).to_dict()
    row = {"stage": stage, "config": json.dumps(cfg, sort_keys=True),
           "best_iteration": best_iteration,
           "logged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    row.update({f"val_{k}": val_metrics[k] for k in
                ("mae", "rmse", "spike_recall", "spike_precision", "spike_f1", "spike_pr_auc")})
    SEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(SEARCH_LOG, mode="a", header=not SEARCH_LOG.exists(), index=False)
    print(f"  [{stage}] val MAE {val_metrics['mae']:7.2f}  F1 {val_metrics['spike_f1']:.3f}  "
          f"{json.dumps(cfg, sort_keys=True)[:150]}", flush=True)
    return row


def evaluate(stage: str, cfg: dict) -> dict:
    if logged(stage, cfg) is not None:
        return logged(stage, cfg).to_dict()
    preds, best, parts = predict_config(cfg, seed=42)
    return log_result(stage, cfg, regression_metrics(parts["val"][TARGET_REG].values, preds["val"]), best)


def ranked(stages: list[str], eligible_only: bool = True) -> pd.DataFrame:
    log = pd.read_csv(SEARCH_LOG)
    log = log[log["stage"].isin(stages)]
    if eligible_only:
        log = log[log["val_spike_f1"] >= faithful_val_f1() - 1e-12]
    return log.sort_values("val_mae")


def write_selection(name: str, stages: list[str]) -> dict:
    elig = ranked(stages)
    note = "lowest 2023 MAE among configurations with 2023 spike F1 >= faithful hybrid"
    if elig.empty:
        elig = ranked(stages, eligible_only=False)
        note = "no configuration met the F1 constraint; lowest 2023 MAE overall"
    top = elig.iloc[0]
    sel = {"config": json.loads(top["config"]), "stage": top["stage"], "selection_rule": note,
           "faithful_val_spike_f1": faithful_val_f1(),
           "val": {k[4:]: float(top[k]) for k in top.index if k.startswith("val_")},
           "n_configs_considered": int(len(ranked(stages, eligible_only=False)))}
    (RUNS_DIR / f"selected_{name}.json").write_text(json.dumps(sel, indent=2))
    print(f"\n  Selected ({name}): {json.dumps(sel, indent=2)}")
    return sel


# --- Phase 2 -------------------------------------------------------------

TARGET_OBJECTIVES = [("raw", "sq"), ("raw", "abs"), ("raw", "huber"), ("asinh", "sq"), ("asinh", "abs"),
                     ("resid_chronos", "sq"), ("resid_chronos", "abs"), ("resid_chronos", "huber")]
GRID = {"depth": [4, 6, 8], "lr": [0.03, 0.1], "mcw": [1, 10]}


def run_search() -> None:
    for source, featset, (target, objective) in itertools.product(
            ["t5_positional", "t5_gap"], ["median", "stats"], TARGET_OBJECTIVES):
        evaluate("p2-A", {**DEFAULTS, "source": source, "featset": featset,
                          "target": target, "objective": objective})

    top = ranked(["p2-A"])
    if len(top) < 3:
        top = pd.concat([top, ranked(["p2-A"], eligible_only=False)]).drop_duplicates("config")
    for cfg_json in top["config"].head(3):
        base_cfg = json.loads(cfg_json)
        for depth, lr, mcw in itertools.product(GRID["depth"], GRID["lr"], GRID["mcw"]):
            cfg = {**base_cfg, "depth": depth, "lr": lr, "mcw": mcw}
            if cfg == base_cfg:
                continue
            evaluate("p2-B", cfg)
    write_selection("improved", ["p2-A", "p2-B"])


# --- Alternatives (the plan's caveat) ------------------------------------

def run_alternatives() -> None:
    best_p2 = json.loads((RUNS_DIR / "selected_improved.json").read_text())["config"]
    p2_top = [json.loads(c) for c in ranked(["p2-A", "p2-B"])["config"].head(3)]
    if not p2_top:
        p2_top = [best_p2]

    # Chronos-Bolt inputs with the best Phase 2 target, objective, and hyperparameters.
    for source in [s for s in CHRONOS_SOURCES if s.startswith("bolt")]:
        for featset in FEATSETS:
            for cfg in p2_top:
                evaluate("alt-bolt", {**cfg, "source": source, "featset": featset})

    # Other second-stage learners on the best input so far.
    best_input = json.loads(ranked(["p2-A", "p2-B", "alt-bolt"])["config"].iloc[0]) \
        if not ranked(["p2-A", "p2-B", "alt-bolt"]).empty else best_p2
    for learner, objective, target in itertools.product(["lgbm", "cat"], ["sq", "abs", "huber"],
                                                        ["raw", "resid_chronos"]):
        evaluate("alt-learner", {**DEFAULTS, "learner": learner, "source": best_input["source"],
                                 "featset": best_input["featset"], "target": target,
                                 "objective": objective})

    # Residual on top of the day-ahead price.
    for source, objective in itertools.product(["t5_positional", best_input["source"]], ["sq", "abs", "huber"]):
        evaluate("alt-da-resid", {**DEFAULTS, "source": source, "featset": "stats",
                                  "target": "resid_da", "objective": objective})

    stages = ["p2-A", "p2-B", "alt-bolt", "alt-learner", "alt-da-resid"]
    best_plain = json.loads(ranked(stages)["config"].iloc[0]) if not ranked(stages).empty else best_p2

    # Weighted blend of zero-shot Chronos and the best tree model.
    preds, best, parts = fit_predict(best_plain, seed=42)
    y_val = parts["val"][TARGET_REG].values
    for w in np.round(np.linspace(0, 1, 11), 1):
        cfg = {"kind": "blend", "component": best_plain, "w": float(w)}
        blend = w * parts["val"]["chronos_pred"].values + (1 - w) * preds["val"]
        log_result("alt-blend", cfg, regression_metrics(y_val, blend), best)

    # Spike-gated two-stage model: a same-hour spike classifier routes each hour
    # to a squared-error spike regressor or to the best MAE-oriented regressor.
    spike_cfg = {**best_plain, "learner": "xgb", "target": "raw", "objective": "sq",
                 "depth": DEFAULTS["depth"], "lr": DEFAULTS["lr"], "mcw": DEFAULTS["mcw"]}
    normal, best_n, parts = fit_predict(best_plain, seed=42)
    spike, _, _ = fit_predict(spike_cfg, seed=42)
    proba = fit_spike_classifier(spike_cfg, seed=42)
    for tau in np.round(np.linspace(0.1, 0.9, 9), 1):
        cfg = {"kind": "gated", "normal": best_plain, "spike": spike_cfg, "tau": float(tau)}
        gated = np.where(proba["val"] > tau, spike["val"], normal["val"])
        log_result("alt-gated", cfg, regression_metrics(y_val, gated), best_n)

    write_selection("alternative", ["alt-bolt", "alt-learner", "alt-da-resid", "alt-blend", "alt-gated"])


# --- Finalize ------------------------------------------------------------

METRIC_KEYS = ("mae", "rmse", "spike_recall", "spike_precision", "spike_f1", "spike_pr_auc")


def finalize(which: str) -> None:
    sel = json.loads((RUNS_DIR / f"selected_{which}.json").read_text())
    cfg = sel["config"]
    per_seed = {"val": [], "test": []}
    frames = {}
    for seed in FINAL_SEEDS:
        preds, best, parts = predict_config(cfg, seed=seed, predict=("val", "test"))
        for s in ("val", "test"):
            per_seed[s].append(regression_metrics(parts[s][TARGET_REG].values, preds[s]))
            frames.setdefault(s, pd.DataFrame({"split": s, "y_true": parts[s][TARGET_REG].values},
                                              index=parts[s].index))
            frames[s][f"y_pred_seed{seed}"] = preds[s]
        print(f"  seed {seed}: test MAE {per_seed['test'][-1]['mae']:.2f}", flush=True)
    for s in frames:
        frames[s]["y_pred"] = frames[s][f"y_pred_seed{FINAL_SEEDS[0]}"]
    metrics = {s: {k: float(np.mean([m[k] for m in per_seed[s]])) for k in METRIC_KEYS} for s in per_seed}
    metrics["test_std"] = {k: float(np.std([m[k] for m in per_seed["test"]])) for k in METRIC_KEYS}
    metrics["test_seeds"] = per_seed["test"]
    save_run(f"chronos_xgb_{which}", pd.concat([frames["val"], frames["test"]]), metrics,
             {"config": cfg, "selection": sel, "seeds": FINAL_SEEDS, "xgb_device": XGB_DEVICE,
              "test_metrics_are": "mean over seeds; y_pred is the first seed"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["search", "alternatives", "finalize"])
    ap.add_argument("--which", choices=["improved", "alternative"], default="improved")
    args = ap.parse_args()
    if args.command == "search":
        run_search()
    elif args.command == "alternatives":
        run_alternatives()
    else:
        finalize(args.which)


if __name__ == "__main__":
    main()
