"""E1 historical-coverage baselines, E2 standalone Chronos, E3 OOF LEAR-style augmentation.

E1 selects R1/R2/R3 on the 2023 five-seed MAE, writes the selection record, and refuses
to score 2024 until that record is committed.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from models.pes2027.common import (
    RUNS, SEEDS, TARGET, TEST_START, TRAIN_START, VAL_START, git_revision, hourly_max,
    index_hash, utc_now, write_csv, write_json,
)
from models.pes2027.features_extra import oof_lear, representation
from models.pes2027.fit import baseline_columns, fit_one, require, save_pred, seed_mean, split_index
from models.pes2027.indep_stats import paired_summary, point_metrics
from models.pes2027.protocol import require_committed
from models.run_pit_ablation import load_sources
from models.run_pit_verified import build_spec, finite_slice, study_sources
from src.features.pit_latency import PRIMARY_PRICE_DELAY_HOURS


def _frame(sources, extra: pd.DataFrame | None = None):
    spec = {"name": "delay_1h", "price": 1, "load": 1, "drop_load": False}
    frame, _, _ = build_spec(sources, spec, "max", False, sources["cache"], sources["rt"][TARGET])
    if extra is not None:
        overlap = [c for c in extra.columns if c in frame.columns]
        extra = extra.drop(columns=overlap)
        frame = frame.join(extra, how="left")
    return frame


def _splits(frame):
    kept = frame.index[(frame.index >= TRAIN_START) & (frame.index < pd.Timestamp("2025-01-01", tz="UTC"))]
    return split_index(finite_slice(frame, kept), VAL_START, TEST_START)


def _load():
    raw = study_sources(load_sources())
    from pathlib import Path
    raw["cache"] = pd.read_parquet(Path(__file__).resolve().parents[2] / "runs" / "pit_verified" / "chronos" / "delay_1h.parquet")
    return raw


def e1_select() -> dict:
    frozen = require()
    sources = _load()
    hourly = hourly_max(sources["rt"][TARGET])
    base_frame = _frame(sources)
    train, val, test = _splits(base_frame)
    base_cols = baseline_columns(base_frame)
    rows, store = [], {}
    for name in ("R1", "R2", "R3"):
        extra = representation(name, hourly, base_frame.index, PRIMARY_PRICE_DELAY_HOURS)
        frame = _frame(sources, extra)
        cols = base_cols + [c for c in extra.columns if c not in base_cols]
        val_preds = []
        print(f"E1 {name}: {len(cols)} columns, train={len(train)} val={len(val)}", flush=True)
        for seed in SEEDS:
            rec = fit_one(frame, cols, train, val, test, seed)
            val_preds.append(rec["val"][1])
            rows.append({"rep": name, "seed": seed, "best_iteration": rec["best_iteration"],
                         "n_columns": rec["n_columns"], "seconds": rec["seconds"],
                         "val_mae": point_metrics(rec["val"][0], rec["val"][1])["mae"]})
            print(f"  {name} seed {seed}: val_mae={rows[-1]['val_mae']:.6f} best={rec['best_iteration']} {rec['seconds']:.1f}s", flush=True)
        y = rec["val"][0]
        avg = seed_mean(val_preds)
        store[name] = {"val_mae_seed_avg": float(np.mean(np.abs(avg - y))), "n_columns": len(cols),
                       "columns": cols}
        rows.append({"rep": name, "seed": "avg", "best_iteration": None, "n_columns": len(cols),
                     "seconds": None, "val_mae": store[name]["val_mae_seed_avg"]})
    chosen = min(store, key=lambda k: store[k]["val_mae_seed_avg"])
    out = RUNS / "E1-H512"
    write_csv(out / "selection_val2023.csv", pd.DataFrame(rows))
    record = {
        "created_utc": utc_now(), "protocol": frozen, "code_revision": git_revision(),
        "selected": chosen, "rule": "minimum 2023 five-seed-average MAE",
        "candidates": {k: {"val_mae_seed_avg": v["val_mae_seed_avg"], "n_columns": v["n_columns"]} for k, v in store.items()},
        "selected_columns": store[chosen]["columns"],
        "train_rows": int(len(train)), "val_rows": int(len(val)),
        "scored_2024": False,
    }
    write_json(out / "selection.json", record)
    print(record)
    return record


def e1_score_2024() -> dict:
    frozen = require()
    sel_path = RUNS / "E1-H512" / "selection.json"
    recorded = require_committed(sel_path)
    selection = __import__("json").loads(sel_path.read_text())
    if selection["scored_2024"]:
        raise SystemExit("2024 already marked scored")
    sources = _load()
    hourly = hourly_max(sources["rt"][TARGET])
    extra = representation(selection["selected"], hourly, _frame(sources).index, PRIMARY_PRICE_DELAY_HOURS)
    frame = _frame(sources, extra)
    train, val, test = _splits(frame)
    cols_b512 = selection["selected_columns"]
    cols_plus = cols_b512 + [c for c in ("chronos_pred", "chronos_q10", "chronos_q90", "chronos_mean",
                                         "chronos_p_gt200", "chronos_spread") if c not in cols_b512]
    orig_plus = baseline_columns(frame) + [c for c in cols_plus if c.startswith("chronos_")]
    preds = {}
    rows = []
    for label, cols in (("B512", cols_b512), ("B512_chronos", cols_plus), ("original_both_on_B512_index", orig_plus)):
        stack = []
        for seed in SEEDS:
            rec = fit_one(frame, cols, train, val, test, seed)
            stack.append(rec["test"][1])
            save_pred(RUNS / "E1-H512" / "preds" / f"{label}_seed{seed}.parquet", test, rec["test"][0], rec["test"][1],
                      {"label": label, "seed": seed, "best_iteration": rec["best_iteration"], "protocol": frozen})
            rows.append({"label": label, "seed": seed, "best_iteration": rec["best_iteration"],
                         "test_mae": point_metrics(rec["test"][0], rec["test"][1])["mae"]})
        preds[label] = (rec["test"][0], seed_mean(stack))
    y = preds["B512"][0]
    summary = {
        "selected": selection["selected"], "selection_commit": recorded,
        "n": int(len(test)), "index_hash": index_hash(test),
        "mae": {k: point_metrics(y, p)["mae"] for k, (_, p) in preds.items()},
        "B512_chronos_minus_B512": paired_summary(y, preds["B512_chronos"][1], preds["B512"][1], test),
        "original_both_minus_B512": paired_summary(y, preds["original_both_on_B512_index"][1], preds["B512"][1], test),
    }
    write_csv(RUNS / "E1-H512" / "fits.csv", pd.DataFrame(rows))
    write_json(RUNS / "E1-H512" / "score_2024.json", {**summary, "protocol": frozen, "created_utc": utc_now()})
    selection["scored_2024"] = True
    # do not rewrite selection.json here; it must stay the committed record
    print({k: summary[k] for k in ("selected", "mae")})
    return summary


def e2_standalone() -> dict:
    frozen = require()
    pred = pd.read_parquet(
        __import__("pathlib").Path(__file__).resolve().parents[2] / "runs" / "pit_verified" / "predictions" /
        "delay_1h__max__asinh__xgb_exog__seed0.parquet"
    )
    cache = pd.read_parquet(
        __import__("pathlib").Path(__file__).resolve().parents[2] / "runs" / "pit_verified" / "chronos" / "delay_1h.parquet"
    )
    y = pred["y_true"]
    p = cache.reindex(pred.index)["chronos_pred"]
    keep = y.notna() & p.notna()
    y, p, idx = y[keep].to_numpy(), p[keep].to_numpy(), pred.index[keep]
    persist = pd.read_parquet(
        __import__("pathlib").Path(__file__).resolve().parents[2] / "runs" / "pes2027" / "E0-REPRO" / "primary_per_origin.parquet"
    )
    out = {
        "n": int(keep.sum()), "index_hash": index_hash(idx),
        "metrics": point_metrics(y, p),
        "vs_persistence": paired_summary(persist.loc[idx, "y_true"], p, persist.loc[idx, "pred_persistence_t_minus_2"], idx),
        "vs_xgb_exog": paired_summary(persist.loc[idx, "y_true"], p, persist.loc[idx, "pred_xgb_exog_seed_avg"], idx),
        "protocol": frozen, "created_utc": utc_now(),
    }
    write_json(RUNS / "E2-CHR" / "standalone.json", out)
    save_pred(RUNS / "E2-CHR" / "chronos_pred.parquet", idx, y, p, {"model": "chronos_pred", "protocol": frozen})
    print(out["metrics"]["mae"])
    return out


def e3_oof() -> dict:
    frozen = require()
    sources = _load()
    hourly = hourly_max(sources["rt"][TARGET])
    pred, folds = oof_lear(hourly, sources["da"], years=range(2019, 2025), delay_hours=1)
    write_json(RUNS / "E3-LEARAUG" / "folds.json", {"folds": [{k: v for k, v in f.items() if k != "path"} for f in folds],
                                                    "paths": {str(f["year"]): f["path"] for f in folds}})
    frame = _frame(sources)
    frame["lear_oof_pred"] = pred.reindex(frame.index)
    frame["miss_lear"] = frame["lear_oof_pred"].isna().astype(float)
    train, val, test = _splits(frame)
    base = baseline_columns(frame)
    lear_cols = base + ["lear_oof_pred", "miss_lear"]
    both_cols = lear_cols + [c for c in ("chronos_pred", "chronos_q10", "chronos_q90", "chronos_mean",
                                         "chronos_p_gt200", "chronos_spread") if c not in lear_cols]
    preds, rows = {}, []
    for label, cols in (("xgb", base), ("xgb_lear", lear_cols), ("xgb_lear_chronos", both_cols)):
        stack = []
        for seed in SEEDS:
            rec = fit_one(frame, cols, train, val, test, seed)
            stack.append(rec["test"][1])
            save_pred(RUNS / "E3-LEARAUG" / "preds" / f"{label}_seed{seed}.parquet", test, rec["test"][0], rec["test"][1],
                      {"label": label, "seed": seed, "best_iteration": rec["best_iteration"]})
            rows.append({"label": label, "seed": seed, "best_iteration": rec["best_iteration"],
                         "test_mae": point_metrics(rec["test"][0], rec["test"][1])["mae"]})
        preds[label] = (rec["test"][0], seed_mean(stack))
    y = preds["xgb"][0]
    summary = {
        "label": "not strictly matched",
        "mae": {k: point_metrics(y, p)["mae"] for k, (_, p) in preds.items()},
        "xgb_lear_minus_xgb": paired_summary(y, preds["xgb_lear"][1], preds["xgb"][1], test),
        "xgb_lear_chronos_minus_xgb_lear": paired_summary(y, preds["xgb_lear_chronos"][1], preds["xgb_lear"][1], test),
        "protocol": frozen, "created_utc": utc_now(),
    }
    write_csv(RUNS / "E3-LEARAUG" / "fits.csv", pd.DataFrame(rows))
    write_json(RUNS / "E3-LEARAUG" / "summary.json", summary)
    print(summary["mae"])
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["e1-select", "e1-score", "e2", "e3"])
    args = p.parse_args()
    {"e1-select": e1_select, "e1-score": e1_score_2024, "e2": e2_standalone, "e3": e3_oof}[args.stage]()


if __name__ == "__main__":
    main()
