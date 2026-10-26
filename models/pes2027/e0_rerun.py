"""E0-RERUN: refit the primary pair (5 seeds x 2 models) with the original code path.

Uses the saved delay-1h Chronos cache, the original frame builder, and the original
fit_predict. Writes only to runs/pes2027/E0-REPRO/rerun/. Compares every forecast with
the saved per-seed parquet and records fit and prediction times.

    python -m models.pes2027.e0_rerun
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models.pes2027.common import (
    RUNS, SEEDS, TARGET, TEST_START, TRAIN_START, VAL_START, VERIFIED, git_revision, index_hash,
    read_json, utc_now, write_csv, write_json, write_parquet,
)
from models.run_pit_verified import build_spec, columns_for, finite_slice, prepare, study_sources
from models.run_pit_ablation import decode_target, fit_xgb, load_sources, FROZEN_DEPTH, FROZEN_LR
from src.features.pit_sets import apply_asinh, feature_subsets, signed_asinh_columns

OUT = RUNS / "E0-REPRO" / "rerun"


def main() -> None:
    t0 = time.time()
    sources = study_sources(load_sources())
    cache = pd.read_parquet(VERIFIED / "chronos" / "delay_1h.parquet")
    spec = {"name": "delay_1h", "price": 1, "load": 1, "drop_load": False, "role": "primary"}
    frame, _prov, _h = build_spec(sources, spec, compute_hash=False, chronos_cache=cache)
    build_seconds = time.time() - t0
    columns = columns_for(frame, False)
    subsets = feature_subsets(columns)
    train = finite_slice(frame, frame.index[(frame.index >= TRAIN_START) & (frame.index < VAL_START)])
    val = finite_slice(frame, frame.index[(frame.index >= VAL_START) & (frame.index < TEST_START)])
    test = finite_slice(frame, frame.index[frame.index >= TEST_START])
    saved_split = read_json(VERIFIED / "splits" / "delay_1h.json")
    split_check = {
        "train_rows": [len(train), saved_split["train_rows"]], "val_rows": [len(val), saved_split["val_rows"]],
        "test_rows": [len(test), saved_split["test_rows"]], "test_hash": [index_hash(test), saved_split["test_hash"]],
        "train_hash": index_hash(train), "val_hash": index_hash(val),
        "train_max_lt_val_min": bool(train.max() < val.min()), "val_max_lt_test_min": bool(val.max() < test.min()),
    }

    rows, medians_out = [], {}
    for model in ("xgb_exog", "xgb_both"):
        cols = subsets[model]
        matrices, usable, med = prepare(frame, cols, train, val, test)
        # Imputation medians must equal medians of the transformed training rows alone.
        tr = apply_asinh(frame.loc[train, cols].copy(), signed_asinh_columns(cols))
        own = tr[usable].median(numeric_only=True)
        medians_out[model] = {"n_usable": len(usable), "usable": usable,
                              "max_abs_median_diff_vs_train_only": float(np.nanmax(np.abs(own - med[usable]))),
                              "dropped_all_nan_in_train": [c for c in cols if c not in usable]}
        y_train = np.arcsinh(frame.loc[train, TARGET].to_numpy(float))
        y_val = frame.loc[val, TARGET].to_numpy(float)
        y_test = frame.loc[test, TARGET].to_numpy(float)
        for seed in SEEDS:
            t1 = time.time()
            booster, best, device = fit_xgb(matrices[0], y_train, matrices[1], np.arcsinh(y_val), seed, FROZEN_DEPTH, FROZEN_LR)
            fit_s = time.time() - t1
            t2 = time.time()
            raw = booster.predict(matrices[2], iteration_range=(0, best + 1))
            pred_s = time.time() - t2
            pred = decode_target(raw)
            val_pred = decode_target(booster.predict(matrices[1], iteration_range=(0, best + 1)))
            key = f"delay_1h__max__asinh__{model}__seed{seed}"
            saved = pd.read_parquet(VERIFIED / "predictions" / f"{key}.parquet")
            saved_meta = read_json(VERIFIED / "predictions" / f"{key}.json")
            if not saved.index.equals(test):
                raise AssertionError(f"{key}: test index differs from the saved forecast")
            diff = np.abs(pred - saved["y_pred"].to_numpy(float))
            write_parquet(OUT / f"{key}.parquet", pd.DataFrame({"split": "test", "y_true": y_test, "y_pred": pred}, index=test))
            write_parquet(OUT / f"{key}__val.parquet", pd.DataFrame({"split": "val", "y_true": y_val, "y_pred": val_pred}, index=val))
            rows.append({
                "model": model, "seed": seed, "n_features": len(usable), "device": device,
                "best_iteration": best, "saved_best_iteration": saved_meta.get("best_iteration"),
                "max_abs_diff_vs_saved": float(diff.max()), "n_rows_differ": int((diff > 0).sum()),
                "test_mae": float(np.mean(np.abs(pred - y_test))), "val_mae": float(np.mean(np.abs(val_pred - y_val))),
                "fit_seconds": fit_s, "predict_seconds_test": pred_s, "predict_ms_per_origin": 1000 * pred_s / len(test),
            })
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    write_csv(OUT / "rerun_fits.csv", res)
    write_json(OUT / "rerun_summary.json", {
        "experiment": "E0-RERUN", "created_utc": utc_now(), "code_revision": git_revision(),
        "frame_build_seconds": build_seconds, "split_check": split_check, "imputation": medians_out,
        "all_forecasts_identical": bool((res["max_abs_diff_vs_saved"] == 0).all()),
        "all_best_iterations_identical": bool((res["best_iteration"] == res["saved_best_iteration"]).all()),
        "xgb_params": {"n_estimators": 2000, "max_depth": FROZEN_DEPTH, "learning_rate": FROZEN_LR,
                       "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
                       "objective": "reg:absoluteerror", "tree_method": "hist", "early_stopping_rounds": 50,
                       "eval_metric": "mae"},
    })
    print(res[["model", "seed", "best_iteration", "saved_best_iteration", "max_abs_diff_vs_saved", "val_mae", "fit_seconds"]].to_string())


if __name__ == "__main__":
    main()
