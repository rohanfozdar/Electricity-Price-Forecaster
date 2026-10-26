"""E4 cache-seed tree fits, E5 cost compilation, E6 saved-artifact sensitivities, E7 rolling windows."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from models.pes2027.common import (
    FINAL_AUDIT, RUNS, SEEDS, TARGET, TEST_START, TRAIN_START, VAL_START, VERIFIED,
    git_revision, hourly_max, index_hash, read_json, saved_seed_frames, seed_average,
    utc_now, write_csv, write_json,
)
from models.pes2027.fit import fit_one, require, save_pred, seed_mean
from models.pes2027.indep_stats import holm_adjust, paired_summary, point_metrics
from models.run_pit_verified import build_spec, columns_for, finite_slice, study_sources
from models.run_pit_ablation import load_sources
from src.features.pit_sets import feature_subsets

CACHE_SEEDS = (0, 1, 2, 3)
CHRONOS_COLS = (
    "chronos_pred", "chronos_q10", "chronos_q90", "chronos_mean", "chronos_p_gt200", "chronos_spread",
)
WINDOWS = {
    "E2022": {
        "train_end": pd.Timestamp("2021-01-01", tz="UTC"),
        "val_end": pd.Timestamp("2022-01-01", tz="UTC"),
        "test_end": pd.Timestamp("2023-01-01", tz="UTC"),
    },
    "E2023": {
        "train_end": pd.Timestamp("2022-01-01", tz="UTC"),
        "val_end": pd.Timestamp("2023-01-01", tz="UTC"),
        "test_end": pd.Timestamp("2024-01-01", tz="UTC"),
    },
    "E2024": {
        "train_end": VAL_START,
        "val_end": TEST_START,
        "test_end": pd.Timestamp("2025-01-01", tz="UTC"),
    },
}


def _sources():
    return study_sources(load_sources())


def _frame(sources, cache, spec=None, target="max", rt_price=None):
    spec = spec or {"name": "delay_1h", "price": 1, "load": 1, "drop_load": False}
    return build_spec(sources, spec, target, False, cache, rt_price)[0]


def _split_years(frame, train_end, val_end, test_end):
    kept = frame.index[(frame.index >= TRAIN_START) & (frame.index < test_end)]
    kept = finite_slice(frame, kept)
    train = kept[kept < train_end]
    val = kept[(kept >= train_end) & (kept < val_end)]
    test = kept[(kept >= val_end) & (kept < test_end)]
    return train, val, test


def _pair(y, a, b, index, drop_segments=True):
    s = paired_summary(y, a, b, index)
    if drop_segments:
        s = {k: v for k, v in s.items() if k != "segments"}
    return s


def e4_fits() -> dict:
    frozen = require()
    sources = _sources()
    cache_dir = RUNS / "E4-CSEED" / "caches"
    missing = [s for s in CACHE_SEEDS if not (cache_dir / f"delay_1h_seed{s}.parquet").exists()]
    if missing:
        raise SystemExit(f"E4 caches missing for seeds {missing}")
    baseline = pd.read_parquet(RUNS / "E0-REPRO" / "primary_per_origin.parquet")
    y_fixed = baseline["y_true"].to_numpy(float)
    exog = baseline["pred_xgb_exog_seed_avg"].to_numpy(float)
    idx_fixed = baseline.index
    rows, pairs, cache_avgs = [], [], {}
    for cseed in CACHE_SEEDS:
        cache = pd.read_parquet(cache_dir / f"delay_1h_seed{cseed}.parquet")
        frame = _frame(sources, cache)
        cols = feature_subsets(columns_for(frame, False))["xgb_both"]
        train, val, test = _split_years(frame, VAL_START, TEST_START, pd.Timestamp("2025-01-01", tz="UTC"))
        if not test.equals(idx_fixed):
            raise AssertionError(f"cache seed {cseed}: test index differs from the frozen 2024 origins")
        stack = []
        for tseed in SEEDS:
            rec = fit_one(frame, cols, train, val, test, tseed)
            stack.append(rec["test"][1])
            save_pred(
                RUNS / "E4-CSEED" / "preds" / f"cache{cseed}_tree{tseed}.parquet",
                test, rec["test"][0], rec["test"][1],
                {"cache_seed": cseed, "tree_seed": tseed, "best_iteration": rec["best_iteration"],
                 "seconds": rec["seconds"], "protocol": frozen},
            )
            mae = point_metrics(rec["test"][0], rec["test"][1])["mae"]
            rows.append({"cache_seed": cseed, "tree_seed": tseed, "best_iteration": rec["best_iteration"],
                         "seconds": rec["seconds"], "test_mae": mae})
            print({"cache": cseed, "tree": tseed, "mae": mae, "seconds": rec["seconds"]}, flush=True)
        avg = seed_mean(stack)
        cache_avgs[cseed] = avg
        pairs.append({"cache_seed": cseed, "vs_fixed_xgb_exog": _pair(y_fixed, avg, exog, idx_fixed),
                      "mae_both": float(np.mean(np.abs(avg - y_fixed)))})
    # Cache-to-cache variance of the seed-averaged forecast, separate from tree-seed variance.
    mat = np.vstack([cache_avgs[s] for s in CACHE_SEEDS])
    tree_mat = np.vstack([
        pd.read_parquet(RUNS / "E4-CSEED" / "preds" / f"cache{c}_tree{t}.parquet")["y_pred"].to_numpy(float)
        for c in CACHE_SEEDS for t in SEEDS
    ])
    summary = {
        "protocol": frozen, "created_utc": utc_now(), "code_revision": git_revision(),
        "n": int(len(idx_fixed)), "index_hash": index_hash(idx_fixed),
        "pairs_vs_fixed_baseline": pairs,
        "cache_seed_sd_of_hourly_forecast": float(mat.std(axis=0, ddof=1).mean()),
        "cache_seed_sd_of_mae": float(np.std([p["mae_both"] for p in pairs], ddof=1)),
        "tree_and_cache_sd_of_hourly_forecast": float(tree_mat.std(axis=0, ddof=1).mean()),
        "note": "Baseline is the original five-seed xgb_exog average. Cache variance is across torch seeds 0-3.",
    }
    write_csv(RUNS / "E4-CSEED" / "fits.csv", pd.DataFrame(rows))
    write_json(RUNS / "E4-CSEED" / "summary.json", summary)
    return summary


def e5_cost() -> dict:
    frozen = require()
    cache_dir = RUNS / "E4-CSEED" / "caches"
    caches = []
    for seed in CACHE_SEEDS:
        path = cache_dir / f"delay_{1}h_seed{seed}.json"
        if path.exists():
            caches.append(read_json(path))
    probe = read_json(RUNS / "E4-CSEED" / "latency_probe.json") if (RUNS / "E4-CSEED" / "latency_probe.json").exists() else None
    rerun = read_json(RUNS / "E0-REPRO" / "rerun" / "rerun_summary.json")
    fits = pd.read_csv(RUNS / "E0-REPRO" / "rerun" / "rerun_fits.csv")
    e4 = RUNS / "E4-CSEED" / "fits.csv"
    e4_fits = pd.read_csv(e4) if e4.exists() else None
    out = {
        "protocol": frozen, "created_utc": utc_now(),
        "one_time": {
            "chronos_cache_seconds_per_seed": {c["seed"]: c["seconds_generate"] for c in caches},
            "chronos_cache_rows": {c["seed"]: c["rows"] for c in caches},
            "chronos_ms_per_origin_full_pass": {c["seed"]: c["ms_per_origin"] for c in caches},
            "xgb_train_seconds_primary_rerun": fits.groupby("model")["fit_seconds"].median().to_dict(),
        },
        "recurring": {
            "chronos_batch1_ms_median": None if probe is None else probe["batch1_ms_median"],
            "chronos_batch256_ms_per_origin": None if probe is None else probe["batch256_ms_per_origin"],
            "xgb_predict_ms_per_origin_primary": float(fits["predict_ms_per_origin"].median()),
        },
        "latency_probe": probe,
        "e4_tree_seconds": None if e4_fits is None else {
            "median": float(e4_fits["seconds"].median()),
            "n_fits": int(len(e4_fits)),
        },
        "rerun_frame_build_seconds": rerun.get("frame_build_seconds"),
        "note": "One-time cost is the cache build and tree training. Recurring cost is a later origin.",
    }
    write_json(RUNS / "E5-COST" / "cost.json", out)
    return out


def e6_sens() -> dict:
    """Reproduce secondary analyses from saved predictions. No new tree is fit."""
    frozen = require()
    rows, holm_groups = [], {}
    scen = ["delay_1h", "delay_2h", "delay_3h", "delay_6h", "delay_1h_load_24h", "delay_1h_load_dropped"]
    for name in scen:
        fb = saved_seed_frames(f"{name}__max__asinh__xgb_both")
        fe = saved_seed_frames(f"{name}__max__asinh__xgb_exog")
        index, y, both, _ = seed_average(fb)
        _, _, exog, _ = seed_average(fe)
        pair = _pair(y, both, exog, index)
        rows.append({"family": "delay" if name.startswith("delay_") and "load" not in name else "load",
                     "scenario": name, "status": "secondary_reproduced", **pair,
                     "index_hash": index_hash(index)})
    index, y, both, exog = None, None, None, None
    fb = saved_seed_frames("delay_1h__max__asinh__xgb_both")
    fe = saved_seed_frames("delay_1h__max__asinh__xgb_exog")
    index, y, both, _ = seed_average(fb)
    _, _, exog, _ = seed_average(fe)
    for block in (12, 24, 48, 72, 168):
        pair = paired_summary(y, both, exog, index, block=block)
        pair = {k: v for k, v in pair.items() if k != "segments"}
        rows.append({"family": "blocks", "scenario": f"block_{block}", "status": "secondary_reproduced",
                     **pair, "index_hash": index_hash(index)})
    cf_b = [pd.read_parquet(FINAL_AUDIT / "complete_four_predictions" / f"xgb_both__seed{s}.parquet") for s in SEEDS]
    cf_e = [pd.read_parquet(FINAL_AUDIT / "complete_four_predictions" / f"xgb_exog__seed{s}.parquet") for s in SEEDS]
    cix, cy, cb, _ = seed_average(cf_b)
    _, _, ce, _ = seed_average(cf_e)
    rows.append({"family": "other", "scenario": "complete_four", "status": "secondary_reproduced",
                 **_pair(cy, cb, ce, cix), "index_hash": index_hash(cix)})
    for name, prefix in (("hourly_mean", "delay_1h__mean__asinh"),
                         ("hb_north", "hb_north_delay_1h__max__asinh")):
        hb = saved_seed_frames(f"{prefix}__xgb_both")
        he = saved_seed_frames(f"{prefix}__xgb_exog")
        hix, hy, hboth, _ = seed_average(hb)
        _, _, hexog, _ = seed_average(he)
        rows.append({"family": "other", "scenario": name, "status": "secondary_reproduced",
                     **_pair(hy, hboth, hexog, hix), "index_hash": index_hash(hix)})
    table = pd.DataFrame(rows)
    families = {
        "E6_delays": ["delay_1h", "delay_2h", "delay_3h", "delay_6h"],
        "E6_load": ["delay_1h_load_24h", "delay_1h_load_dropped"],
        "E6_blocks": [f"block_{b}" for b in (12, 24, 48, 72, 168)],
        "E6_other": ["complete_four", "hourly_mean", "hb_north"],
    }
    holm = {}
    for fam, names in families.items():
        sub = table.set_index("scenario").loc[names]
        holm[fam] = {
            "scenarios": names,
            "p_boot": list(sub["p_boot_interval_inversion"]),
            "p_holm": holm_adjust(sub["p_boot_interval_inversion"]),
        }
    write_csv(RUNS / "E6-SENS" / "sensitivities.csv", table)
    write_json(RUNS / "E6-SENS" / "summary.json", {
        "protocol": frozen, "created_utc": utc_now(), "code_revision": git_revision(),
        "load_daily_posting_gate": "not added; NP6-345-CD posting minute is not documented",
        "holm": holm,
        "status": "secondary",
        "source": "saved pit_verified and final_audit predictions; independent stats only",
    })
    print(table[["family", "scenario", "gap_a_minus_b", "p_boot_interval_inversion"]].to_string())
    return {"n": len(table)}


def e7_roll() -> dict:
    frozen = require()
    sources = _sources()
    cache = pd.read_parquet(VERIFIED / "chronos" / "delay_1h.parquet")
    frame = _frame(sources, cache)
    subsets = feature_subsets(columns_for(frame, False))
    hourly = hourly_max(sources["rt"][TARGET])
    persist = frame["hubavg_lag_1h"]
    rows, summaries = [], {}
    for name, bounds in WINDOWS.items():
        train, val, test = _split_years(frame, bounds["train_end"], bounds["val_end"], bounds["test_end"])
        y = frame.loc[test, TARGET].to_numpy(float)
        preds = {}
        for model in ("xgb_exog", "xgb_both"):
            stack = []
            for seed in SEEDS:
                rec = fit_one(frame, subsets[model], train, val, test, seed)
                stack.append(rec["test"][1])
                save_pred(
                    RUNS / "E7-ROLL" / "preds" / f"{name}_{model}_seed{seed}.parquet",
                    test, rec["test"][0], rec["test"][1],
                    {"window": name, "model": model, "seed": seed, "best_iteration": rec["best_iteration"],
                     "seconds": rec["seconds"], "protocol": frozen},
                )
                rows.append({"window": name, "model": model, "seed": seed, "best_iteration": rec["best_iteration"],
                             "seconds": rec["seconds"], "n_train": len(train), "n_val": len(val),
                             "n_test": len(test), "test_mae": point_metrics(rec["test"][0], rec["test"][1])["mae"]})
                print(rows[-1], flush=True)
            preds[model] = seed_mean(stack)
        chrono = cache.reindex(test)["chronos_pred"].to_numpy(float)
        pers = persist.reindex(test).to_numpy(float)
        keep = np.isfinite(chrono) & np.isfinite(pers)
        summaries[name] = {
            "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
            "index_hash": index_hash(test),
            "mae": {m: point_metrics(y, preds[m])["mae"] for m in preds},
            "mae_chronos": point_metrics(y[keep], chrono[keep])["mae"],
            "mae_persistence": point_metrics(y[keep], pers[keep])["mae"],
            "matched_both_minus_exog": _pair(y, preds["xgb_both"], preds["xgb_exog"], test),
            "label": "retrospective robustness",
        }
    write_csv(RUNS / "E7-ROLL" / "fits.csv", pd.DataFrame(rows))
    write_json(RUNS / "E7-ROLL" / "summary.json", {
        "protocol": frozen, "created_utc": utc_now(), "windows": summaries,
        "holm_E7_windows": holm_adjust([summaries[w]["matched_both_minus_exog"]["p_boot_interval_inversion"]
                                        for w in ("E2022", "E2023", "E2024")]),
        "note": "Shorter early windows. E2022 validation year contains the 2021 crisis. Not an untouched holdout.",
    })
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["e4", "e5", "e6", "e7"])
    args = p.parse_args()
    {"e4": e4_fits, "e5": e5_cost, "e6": e6_sens, "e7": e7_roll}[args.stage]()


if __name__ == "__main__":
    main()
