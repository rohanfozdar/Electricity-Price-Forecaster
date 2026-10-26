"""E0-REPRO: rebuild every manuscript table from saved per-origin predictions.

No model is fit here. Each number is recomputed with models.pes2027.indep_stats and
compared with (a) the saved JSON/CSV value and (b) the original repository functions.

    python models/pes2027/e0_reproduce.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.pes2027.common import (
    CRISIS_END, CRISIS_START, FINAL_AUDIT, FROZEN_HASH, RUNS, SEEDS, TARGET, VERIFIED,
    git_revision, hourly_max, index_hash, read_json, raw_rt, saved_seed_frames, seed_average,
    utc_now, write_csv, write_json, write_parquet,
)
from models.pes2027.indep_stats import paired_summary, point_metrics, segment_lengths
from src.evaluation.statistical_tests import diebold_mariano, paired_bootstrap

OUT = RUNS / "E0-REPRO"
CHECKS: list[dict] = []


def check(name: str, recomputed, saved, tol: float = 1e-9, source: str = "") -> None:
    if saved is None:
        CHECKS.append({"check": name, "recomputed": recomputed, "saved": None,
                       "abs_diff": None, "match": None, "source": source})
        return
    diff = abs(float(recomputed) - float(saved))
    CHECKS.append({"check": name, "recomputed": float(recomputed), "saved": float(saved),
                   "abs_diff": diff, "match": bool(diff <= tol), "source": source})


def original_functions(y, a, b, index, block=24):
    """The repository's own bootstrap and DM, for a second cross-check."""
    sizes = [int(x) for x in segment_lengths(index)]
    boot = paired_bootstrap(lambda yy, pp: float(np.mean(np.abs(pp - yy))), y, a, b, sizes, block, 2000, seed=0)
    dm = diebold_mariano(np.abs(a - y), np.abs(b - y), block)
    return boot, dm


def pair_rows(label, y, both, exog, index, saved: dict | None, source: str, block=24):
    s = paired_summary(y, both, exog, index, block=block)
    boot, dm = original_functions(y, both, exog, index, block)
    check(f"{label}: gap vs original paired_bootstrap", s["gap_a_minus_b"], boot["diff"], 1e-12, "code")
    check(f"{label}: ci_low vs original paired_bootstrap", s["ci_low"], boot["ci_low"], 1e-9, "code")
    check(f"{label}: ci_high vs original paired_bootstrap", s["ci_high"], boot["ci_high"], 1e-9, "code")
    check(f"{label}: boot p vs original", s["p_boot_interval_inversion"], boot["p_value"], 1e-12, "code")
    check(f"{label}: DM(h=lag+1) p vs original diebold_mariano", s["dm_p_hln_h_lag_plus_1"], dm["p_value"], 1e-9, "code")
    check(f"{label}: DM(h=lag+1) stat vs original", s["dm_stat_hln_h_lag_plus_1"], dm["stat"], 1e-9, "code")
    if saved:
        for mine, theirs in (("gap_a_minus_b", "gap"), ("ci_low", "ci_low"), ("ci_high", "ci_high"),
                             ("p_boot_interval_inversion", "p"), ("dm_p_hln_h_lag_plus_1", "dm_p"),
                             ("mae_a", "mae_a"), ("mae_b", "mae_b")):
            if theirs in saved and saved[theirs] is not None:
                check(f"{label}: {mine} vs saved", s[mine], saved[theirs], 1e-9, source)
    s = {k: v for k, v in s.items() if k != "segments"}
    s["n_segments"] = len(segment_lengths(index))
    return {"comparison": label, **s, "source": source}


def comparisons_json() -> dict:
    rows = read_json(VERIFIED / "comparisons.json")
    return {row["name"]: row for row in rows}


def saved_pair(row: dict) -> dict:
    return {"gap": row["mae_gap_a_minus_b"], "ci_low": row["bootstrap_ci_low"],
            "ci_high": row["bootstrap_ci_high"], "p": row["bootstrap_p"], "dm_p": row["dm_p"],
            "mae_a": row["mae_a"], "mae_b": row["mae_b"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comp = comparisons_json()
    pairs, levels, per_seed = [], [], []

    # Primary and latency/load scenarios.
    scen = ["delay_1h", "delay_2h", "delay_3h", "delay_6h", "delay_1h_load_24h", "delay_1h_load_dropped"]
    vectors = {}
    for name in scen:
        fb = saved_seed_frames(f"{name}__max__asinh__xgb_both")
        fe = saved_seed_frames(f"{name}__max__asinh__xgb_exog")
        if not fb[0].index.equals(fe[0].index):
            raise AssertionError(f"{name}: paired models are on different origins")
        index, y, both, stack_b = seed_average(fb)
        _, y2, exog, stack_e = seed_average(fe)
        if not np.array_equal(y, y2):
            raise AssertionError(f"{name}: paired targets differ")
        vectors[name] = (index, y, both, exog, stack_b, stack_e)
        row = comp[f"{name}__mean_seed__both_minus_exog"]
        pairs.append({"scenario": name, "test_hash": index_hash(index),
                      **pair_rows(name, y, both, exog, index, saved_pair(row), "runs/pit_verified/comparisons.json")})
        for model, pred, key in (("xgb_both", both, "seed_averaged_metrics_both"),
                                 ("xgb_exog", exog, "seed_averaged_metrics_exog")):
            m = point_metrics(y, pred)
            savedm = row[key]
            for k_mine, k_saved in (("mae", "mae"), ("rmse", "rmse"), ("medae", "medae"),
                                    ("bias_pred_minus_y", "bias"), ("spike_precision", "spike_precision"),
                                    ("spike_recall", "spike_recall"), ("spike_f1", "spike_f1"),
                                    ("spike_mae", "spike_mae"), ("nonspike_mae", "nonspike_mae")):
                check(f"{name} {model}: {k_mine}", m[k_mine], savedm[k_saved], 1e-9, "comparisons.json")
            levels.append({"scenario": name, "model": model, "estimand": "seed_averaged_forecast", **m})
        for i, seed in enumerate(SEEDS):
            mb = float(np.mean(np.abs(stack_b[i] - y)))
            me = float(np.mean(np.abs(stack_e[i] - y)))
            per_seed.append({"scenario": name, "seed": seed, "mae_both": mb, "mae_exog": me, "gap": mb - me})
            check(f"{name} seed {seed}: per-seed gap", mb - me, row["per_seed_gap"][i], 1e-9, "comparisons.json")
        check(f"{name}: mean of seed MAEs both", np.mean(stack_b_mae := [np.mean(np.abs(r - y)) for r in stack_b]),
              row["mean_of_seed_maes_both"], 1e-9, "comparisons.json")
        check(f"{name}: mean of seed MAEs exog", np.mean([np.mean(np.abs(r - y)) for r in stack_e]),
              row["mean_of_seed_maes_exog"], 1e-9, "comparisons.json")
        del stack_b_mae

    index, y, both, exog, stack_b, stack_e = vectors["delay_1h"]
    check("primary: evaluation index hash", int(index_hash(index) == FROZEN_HASH), 1, 0, "manuscript")
    check("primary: n", len(y), 8767, 0, "manuscript")

    # Seed 42 alone.
    s42 = comp["delay_1h__seed42__both_minus_exog"]
    pairs.append({"scenario": "delay_1h_seed42_only", "test_hash": index_hash(index),
                  **pair_rows("delay_1h seed 42 only", y, stack_b[SEEDS.index(42)], stack_e[SEEDS.index(42)], index,
                              saved_pair(s42), "runs/pit_verified/comparisons.json")})

    # Block lengths.
    blocks_v = pd.read_csv(VERIFIED / "block_lengths.csv").set_index("block_hours")
    blocks_f = pd.read_csv(FINAL_AUDIT / "block_lengths.csv").set_index("block_hours")
    block_rows = []
    for block in (12, 24, 48, 72, 168):
        src = blocks_v if block in blocks_v.index else blocks_f
        srow = src.loc[block]
        saved = {"gap": srow.get("mae_gap", srow.get("gap")), "ci_low": srow["ci_low"],
                 "ci_high": srow["ci_high"], "p": srow["bootstrap_p"], "dm_p": srow["dm_p"]}
        name = "runs/pit_verified/block_lengths.csv" if src is blocks_v else "runs/final_audit/block_lengths.csv"
        block_rows.append({"block_hours": block, **pair_rows(f"block {block}h", y, both, exog, index, saved, name, block)})

    # Complete-target refit (final audit) and the saved forecasts restricted to the same hours.
    cf = read_json(FINAL_AUDIT / "complete_four_sensitivity.json")
    cb = [pd.read_parquet(FINAL_AUDIT / "complete_four_predictions" / f"xgb_both__seed{s}.parquet") for s in SEEDS]
    ce = [pd.read_parquet(FINAL_AUDIT / "complete_four_predictions" / f"xgb_exog__seed{s}.parquet") for s in SEEDS]
    cindex, cy, cboth, _ = seed_average(cb)
    _, _, cexog, _ = seed_average(ce)
    mask = index.isin(cindex)
    check("complete-four: refit both equals saved subset (max abs diff)", float(np.max(np.abs(cboth - both[mask]))), 0.0, 0, "final_audit")
    check("complete-four: refit exog equals saved subset (max abs diff)", float(np.max(np.abs(cexog - exog[mask]))), 0.0, 0, "final_audit")
    pairs.append({"scenario": "complete_four_refit", "test_hash": index_hash(cindex),
                  **pair_rows("complete-four refit", cy, cboth, cexog, cindex,
                              {"gap": cf["gap"], "ci_low": cf["ci_low"], "ci_high": cf["ci_high"],
                               "p": cf["bootstrap_p"], "dm_p": cf["dm_p"]},
                              "runs/final_audit/complete_four_sensitivity.json")})
    pairs.append({"scenario": "complete_four_saved_subset", "test_hash": index_hash(index[mask]),
                  **pair_rows("complete-four saved subset", y[mask], both[mask], exog[mask], index[mask], None, "recomputed")})

    # Hourly mean and HB_NORTH.
    for name, prefix, path in (("hourly_mean", "delay_1h__mean__asinh", "hourly_mean.json"),
                               ("hb_north", "hb_north_delay_1h__max__asinh", "hub_north.json")):
        fb = saved_seed_frames(f"{prefix}__xgb_both")
        fe = saved_seed_frames(f"{prefix}__xgb_exog")
        hix, hy, hb, sb = seed_average(fb)
        _, _, he, se = seed_average(fe)
        saved = read_json(VERIFIED / path)
        pairs.append({"scenario": name, "test_hash": index_hash(hix),
                      **pair_rows(name, hy, hb, he, hix, saved_pair(saved), f"runs/pit_verified/{path}")})
        for model, pred in (("xgb_both", hb), ("xgb_exog", he)):
            levels.append({"scenario": name, "model": model, "estimand": "seed_averaged_forecast", **point_metrics(hy, pred)})
        for i, seed in enumerate(SEEDS):
            mb, me = float(np.mean(np.abs(sb[i] - hy))), float(np.mean(np.abs(se[i] - hy)))
            per_seed.append({"scenario": name, "seed": seed, "mae_both": mb, "mae_exog": me, "gap": mb - me})

    # Baselines on the frozen index.
    rt = raw_rt()
    hourly = hourly_max(rt[TARGET])
    base_rows = []
    saved_base = pd.read_csv(VERIFIED / "baselines.csv").set_index("model")
    lear = read_json(FINAL_AUDIT / "lear_style.json")
    preds = {
        "seasonal_naive_168": hourly.reindex(index - pd.Timedelta(hours=168)).to_numpy(float),
        "seasonal_naive_24": hourly.reindex(index - pd.Timedelta(hours=24)).to_numpy(float),
        "persistence_latency": hourly.reindex(index - pd.Timedelta(hours=2)).to_numpy(float),
    }
    lasso = pd.read_parquet(VERIFIED / "predictions" / "lasso_delay_1h.parquet")
    lear_p = pd.read_parquet(FINAL_AUDIT / "lear_style_predictions.parquet")
    for name, frame in (("lasso_lags_calendar", lasso), ("lear_style", lear_p)):
        if not frame.index.equals(index):
            raise AssertionError(f"{name} is not on the frozen index")
        if not np.array_equal(frame["y_true"].to_numpy(float), y):
            raise AssertionError(f"{name} target differs from the tree target")
        preds[name] = frame["y_pred"].to_numpy(float)
    preds["xgb_exog_seed_avg"] = exog
    preds["xgb_both_seed_avg"] = both
    for name, pred in preds.items():
        ok = np.isfinite(pred)
        m = point_metrics(y[ok], pred[ok])
        base_rows.append({"model": name, "n_scored": int(ok.sum()), "n_frozen": len(y), **m})
        if name in saved_base.index:
            check(f"baseline {name}: mae", m["mae"], saved_base.loc[name, "mae"], 1e-9, "baselines.csv")
            check(f"baseline {name}: rmse", m["rmse"], saved_base.loc[name, "rmse"], 1e-9, "baselines.csv")
            check(f"baseline {name}: n", m["n"], saved_base.loc[name, "n"], 0, "baselines.csv")
    check("lear_style: mae", base_rows[4]["mae"], lear["metrics"]["mae"], 1e-9, "lear_style.json")
    check("lear_style: rmse", base_rows[4]["rmse"], lear["metrics"]["rmse"], 1e-9, "lear_style.json")
    for name in ("persistence_latency", "lear_style", "lasso_lags_calendar"):
        pairs.append({"scenario": f"xgb_exog_vs_{name}",
                      **pair_rows(f"xgb_exog minus {name}", y, exog, preds[name], index, None, "recomputed")})
        pairs.append({"scenario": f"xgb_both_vs_{name}",
                      **pair_rows(f"xgb_both minus {name}", y, both, preds[name], index, None, "recomputed")})

    # February 2021 diagnostic (single seed 42 trees, as saved).
    crisis_rows = []
    ce42 = pd.read_parquet(VERIFIED / "predictions" / "crisis2021__asinh__xgb_exog__seed42.parquet")
    cb42 = pd.read_parquet(VERIFIED / "predictions" / "crisis2021__asinh__xgb_both__seed42.parquet")
    cr42 = pd.read_parquet(VERIFIED / "predictions" / "crisis2021__raw__xgb_exog__seed42.parquet")
    if not (ce42.index.equals(cb42.index) and ce42.index.equals(cr42.index)):
        raise AssertionError("crisis predictions are on different hours")
    cix = ce42.index
    cyy = ce42["y_true"].to_numpy(float)
    check("crisis: target equals raw hourly max", float(np.max(np.abs(hourly.reindex(cix).to_numpy(float) - cyy))), 0.0, 1e-9, "raw")
    pers = hourly.reindex(cix - pd.Timedelta(hours=2)).to_numpy(float)
    in_crisis = (cix >= CRISIS_START) & (cix < CRISIS_END)
    saved_c = read_json(FINAL_AUDIT / "crisis_diagnostics.json")
    for label, mask in (("crisis", in_crisis), ("outside", ~in_crisis), ("all_2021", np.ones(len(cix), bool))):
        for model, pred in (("xgb_exog_seed42", ce42["y_pred"].to_numpy(float)),
                            ("xgb_both_seed42", cb42["y_pred"].to_numpy(float)),
                            ("xgb_exog_raw_target_seed42", cr42["y_pred"].to_numpy(float)),
                            ("persistence_t_minus_2", pers)):
            m = point_metrics(cyy[mask], pred[mask])
            crisis_rows.append({"window": label, "model": model, **m})
            key = {"xgb_exog_seed42": "xgb_exog", "xgb_both_seed42": "xgb_both",
                   "persistence_t_minus_2": "persistence", "xgb_exog_raw_target_seed42": "raw_target_xgb_exog"}[model]
            if label in saved_c and key in saved_c[label]:
                check(f"crisis {label} {model}: mae", m["mae"], saved_c[label][key]["mae"], 1e-9, "crisis_diagnostics.json")
                check(f"crisis {label} {model}: rmse", m["rmse"], saved_c[label][key]["rmse"], 1e-9, "crisis_diagnostics.json")
    check("crisis: n hours in window", int(in_crisis.sum()), 264, 0, "manuscript")
    check("crisis: n outside", int((~in_crisis).sum()), 8299, 0, "manuscript")

    write_csv(OUT / "paired_comparisons.csv", pd.DataFrame(pairs))
    write_csv(OUT / "block_lengths.csv", pd.DataFrame(block_rows))
    write_csv(OUT / "levels.csv", pd.DataFrame(levels))
    write_csv(OUT / "per_seed.csv", pd.DataFrame(per_seed))
    write_csv(OUT / "baselines.csv", pd.DataFrame(base_rows))
    write_csv(OUT / "crisis_2021.csv", pd.DataFrame(crisis_rows))
    checks = pd.DataFrame(CHECKS)
    write_csv(OUT / "checks.csv", checks)

    # Per-origin forecasts and losses for the primary comparison.
    per_origin = pd.DataFrame({
        "y_true": y, "pred_xgb_exog_seed_avg": exog, "pred_xgb_both_seed_avg": both,
        **{f"pred_xgb_exog_seed{s}": stack_e[i] for i, s in enumerate(SEEDS)},
        **{f"pred_xgb_both_seed{s}": stack_b[i] for i, s in enumerate(SEEDS)},
        "pred_persistence_t_minus_2": preds["persistence_latency"],
        "pred_seasonal_naive_24": preds["seasonal_naive_24"],
        "pred_seasonal_naive_168": preds["seasonal_naive_168"],
        "pred_lasso_lags_calendar": preds["lasso_lags_calendar"],
        "pred_lear_style": preds["lear_style"],
    }, index=index)
    per_origin["abs_err_xgb_exog"] = np.abs(exog - y)
    per_origin["abs_err_xgb_both"] = np.abs(both - y)
    per_origin["loss_diff_both_minus_exog"] = per_origin["abs_err_xgb_both"] - per_origin["abs_err_xgb_exog"]
    per_origin.index.name = "origin_utc"
    write_parquet(OUT / "primary_per_origin.parquet", per_origin)
    crisis_origin = pd.DataFrame({"y_true": cyy, "pred_xgb_exog_seed42": ce42["y_pred"].to_numpy(float),
                                  "pred_xgb_both_seed42": cb42["y_pred"].to_numpy(float),
                                  "pred_xgb_exog_raw_target_seed42": cr42["y_pred"].to_numpy(float),
                                  "pred_persistence_t_minus_2": pers, "in_crisis_window": in_crisis}, index=cix)
    crisis_origin.index.name = "origin_utc"
    write_parquet(OUT / "crisis2021_per_origin.parquet", crisis_origin)

    failed = checks[checks["match"] == False]  # noqa: E712
    write_json(OUT / "summary.json", {
        "experiment": "E0-REPRO", "created_utc": utc_now(), "code_revision": git_revision(),
        "n_checks": int(len(checks)), "n_failed": int(len(failed)),
        "failed": failed.to_dict(orient="records"),
        "frozen_index_hash": index_hash(index),
        "inputs": ["runs/pit_verified/predictions/*.parquet", "runs/final_audit/complete_four_predictions/*.parquet",
                   "runs/final_audit/lear_style_predictions.parquet", "data/raw/ercot_rt_prices.parquet"],
    })
    print(f"checks {len(checks)}, failed {len(failed)}")
    if len(failed):
        print(failed.to_string())


if __name__ == "__main__":
    main()
