"""Compare every numeric claim in manuscript copy B with its full-precision source.

    python -m models.pes2027.e0_precision
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from models.pes2027.common import FINAL_AUDIT, RUNS, VERIFIED, read_json, utc_now, write_csv, write_json

E0 = RUNS / "E0-REPRO"


def rounded(value: float, decimals: int) -> float:
    q = Decimal(1).scaleb(-decimals)
    return float(Decimal(repr(float(value))).quantize(q, rounding=ROUND_HALF_UP))


def main() -> None:
    pairs = pd.read_csv(E0 / "paired_comparisons.csv").set_index("scenario")
    levels = pd.read_csv(E0 / "levels.csv")
    lv = {(r.scenario, r.model): r for r in levels.itertuples()}
    seeds = pd.read_csv(E0 / "per_seed.csv")
    blocks = pd.read_csv(E0 / "block_lengths.csv").set_index("block_hours")
    base = pd.read_csv(E0 / "baselines.csv").set_index("model")
    crisis = pd.read_csv(E0 / "crisis_2021.csv")
    cz = {(r.window, r.model): r for r in crisis.itertuples()}
    cdiag = read_json(FINAL_AUDIT / "crisis_diagnostics.json")
    lear = read_json(FINAL_AUDIT / "lear_style.json")
    recon = read_json(FINAL_AUDIT / "target_reconciliation.json")
    p = pairs.loc["delay_1h"]
    ps = seeds[seeds.scenario == "delay_1h"].set_index("seed")
    b1, e1 = lv[("delay_1h", "xgb_both")], lv[("delay_1h", "xgb_exog")]

    rows = []

    def add(cid, where, quoted, full, decimals, source, note=""):
        rows.append({"claim_id": cid, "manuscript_location": where, "quoted": quoted,
                     "full_precision": full, "decimals_quoted": decimals, "source": source, "note": note})

    pc = "runs/pit_verified/comparisons.json (recomputed: runs/pes2027/E0-REPRO)"
    add("N_EVAL", "Abstract; Table II; Sec. III-B", 8767, p["n"], 0, pc)
    add("MAE_BOTH", "Abstract; Table II; Table III; Table V", 14.92, p["mae_a"], 2, pc)
    add("MAE_EXOG", "Abstract; Table II; Table III; Table V", 15.07, p["mae_b"], 2, pc)
    add("GAP", "Abstract (as 0.15 improvement); Table II; Table V", -0.15, p["gap_a_minus_b"], 2, pc)
    add("CI_LOW", "Abstract; Table II; Table V", -0.32, p["ci_low"], 2, pc)
    add("CI_HIGH", "Abstract; Table II; Table V", 0.06, p["ci_high"], 2, pc)
    add("P_BOOT", "Abstract; Table II; Sec. V-A", 0.131, p["p_boot_interval_inversion"], 3, pc,
        "Interval-inversion bootstrap p-value; not a null-centered p-value")
    add("P_DM", "Abstract; Table II; Sec. V-A; Sec. VI-A", 0.124, p["dm_p_hln_h_lag_plus_1"], 3, pc,
        "HLN factor applied with h = 25 (lag + 1); with h = 1 the value is %.6f" % p["dm_p_hln_h1"])
    add("GAP_PCT", "Sec. V-A 'approximately 1.0%'", 1.0, 100 * -p["gap_a_minus_b"] / p["mae_b"], 1, pc)
    add("RMSE_EXOG", "Table II; Table III", 121.60, e1.rmse, 2, pc)
    add("RMSE_BOTH", "Table II; Table III", 120.73, b1.rmse, 2, pc)
    add("MEDAE_EXOG", "Table II", 3.00, e1.medae, 2, pc)
    add("MEDAE_BOTH", "Table II", 2.93, b1.medae, 2, pc)
    add("BIAS_EXOG", "Table II", -10.78, e1.bias_pred_minus_y, 2, pc)
    add("BIAS_BOTH", "Table II", -10.38, b1.bias_pred_minus_y, 2, pc)
    add("N_SPIKES", "Table II; Sec. V-C", 114, b1.n_spikes, 0, pc)
    msb = seeds[seeds.scenario == "delay_1h"]
    add("MEAN_SEED_MAE_BOTH", "Sec. V-A", 15.03, msb.mae_both.mean(), 2, pc)
    add("MEAN_SEED_MAE_EXOG", "Sec. V-A", 15.17, msb.mae_exog.mean(), 2, pc)
    add("MEAN_SEED_GAP", "Sec. V-A", -0.14, msb.mae_both.mean() - msb.mae_exog.mean(), 2, pc)
    for seed, q in ((0, 0.13), (1, -0.15), (2, -0.18), (3, -0.30), (42, -0.20)):
        add(f"SEED{seed}_GAP", "Sec. V-A", q, ps.loc[seed, "gap"], 2, pc)
    bsrc = "runs/pit_verified/baselines.csv; runs/final_audit/lear_style.json"
    add("SN168_MAE", "Table III", 32.35, base.loc["seasonal_naive_168", "mae"], 2, bsrc)
    add("SN24_MAE", "Table III", 29.58, base.loc["seasonal_naive_24", "mae"], 2, bsrc)
    add("PERS_MAE", "Table III", 25.84, base.loc["persistence_latency", "mae"], 2, bsrc)
    add("LASSO_MAE", "Table III", 26.20, base.loc["lasso_lags_calendar", "mae"], 2, bsrc)
    add("LEAR_MAE", "Table III; Sec. V-B", 16.32, base.loc["lear_style", "mae"], 2, bsrc)
    add("LEAR_RMSE", "Table III; Sec. V-B", 110.17, base.loc["lear_style", "rmse"], 2, bsrc)
    add("LEAR_ALPHA", "Sec. V-B", 1e-6, lear["alpha"], 9, "runs/final_audit/lear_style.json")
    for name, label in (("seasonal_naive_168", "SN168_RMSE"), ("seasonal_naive_24", "SN24_RMSE"),
                        ("persistence_latency", "PERS_RMSE"), ("lasso_lags_calendar", "LASSO_RMSE")):
        add(label, "Table III shows a dash ('not supplied by the final research audit')", None,
            base.loc[name, "rmse"], 2, bsrc, "Value exists in runs/pit_verified/baselines.csv")
    add("SPIKE_PREC_BOTH", "Sec. V-C; Table IV", 0.622, b1.spike_precision, 3, pc)
    add("SPIKE_REC_BOTH", "Sec. V-C; Table IV", 0.202, b1.spike_recall, 3, pc)
    add("SPIKE_F1_BOTH", "Sec. V-C; Table IV", 0.305, b1.spike_f1, 3, pc)
    add("SPIKE_PREC_EXOG", "Sec. V-C; Table IV", 0.559, e1.spike_precision, 3, pc)
    add("SPIKE_REC_EXOG", "Sec. V-C; Table IV", 0.167, e1.spike_recall, 3, pc)
    add("SPIKE_F1_EXOG", "Sec. V-C; Table IV", 0.257, e1.spike_f1, 3, pc)
    add("SPIKE_MAE_BOTH", "Sec. V-C; Table IV", 562.28, b1.spike_mae, 2, pc)
    add("SPIKE_MAE_EXOG", "Sec. V-C; Table IV", 565.56, e1.spike_mae, 2, pc)
    add("NONSPIKE_MAE_BOTH", "Sec. V-C; Table IV", 7.71, b1.nonspike_mae, 2, pc)
    add("NONSPIKE_MAE_EXOG", "Sec. V-C; Table IV", 7.82, e1.nonspike_mae, 2, pc)
    for scen, label, (qa, qb, qg, ql, qh) in (
        ("delay_2h", "DELAY2", (16.23, 16.39, -0.17, -0.34, 0.04)),
        ("delay_3h", "DELAY3", (16.55, 16.65, -0.10, -0.32, 0.20)),
        ("delay_6h", "DELAY6", (17.03, 17.12, -0.09, -0.18, 0.01)),
        ("delay_1h_load_24h", "LOAD24", (15.13, 15.24, -0.11, -0.34, 0.24)),
        ("delay_1h_load_dropped", "LOADDROP", (14.79, 14.97, -0.18, -0.44, 0.18)),
    ):
        r = pairs.loc[scen]
        for suffix, q, full in (("MAE_BOTH", qa, r.mae_a), ("MAE_EXOG", qb, r.mae_b), ("GAP", qg, r.gap_a_minus_b),
                                ("CI_LOW", ql, r.ci_low), ("CI_HIGH", qh, r.ci_high)):
            add(f"{label}_{suffix}", "Table V", q, full, 2, pc)
    for block, (ql, qh) in ((12, (-0.32, 0.05)), (48, (-0.33, 0.06)), (72, (-0.34, 0.08)), (168, (-0.34, 0.09))):
        r = blocks.loc[block]
        add(f"BLOCK{block}_CI_LOW", "Sec. VI-A", ql, r.ci_low, 2, r.source)
        add(f"BLOCK{block}_CI_HIGH", "Sec. VI-A", qh, r.ci_high, 2, r.source)
    for block, q in ((24, 0.124), (48, 0.149), (72, 0.165), (168, 0.188)):
        r = blocks.loc[block]
        add(f"BLOCK{block}_DM_P", "Sec. VI-A", q, r.dm_p_hln_h_lag_plus_1, 3, r.source,
            f"DM HAC lag equals the block length ({block}); HLN h = lag + 1")
    for scen, label, (qg, ql, qh) in (("hourly_mean", "HMEAN", (-0.18, -0.28, -0.07)),
                                      ("hb_north", "NORTH", (-0.15, -0.29, -0.02))):
        r = pairs.loc[scen]
        add(f"{label}_GAP", "Sec. VI-B", qg, r.gap_a_minus_b, 2, r.source)
        add(f"{label}_CI_LOW", "Sec. VI-B", ql, r.ci_low, 2, r.source)
        add(f"{label}_CI_HIGH", "Sec. VI-B", qh, r.ci_high, 2, r.source)
    csrc = "runs/final_audit/crisis_diagnostics.json (recomputed: runs/pes2027/E0-REPRO/crisis_2021.csv)"
    ce, cb = cz[("crisis", "xgb_exog_seed42")], cz[("crisis", "xgb_both_seed42")]
    cp, cr = cz[("crisis", "persistence_t_minus_2")], cz[("crisis", "xgb_exog_raw_target_seed42")]
    oe, ob, op = cz[("outside", "xgb_exog_seed42")], cz[("outside", "xgb_both_seed42")], cz[("outside", "persistence_t_minus_2")]
    add("CRISIS_N", "Sec. VI-C; Table VI", 264, ce.n, 0, csrc)
    add("OUTSIDE_N", "Table VI caption", 8299, oe.n, 0, csrc)
    add("CRISIS_YMAX", "Sec. VI-C", 9051.55, ce.y_max, 2, csrc)
    add("CRISIS_EXOG_PREDMAX", "Sec. VI-C", 697.21, ce.pred_max, 2, csrc)
    add("CRISIS_EXOG_MAE", "Sec. VI-C; Table VI", 3836.73, ce.mae, 2, csrc)
    add("CRISIS_BOTH_PREDMAX", "Sec. VI-C", 2366.94, cb.pred_max, 2, csrc)
    add("CRISIS_BOTH_MAE", "Sec. VI-C; Table VI", 3604.11, cb.mae, 2, csrc)
    add("CRISIS_PERS_MAE", "Sec. VI-C; Table VI", 540.13, cp.mae, 2, csrc)
    add("CRISIS_EXOG_RMSE", "Table VI", 5523.16, ce.rmse, 2, csrc)
    add("CRISIS_BOTH_RMSE", "Table VI", 5239.81, cb.rmse, 2, csrc)
    add("CRISIS_PERS_RMSE", "Table VI", 1446.69, cp.rmse, 2, csrc)
    add("OUTSIDE_EXOG_MAE", "Sec. VI-C; Table VI", 16.93, oe.mae, 2, csrc)
    add("OUTSIDE_BOTH_MAE", "Sec. VI-C; Table VI", 17.20, ob.mae, 2, csrc)
    add("OUTSIDE_PERS_MAE", "Sec. VI-C; Table VI", 25.83, op.mae, 2, csrc)
    add("CRISIS_RAW_MAE", "Sec. VI-C 'approximately $3,824'", 3824, cr.mae, 0, csrc)
    add("TRAIN_MAX", "Sec. VI-C", 9005.29, cdiag["train_prices"]["max"], 2, "runs/final_audit/crisis_diagnostics.json",
        "At %s" % cdiag["train_prices"]["max_timestamp"])
    add("TRAIN_N_GT5000", "Sec. VI-C 'ten hours above $5,000'", 10, cdiag["train_prices"]["hours_above_5000"], 0,
        "runs/final_audit/crisis_diagnostics.json")
    add("UTC_HOURS_2024", "Sec. III-B; App. A-C", 8784, recon["utc_2024_hours"], 0, "runs/final_audit/target_reconciliation.json")
    add("ABSENT_2024", "Sec. III-B; App. A-C", 17, recon["absent_from_evaluation"], 0, "runs/final_audit/target_reconciliation.json")
    add("COMPLETE_N", "Sec. VI-B; App. A-C", 8766, pairs.loc["complete_four_refit", "n"], 0, "runs/final_audit/complete_four_sensitivity.json")
    add("SINH10", "Sec. IV-A", 11013, 11013.232874703393, 0, "numpy.sinh(10)")

    out = pd.DataFrame(rows)
    def classify(r):
        if r.quoted is None or pd.isna(r.quoted):
            return "value omitted in manuscript"
        rq = rounded(r.full_precision, int(r.decimals_quoted))
        if abs(rq - float(r.quoted)) < 1e-12:
            return "consistent with rounding"
        if abs(float(r.full_precision) - float(r.quoted)) <= 0.5 * 10 ** (-int(r.decimals_quoted)) + 1e-12:
            return "consistent with rounding (boundary)"
        return "substantive difference"
    out["rounded_full"] = [None if pd.isna(r.quoted) else rounded(r.full_precision, int(r.decimals_quoted)) for r in out.itertuples()]
    out["classification"] = [classify(r) for r in out.itertuples()]
    write_csv(E0 / "manuscript_precision_check.csv", out)
    write_json(E0 / "manuscript_precision_summary.json", {
        "created_utc": utc_now(), "n_claims": int(len(out)),
        "by_class": out["classification"].value_counts().to_dict(),
        "not_consistent": out[~out["classification"].str.startswith("consistent")].to_dict(orient="records"),
    })
    print(out["classification"].value_counts().to_string())
    print(out[~out["classification"].str.startswith("consistent")].to_string())


if __name__ == "__main__":
    main()
