"""E8 February 2021 diagnostic: keep seed-42 originals, add a five-seed refit, official sub-windows."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from models.pes2027.common import (
    CRISIS_END, CRISIS_START, EVIDENCE, RUNS, SEEDS, TARGET, VERIFIED,
    git_revision, index_hash, read_json, utc_now, write_csv, write_json, write_parquet,
)
from models.pes2027.fit import fit_one, require, save_pred, seed_mean
from models.pes2027.indep_stats import point_metrics
from models.pes2027.protocol import require_committed
from models.run_pit_verified import build_spec, columns_for, finite_slice, study_sources
from models.run_pit_ablation import load_sources
from src.features.pit_sets import feature_subsets

WINDOWS_FILE = EVIDENCE / "protocol" / "e8_official_subwindows.yaml"
# Corrected names. The code label xgb_exog is the 37-feature delayed-history baseline,
# not an exogenous-only model. Persistence is the latest legal lag (t-2 under d=1).
NAMES = {
    "xgb_exog": "xgb_delayed_history_baseline",
    "xgb_both": "xgb_delayed_history_plus_chronos",
    "persistence": "persistence_latest_legal_t_minus_2",
}


def _crisis_index(frame):
    test_start = pd.Timestamp("2021-01-01", tz="UTC")
    test_end = pd.Timestamp("2022-01-01", tz="UTC")
    es_end = test_start
    es_start = es_end - pd.Timedelta(days=90)
    train = finite_slice(frame, frame.index[frame.index < es_start])
    val = finite_slice(frame, frame.index[(frame.index >= es_start) & (frame.index < es_end)])
    test = finite_slice(frame, frame.index[(frame.index >= test_start) & (frame.index < test_end)])
    return train, val, test


def _windows() -> dict:
    require_committed(WINDOWS_FILE)
    spec = yaml.safe_load(WINDOWS_FILE.read_text())
    out = {}
    for name, block in spec["windows"].items():
        out[name] = (
            pd.Timestamp(block["start_utc"]),
            pd.Timestamp(block["end_utc"]),
        )
    return out, spec


def _mask(index, start, end):
    return (index >= start) & (index < end)


def original_seed42() -> dict:
    frozen = require()
    windows, spec = _windows()
    exog = pd.read_parquet(VERIFIED / "predictions" / "crisis2021__asinh__xgb_exog__seed42.parquet")
    both = pd.read_parquet(VERIFIED / "predictions" / "crisis2021__asinh__xgb_both__seed42.parquet")
    raw = pd.read_parquet(VERIFIED / "predictions" / "crisis2021__raw__xgb_exog__seed42.parquet")
    if not exog.index.equals(both.index):
        raise AssertionError("original seed-42 crisis indexes differ")
    sources = study_sources(load_sources())
    cache = pd.read_parquet(VERIFIED / "chronos" / "delay_1h.parquet")
    frame = build_spec(sources, {"name": "delay_1h", "price": 1, "load": 1, "drop_load": False},
                       "max", False, cache)[0]
    persist = frame["hubavg_lag_1h"].reindex(exog.index)
    if persist.isna().any():
        raise AssertionError("persistence missing on a scored 2021 hour")
    index = exog.index
    y = exog["y_true"].to_numpy(float)
    preds = {
        "xgb_delayed_history_baseline": exog["y_pred"].to_numpy(float),
        "xgb_delayed_history_plus_chronos": both["y_pred"].to_numpy(float),
        "persistence_latest_legal_t_minus_2": persist.to_numpy(float),
    }
    slices = {
        "inherited_crisis": _mask(index, CRISIS_START, CRISIS_END),
        "all_scored_2021": np.ones(len(index), dtype=bool),
        "outside_inherited_crisis": ~_mask(index, CRISIS_START, CRISIS_END),
    }
    for name, (start, end) in windows.items():
        slices[f"official_{name}"] = _mask(index, start, end)
    rows = []
    for sl_name, mask in slices.items():
        for model, pred in preds.items():
            m = point_metrics(y[mask], pred[mask])
            rows.append({"source": "original_seed42", "slice": sl_name, "model": model, **m,
                         "pred_max": float(pred[mask].max()), "y_max": float(y[mask].max())})
        if sl_name == "inherited_crisis":
            m = point_metrics(raw["y_true"].to_numpy(float)[mask], raw["y_pred"].to_numpy(float)[mask])
            rows.append({"source": "original_seed42", "slice": sl_name,
                         "model": "xgb_delayed_history_baseline_raw_target", **m,
                         "pred_max": float(raw["y_pred"].to_numpy(float)[mask].max()),
                         "y_max": float(raw["y_true"].to_numpy(float)[mask].max())})
    write_csv(RUNS / "E8-CRISIS" / "original_seed42.csv", pd.DataFrame(rows))
    write_parquet(RUNS / "E8-CRISIS" / "original_seed42_per_origin.parquet", pd.DataFrame({
        "y_true": y,
        "pred_xgb_delayed_history_baseline": preds["xgb_delayed_history_baseline"],
        "pred_xgb_delayed_history_plus_chronos": preds["xgb_delayed_history_plus_chronos"],
        "pred_persistence_latest_legal_t_minus_2": preds["persistence_latest_legal_t_minus_2"],
    }, index=index))
    write_json(RUNS / "E8-CRISIS" / "windows_used.json", {
        "protocol": frozen, "windows_file": require_committed(WINDOWS_FILE),
        "spec": spec, "created_utc": utc_now(),
        "note": "Sub-window metrics were not computed until this file was committed.",
    })
    return {"n_rows": len(rows)}


def five_seed_refit() -> dict:
    frozen = require()
    _windows()  # refuse if the official cuts are not committed
    sources = study_sources(load_sources())
    cache = pd.read_parquet(VERIFIED / "chronos" / "delay_1h.parquet")
    frame = build_spec(sources, {"name": "delay_1h", "price": 1, "load": 1, "drop_load": False},
                       "max", False, cache)[0]
    subsets = feature_subsets(columns_for(frame, False))
    train, val, test = _crisis_index(frame)
    rows, preds = [], {}
    for model in ("xgb_exog", "xgb_both"):
        stack = []
        for seed in SEEDS:
            rec = fit_one(frame, subsets[model], train, val, test, seed)
            stack.append(rec["test"][1])
            save_pred(
                RUNS / "E8-CRISIS" / "preds" / f"{model}_seed{seed}.parquet",
                test, rec["test"][0], rec["test"][1],
                {"model": model, "seed": seed, "best_iteration": rec["best_iteration"],
                 "seconds": rec["seconds"], "protocol": frozen, "display_name": NAMES[model]},
            )
            rows.append({"model": NAMES[model], "seed": seed, "best_iteration": rec["best_iteration"],
                         "seconds": rec["seconds"], "test_mae_2021": point_metrics(rec["test"][0], rec["test"][1])["mae"]})
            print(rows[-1], flush=True)
        preds[NAMES[model]] = seed_mean(stack)
    persist = frame["hubavg_lag_1h"].reindex(test).to_numpy(float)
    preds[NAMES["persistence"]] = persist
    y = frame.loc[test, TARGET].to_numpy(float)
    windows, _spec = _windows()
    slices = {
        "inherited_crisis": _mask(test, CRISIS_START, CRISIS_END),
        "all_scored_2021": np.ones(len(test), dtype=bool),
        "outside_inherited_crisis": ~_mask(test, CRISIS_START, CRISIS_END),
    }
    for name, (start, end) in windows.items():
        slices[f"official_{name}"] = _mask(test, start, end)
    scored = []
    for sl_name, mask in slices.items():
        for model, pred in preds.items():
            m = point_metrics(y[mask], pred[mask])
            scored.append({"source": "five_seed_average", "slice": sl_name, "model": model, **m,
                           "pred_max": float(pred[mask].max()), "y_max": float(y[mask].max())})
    write_csv(RUNS / "E8-CRISIS" / "fits.csv", pd.DataFrame(rows))
    write_csv(RUNS / "E8-CRISIS" / "five_seed.csv", pd.DataFrame(scored))
    write_parquet(RUNS / "E8-CRISIS" / "five_seed_per_origin.parquet", pd.DataFrame({
        "y_true": y,
        "pred_xgb_delayed_history_baseline": preds[NAMES["xgb_exog"]],
        "pred_xgb_delayed_history_plus_chronos": preds[NAMES["xgb_both"]],
        "pred_persistence_latest_legal_t_minus_2": persist,
    }, index=test))
    write_json(RUNS / "E8-CRISIS" / "summary.json", {
        "protocol": frozen, "created_utc": utc_now(), "code_revision": git_revision(),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "index_hash": index_hash(test),
        "names": NAMES,
        "note": "Diagnostic only. No causal or resilience claim. 2021 is retrospective stress.",
    })
    return {"n_test": int(len(test))}


def figure() -> None:
    """Write a simple actual-vs-forecast PNG for the inherited crisis hours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig = pd.read_parquet(RUNS / "E8-CRISIS" / "original_seed42_per_origin.parquet")
    mask = (orig.index >= CRISIS_START) & (orig.index < CRISIS_END)
    part = orig.loc[mask]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(part.index, part["y_true"], color="black", lw=1.2, label="actual hourly max")
    ax.plot(part.index, part["pred_xgb_delayed_history_baseline"], color="C0", lw=1.0,
            label="delayed-history XGB (seed 42)")
    ax.plot(part.index, part["pred_xgb_delayed_history_plus_chronos"], color="C1", lw=1.0,
            label="delayed-history + Chronos (seed 42)")
    ax.plot(part.index, part["pred_persistence_latest_legal_t_minus_2"], color="C2", lw=1.0,
            label="persistence, latest legal t-2")
    ax.set_ylabel("$/MWh")
    ax.set_title("2021-02-10 to 2021-02-20 Central: actual vs forecasts (original seed 42)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    out = RUNS / "E8-CRISIS" / "figures" / "crisis_actual_vs_forecast.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_json(out.with_suffix(".json"), {
        "file": "runs/pes2027/E8-CRISIS/figures/crisis_actual_vs_forecast.png",
        "source": "runs/pes2027/E8-CRISIS/original_seed42_per_origin.parquet",
        "slice": "inherited_crisis",
        "created_utc": utc_now(),
    })


def write_target_and_use() -> None:
    text = """# Target and use

This note states what the forecast is and what it is not. It does not estimate trading
returns, settlement cost, or dispatch value.

## Who would use the forecast

A one-hour-ahead forecast of the forthcoming ERCOT hourly maximum hub-average
real-time price is a descriptive input that a short-horizon analyst, a load-serving
entity, or a researcher could consult. No user, position, or decision rule is
identified in the study. The paper therefore reports accuracy, tails, bias, and
event detection as separate objects.

## What the target measures

The target at issue time t (UTC hour start) is the maximum of the HB_HUBAVG
15-minute settlement-point prices whose interval starts are t, t+15, t+30, and
t+45 minutes. Incomplete hours keep the maximum of the intervals that exist.
HB_HUBAVG is a hub average, not a zonal or nodal settlement price and not a
volume-weighted exposure.

## Why this is not settlement cost or exposure

ERCOT real-time energy is settled at 15-minute intervals at settlement points.
An hourly maximum of a hub average:

- discards the other three intervals in a complete hour;
- ignores quantity, so it is not a mark-to-market P&L;
- ignores day-ahead positions, congestion, and ancillary-service settlements;
- is not the price paid by any specific QSE or retail customer.

A $200/MWh threshold is an inherited code constant. Its scientific rationale is
[MISSING] beyond that constant. Spike precision and recall are descriptive.

## What the February 2021 diagnostic is

The 2021 evaluation is a chronological stress check with a pre-2021 fit. Official
FERC/NERC, ERCOT, and PUC timestamps define onset, emergency, and recovery
sub-windows after those cuts were committed and before the sub-window metrics
were computed. The diagnostic does not claim a causal mechanism, a resilience
result, or a recommended operating policy.

## What is not estimated

No trading strategy, no dispatch recommendation, no scarcity adder, and no
consumer-bill calculation is produced. Accuracy on ordinary hours and errors on
scarcity hours are reported separately so they are not substituted for each other.
"""
    path = EVIDENCE / "TARGET_AND_USE.md"
    path.write_text(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["original", "refit", "figure", "target-doc", "all"])
    args = p.parse_args()
    if args.stage in ("original", "all"):
        original_seed42()
    if args.stage in ("refit", "all"):
        five_seed_refit()
    if args.stage in ("figure", "all"):
        figure()
    if args.stage in ("target-doc", "all"):
        write_target_and_use()


if __name__ == "__main__":
    main()
