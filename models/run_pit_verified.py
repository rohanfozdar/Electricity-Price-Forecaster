"""Corrected point-in-time ablation. Does not write 2025-2026 scores.

The primary delay is one hour. That choice is the shortest pre-specified
assumption that excludes the interval ending at the issue time. It is fixed
before any corrected MAE is read. Outputs go to runs/pit_verified/.

    python models/run_pit_verified.py all
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.run_benchmark import SPIKE_PRICE, index_hash  # noqa: E402
from models.run_pit_ablation import (  # noqa: E402
    ASINH_CLIP,
    BOOT_BLOCK,
    BOOT_DRAWS,
    FROZEN_DEPTH,
    FROZEN_LR,
    PRIMARY_SEED,
    SEEDS,
    TARGET,
    TEST_START,
    TRAIN_START,
    VAL_START,
    compare_pair,
    contiguous_sizes,
    decode_target,
    fit_xgb,
    load_sources,
    score_pair,
)
from src.evaluation.statistical_tests import block_indices, diebold_mariano, paired_bootstrap  # noqa: E402
from src.features.pit_latency import (  # noqa: E402
    PRIMARY_LOAD_DELAY_HOURS,
    PRIMARY_PRICE_DELAY_HOURS,
    availability_manifest,
    build_delayed_frame,
    hourly_max_from_intervals,
    hourly_mean_from_intervals,
)
from src.features.pit_sets import (  # noqa: E402
    apply_asinh,
    feature_subsets,
    signed_asinh_columns,
    train_median_impute,
)

PROTOCOL = "2026-10-13-pit-verified-v1"
OUT = ROOT / "runs" / "pit_verified"
CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
# Bins were written here before the corrected 2024 predictions existed.
REGIMES = (
    ("negative", lambda y: y < 0),
    ("0_to_50", lambda y: (y >= 0) & (y <= 50)),
    ("50_to_100", lambda y: (y > 50) & (y <= 100)),
    ("100_to_200", lambda y: (y > 100) & (y <= 200)),
    ("200_to_500", lambda y: (y > 200) & (y <= 500)),
    ("above_500", lambda y: y > 500),
)
CRISIS_START = pd.Timestamp("2021-02-10", tz="US/Central").tz_convert("UTC")
CRISIS_END = pd.Timestamp("2021-02-21", tz="US/Central").tz_convert("UTC")
BLOCK_LENGTHS = (24, 48, 72, 168)
LOAD_COLUMNS = ("load_lag_1h", "load_rollmean_24h", "miss_load")
SPECS = (
    {"name": "delay_1h", "price": 1, "load": 1, "drop_load": False, "role": "primary"},
    {"name": "delay_2h", "price": 2, "load": 2, "drop_load": False, "role": "latency"},
    {"name": "delay_3h", "price": 3, "load": 3, "drop_load": False, "role": "latency"},
    {"name": "delay_6h", "price": 6, "load": 6, "drop_load": False, "role": "latency"},
    {"name": "delay_1h_load_24h", "price": 1, "load": 24, "drop_load": False, "role": "load_sensitivity"},
    {"name": "delay_1h_load_dropped", "price": 1, "load": 1, "drop_load": True, "role": "load_sensitivity"},
)
MASTER_COLUMNS = (
    "analysis", "scenario", "price_delay_hours", "load_delay_hours", "load_mode",
    "model", "seed", "estimand", "split", "subset", "block_hours", "n", "n_spikes",
    "mae", "rmse", "bias", "medae", "spike_precision", "spike_recall", "spike_f1",
    "spike_mae", "nonspike_mae", "mae_gap", "gap_ci_low", "gap_ci_high",
    "bootstrap_p", "dm_stat", "dm_p", "interval_excludes_zero", "artifact",
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _read_json(path: Path):
    return json.loads(path.read_text())


def _na(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return value


def master_row(**kwargs) -> dict:
    row = {col: "NA" for col in MASTER_COLUMNS}
    for key, value in kwargs.items():
        if key not in row:
            raise KeyError(key)
        row[key] = _na(value)
    for key in ("analysis", "scenario", "split", "subset", "artifact", "model"):
        text = str(row[key])
        if "2025" in text or "2026" in text:
            raise AssertionError(f"{key} mentions 2025 or 2026: {text}")
    return row


def enrich(y, pred) -> dict:
    metrics = score_pair(y, pred)
    err = np.asarray(pred, float) - np.asarray(y, float)
    spike = np.asarray(y, float) > SPIKE_PRICE
    metrics["bias"] = float(np.mean(err))
    metrics["spike_mae"] = float(np.mean(np.abs(err[spike]))) if spike.any() else None
    metrics["nonspike_mae"] = float(np.mean(np.abs(err[~spike]))) if (~spike).any() else None
    metrics["n_spikes"] = int(spike.sum())
    return metrics


def finite_slice(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    values = frame.loc[index, TARGET].to_numpy(dtype=float)
    return index[np.isfinite(values)]


def prepare(frame, columns, train_idx, eval_idx, test_idx):
    parts = [frame.loc[ix, columns].copy() for ix in (train_idx, eval_idx, test_idx)]
    transformed = [apply_asinh(part, signed_asinh_columns(columns)) for part in parts]
    usable = [col for col in columns if transformed[0][col].notna().any()]
    filled, medians = train_median_impute(transformed[0], transformed, usable)
    return [part[usable] for part in filled], usable, medians


def fit_predict(frame, columns, train_idx, eval_idx, test_idx, seed: int, transform: str):
    matrices, usable, medians = prepare(frame, columns, train_idx, eval_idx, test_idx)
    y_train = frame.loc[train_idx, TARGET].to_numpy(float)
    y_eval = frame.loc[eval_idx, TARGET].to_numpy(float)
    y_test = frame.loc[test_idx, TARGET].to_numpy(float)
    if transform == "asinh":
        y_fit, y_stop = np.arcsinh(y_train), np.arcsinh(y_eval)
        decode = decode_target
    elif transform == "raw":
        y_fit, y_stop = y_train, y_eval
        decode = lambda pred: np.asarray(pred, float)
    else:
        raise ValueError(transform)
    model, best, device = fit_xgb(
        matrices[0], y_fit, matrices[1], y_stop, seed, FROZEN_DEPTH, FROZEN_LR,
    )
    scored = {}
    for name, matrix, y in (("val", matrices[1], y_eval), ("test", matrices[2], y_test)):
        raw = model.predict(matrix, iteration_range=(0, best + 1))
        scored[name] = (y, decode(raw))
    return scored, int(best), device, usable, medians


def pred_path(key: str) -> Path:
    return OUT / "predictions" / f"{key}.parquet"


def load_prediction(key: str) -> pd.DataFrame | None:
    path = pred_path(key)
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        return None
    if _read_json(meta).get("protocol") != PROTOCOL:
        return None
    return pd.read_parquet(path)


def save_prediction(key: str, index, y, pred, split: str, meta: dict) -> None:
    frame = pd.DataFrame({"split": split, "y_true": y, "y_pred": pred}, index=index)
    pred_path(key).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(pred_path(key))
    _write_json(pred_path(key).with_suffix(".json"), {"protocol": PROTOCOL, **meta})


def study_sources(raw: dict) -> dict:
    """Drop source rows stamped in 2025 or later. Prices on disk already end in 2024."""
    out = dict(raw)
    for key in ("matrix", "chronos", "rt", "da", "load"):
        obj = out[key]
        if obj.index.max() >= CUTOFF:
            raise AssertionError(f"{key} extends into 2025")
    dropped = {}
    for key in ("gas", "storage", "nrc"):
        before = len(out[key])
        out[key] = out[key].loc[out[key].index < CUTOFF]
        dropped[key] = before - len(out[key])
    out["rows_dropped_after_2024"] = dropped
    return out


def columns_for(frame: pd.DataFrame, drop_load: bool) -> list[str]:
    saved = json.loads((ROOT / "runs" / "pit_ablation" / "matrices" / "conservative_columns.json").read_text())
    names = [name for name in saved if not (drop_load and name in LOAD_COLUMNS)]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"delayed frame is missing {missing}")
    return names


def assert_delay(frame: pd.DataFrame, hourly: pd.Series, delay_hours: int) -> None:
    sample = frame.index[frame.index >= pd.Timestamp("2024-06-15", tz="UTC")][0]
    expected = hourly.loc[sample - pd.Timedelta(hours=1 + delay_hours)]
    actual = frame.loc[sample, "hubavg_lag_1h"]
    if not np.isclose(actual, expected, equal_nan=True):
        raise AssertionError(f"delay check failed at {sample}: {actual} != {expected}")


def build_spec(sources: dict, spec: dict, target: str = "max", compute_hash: bool = True,
               chronos_cache: pd.DataFrame | None = None, rt_price: pd.Series | None = None):
    index = sources["matrix"].index
    price = sources["rt"][TARGET] if rt_price is None else rt_price
    cache = sources["chronos"] if chronos_cache is None else chronos_cache
    chronos_delay = spec["price"] if chronos_cache is None else 0
    frame, provenance = build_delayed_frame(
        index, price, sources["da"], sources["load"], sources["storage"],
        sources["gas"], sources["nrc"], cache,
        price_delay_hours=spec["price"], load_delay_hours=spec["load"],
        drop_load=spec["drop_load"], target=target, compute_hash=compute_hash,
        chronos_delay_hours=chronos_delay,
    )
    hourly = hourly_max_from_intervals(price)
    assert_delay(frame, hourly, spec["price"])
    return frame, provenance, hourly


def timestamp_check(sources: dict) -> dict:
    columns = [str(col) for col in sources["rt"].columns]
    publication = [col for col in columns if any(tok in col.lower() for tok in ("pub", "post", "release", "available"))]
    payload = {
        "rt_columns": columns,
        "publication_timestamp_columns": publication,
        "outcome": "no publication timestamp column in ercot_rt_prices.parquet",
        "load_columns": [str(col) for col in sources["load"].index.names] if False else ["load_actual_mw"],
        "load_name": str(sources["load"].name),
    }
    if publication:
        payload["outcome"] = "unexpected publication-like columns: " + ", ".join(publication)
    _write_json(OUT / "timestamp_check.json", payload)
    return payload


def series_check(sources: dict) -> dict:
    hourly = hourly_max_from_intervals(sources["rt"][TARGET])
    matrix = sources["matrix"][TARGET]
    both = pd.concat([hourly.rename("interval"), matrix.rename("matrix")], axis=1).dropna()
    gap = (both["interval"] - both["matrix"]).abs()
    payload = {
        "overlap_hours": int(len(both)),
        "hours_abs_diff_gt_0_01": int((gap > 0.01).sum()),
        "max_abs_diff": float(gap.max()) if len(gap) else None,
        "note": (
            "Lag features and the new Chronos context both use the 15-minute hourly maximum. "
            "The old cache in data/features/chronos_hourly was built on the matrix series and is not reused."
        ),
    }
    _write_json(OUT / "series_check.json", payload)
    return payload


def write_reproduction() -> dict:
    metrics = pd.read_csv(ROOT / "runs" / "pit_ablation" / "ablation_metrics.csv")
    comparisons = _read_json(ROOT / "runs" / "pit_ablation" / "comparisons.json")
    replay = _read_json(ROOT / "runs" / "pit_ablation" / "saved_booster_replay.json")
    manifest = _read_json(ROOT / "runs" / "pit_ablation" / "build_manifest.json")
    primary = metrics[(metrics["role"] == "primary") & (metrics["split"] == "test") & (metrics["info_set"] == "conservative")]
    means = primary.groupby("model")[["mae", "rmse", "medae"]].mean()
    by_name = {row["name"]: row for row in comparisons}
    mean_seed = by_name["conservative__mean_seed__both_minus_exog"]
    seed42 = by_name["conservative__seed42__both_minus_exog"]
    payload = {
        "refit": False,
        "used_2025_or_2026": False,
        "test_hash": manifest["test_hash"],
        "replay_mae": replay["mae"],
        "replay_max_abs_error": replay["max_abs_error"],
        "seed_averaged_forecast": {
            "mae_with_chronos": mean_seed["mae_a"],
            "mae_without_chronos": mean_seed["mae_b"],
            "gap": mean_seed["mae_gap_a_minus_b"],
            "ci": [mean_seed["bootstrap_ci_low"], mean_seed["bootstrap_ci_high"]],
            "bootstrap_p": mean_seed["bootstrap_p"],
            "dm_stat": mean_seed["dm_stat"],
            "dm_p": mean_seed["dm_p"],
            "interval_excludes_zero": mean_seed["interval_excludes_zero"],
            "per_seed_gaps": mean_seed.get("per_seed_mae_gap_both_minus_exog"),
        },
        "mean_of_seed_maes": {
            "with_chronos": float(means.loc["xgb_both", "mae"]),
            "without_chronos": float(means.loc["xgb_exog", "mae"]),
            "gap": float(means.loc["xgb_both", "mae"] - means.loc["xgb_exog", "mae"]),
        },
        "seed42_gap": seed42["mae_gap_a_minus_b"],
        "seed42_ci": [seed42["bootstrap_ci_low"], seed42["bootstrap_ci_high"]],
        "seed42_excludes_zero": seed42["interval_excludes_zero"],
    }
    _write_json(OUT / "reproduction_summary.json", payload)
    gap = payload["seed_averaged_forecast"]
    text = f"""# Reproduction of the saved conservative ablation

No model was refit for this summary. No 2025 or 2026 score is included. The files read were `runs/pit_ablation/ablation_metrics.csv`, `runs/pit_ablation/comparisons.json`, `runs/pit_ablation/saved_booster_replay.json`, and `runs/pit_ablation/build_manifest.json`.

The saved honest-tree booster replays at MAE ${replay['mae']:.4f} with a maximum absolute difference of {replay['max_abs_error']}. The frozen 2024 index hash is `{manifest['test_hash']}`.

Two different estimands were stored in the earlier run:

- MAE of the seed-averaged forecast: ${gap['mae_with_chronos']:.4f} with Chronos and ${gap['mae_without_chronos']:.4f} without. The gap is ${gap['gap']:.4f}. The 24-hour block bootstrap interval is [{gap['ci'][0]:.4f}, {gap['ci'][1]:.4f}], bootstrap p {gap['bootstrap_p']:.3f}, Diebold-Mariano p {gap['dm_p']:.3f}. The interval covers zero.
- Mean of the five seed-level MAEs: ${payload['mean_of_seed_maes']['with_chronos']:.4f} with Chronos and ${payload['mean_of_seed_maes']['without_chronos']:.4f} without. The difference of those means is ${payload['mean_of_seed_maes']['gap']:.4f}. That number is not the gap above.

Per-seed gaps, Chronos minus exogenous, in seed order 0, 1, 2, 3, 42: {', '.join(f'{v:.4f}' for v in gap['per_seed_gaps'])}. The seed-42 pair alone has gap ${payload['seed42_gap']:.4f} and an interval that excludes zero. That pair was not the pre-registered summary.

2024 had already been inspected when that ablation was run. This file confirms the saved arithmetic. It does not re-establish that every feature was public at the issue time. The zero-delay label on real-time prices is withdrawn in `reports/source_evidence_register.md`.
"""
    path = ROOT / "reports" / "pit_verified_reproduction.md"
    path.write_text(text)
    return payload


def scenario_key(spec, model, seed, transform="asinh", target="max") -> str:
    return f"{spec['name']}__{target}__{transform}__{model}__seed{seed}"


def run_fits(sources: dict, master: list, caches: dict) -> None:
    rows = []
    for spec in SPECS:
        print(f"scenario {spec['name']}", flush=True)
        frame, provenance, _hourly = build_spec(
            sources, spec, compute_hash=spec["role"] == "primary",
            chronos_cache=caches[spec["price"]],
        )
        if spec["role"] == "primary":
            provenance.to_parquet(OUT / "chronos" / "delay_1h_aligned.parquet")
        columns = columns_for(frame, spec["drop_load"])
        subsets = feature_subsets(columns)
        train = finite_slice(frame, frame.index[(frame.index >= TRAIN_START) & (frame.index < VAL_START)])
        eval_idx = finite_slice(frame, frame.index[(frame.index >= VAL_START) & (frame.index < TEST_START)])
        test = finite_slice(frame, frame.index[frame.index >= TEST_START])
        if not (train.max() < eval_idx.min() and eval_idx.max() < test.min()):
            raise AssertionError("split overlap")
        _write_json(OUT / "splits" / f"{spec['name']}.json", {
            "train_rows": int(len(train)), "val_rows": int(len(eval_idx)), "test_rows": int(len(test)),
            "test_hash": index_hash(test), "test_start": str(test.min()), "test_end": str(test.max()),
        })
        for model, model_cols in (("xgb_exog", subsets["xgb_exog"]), ("xgb_both", subsets["xgb_both"])):
            for seed in SEEDS:
                key = scenario_key(spec, model, seed)
                existing = load_prediction(key)
                if existing is None:
                    t0 = time.time()
                    scored, best, device, usable, _med = fit_predict(
                        frame, model_cols, train, eval_idx, test, seed, "asinh",
                    )
                    y, pred = scored["test"]
                    save_prediction(key, test, y, pred, "test", {
                        "scenario": spec["name"], "model": model, "seed": seed,
                        "best_iteration": best, "device": device, "n_columns": len(usable),
                        "seconds": round(time.time() - t0, 2), "transform": "asinh", "target": "max",
                    })
                    print(f"  fit {key} {time.time() - t0:.1f}s best {best}", flush=True)
                else:
                    y = existing["y_true"].to_numpy()
                    pred = existing["y_pred"].to_numpy()
                metrics = enrich(y, pred)
                rows.append({
                    "scenario": spec["name"], "model": model, "seed": seed,
                    "price_delay_hours": spec["price"], "load_delay_hours": spec["load"],
                    "load_mode": "dropped" if spec["drop_load"] else "delayed",
                    **{k: metrics[k] for k in (
                        "n", "n_spikes", "mae", "rmse", "bias", "medae", "spike_precision",
                        "spike_recall", "spike_f1", "spike_mae", "nonspike_mae",
                    )},
                })
                master.append(master_row(
                    analysis="latency_fit", scenario=spec["name"], price_delay_hours=spec["price"],
                    load_delay_hours=spec["load"],
                    load_mode="dropped" if spec["drop_load"] else "delayed",
                    model=model, seed=seed, estimand="single_seed", split="test_2024",
                    subset="all_hours", n=metrics["n"], n_spikes=metrics["n_spikes"],
                    mae=metrics["mae"], rmse=metrics["rmse"], bias=metrics["bias"],
                    medae=metrics["medae"], spike_precision=metrics["spike_precision"],
                    spike_recall=metrics["spike_recall"], spike_f1=metrics["spike_f1"],
                    spike_mae=metrics["spike_mae"], nonspike_mae=metrics["nonspike_mae"],
                    artifact=str(pred_path(key).relative_to(ROOT)),
                ))
        del frame
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)


def averaged(spec, model) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, list[float]]:
    frames = []
    maes = []
    for seed in SEEDS:
        frame = load_prediction(scenario_key(spec, model, seed))
        if frame is None:
            raise FileNotFoundError(scenario_key(spec, model, seed))
        frames.append(frame)
        maes.append(float(np.mean(np.abs(frame["y_pred"] - frame["y_true"]))))
    if any(not frames[0].index.equals(frame.index) for frame in frames[1:]):
        raise AssertionError("seed indexes differ")
    y = frames[0]["y_true"].to_numpy()
    pred = np.mean([frame["y_pred"].to_numpy() for frame in frames], axis=0)
    return frames[0].index, y, pred, maes


def run_comparisons(master: list) -> list[dict]:
    rows = []
    for spec in SPECS:
        index, y, both, both_maes = averaged(spec, "xgb_both")
        _, _, exog, exog_maes = averaged(spec, "xgb_exog")
        compared = compare_pair(
            f"{spec['name']}__mean_seed__both_minus_exog", spec["name"], y, both, exog, index,
            "xgb_both_mean_seed", "xgb_exog_mean_seed",
        )
        compared.update({
            "price_delay_hours": spec["price"],
            "load_delay_hours": spec["load"],
            "load_mode": "dropped" if spec["drop_load"] else "delayed",
            "role": spec["role"],
            "estimand": "mae_of_seed_averaged_forecast",
            "mean_of_seed_maes_both": float(np.mean(both_maes)),
            "mean_of_seed_maes_exog": float(np.mean(exog_maes)),
            "gap_of_mean_seed_maes": float(np.mean(both_maes) - np.mean(exog_maes)),
            "per_seed_mae_both": both_maes,
            "per_seed_mae_exog": exog_maes,
            "per_seed_gap": [a - b for a, b in zip(both_maes, exog_maes)],
        })
        both_m = enrich(y, both)
        exog_m = enrich(y, exog)
        compared["seed_averaged_metrics_both"] = both_m
        compared["seed_averaged_metrics_exog"] = exog_m
        rows.append(compared)
        for model, metrics in (("xgb_both", both_m), ("xgb_exog", exog_m)):
            master.append(master_row(
                analysis="latency_gap" if model == "xgb_both" else "latency_level",
                scenario=spec["name"], price_delay_hours=spec["price"],
                load_delay_hours=spec["load"],
                load_mode=compared["load_mode"], model=model, seed="mean_of_seeds",
                estimand="mae_of_seed_averaged_forecast", split="test_2024", subset="all_hours",
                block_hours=BOOT_BLOCK, n=metrics["n"], n_spikes=metrics["n_spikes"],
                mae=metrics["mae"], rmse=metrics["rmse"], bias=metrics["bias"], medae=metrics["medae"],
                spike_precision=metrics["spike_precision"], spike_recall=metrics["spike_recall"],
                spike_f1=metrics["spike_f1"], spike_mae=metrics["spike_mae"],
                nonspike_mae=metrics["nonspike_mae"],
                mae_gap=compared["mae_gap_a_minus_b"] if model == "xgb_both" else "NA",
                gap_ci_low=compared["bootstrap_ci_low"] if model == "xgb_both" else "NA",
                gap_ci_high=compared["bootstrap_ci_high"] if model == "xgb_both" else "NA",
                bootstrap_p=compared["bootstrap_p"] if model == "xgb_both" else "NA",
                dm_stat=compared["dm_stat"] if model == "xgb_both" else "NA",
                dm_p=compared["dm_p"] if model == "xgb_both" else "NA",
                interval_excludes_zero=compared["interval_excludes_zero"] if model == "xgb_both" else "NA",
                artifact="runs/pit_verified/comparisons.json",
            ))
        master.append(master_row(
            analysis="mean_of_seed_maes", scenario=spec["name"], price_delay_hours=spec["price"],
            load_delay_hours=spec["load"], load_mode=compared["load_mode"],
            model="xgb_both_minus_xgb_exog", seed="mean_of_five",
            estimand="difference_of_mean_seed_maes", split="test_2024", subset="all_hours",
            mae_gap=compared["gap_of_mean_seed_maes"],
            artifact="runs/pit_verified/comparisons.json",
        ))
        if spec["role"] == "primary":
            both42 = load_prediction(scenario_key(spec, "xgb_both", PRIMARY_SEED))
            exog42 = load_prediction(scenario_key(spec, "xgb_exog", PRIMARY_SEED))
            one = compare_pair(
                f"{spec['name']}__seed42__both_minus_exog", spec["name"],
                both42["y_true"].to_numpy(), both42["y_pred"].to_numpy(), exog42["y_pred"].to_numpy(),
                both42.index, "xgb_both_seed42", "xgb_exog_seed42",
            )
            one["estimand"] = "single_seed_42"
            rows.append(one)
    _write_json(OUT / "comparisons.json", rows)
    return rows


def run_block_and_spikes(master: list) -> None:
    spec = SPECS[0]
    index, y, both, _ = averaged(spec, "xgb_both")
    _, _, exog, _ = averaged(spec, "xgb_exog")
    sizes = contiguous_sizes(index)
    block_rows = []
    for block in BLOCK_LENGTHS:
        boot = paired_bootstrap(
            lambda yy, pp: float(np.mean(np.abs(pp - yy))),
            y, both, exog, sizes, block, BOOT_DRAWS, seed=0,
        )
        dm = diebold_mariano(np.abs(both - y), np.abs(exog - y), block)
        block_rows.append({
            "block_hours": block, "mae_gap": boot["diff"], "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"], "bootstrap_p": boot["p_value"], "n_boot": boot["n_boot"],
            "dm_stat": dm["stat"], "dm_p": dm["p_value"],
            "interval_excludes_zero": bool(boot["ci_high"] < 0 or boot["ci_low"] > 0),
        })
        master.append(master_row(
            analysis="block_length", scenario=spec["name"], price_delay_hours=1, load_delay_hours=1,
            load_mode="delayed", model="xgb_both_minus_xgb_exog", seed="mean_of_seeds",
            estimand="mae_of_seed_averaged_forecast", split="test_2024", subset="all_hours",
            block_hours=block, n=len(y), n_spikes=int((y > SPIKE_PRICE).sum()),
            mae_gap=boot["diff"], gap_ci_low=boot["ci_low"], gap_ci_high=boot["ci_high"],
            bootstrap_p=boot["p_value"], dm_stat=dm["stat"], dm_p=dm["p_value"],
            interval_excludes_zero=bool(boot["ci_high"] < 0 or boot["ci_low"] > 0),
            artifact="runs/pit_verified/block_lengths.csv",
        ))
    pd.DataFrame(block_rows).to_csv(OUT / "block_lengths.csv", index=False)

    spike_rows = []
    rule = (
        "A draw with no true spike leaves recall and F1 undefined. A draw with no "
        "predicted spike leaves precision and F1 undefined. Undefined draws are counted "
        "and omitted from that metric's quantiles. They are not filled with zero."
    )
    for model, pred in (("xgb_both", both), ("xgb_exog", exog)):
        for block in BLOCK_LENGTHS:
            stats = _spike_bootstrap(y, pred, sizes, block)
            stats.update({"model": model, "block_hours": block, "rule": rule})
            spike_rows.append(stats)
            master.append(master_row(
                analysis="spike_uncertainty", scenario="delay_1h", price_delay_hours=1,
                load_delay_hours=1, load_mode="delayed", model=model, seed="mean_of_seeds",
                estimand="mae_of_seed_averaged_forecast", split="test_2024", subset="spikes",
                block_hours=block, n=len(y), n_spikes=int((y > SPIKE_PRICE).sum()),
                spike_precision=stats["precision"], spike_recall=stats["recall"],
                spike_f1=stats["f1"], artifact="runs/pit_verified/spike_uncertainty.csv",
            ))
    pd.DataFrame(spike_rows).to_csv(OUT / "spike_uncertainty.csv", index=False)


def _rates(y, pred):
    true_s = y > SPIKE_PRICE
    pred_s = pred > SPIKE_PRICE
    tp = float(np.sum(true_s & pred_s))
    fp = float(np.sum(~true_s & pred_s))
    fn = float(np.sum(true_s & ~pred_s))
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif np.isfinite(precision) and np.isfinite(recall):
        f1 = 0.0
    else:
        f1 = np.nan
    return precision, recall, f1


def _spike_bootstrap(y, pred, sizes, block: int) -> dict:
    rng = np.random.default_rng(0)
    precision, recall, f1 = _rates(y, pred)
    draws = np.empty((BOOT_DRAWS, 3))
    for i in range(BOOT_DRAWS):
        take = block_indices(sizes, block, rng)
        draws[i] = _rates(y[take], pred[take])
    out = {"precision": precision, "recall": recall, "f1": f1}
    for j, name in enumerate(("precision", "recall", "f1")):
        defined = draws[:, j][np.isfinite(draws[:, j])]
        out[f"n_undefined_{name}"] = int(np.isnan(draws[:, j]).sum())
        if len(defined) >= 100:
            lo, hi = np.quantile(defined, [0.025, 0.975])
            out[f"{name}_ci_low"] = float(lo)
            out[f"{name}_ci_high"] = float(hi)
        else:
            out[f"{name}_ci_low"] = None
            out[f"{name}_ci_high"] = None
    return out


def run_regimes(master: list) -> None:
    spec = SPECS[0]
    _index, y, both, _ = averaged(spec, "xgb_both")
    _, _, exog, _ = averaged(spec, "xgb_exog")
    sizes_all = None
    rows = []
    for name, mask_fn in REGIMES:
        mask = mask_fn(y)
        n = int(mask.sum())
        if n == 0:
            rows.append({"regime": name, "n": 0, "benefit_claim_allowed": False})
            continue
        gap = float(np.mean(np.abs(both[mask] - y[mask])) - np.mean(np.abs(exog[mask] - y[mask])))
        # Bootstrap the paired gap inside the regime by restricting the series.
        # Contiguous structure is not preserved inside a price bin, so the block
        # bootstrap here resamples the regime hours as one group.
        boot = paired_bootstrap(
            lambda yy, pp: float(np.mean(np.abs(pp - yy))),
            y[mask], both[mask], exog[mask], [n], BOOT_BLOCK, BOOT_DRAWS, seed=0,
        )
        claim = bool(n >= 100 and boot["ci_high"] < 0)
        rows.append({
            "regime": name, "n": n, "n_spikes": int((y[mask] > SPIKE_PRICE).sum()),
            "mae_both": float(np.mean(np.abs(both[mask] - y[mask]))),
            "mae_exog": float(np.mean(np.abs(exog[mask] - y[mask]))),
            "mae_gap": boot["diff"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "bootstrap_p": boot["p_value"], "interval_excludes_zero": bool(boot["ci_high"] < 0 or boot["ci_low"] > 0),
            "benefit_claim_allowed": claim,
            "note": "Realized-price subset. Not a model input. Claim withheld when n < 100 or the interval covers zero.",
        })
        master.append(master_row(
            analysis="price_regime", scenario="delay_1h", price_delay_hours=1, load_delay_hours=1,
            load_mode="delayed", model="xgb_both_minus_xgb_exog", seed="mean_of_seeds",
            estimand="mae_of_seed_averaged_forecast", split="test_2024", subset=name,
            block_hours=BOOT_BLOCK, n=n, n_spikes=int((y[mask] > SPIKE_PRICE).sum()),
            mae_gap=boot["diff"], gap_ci_low=boot["ci_low"], gap_ci_high=boot["ci_high"],
            bootstrap_p=boot["p_value"],
            interval_excludes_zero=bool(boot["ci_high"] < 0 or boot["ci_low"] > 0),
            artifact="runs/pit_verified/regimes.csv",
        ))
    del sizes_all
    pd.DataFrame(rows).to_csv(OUT / "regimes.csv", index=False)


def run_crisis(sources: dict, master: list, caches: dict) -> dict:
    spec = SPECS[0]
    frame, _prov, hourly = build_spec(sources, spec, compute_hash=False, chronos_cache=caches[1])
    columns = columns_for(frame, False)
    subsets = feature_subsets(columns)
    test_start = pd.Timestamp("2021-01-01", tz="UTC")
    test_end = pd.Timestamp("2022-01-01", tz="UTC")
    es_end = test_start
    es_start = es_end - pd.Timedelta(days=90)
    train = finite_slice(frame, frame.index[frame.index < es_start])
    eval_idx = finite_slice(frame, frame.index[(frame.index >= es_start) & (frame.index < es_end)])
    test = finite_slice(frame, frame.index[(frame.index >= test_start) & (frame.index < test_end)])
    train_prices = frame.loc[train, TARGET]
    payload = {
        "window": "2021-02-10 through 2021-02-20 US/Central, chosen before reading the corrected errors",
        "train_end": str(train.max()), "train_max_price": float(train_prices.max()),
        "train_p99_price": float(train_prices.quantile(0.99)),
        "asinh_clip_dollars": float(np.sinh(ASINH_CLIP)),
        "scored_2021_hours": int(len(test)),
    }
    fits = {}
    for model in ("xgb_exog", "xgb_both"):
        key = f"crisis2021__asinh__{model}__seed{PRIMARY_SEED}"
        existing = load_prediction(key)
        if existing is None:
            scored, best, device, usable, _med = fit_predict(
                frame, subsets[model], train, eval_idx, test, PRIMARY_SEED, "asinh",
            )
            y, pred = scored["test"]
            save_prediction(key, test, y, pred, "test", {
                "best_iteration": best, "device": device, "transform": "asinh", "model": model,
            })
        else:
            y = existing["y_true"].to_numpy()
            pred = existing["y_pred"].to_numpy()
            test = existing.index
        fits[model] = (pd.DatetimeIndex(test), np.asarray(y, float), np.asarray(pred, float))
    raw_key = f"crisis2021__raw__xgb_exog__seed{PRIMARY_SEED}"
    existing = load_prediction(raw_key)
    if existing is None:
        scored, best, device, _usable, _med = fit_predict(
            frame, subsets["xgb_exog"], train, eval_idx, test, PRIMARY_SEED, "raw",
        )
        save_prediction(raw_key, test, scored["test"][0], scored["test"][1], "test", {
            "best_iteration": best, "device": device, "transform": "raw", "model": "xgb_exog",
        })
        raw_y, raw_pred, raw_index = scored["test"][0], scored["test"][1], test
    else:
        raw_index, raw_y, raw_pred = existing.index, existing["y_true"].to_numpy(), existing["y_pred"].to_numpy()
    persistence = frame["hubavg_lag_1h"].reindex(fits["xgb_exog"][0]).to_numpy(dtype=float)
    if not np.isfinite(persistence).all():
        raise AssertionError("latency-valid persistence is missing on a scored 2021 hour")
    subsets_out = {}
    index = fits["xgb_exog"][0]
    masks = {
        "crisis_hours": (index >= CRISIS_START) & (index < CRISIS_END),
        "all_scored_2021": np.ones(len(index), dtype=bool),
        "outside_crisis": ~((index >= CRISIS_START) & (index < CRISIS_END)),
    }
    for label, mask in masks.items():
        block = {"n": int(mask.sum())}
        for model in ("xgb_exog", "xgb_both"):
            _ix, y, pred = fits[model]
            block[model] = enrich(y[mask], pred[mask])
            block[model]["pred_max"] = float(np.max(pred[mask]))
            block[model]["y_max"] = float(np.max(y[mask]))
            block[model]["n_pred_above_10000"] = int(np.sum(np.abs(pred[mask]) > 10000))
        block["persistence"] = enrich(fits["xgb_exog"][1][mask], persistence[mask])
        raw_mask = (raw_index >= CRISIS_START) & (raw_index < CRISIS_END) if label == "crisis_hours" else None
        if label == "crisis_hours":
            block["raw_target_xgb_exog"] = enrich(raw_y[raw_mask], raw_pred[raw_mask])
            block["raw_target_xgb_exog"]["pred_max"] = float(np.max(raw_pred[raw_mask]))
        subsets_out[label] = block
        for model in ("xgb_exog", "xgb_both", "persistence"):
            metrics = block[model]
            master.append(master_row(
                analysis="february_2021", scenario="delay_1h", price_delay_hours=1,
                load_delay_hours=1, load_mode="delayed", model=model, seed=PRIMARY_SEED,
                estimand="single_seed", split="expanding_2021", subset=label,
                n=metrics["n"], n_spikes=metrics["n_spikes"], mae=metrics["mae"],
                rmse=metrics["rmse"], bias=metrics["bias"], medae=metrics["medae"],
                spike_precision=metrics["spike_precision"], spike_recall=metrics["spike_recall"],
                spike_f1=metrics["spike_f1"], spike_mae=metrics["spike_mae"],
                nonspike_mae=metrics["nonspike_mae"],
                artifact=f"runs/pit_verified/predictions/crisis2021__asinh__{model}__seed42.parquet"
                if model != "persistence" else "runs/pit_verified/crisis_2021.json",
            ))
    payload["subsets"] = subsets_out
    full_2021 = pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h", tz="UTC")
    payload["calendar_hours_missing_from_matrix_index"] = int(len(full_2021.difference(sources["matrix"].index)))
    _write_json(OUT / "crisis_2021.json", payload)
    return payload


def run_baselines(sources: dict, master: list, caches: dict) -> None:
    spec = SPECS[0]
    frame, _prov, hourly = build_spec(sources, spec, compute_hash=False, chronos_cache=caches[1])
    test = frame.index[frame.index >= TEST_START]
    y = frame.loc[test, TARGET]
    keep = y.notna()
    y = y.loc[keep]
    test = y.index
    rows = []
    seasonal = {}
    for hours, name in ((24, "seasonal_naive_24"), (168, "seasonal_naive_168")):
        moved = hourly.copy()
        moved.index = hourly.index + pd.Timedelta(hours=hours)
        seasonal[name] = moved.reindex(test)
    seasonal["persistence_latency"] = frame.loc[test, "hubavg_lag_1h"]
    for name, pred in seasonal.items():
        mask = pred.notna() & y.notna()
        metrics = enrich(y.loc[mask].to_numpy(), pred.loc[mask].to_numpy())
        rows.append({"model": name, "information_set": "delay_1h", **metrics})
        master.append(master_row(
            analysis="baseline", scenario="delay_1h", price_delay_hours=1, load_delay_hours=1,
            load_mode="delayed", model=name, seed="na", estimand="point_forecast",
            split="test_2024", subset="all_hours", n=metrics["n"], n_spikes=metrics["n_spikes"],
            mae=metrics["mae"], rmse=metrics["rmse"], bias=metrics["bias"], medae=metrics["medae"],
            spike_precision=metrics["spike_precision"], spike_recall=metrics["spike_recall"],
            spike_f1=metrics["spike_f1"], spike_mae=metrics["spike_mae"],
            nonspike_mae=metrics["nonspike_mae"], artifact="runs/pit_verified/baselines.csv",
        ))
    lasso = _fit_lasso(frame, test)
    rows.append(lasso["row"])
    master.append(lasso["master"])
    historical = _score_historical(test)
    rows.extend(historical)
    for row in historical:
        master.append(master_row(
            analysis="historical_reference", scenario="not_latency_matched",
            price_delay_hours="NA", load_delay_hours="NA", load_mode="NA",
            model=row["model"], seed="na", estimand="saved_prediction",
            split="test_2024", subset="all_hours", n=row.get("n"), mae=row.get("mae"),
            rmse=row.get("rmse"), artifact=row.get("path", "NA"),
        ))
    pd.DataFrame(rows).to_csv(OUT / "baselines.csv", index=False)


def _fit_lasso(frame: pd.DataFrame, test: pd.DatetimeIndex) -> dict:
    from sklearn.linear_model import Lasso

    columns = [
        col for col in frame.columns
        if col.startswith("hubavg_lag_") or col.endswith("_local")
    ]
    train = frame.index[(frame.index >= TRAIN_START) & (frame.index < VAL_START)]
    val = frame.index[(frame.index >= VAL_START) & (frame.index < TEST_START)]
    train = finite_slice(frame, train)
    val = finite_slice(frame, val)
    test = finite_slice(frame, test)
    usable = list(columns)
    raw_parts = [frame.loc[ix, usable].copy() for ix in (train, val, test)]
    filled, _med = train_median_impute(raw_parts[0], raw_parts, usable)
    train_x = filled[0][usable].to_numpy(float)
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma == 0] = 1.0
    def scale(part):
        return (part[usable].to_numpy(float) - mu) / sigma
    y_train = frame.loc[train, TARGET].to_numpy(float)
    y_val = frame.loc[val, TARGET].to_numpy(float)
    y_test = frame.loc[test, TARGET].to_numpy(float)
    best_alpha, best_mae = None, np.inf
    for alpha in np.logspace(-4, 2, 19):
        model = Lasso(alpha=float(alpha), max_iter=8000)
        model.fit(scale(filled[0]), y_train)
        pred = model.predict(scale(filled[1]))
        mae = float(np.mean(np.abs(pred - y_val)))
        if mae < best_mae:
            best_alpha, best_mae = float(alpha), mae
    model = Lasso(alpha=best_alpha, max_iter=8000)
    model.fit(scale(filled[0]), y_train)
    pred = model.predict(scale(filled[2]))
    metrics = enrich(y_test, pred)
    save_prediction("lasso_delay_1h", test, y_test, pred, "test", {
        "alpha": best_alpha, "val_mae": best_mae, "columns": usable,
    })
    row = {"model": "lasso_lags_calendar", "information_set": "delay_1h", "alpha": best_alpha,
           "val_mae_for_alpha": best_mae, **metrics}
    return {
        "row": row,
        "master": master_row(
            analysis="baseline", scenario="delay_1h", price_delay_hours=1, load_delay_hours=1,
            load_mode="delayed", model="lasso_lags_calendar", seed="na",
            estimand="point_forecast", split="test_2024", subset="all_hours",
            n=metrics["n"], n_spikes=metrics["n_spikes"], mae=metrics["mae"], rmse=metrics["rmse"],
            bias=metrics["bias"], medae=metrics["medae"], spike_precision=metrics["spike_precision"],
            spike_recall=metrics["spike_recall"], spike_f1=metrics["spike_f1"],
            spike_mae=metrics["spike_mae"], nonspike_mae=metrics["nonspike_mae"],
            artifact="runs/pit_verified/predictions/lasso_delay_1h.parquet",
        ),
    }


def _score_historical(test: pd.DatetimeIndex) -> list[dict]:
    candidates = {
        "simplified_three_lag_lasso_historical": ROOT / "artifacts" / "honest" / "predictions" / "lear.parquet",
        "honest_tree_historical": ROOT / "artifacts" / "honest" / "predictions" / "honest_tree.parquet",
        "persistence_historical": ROOT / "artifacts" / "honest" / "predictions" / "persistence.parquet",
    }
    rows = []
    for name, path in candidates.items():
        if not path.exists():
            rows.append({"model": name, "mae": None, "note": "file missing", "path": str(path)})
            continue
        frame = pd.read_parquet(path)
        pred_col = "y_pred" if "y_pred" in frame.columns else "prediction" if "prediction" in frame.columns else None
        if pred_col is None:
            rows.append({"model": name, "mae": None, "note": f"columns {list(frame.columns)}", "path": str(path)})
            continue
        aligned = frame.reindex(test)
        y = aligned["y_true"] if "y_true" in aligned.columns else None
        if y is None or y.isna().all():
            rows.append({"model": name, "mae": None, "note": "index did not overlap the 2024 test", "path": str(path)})
            continue
        mask = y.notna() & aligned[pred_col].notna()
        metrics = enrich(y.loc[mask].to_numpy(), aligned.loc[mask, pred_col].to_numpy())
        rows.append({
            "model": name, "information_set": "historical_not_latency_matched",
            "path": str(path.relative_to(ROOT)), "note": "not Lago et al. LEAR; not the delayed information set",
            **metrics,
        })
    return rows


def run_hourly_mean(sources: dict, master: list, caches: dict) -> None:
    spec = SPECS[0]
    frame, _prov, _hourly = build_spec(
        sources, spec, target="mean", compute_hash=False, chronos_cache=caches[1],
    )
    columns = columns_for(frame, False)
    subsets = feature_subsets(columns)
    train = finite_slice(frame, frame.index[(frame.index >= TRAIN_START) & (frame.index < VAL_START)])
    eval_idx = finite_slice(frame, frame.index[(frame.index >= VAL_START) & (frame.index < TEST_START)])
    test = finite_slice(frame, frame.index[frame.index >= TEST_START])
    for model in ("xgb_exog", "xgb_both"):
        for seed in SEEDS:
            key = scenario_key(spec, model, seed, target="mean")
            existing = load_prediction(key)
            if existing is None:
                scored, best, device, _usable, _med = fit_predict(
                    frame, subsets[model], train, eval_idx, test, seed, "asinh",
                )
                y, pred = scored["test"]
                save_prediction(key, test, y, pred, "test", {
                    "target": "mean", "best_iteration": best, "device": device, "model": model, "seed": seed,
                })
            else:
                y, pred = existing["y_true"].to_numpy(), existing["y_pred"].to_numpy()
            metrics = enrich(y, pred)
            master.append(master_row(
                analysis="hourly_mean_target", scenario="delay_1h", price_delay_hours=1,
                load_delay_hours=1, load_mode="delayed", model=model, seed=seed,
                estimand="single_seed", split="test_2024", subset="hourly_mean",
                n=metrics["n"], n_spikes=metrics["n_spikes"], mae=metrics["mae"], rmse=metrics["rmse"],
                bias=metrics["bias"], medae=metrics["medae"], spike_precision=metrics["spike_precision"],
                spike_recall=metrics["spike_recall"], spike_f1=metrics["spike_f1"],
                spike_mae=metrics["spike_mae"], nonspike_mae=metrics["nonspike_mae"],
                artifact=str(pred_path(key).relative_to(ROOT)),
            ))
    index, y, both, both_maes = averaged_target(spec, "xgb_both", "mean")
    _, _, exog, exog_maes = averaged_target(spec, "xgb_exog", "mean")
    compared = compare_pair(
        "delay_1h_hourly_mean__both_minus_exog", "delay_1h_hourly_mean", y, both, exog, index,
        "xgb_both", "xgb_exog",
    )
    compared["mean_of_seed_maes_gap"] = float(np.mean(both_maes) - np.mean(exog_maes))
    compared["note"] = (
        "Secondary target. Chronos statistics remain forecasts of the hourly maximum. "
        "This row does not replace the hourly-maximum headline."
    )
    _write_json(OUT / "hourly_mean.json", compared)
    master.append(master_row(
        analysis="hourly_mean_target", scenario="delay_1h", price_delay_hours=1,
        load_delay_hours=1, load_mode="delayed", model="xgb_both_minus_xgb_exog",
        seed="mean_of_seeds", estimand="mae_of_seed_averaged_forecast", split="test_2024",
        subset="hourly_mean", block_hours=BOOT_BLOCK, n=len(y),
        mae_gap=compared["mae_gap_a_minus_b"], gap_ci_low=compared["bootstrap_ci_low"],
        gap_ci_high=compared["bootstrap_ci_high"], bootstrap_p=compared["bootstrap_p"],
        dm_stat=compared["dm_stat"], dm_p=compared["dm_p"],
        interval_excludes_zero=compared["interval_excludes_zero"],
        artifact="runs/pit_verified/hourly_mean.json",
    ))


def averaged_target(spec, model, target):
    frames, maes = [], []
    for seed in SEEDS:
        frame = load_prediction(scenario_key(spec, model, seed, target=target))
        frames.append(frame)
        maes.append(float(np.mean(np.abs(frame["y_pred"] - frame["y_true"]))))
    y = frames[0]["y_true"].to_numpy()
    pred = np.mean([frame["y_pred"].to_numpy() for frame in frames], axis=0)
    return frames[0].index, y, pred, maes


def write_manifest() -> None:
    manifest = availability_manifest(PRIMARY_PRICE_DELAY_HOURS, PRIMARY_LOAD_DELAY_HOURS)
    path = ROOT / "reports" / "feature_availability_manifest.csv"
    manifest.to_csv(path, index=False)
    if list(manifest.columns) != list(availability_manifest().columns):
        raise AssertionError("manifest columns drifted")


def write_master(master: list) -> None:
    frame = pd.DataFrame(master, columns=MASTER_COLUMNS)
    frame.to_csv(OUT / "master_metrics.csv", index=False)
    text = frame.to_csv(index=False)
    if "\n2025" in text or ",2025," in text or ",2026," in text:
        raise AssertionError("master metrics contain a 2025 or 2026 field")


def load_caches(sources: dict, delays: tuple[int, ...]) -> dict:
    from models.chronos_delayed_cache import cache_path, ensure_cache

    caches = {}
    for delay in delays:
        caches[delay] = ensure_cache(
            sources["rt"][TARGET], delay, cache_path(ROOT, f"delay_{delay}h"), PROTOCOL,
        )
    return caches


def run_hub(sources: dict, master: list) -> None:
    """HB_NORTH with its own latency-filtered Chronos cache. Not another market."""
    from models.chronos_delayed_cache import cache_path, ensure_cache

    if "HB_NORTH" not in sources["rt"].columns:
        _write_json(OUT / "hub_north.json", {"status": "HB_NORTH column missing"})
        return
    cache = ensure_cache(
        sources["rt"]["HB_NORTH"], 1, cache_path(ROOT, "hb_north_delay_1h"), PROTOCOL,
    )
    spec = {"name": "hb_north_delay_1h", "price": 1, "load": 1, "drop_load": False, "role": "hub"}
    frame, _prov, _hourly = build_spec(
        sources, spec, compute_hash=False, chronos_cache=cache, rt_price=sources["rt"]["HB_NORTH"],
    )
    columns = columns_for(frame, False)
    subsets = feature_subsets(columns)
    train = finite_slice(frame, frame.index[(frame.index >= TRAIN_START) & (frame.index < VAL_START)])
    eval_idx = finite_slice(frame, frame.index[(frame.index >= VAL_START) & (frame.index < TEST_START)])
    test = finite_slice(frame, frame.index[frame.index >= TEST_START])
    for model in ("xgb_exog", "xgb_both"):
        for seed in SEEDS:
            key = scenario_key(spec, model, seed)
            existing = load_prediction(key)
            if existing is None:
                scored, best, device, _usable, _med = fit_predict(
                    frame, subsets[model], train, eval_idx, test, seed, "asinh",
                )
                y, pred = scored["test"]
                save_prediction(key, test, y, pred, "test", {
                    "model": model, "seed": seed, "hub": "HB_NORTH", "best_iteration": best, "device": device,
                })
            else:
                y, pred = existing["y_true"].to_numpy(), existing["y_pred"].to_numpy()
            metrics = enrich(y, pred)
            master.append(master_row(
                analysis="hb_north", scenario="hb_north_delay_1h", price_delay_hours=1,
                load_delay_hours=1, load_mode="delayed", model=model, seed=seed,
                estimand="single_seed", split="test_2024", subset="HB_NORTH",
                n=metrics["n"], n_spikes=metrics["n_spikes"], mae=metrics["mae"],
                rmse=metrics["rmse"], bias=metrics["bias"], medae=metrics["medae"],
                spike_precision=metrics["spike_precision"], spike_recall=metrics["spike_recall"],
                spike_f1=metrics["spike_f1"], spike_mae=metrics["spike_mae"],
                nonspike_mae=metrics["nonspike_mae"],
                artifact=str(pred_path(key).relative_to(ROOT)),
            ))
    index, y, both, both_maes = averaged(spec, "xgb_both")
    _, _, exog, exog_maes = averaged(spec, "xgb_exog")
    compared = compare_pair(
        "hb_north_delay_1h__both_minus_exog", "hb_north_delay_1h", y, both, exog, index,
        "xgb_both", "xgb_exog",
    )
    compared["mean_of_seed_maes_gap"] = float(np.mean(both_maes) - np.mean(exog_maes))
    compared["note"] = (
        "HB_NORTH prices and an HB_NORTH Chronos cache. Another ERCOT hub, not another market. "
        "The older transfer that reused the hub-average cache is not this result."
    )
    _write_json(OUT / "hub_north.json", compared)
    master.append(master_row(
        analysis="hb_north", scenario="hb_north_delay_1h", price_delay_hours=1,
        load_delay_hours=1, load_mode="delayed", model="xgb_both_minus_xgb_exog",
        seed="mean_of_seeds", estimand="mae_of_seed_averaged_forecast", split="test_2024",
        subset="HB_NORTH", block_hours=BOOT_BLOCK, n=len(y),
        n_spikes=int((y > SPIKE_PRICE).sum()),
        mae_gap=compared["mae_gap_a_minus_b"], gap_ci_low=compared["bootstrap_ci_low"],
        gap_ci_high=compared["bootstrap_ci_high"], bootstrap_p=compared["bootstrap_p"],
        dm_stat=compared["dm_stat"], dm_p=compared["dm_p"],
        interval_excludes_zero=compared["interval_excludes_zero"],
        artifact="runs/pit_verified/hub_north.json",
    ))


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    if command in {"reproduce", "all"}:
        write_reproduction()
        write_manifest()
        print("reproduction and manifest written", flush=True)
        if command == "reproduce":
            return
    sources = study_sources(load_sources())
    timestamp_check(sources)
    series_check(sources)
    master: list[dict] = []
    delays = (1, 2, 3, 6) if command in {"ablation", "all"} else (1,)
    caches = load_caches(sources, delays) if command != "reports" else {}
    if command in {"ablation", "all"}:
        run_fits(sources, master, caches)
        run_comparisons(master)
    if command in {"robustness", "all"}:
        if not (OUT / "metrics.csv").exists() and command != "all":
            raise SystemExit("ablation outputs are missing")
        run_block_and_spikes(master)
        run_regimes(master)
        run_crisis(sources, master, caches)
        run_baselines(sources, master, caches)
    if command in {"hourly-mean", "all"}:
        run_hourly_mean(sources, master, caches)
    if command in {"hub", "all"}:
        run_hub(sources, master)
    if master:
        write_master(master)
    if command in {"reports", "all"}:
        from models.pit_verified_reports import write_reports
        write_reports()
    print("done", command, flush=True)


if __name__ == "__main__":
    main()
