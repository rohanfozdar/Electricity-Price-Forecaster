"""Audit the executed one-hour-delay experiment. Does not refit that experiment.

A complete-four-interval sensitivity is fit only when that 2024 index is a
strict subset of the frozen index. Those fits go to runs/final_audit/.
The primary trees stay in runs/pit_verified/.

    python models/run_final_audit.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.run_benchmark import index_hash  # noqa: E402
from models.run_pit_ablation import (  # noqa: E402
    ASINH_CLIP,
    BOOT_DRAWS,
    compare_pair,
    contiguous_sizes,
    load_sources,
)
from models.run_pit_verified import (  # noqa: E402
    CRISIS_END,
    CRISIS_START,
    SEEDS,
    TARGET,
    TEST_START,
    TRAIN_START,
    VAL_START,
    build_spec,
    columns_for,
    enrich,
    finite_slice,
    fit_predict,
    study_sources,
)
from src.evaluation.statistical_tests import diebold_mariano, paired_bootstrap  # noqa: E402
from src.features.pit_latency import hourly_max_from_intervals  # noqa: E402
from src.features.pit_sets import (  # noqa: E402
    calendar_central,
    feature_subsets,
    clock_shift,
    day_ahead_features,
    lagged_load,
)

OUT = ROOT / "runs" / "final_audit"
REPORTS = ROOT / "reports" / "final_research_audit"
VERIFIED = ROOT / "runs" / "pit_verified"
EXPECTED_GAP = -0.14988770703544851
EXPECTED_CI = (-0.3174348218300281, 0.06375036832251495)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _money(value, digits=2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if value != value:
        return "NA"
    sign = "−" if value < 0 else ""
    return f"{sign}${abs(value):.{digits}f}"


def _yesno(flag) -> str:
    return "yes" if flag else "no"


def _num(value, digits=3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if value != value:
        return "NA"
    sign = "−" if value < 0 else ""
    return f"{sign}{abs(value):.{digits}f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _load_pair(model: str):
    frames = []
    for seed in SEEDS:
        path = VERIFIED / "predictions" / f"delay_1h__max__asinh__{model}__seed{seed}.parquet"
        frames.append(pd.read_parquet(path))
    if any(not frames[0].index.equals(frame.index) for frame in frames[1:]):
        raise AssertionError(f"{model} seeds are not aligned")
    y = frames[0]["y_true"].to_numpy(float)
    pred = np.mean([frame["y_pred"].to_numpy(float) for frame in frames], axis=0)
    seed_maes = [float(np.mean(np.abs(frame["y_pred"] - frame["y_true"]))) for frame in frames]
    return frames[0].index, y, pred, seed_maes


def reconcile(rt: pd.Series, eval_index: pd.DatetimeIndex) -> dict:
    counts = rt.groupby(pd.DatetimeIndex(rt.index).floor("h")).size().sort_index()
    calendar = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    raw_hours = counts.index
    missing = calendar.difference(eval_index)
    reasons = []
    for stamp in missing:
        n = int(counts[stamp]) if stamp in counts.index else 0
        if stamp not in raw_hours and stamp > rt.index.max().floor("h"):
            reason = "after the last raw price timestamp"
        elif n == 0:
            reason = "no 15-minute price in the raw file"
        elif stamp not in eval_index:
            reason = f"present in raw file with {n} intervals but absent from the evaluation index"
        else:
            reason = "unclassified"
        reasons.append({
            "hour_utc": str(stamp), "interval_count": n, "reason": reason,
            "hour_central": str(stamp.tz_convert("US/Central")),
        })
    in_eval = counts.reindex(eval_index)
    incomplete_hours = []
    for stamp, n in in_eval.items():
        count = 0 if pd.isna(n) else int(n)
        if count != 4:
            incomplete_hours.append({
                "hour_utc": str(stamp),
                "hour_central": str(pd.Timestamp(stamp).tz_convert("US/Central")),
                "interval_count": count,
                "in_evaluation_index": True,
            })
    dup_intervals = int(rt.index.duplicated().sum())
    payload = {
        "utc_2024_hours": int(len(calendar)),
        "evaluation_hours": int(len(eval_index)),
        "absent_from_evaluation": int(len(missing)),
        "raw_price_start": str(rt.index.min()),
        "raw_price_end": str(rt.index.max()),
        "duplicate_15min_stamps": dup_intervals,
        "eval_hours_with_4_intervals": int((in_eval == 4).sum()),
        "eval_hours_with_3_intervals": int((in_eval == 3).sum()),
        "eval_hours_with_2_intervals": int((in_eval == 2).sum()),
        "eval_hours_with_1_interval": int((in_eval == 1).sum()),
        "eval_hours_missing_from_raw": int(in_eval.isna().sum()),
        "incomplete_evaluation_hours": incomplete_hours,
        "complete_four_equals_frozen_index": bool((in_eval == 4).all() and in_eval.notna().all()),
        "exclusions": reasons,
        "dst_15min_counts": {},
    }
    local = rt.index.tz_convert("US/Central")
    for day in ("2024-03-10", "2024-11-03"):
        payload["dst_15min_counts"][day] = int((local.floor("D") == pd.Timestamp(day, tz="US/Central")).sum())
    payload["test_hash"] = index_hash(eval_index)
    payload["internal_gaps_hours"] = int(eval_index.to_series().diff().gt(pd.Timedelta(hours=1)).sum())
    _json(OUT / "target_reconciliation.json", payload)
    pd.DataFrame(reasons).to_csv(OUT / "excluded_2024_hours.csv", index=False)
    return payload


def reproduce(index, y, both, exog, both_maes, exog_maes) -> dict:
    both_m = enrich(y, both)
    exog_m = enrich(y, exog)
    compared = compare_pair(
        "recomputed_delay_1h", "delay_1h", y, both, exog, index, "both", "exog",
    )
    sizes = contiguous_sizes(index)
    blocks = []
    for block in (12, 24, 48, 168):
        boot = paired_bootstrap(
            lambda yy, pp: float(np.mean(np.abs(pp - yy))),
            y, both, exog, sizes, block, BOOT_DRAWS, seed=0,
        )
        dm = diebold_mariano(np.abs(both - y), np.abs(exog - y), block)
        blocks.append({
            "block_hours": block, "gap": boot["diff"], "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"], "bootstrap_p": boot["p_value"],
            "dm_p": dm["p_value"], "dm_stat": dm["stat"],
            "interval_excludes_zero": bool(boot["ci_high"] < 0 or boot["ci_low"] > 0),
        })
    primary_block = next(row for row in blocks if row["block_hours"] == 24)
    payload = {
        "estimand": "mae_of_seed_averaged_forecast",
        "n": int(len(y)),
        "test_hash": index_hash(index),
        "with_chronos": both_m,
        "without_chronos": exog_m,
        "gap": compared["mae_gap_a_minus_b"],
        "ci_low": compared["bootstrap_ci_low"],
        "ci_high": compared["bootstrap_ci_high"],
        "bootstrap_p": compared["bootstrap_p"],
        "dm_stat": compared["dm_stat"],
        "dm_p": compared["dm_p"],
        "interval_excludes_zero": compared["interval_excludes_zero"],
        "mean_of_seed_maes_gap": float(np.mean(both_maes) - np.mean(exog_maes)),
        "per_seed_gap": [a - b for a, b in zip(both_maes, exog_maes)],
        "matches_saved_gap": abs(compared["mae_gap_a_minus_b"] - EXPECTED_GAP) < 1e-9,
        "matches_saved_ci": (
            abs(compared["bootstrap_ci_low"] - EXPECTED_CI[0]) < 1e-9
            and abs(compared["bootstrap_ci_high"] - EXPECTED_CI[1]) < 1e-9
        ),
        "block_24_matches_compare_pair": abs(primary_block["gap"] - compared["mae_gap_a_minus_b"]) < 1e-9,
        "blocks": blocks,
    }
    _json(OUT / "reproduction.json", payload)
    pd.DataFrame(blocks).to_csv(OUT / "block_lengths.csv", index=False)
    return payload


def audit_caches() -> list[dict]:
    rows = []
    sample = pd.Timestamp("2024-06-15 12:00", tz="UTC")
    for delay in (1, 2, 3, 6):
        meta = json.loads((VERIFIED / "chronos" / f"delay_{delay}h.json").read_text())
        frame = pd.read_parquet(
            VERIFIED / "chronos" / f"delay_{delay}h.parquet",
            columns=["latest_input_timestamp"],
        )
        latest = pd.Timestamp(frame.loc[sample, "latest_input_timestamp"])
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        expected = sample - pd.Timedelta(hours=1 + delay)
        rows.append({
            "delay_hours": delay,
            "series": "HB_HUBAVG",
            "horizon": meta["horizon"],
            "step_index": meta["step_index"],
            "rows": meta["rows"],
            "index_end": str(frame.index.max()),
            "horizon_is_one_plus_delay": meta["horizon"] == 1 + delay,
            "step_is_delay": meta["step_index"] == delay,
            "sample_issue": str(sample),
            "latest_input": str(latest),
            "expected_latest": str(expected),
            "latest_matches": latest == expected,
            "sample_target_present": sample in frame.index,
        })
    north = json.loads((VERIFIED / "chronos" / "hb_north_delay_1h.json").read_text())
    north_frame = pd.read_parquet(
        VERIFIED / "chronos" / "hb_north_delay_1h.parquet",
        columns=["latest_input_timestamp"],
    )
    north_latest = pd.Timestamp(north_frame.loc[sample, "latest_input_timestamp"])
    if north_latest.tzinfo is None:
        north_latest = north_latest.tz_localize("UTC")
    rows.append({
        "delay_hours": 1, "series": "HB_NORTH", "horizon": north["horizon"],
        "step_index": north["step_index"], "rows": north["rows"],
        "index_end": str(north_frame.index.max()),
        "horizon_is_one_plus_delay": north["horizon"] == 2,
        "step_is_delay": north["step_index"] == 1,
        "sample_issue": str(sample),
        "latest_input": str(north_latest),
        "expected_latest": str(sample - pd.Timedelta(hours=2)),
        "latest_matches": north_latest == sample - pd.Timedelta(hours=2),
        "sample_target_present": sample in north_frame.index,
    })
    ends = {row["index_end"] for row in rows}
    if len(ends) != 1:
        raise AssertionError(f"Chronos target index ends differ: {ends}")
    if not all(row["latest_matches"] and row["sample_target_present"] for row in rows):
        raise AssertionError("a Chronos cache does not forecast the sample hour from the legal context")
    _json(OUT / "chronos_cache_audit.json", rows)
    return rows


def diagnose_crisis(rt: pd.Series) -> dict:
    hourly = hourly_max_from_intervals(rt)
    exog = pd.read_parquet(VERIFIED / "predictions" / "crisis2021__asinh__xgb_exog__seed42.parquet")
    both = pd.read_parquet(VERIFIED / "predictions" / "crisis2021__asinh__xgb_both__seed42.parquet")
    raw = pd.read_parquet(VERIFIED / "predictions" / "crisis2021__raw__xgb_exog__seed42.parquet")
    if not exog.index.equals(both.index):
        raise AssertionError("crisis models are not on the same hours")
    y_from_raw = hourly.reindex(exog.index)
    persistence = clock_shift(hourly, 2, exog.index)
    crisis = (exog.index >= CRISIS_START) & (exog.index < CRISIS_END)
    outside = ~crisis
    clip = float(np.sinh(ASINH_CLIP))

    def pack(frame, mask):
        y = frame.loc[mask, "y_true"].to_numpy(float)
        pred = frame.loc[mask, "y_pred"].to_numpy(float)
        metrics = enrich(y, pred)
        metrics["pred_max"] = float(pred.max())
        metrics["y_max"] = float(y.max())
        metrics["n_at_clip"] = int(np.sum(np.abs(pred) > clip - 1))
        order = np.argsort(np.abs(pred - y))[::-1][:10]
        metrics["top10_abs_error_share"] = float(np.abs(pred - y)[order].sum() / np.abs(pred - y).sum())
        return metrics

    y = exog["y_true"].to_numpy(float)
    pers = persistence.to_numpy(float)
    payload = {
        "same_index": True,
        "n": int(len(exog)),
        "y_matches_hourly_max_max_abs": float(np.nanmax(np.abs(y_from_raw.to_numpy(float) - y))),
        "persistence_equals_hour_t_minus_2": True,
        "persistence_finite": bool(np.isfinite(pers).all()),
        "asinh_clip_dollars": clip,
        "exog_best_iteration": json.loads(
            (VERIFIED / "predictions" / "crisis2021__asinh__xgb_exog__seed42.json").read_text()
        )["best_iteration"],
        "both_best_iteration": json.loads(
            (VERIFIED / "predictions" / "crisis2021__asinh__xgb_both__seed42.json").read_text()
        )["best_iteration"],
        "crisis": {
            "xgb_exog": pack(exog, crisis),
            "xgb_both": pack(both, crisis),
            "persistence": enrich(y[crisis], pers[crisis]),
            "raw_target_xgb_exog": pack(raw, crisis) if raw.index.equals(exog.index) else {},
        },
        "outside": {
            "xgb_exog": pack(exog, outside),
            "xgb_both": pack(both, outside),
            "persistence": enrich(y[outside], pers[outside]),
        },
        "alignment_ok": float(np.nanmax(np.abs(y_from_raw.to_numpy(float) - y))) < 1e-6,
    }
    # Persistence MAE must use the same crisis mask.
    payload["crisis"]["persistence"]["pred_max"] = float(np.nanmax(pers[crisis]))
    es_start = pd.Timestamp("2021-01-01", tz="UTC") - pd.Timedelta(days=90)
    train_prices = hourly[hourly.index < es_start]
    payload["train_prices"] = {
        "end": str(train_prices.index.max()),
        "max": float(train_prices.max()),
        "max_timestamp": str(train_prices.idxmax()),
        "p99": float(train_prices.quantile(0.99)),
        "hours_above_5000": int((train_prices > 5000).sum()),
        "hours_above_8000": int((train_prices > 8000).sum()),
    }
    payload["note"] = (
        "Persistence predicts hour t with the hourly maximum at t-2, the last hour legal "
        "under a one-hour delay. The trees predict the same hour t on the same index. "
        "Neither asinh forecast reaches the sinh(10) clip."
    )
    _json(OUT / "crisis_diagnostics.json", payload)
    return payload


def fit_lear(hourly: pd.Series, da_prices: pd.DataFrame, load: pd.Series,
             eval_index: pd.DatetimeIndex) -> dict:
    from sklearn.linear_model import Lasso

    index = hourly.index[(hourly.index >= TRAIN_START) & (hourly.index < pd.Timestamp("2025-01-01", tz="UTC"))]
    missing = eval_index.difference(index)
    if len(missing):
        raise AssertionError(f"{len(missing)} evaluation hours are absent from the hourly price series")
    delay = pd.Timedelta(hours=1)
    frame = pd.DataFrame(index=index)
    for lag in list(range(1, 25)) + [48, 168]:
        frame[f"p_lag_{lag}h"] = clock_shift(hourly, lag + 1, index)
    gated = day_ahead_features(hourly, da_prices, price_delay=delay)
    frame["da_HB_HUBAVG"] = gated["da_HB_HUBAVG"].reindex(index)
    frame["load_lag_1h"] = lagged_load(load, index, delay=delay)["load_lag_1h"]
    cal = calendar_central(index)
    dummies = pd.get_dummies(cal[["hour_local", "day_of_week_local", "month_local"]].astype(int).astype(str), dtype=float)
    dummies.columns = [f"cal_{col}" for col in dummies.columns]
    frame = pd.concat([frame, dummies], axis=1)
    frame[TARGET] = hourly.reindex(index)
    sample = pd.Timestamp("2024-06-15 12:00", tz="UTC")
    legal = hourly.loc[sample - pd.Timedelta(hours=2)]
    if not np.isclose(frame.loc[sample, "p_lag_1h"], legal):
        raise AssertionError("the shortest price lag is not hour t-2")
    price_cols = [col for col in frame.columns if col.startswith("p_lag_") or col == "da_HB_HUBAVG"]
    train = frame.index[(frame.index >= TRAIN_START) & (frame.index < VAL_START)]
    val = frame.index[(frame.index >= VAL_START) & (frame.index < TEST_START)]
    test = eval_index
    cols = [col for col in frame.columns if col != TARGET]

    def transform(idx):
        part = frame.loc[idx, cols].copy()
        for col in price_cols:
            part[col] = np.arcsinh(part[col].astype(float))
        return part

    parts = [transform(idx) for idx in (train, val, test)]
    med = parts[0].median(numeric_only=True)
    filled = []
    for part in parts:
        filled.append(part.fillna(med))
    mu = filled[0].mean()
    sigma = filled[0].std().replace(0, 1.0)

    def scale(part):
        return ((part - mu) / sigma).to_numpy(float)

    y_train = np.arcsinh(frame.loc[train, TARGET].to_numpy(float))
    y_val = frame.loc[val, TARGET].to_numpy(float)
    y_test = frame.loc[test, TARGET].to_numpy(float)
    x_train, x_val, x_test = scale(filled[0]), scale(filled[1]), scale(filled[2])

    def validation_mae(alpha):
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        model = Lasso(alpha=float(alpha), max_iter=100000, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train)
        pred = np.sinh(np.clip(model.predict(x_val), -ASINH_CLIP, ASINH_CLIP))
        return {
            "alpha": float(alpha),
            "val_mae": float(np.mean(np.abs(pred - y_val))),
            "converged": not any(issubclass(item.category, ConvergenceWarning) for item in caught),
            "n_iter": int(np.max(model.n_iter_)),
        }

    path = [validation_mae(alpha) for alpha in np.geomspace(1e-6, 10, 15)]
    converged = [row for row in path if row["converged"]]
    if not converged:
        raise AssertionError("no lasso penalty converged on 2023")
    best = min(converged, key=lambda row: row["val_mae"])
    while best["alpha"] <= min(row["alpha"] for row in path) and best["alpha"] > 1e-8:
        smaller = validation_mae(best["alpha"] / 10)
        path.append(smaller)
        converged = [row for row in path if row["converged"]]
        best = min(converged, key=lambda row: row["val_mae"])
    path = sorted(path, key=lambda row: row["alpha"])
    best_alpha, best_mae = best["alpha"], best["val_mae"]
    endpoint = best_alpha in (path[0]["alpha"], path[-1]["alpha"])
    within_cent = [row["alpha"] for row in path if row["converged"] and row["val_mae"] - best_mae <= 0.01]
    model = Lasso(alpha=best_alpha, max_iter=100000, random_state=0)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise AssertionError(f"selected lasso penalty {best_alpha} did not converge")
    pred = np.sinh(np.clip(model.predict(x_test), -ASINH_CLIP, ASINH_CLIP))
    metrics = enrich(y_test, pred)
    persistence = enrich(y_test, clock_shift(hourly, 2, test).to_numpy(float))
    out = pd.DataFrame({"y_true": y_test, "y_pred": pred}, index=test)
    out.to_parquet(OUT / "lear_style_predictions.parquet")
    payload = {
        "name": "latency-valid LEAR-style lasso",
        "not": "Lago, Marcjasz, De Schutter, and Weron LEAR",
        "alpha": best_alpha,
        "val_mae": best_mae,
        "alpha_path": path,
        "alpha_is_grid_endpoint": endpoint,
        "alphas_within_one_cent_val_mae": within_cent,
        "n_features": len(cols),
        "features": "price lags 1-24, 48, 168 under a one-hour delay; gated day-ahead HB_HUBAVG; lagged realized load; Central hour, weekday, and month dummies",
        "missing_versus_full_lear": "no publication-timed zonal load forecast and no renewable forecast",
        "target": "asinh, decoded with sinh clipped at ±10",
        "penalty_selected_on": "2023 MAE only",
        "fit_on": "2019-2022",
        "test_da_finite_fraction": float(frame.loc[test, "da_HB_HUBAVG"].notna().mean()),
        "test_load_finite_fraction": float(frame.loc[test, "load_lag_1h"].notna().mean()),
        "shortest_lag_checked": "2024-06-15 12:00 UTC uses the hourly maximum at 10:00 UTC",
        "metrics": metrics,
        "persistence": persistence,
    }
    _json(OUT / "lear_style.json", payload)
    return payload


def refit_complete_hours(rt: pd.Series, eval_index: pd.DatetimeIndex,
                         y: np.ndarray, both: np.ndarray, exog: np.ndarray) -> dict:
    """Refit both trees on 2024 hours that contain four 15-minute prices.

    Training stays 2019-2022 and early stopping stays on 2023. The frozen
    8,767-hour result is not replaced.
    """
    counts = rt.groupby(pd.DatetimeIndex(rt.index).floor("h")).size()
    keep = eval_index[counts.reindex(eval_index).fillna(0).to_numpy() == 4]
    mask = eval_index.isin(keep)
    saved_both = enrich(y[mask], both[mask])
    saved_exog = enrich(y[mask], exog[mask])
    sources = study_sources(load_sources())
    cache = pd.read_parquet(VERIFIED / "chronos" / "delay_1h.parquet")
    spec = {"name": "delay_1h", "price": 1, "load": 1, "drop_load": False, "role": "sensitivity"}
    frame, _provenance, _hourly = build_spec(
        sources, spec, compute_hash=False, chronos_cache=cache,
    )
    columns = columns_for(frame, False)
    subsets = feature_subsets(columns)
    train = finite_slice(frame, frame.index[(frame.index >= TRAIN_START) & (frame.index < VAL_START)])
    val = finite_slice(frame, frame.index[(frame.index >= VAL_START) & (frame.index < TEST_START)])
    test = finite_slice(frame, keep.intersection(frame.index))
    if len(test) != len(keep):
        raise AssertionError("a complete-four hour is missing from the delayed frame")
    averaged = {}
    seed_rows = []
    pred_dir = OUT / "complete_four_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for model, model_cols in (("xgb_exog", subsets["xgb_exog"]), ("xgb_both", subsets["xgb_both"])):
        preds = []
        y_test = None
        for seed in SEEDS:
            scored, best, device, _usable, _med = fit_predict(
                frame, model_cols, train, val, test, seed, "asinh",
            )
            y_test, pred = scored["test"]
            pd.DataFrame({"y_true": y_test, "y_pred": pred}, index=test).to_parquet(
                pred_dir / f"{model}__seed{seed}.parquet")
            preds.append(pred)
            seed_rows.append({
                "model": model, "seed": int(seed), "best_iteration": int(best),
                "device": device, "mae": float(np.mean(np.abs(pred - y_test))),
            })
            print(f"complete-four {model} seed {seed} iteration {best}", flush=True)
        averaged[model] = (y_test, np.mean(preds, axis=0))
    y_both, pred_both = averaged["xgb_both"]
    y_exog, pred_exog = averaged["xgb_exog"]
    if not np.allclose(y_both, y_exog):
        raise AssertionError("complete-four models do not share a target")
    compared = compare_pair(
        "complete_four_refit", "delay_1h", y_both, pred_both, pred_exog, test, "both", "exog",
    )
    payload = {
        "refit": True,
        "n": int(len(test)),
        "dropped_hours": int(len(eval_index) - len(test)),
        "test_hash": index_hash(test),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "with_chronos": enrich(y_both, pred_both),
        "without_chronos": enrich(y_exog, pred_exog),
        "gap": compared["mae_gap_a_minus_b"],
        "ci_low": compared["bootstrap_ci_low"],
        "ci_high": compared["bootstrap_ci_high"],
        "bootstrap_p": compared["bootstrap_p"],
        "dm_p": compared["dm_p"],
        "interval_excludes_zero": compared["interval_excludes_zero"],
        "saved_predictions_on_same_hours": {
            "with_chronos_mae": saved_both["mae"],
            "without_chronos_mae": saved_exog["mae"],
            "gap": saved_both["mae"] - saved_exog["mae"],
        },
        "max_abs_diff_vs_saved_seed_average": {
            "both": float(np.max(np.abs(pred_both - both[mask]))),
            "exog": float(np.max(np.abs(pred_exog - exog[mask]))),
        },
        "seeds": seed_rows,
        "does_not_replace_primary": True,
    }
    _json(OUT / "complete_four_sensitivity.json", payload)
    return payload


def environment() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "scipy", "torch"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            versions[name] = getattr(mod, "__version__", "?")
        except Exception:
            versions[name] = None
    files = {
        "comparisons": VERIFIED / "comparisons.json",
        "delay_1h_cache": VERIFIED / "chronos" / "delay_1h.parquet",
        "prediction_exog_seed42": VERIFIED / "predictions" / "delay_1h__max__asinh__xgb_exog__seed42.parquet",
        "prediction_both_seed42": VERIFIED / "predictions" / "delay_1h__max__asinh__xgb_both__seed42.parquet",
    }
    payload = {"python": sys.version.split()[0], "versions": versions,
               "sha256_16": {name: _sha256(path) for name, path in files.items()}}
    _json(OUT / "environment.json", payload)
    return payload


def write_reports(recon, repro, caches, crisis, lear, env, sensitivity) -> None:
    hub = json.loads((VERIFIED / "hub_north.json").read_text())
    hourly = json.loads((VERIFIED / "hourly_mean.json").read_text())
    manifest_src = ROOT / "reports" / "feature_availability_manifest.csv"
    manifest = pd.read_csv(manifest_src)
    prior_crisis = json.loads((VERIFIED / "crisis_2021.json").read_text())
    manifest.to_csv(REPORTS / "feature_availability.csv", index=False)
    if lear["alpha_is_grid_endpoint"]:
        penalty_location = "The selected penalty sits on the boundary of the search."
    else:
        penalty_location = "The selected penalty is interior to that search."
    both = repro["with_chronos"]
    exog = repro["without_chronos"]
    exclusions = pd.DataFrame(recon["exclusions"])
    reason_counts = exclusions["reason"].value_counts().to_dict() if len(exclusions) else {}
    incomplete_lines = "\n".join(
        f"| {row['hour_utc']} | {row['hour_central']} | {row['interval_count']} |"
        for row in recon.get("incomplete_evaluation_hours", [])
    )
    if sensitivity.get("refit"):
        saved_same = sensitivity["saved_predictions_on_same_hours"]
        diffs = sensitivity["max_abs_diff_vs_saved_seed_average"]
        if diffs["both"] == 0 and diffs["exog"] == 0:
            diff_sentence = (
                "The largest absolute difference between the refit seed-averaged forecast "
                "and the saved primary forecast, on the shared hours, is zero for both models."
            )
        else:
            diff_sentence = (
                "The largest absolute difference between the refit seed-averaged forecast "
                f"and the saved primary forecast is {_num(diffs['both'], 6)} for the Chronos "
                f"model and {_num(diffs['exog'], 6)} for the exogenous model."
            )
        sensitivity_text = f"""The complete-four-interval index is a strict subset of the frozen index. The frozen index keeps the hour below. Its target is the maximum of the intervals that exist, which is one interval.

| Hour UTC | Hour Central | Intervals |
| --- | --- | ---: |
{incomplete_lines}

Both trees were refit on the other {sensitivity['n']} hours. Depth, learning rate, asinh target, seeds, training window, and 2023 early stopping are unchanged. The refit does not replace the {recon['evaluation_hours']}-hour primary result. {diff_sentence} The dropped hour is the last timestamp in the raw file, so the file ends inside that hour. It is a 2024 evaluation hour, so it was outside the training window.

| | With Chronos | Without Chronos | Gap | 95% interval | Excludes zero |
| --- | ---: | ---: | ---: | --- | --- |
| Refit on complete hours | {_money(sensitivity['with_chronos']['mae'])} | {_money(sensitivity['without_chronos']['mae'])} | {_money(sensitivity['gap'])} | [{_money(sensitivity['ci_low'])}, {_money(sensitivity['ci_high'])}] | {_yesno(sensitivity['interval_excludes_zero'])} |
| Saved primary predictions, same hours | {_money(saved_same['with_chronos_mae'])} | {_money(saved_same['without_chronos_mae'])} | {_money(saved_same['gap'])} |  |  |

On the complete hours the target is the maximum of four 15-minute prices. The headline remains the frozen index.
"""
    else:
        sensitivity_text = (
            "Every hour in the frozen index has four 15-minute prices. "
            "The complete-four index is the same index, so no refit was run."
        )

    _write(REPORTS / "01_experiment_specification.md", f"""# Experiment specification

This is the configuration that was executed, not a target the code was forced to match. The run is `models/run_pit_verified.py` protocol `2026-10-13-pit-verified-v1`. Outputs remain in `runs/pit_verified/`. This audit does not refit those trees.

| Decision | Executed value | Code |
| --- | --- | --- |
| Target | Hourly maximum of HB_HUBAVG 15-minute settlement prices | `src/features/pit_latency.py:hourly_max_from_intervals` |
| Issue time | Start of hour t, UTC | Row index in `models/run_pit_verified.py` |
| Horizon | One hour ahead. The outcome is the maximum inside hour t. | Same |
| Price delay | 1 hour after interval end, so the latest legal completed hour at t is t−2 | `src/features/pit_sets.py:price_history` |
| Load delay | The same one-hour assumption, applied separately | `src/features/pit_sets.py:lagged_load` |
| Day-ahead gate | 14:00 US/Central on the previous local day | `src/data/asof.py` `dam_publication` |
| Storage gate | Friday 18:00 UTC after the week-ending Friday | `configs/feature_availability.yaml` `eia_storage` |
| Calendar | US/Central. Peak is weekday 16:00–22:00 | `src/features/pit_sets.py:calendar_central` |
| Training | 2019-01-01 through 2022 | `TRAIN_START`, `VAL_START` |
| Validation | 2023, early stopping only | `VAL_START`, `TEST_START` |
| Evaluation | Frozen 2024 index, {recon['evaluation_hours']} hours, hash `{recon['test_hash']}` | Saved predictions |
| Chronos | amazon/chronos-t5-small, context 512, 20 paths, horizon 2, step index 1 for the one-hour delay | `models/chronos_delayed_cache.py` |
| Chronos features | pred, q10, q90, mean, p_gt200, spread | `src/features/pit_sets.py` `CHRONOS_STATS` |
| XGBoost | Depth 6, learning rate 0.05, min_child_weight 1, subsample 0.8, colsample 0.8, 2000 trees, early stopping 50, `reg:absoluteerror`, `eval_metric=mae` | `models/improve_hybrid.py:make_model` |
| Seeds | 0, 1, 2, 3, 42 | `SEEDS` |
| Target transform | asinh, decoded with sinh clipped at ±10 | `decode_target` |
| Primary estimand | MAE of the seed-averaged forecast. Gap is with Chronos minus without. | `run_comparisons` |
| Uncertainty | 2000 moving-block draws, seed 0, block 24 hours, plus Diebold–Mariano with lag 24 | `src/evaluation/statistical_tests.py` |
| Spike rule | Price above $200/MWh, fixed before this audit | `models/run_benchmark.py` `SPIKE_PRICE` |

Gas and NRC are not in the primary columns. `conservative_columns` drops them. The feature list is `runs/pit_ablation/matrices/conservative_columns.json` with values rebuilt under the delay.

Earlier hours, back to 2016, enter only as lags, rolling windows, and the 512-hour Chronos context. They are not evaluation hours.
""")

    _write(REPORTS / "02_temporal_validity.md", f"""# Temporal validity

Classification: retrospective, point-in-time-constrained evaluation under assumed publication delays. It is not fully point-in-time verified. First-publication vintages are not in the repository. The source pages are listed in `reports/source_evidence_register.md`, accessed 2026-10-13.

The machine-readable feature table is `reports/final_research_audit/feature_availability.csv`. It is the manifest generated by `src/features/pit_latency.py` for the one-hour price delay and the one-hour load delay. Status counts in that file:

| Status | Features |
| --- | ---: |
{chr(10).join(f"| {name} | {count} |" for name, count in manifest['verification_status'].value_counts().items())}

## Price delay versus observation lag

An observation lag of one hour would use hour t−1 at issue t. The code does not do that when the delay is positive. `price_history` shifts by `lag + delay_hours`. For `hubavg_lag_1h` and a one-hour delay, the value at t is the hourly maximum at t−2. Hour t−1 ends at t, so a one-hour publication delay makes it unavailable. `previous_hour_intervals` uses the four intervals of that same earlier hour. A unit test in `tests/test_pit_latency.py` fails if the interval that ends at t enters the row.

Load uses the same numeric delay, but the source is a different product. NP6-345-CD is a daily archive. Sharing the one-hour assumption with prices is explicit and unverified. The 24-hour load sensitivity and the dropped-load sensitivity are the checks against that assumption. They do not show a Chronos benefit either.

## Other inputs

Day-ahead rows are kept only when `available_at` is at or before the issue time. The gate is 14:00 US/Central on the previous local day, after the documented 13:30 deadline. Daylight-saving time is handled by `US/Central`. Prices can still be corrected before 10:00 on the next business day. Those corrections are not marked.

Storage uses the Friday 18:00 UTC buffer. That is later than Thursday 10:30 a.m. Eastern. The 2019–2024 holiday calendar was not applied hour by hour.

Gas and NRC are not features of the primary model.

Calendar fields are functions of the issue timestamp in US/Central. On the 2024 spring-forward and fall-back days the raw file has {recon['dst_15min_counts'].get('2024-03-10')} and {recon['dst_15min_counts'].get('2024-11-03')} fifteen-minute stamps. UTC hours still have four interval starts. The delay rule does not depend on the local clock.

## Chronos

Each delay has its own cache. The context is 512 clock hours of the latency-filtered hourly maximum, ending at t−(1+delay). The selected forecast step is hour t. Cache checks are in `08_latency_robustness.md`. Shifting the old zero-delay cache is not what this experiment did.
""")

    excl_lines = "\n".join(
        f"| {row['hour_utc']} | {row['hour_central']} | {row['interval_count']} | {row['reason']} |"
        for row in recon["exclusions"]
    )
    _write(REPORTS / "03_target_integrity.md", f"""# Target integrity

UTC 2024 has {recon['utc_2024_hours']} hours. The evaluation index has {recon['evaluation_hours']}. The difference is {recon['absent_from_evaluation']} hours. The raw price file runs from {recon['raw_price_start']} to {recon['raw_price_end']}. Duplicate 15-minute stamps: {recon['duplicate_15min_stamps']}. Internal gaps inside the evaluation index: {recon['internal_gaps_hours']}.

Interval counts inside the evaluation index:

| Intervals in the hour | Hours |
| --- | ---: |
| 4 | {recon['eval_hours_with_4_intervals']} |
| 3 | {recon['eval_hours_with_3_intervals']} |
| 2 | {recon['eval_hours_with_2_intervals']} |
| 1 | {recon['eval_hours_with_1_interval']} |
| missing from the raw file | {recon['eval_hours_missing_from_raw']} |

{sensitivity_text}

Exclusion reasons: {reason_counts}.

| Hour UTC | Hour Central | Intervals | Reason |
| --- | --- | ---: | --- |
{excl_lines}

These hours are the end of the raw file. They were not dropped by a feature filter, by Chronos warm-up, or by a daylight-saving conversion. Recovering them would require prices that are not in `data/raw/ercot_rt_prices.parquet`. They were not fetched.

The frozen index is therefore not a complete UTC year and not a complete Central year. It is every hour from 2024-01-01 00:00 UTC through the last hour present in the price file, 2024-12-31 06:00 UTC. That last stamp is midnight Central at the end of December 31. The first hours of the index are still the evening of December 31, 2023, in Central time. The paper should say that, rather than "calendar year 2024."

Central daylight-saving days in the raw file have {recon['dst_15min_counts'].get('2024-03-10')} fifteen-minute stamps on 2024-03-10 and {recon['dst_15min_counts'].get('2024-11-03')} on 2024-11-03. Those days are short and long in local time. They do not remove UTC hours from this index.
""")

    block_lines = "\n".join(
        f"| {row['block_hours']} | {_money(row['gap'])} | [{_money(row['ci_low'])}, {_money(row['ci_high'])}] | {_yesno(row['interval_excludes_zero'])} | {_num(row['dm_p'])} |"
        for row in repro["blocks"]
    )
    _write(REPORTS / "04_ablation_reproduction.md", f"""# Ablation reproduction

Recomputed from the five saved prediction files for `delay_1h`, without refitting. The estimand is the MAE of the seed-averaged forecast. The average of the seed-level MAEs is reported beside it and is not the primary comparison.

| | With Chronos | Without Chronos |
| --- | ---: | ---: |
| MAE of the averaged forecast | {_money(both['mae'])} | {_money(exog['mae'])} |
| RMSE | {_money(both['rmse'])} | {_money(exog['rmse'])} |
| Median absolute error | {_money(both['medae'])} | {_money(exog['medae'])} |
| Spike recall at $200 | {_num(both['spike_recall'])} | {_num(exog['spike_recall'])} |
| Spike precision | {_num(both['spike_precision'])} | {_num(exog['spike_precision'])} |

Gap, with Chronos minus without: {_money(repro['gap'], 4)}. Interval [{_money(repro['ci_low'])}, {_money(repro['ci_high'])}]. Bootstrap p {_num(repro['bootstrap_p'])}. Diebold–Mariano statistic {_num(repro['dm_stat'], 2)}, p {_num(repro['dm_p'])}. Interval excludes zero: {_yesno(repro['interval_excludes_zero'])}.

Match to the saved comparison file: gap {_yesno(repro['matches_saved_gap'])}, interval {_yesno(repro['matches_saved_ci'])}. Difference of the mean seed MAEs: {_money(repro['mean_of_seed_maes_gap'])}. Per-seed gaps, seeds 0, 1, 2, 3, 42: {', '.join(_money(v) for v in repro['per_seed_gap'])}.

Hours {repro['n']}. Test hash `{repro['test_hash']}`.

## Block length

The prespecified block is 24 hours. The other lengths were computed from the same predictions and the same 2000 draws. None is selected because it favors Chronos.

| Block hours | Gap | 95% interval | Excludes zero | DM p |
| --- | ---: | --- | --- | ---: |
{block_lines}

A 24-hour block does not cover a weekly cycle. Price errors can remain dependent across several days, especially around a heat wave or an outage. The 168-hour row is the check for that. It still covers zero. A longer block widens the interval. It does not create a significant result, and it does not prove the dependence has been fully captured.

The Diebold–Mariano test here uses absolute error and a Bartlett lag equal to the block length, with the Harvey–Leybourne–Newbold correction. A non-significant result is not a proof that the effect is exactly zero.

## Manifest

- Predictions: `runs/pit_verified/predictions/delay_1h__max__asinh__{{xgb_exog,xgb_both}}__seed{{0,1,2,3,42}}.parquet`
- Comparison file this reproduction matches: `runs/pit_verified/comparisons.json`
- Recomputed artifact: `runs/final_audit/reproduction.json`
- Checksums: `runs/final_audit/environment.json`
""")

    _write(REPORTS / "05_benchmark_validity.md", f"""# Benchmark validity

The matched tree without Chronos, MAE {_money(exog['mae'])} on the seed-averaged forecast, is the strong tree benchmark. No second tree family was added. Adding one would change the question.

The historical three-lag lasso at about $13.39 and the honest tree at about $11.56 used a different information set. They are not latency-valid competitors. The earlier one-hour persistence at about $17.04 used hour t−1. Under the one-hour publication delay the last legal hour is t−2. That persistence MAE, recomputed on the frozen index, is {_money(lear['persistence']['mae'])}.

## Latency-valid LEAR-style lasso

This is not the Lago, Marcjasz, De Schutter, and Weron LEAR. The full specification uses day-ahead forecasts of load and renewables that this repository does not have as publication-timed series. The model that was fit uses only information that is already in the primary set:

- Price lags 1 through 24, plus 48 and 168, after the one-hour delay. The shortest lag is hour t−2.
- The publication-gated day-ahead HB_HUBAVG price.
- Realized load lagged by the same one-hour assumption.
- US/Central hour, weekday, and month dummies.

The target is asinh, decoded with sinh clipped at ±10, matching the trees. The penalty is the 2023 MAE minimum among penalties whose fit converged, {lear['alpha']:.1e}, validation MAE {_money(lear['val_mae'])}. Penalties whose 2023 MAE is within one cent of that minimum run from {min(lear['alphas_within_one_cent_val_mae']):.1e} to {max(lear['alphas_within_one_cent_val_mae']):.1e}. {penalty_location} The fit uses 2019–2022 only. 2024 was not used to choose the penalty. Feature count: {lear['n_features']}. On the evaluation hours, the day-ahead price is finite for {_num(100 * lear['test_da_finite_fraction'], 1)} percent of hours and lagged load is finite for {_num(100 * lear['test_load_finite_fraction'], 1)} percent. Missing values use the training median.

| | MAE | RMSE | Median AE | n |
| --- | ---: | ---: | ---: | ---: |
| LEAR-style lasso | {_money(lear['metrics']['mae'])} | {_money(lear['metrics']['rmse'])} | {_money(lear['metrics']['medae'])} | {lear['metrics']['n']} |
| XGBoost without Chronos | {_money(exog['mae'])} | {_money(exog['rmse'])} | {_money(exog['medae'])} | {exog['n']} |
| Latency-valid persistence | {_money(lear['persistence']['mae'])} | {_money(lear['persistence']['rmse'])} | {_money(lear['persistence']['medae'])} | {lear['persistence']['n']} |

Predictions: `runs/final_audit/lear_style_predictions.parquet`.

On MAE, the tree is lower than this linear model by about {_money(lear['metrics']['mae'] - exog['mae'])}. On RMSE, the linear model is lower, {_money(lear['metrics']['rmse'])} against {_money(exog['rmse'])}. Both are more accurate on MAE than latency-valid persistence. That supports competitiveness against a LEAR-style linear model built from the available series. It does not establish competitiveness against a LEAR that also sees a real load forecast and renewable forecasts. The comparison is not a reason to replace the primary Chronos test.
""")

    c_ex = crisis["crisis"]["xgb_exog"]
    c_both = crisis["crisis"]["xgb_both"]
    c_pers = crisis["crisis"]["persistence"]
    o_ex = crisis["outside"]["xgb_exog"]
    o_pers = crisis["outside"]["persistence"]
    _write(REPORTS / "06_crisis_diagnostics.md", f"""# February 2021 diagnostics

The window is 2021-02-10 00:00 through 2021-02-21 00:00 US/Central. The fit uses only hours before 2021, with early stopping on the last 90 days of 2020. No hyperparameter was changed for this audit.

## Alignment

The exogenous and Chronos models share one index of {crisis['n']} scored 2021 hours. The saved target matches the hourly maximum of the raw 15-minute prices. Maximum absolute difference: {crisis['y_matches_hourly_max_max_abs']}. Persistence is the hourly maximum at t−2, and it is finite on every scored hour. It predicts the same hour t. The information cutoff is the same one-hour delay.

The asinh decode is clipped at ±{ _money(crisis['asinh_clip_dollars']) }. Exogenous forecasts at the clip: {c_ex['n_at_clip']}. Chronos-model forecasts at the clip: {c_both['n_at_clip']}. The exogenous maximum of {_money(c_ex['pred_max'])} is far below the clip. The inverse transform is not what caps the forecast near $700.

Early stopping selected iteration {crisis['exog_best_iteration']} for the exogenous tree and {crisis['both_best_iteration']} for the Chronos tree, out of a 2000-tree cap and a patience of 50. The stopping rule ran. It was not a silent one-tree fit. It was also not retuned.

## Same hours, two regimes

| Slice | Model | n | MAE | Pred max | Price max | Share of absolute error in the 10 largest hours |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Crisis | XGBoost without Chronos | {c_ex['n']} | {_money(c_ex['mae'])} | {_money(c_ex['pred_max'])} | {_money(c_ex['y_max'])} | {_num(c_ex['top10_abs_error_share'], 2)} |
| Crisis | XGBoost with Chronos | {c_both['n']} | {_money(c_both['mae'])} | {_money(c_both['pred_max'])} | {_money(c_both['y_max'])} | {_num(c_both['top10_abs_error_share'], 2)} |
| Crisis | Persistence at t−2 | {c_pers['n']} | {_money(c_pers['mae'])} | {_money(c_pers.get('pred_max'))} |  |  |
| Outside | XGBoost without Chronos | {o_ex['n']} | {_money(o_ex['mae'])} | {_money(o_ex['pred_max'])} | {_money(o_ex['y_max'])} | {_num(o_ex['top10_abs_error_share'], 2)} |
| Outside | Persistence at t−2 | {o_pers['n']} | {_money(o_pers['mae'])} |  |  |  |

Persistence is better inside the window because the last legal price is already a storm price. The trees forecast a few hundred dollars while the realized maximum is about $9,052. Outside the window the trees are better than persistence, MAE {_money(o_ex['mae'])} versus {_money(o_pers['mae'])}.

The fitted model's training rows reach {_money(prior_crisis['train_max_price'])} at {prior_crisis['train_max_timestamp']}, and their 99th percentile is {_money(prior_crisis['train_p99_price'])}. On the hourly price series before the same early-stopping cutoff, {crisis['train_prices']['hours_above_5000']} hours exceed $5,000 and {crisis['train_prices']['hours_above_8000']} exceed $8,000, including 13 and 15 August 2019. The series had already reached this price level. It had not contained an eleven-day stretch like February 2021. The failure is consistent with a tree that does not carry a rare spike forward, not with a misaligned target or a clipped inverse transform. A raw-target exogenous fit on the same split has crisis MAE {_money(crisis['crisis']['raw_target_xgb_exog']['mae'])} and maximum prediction {_money(crisis['crisis']['raw_target_xgb_exog']['pred_max'])}, so changing only the target transform does not repair the window.

No remedy was fit. Changing depth, loss, or the training window to chase February 2021 would be a different study.
""")

    _write(REPORTS / "07_secondary_analyses.md", f"""# Secondary analyses

The only primary test is the hub-average, hourly-maximum, one-hour-delay, seed-averaged MAE gap, with the 24-hour block bootstrap. Its interval covers zero.

## HB_NORTH

The cache file is `runs/pit_verified/chronos/hb_north_delay_1h.parquet`, built from HB_NORTH prices, horizon 2, step index 1. It is not the HB_HUBAVG cache. The price lags in that fit are HB_NORTH hourly maxima under the same one-hour delay. The protocol is the same five seeds and the same hyperparameters. The paired gap is {_money(hub['mae_gap_a_minus_b'])}, interval [{_money(hub['bootstrap_ci_low'])}, {_money(hub['bootstrap_ci_high'])}]. The interval excludes zero: {_yesno(hub['interval_excludes_zero'])}. MAE with Chronos {_money(hub['mae_a'])}, without {_money(hub['mae_b'])}.

HB_NORTH is another ERCOT hub. This is not evidence about another market. The interval is a nominal 95 percent interval for one secondary comparison.

## Hourly mean

Same one-hour-delay features. The outcome is the mean of the four 15-minute prices. Chronos features remain forecasts of the hourly maximum. The gap is {_money(hourly['mae_gap_a_minus_b'])}, interval [{_money(hourly['bootstrap_ci_low'])}, {_money(hourly['bootstrap_ci_high'])}]. The interval excludes zero: {_yesno(hourly['interval_excludes_zero'])}. This does not replace the hourly-maximum headline.

## What else was looked at

Prespecified alongside the primary test, and not promoted when the interval excludes zero:

- Price and load delays of 2, 3, and 6 hours.
- A 24-hour load lag and a dropped-load case, both at the one-hour price delay.
- Block lengths 12, 24, 48, and 168 hours.
- Six realized-price bins.
- Spike precision, recall, and F1.
- The February 2021 window.
- The LEAR-style lasso.
- HB_NORTH and the hourly mean.

A secondary interval that excludes zero is not a correction of the primary test. The primary interval covers zero. Selecting the hourly-mean or HB_NORTH result because it excludes zero would be a selection effect. Neither is the paper's main claim.
""")

    cache_lines = "\n".join(
        f"| {row.get('series', 'HB_HUBAVG')} | {row['delay_hours']} | {row['horizon']} | {row['step_index']} | {_yesno(row['horizon_is_one_plus_delay'])} | {_yesno(row.get('latest_matches', False))} |"
        for row in caches
    )
    _write(REPORTS / "08_latency_robustness.md", f"""# Latency robustness

Each delay was fit as its own pair of trees on its own Chronos cache. The target hour does not move. At issue t the model still predicts the maximum inside hour t. Only the latest legal history changes. For delay D the latest legal hour is t−(1+D), and the cache horizon is 1+D so that the selected step is hour t.

| Series | Delay | Horizon | Step | Horizon is 1+delay | Sample latest input matches |
| --- | ---: | ---: | ---: | --- | --- |
{cache_lines}

The sample issue time for the hub-average rows is 2024-06-15 12:00 UTC. A match means the stored latest input is exactly t−(1+delay).

The saved gaps, MAE of the seed-averaged forecast, are unchanged by this audit because the models were not refit:

| Delay | With Chronos | Without | Gap | 95% interval | Excludes zero |
| --- | ---: | ---: | ---: | --- | --- |
| 1 hour | $14.92 | $15.07 | −$0.15 | [−$0.32, $0.06] | no |
| 2 hours | $16.23 | $16.39 | −$0.17 | [−$0.34, $0.04] | no |
| 3 hours | $16.55 | $16.65 | −$0.10 | [−$0.32, $0.20] | no |
| 6 hours | $17.03 | $17.12 | −$0.09 | [−$0.18, $0.01] | no |
| 1 hour price, 24 hour load | $15.13 | $15.24 | −$0.11 | [−$0.34, $0.24] | no |
| 1 hour price, load dropped | $14.79 | $14.97 | −$0.18 | [−$0.44, $0.18] | no |

Within each row the two models see the same delay. No row's interval lies entirely below zero. The one-hour row remains the primary result because it was the shortest pre-specified assumption, not because its gap is the largest.
""")

    _write(REPORTS / "09_reproducibility.md", f"""# Reproducibility

Three layers stay separate.

1. Original capstone artifacts, including `artifacts/honest/` and `docs/static/pdfs/paper.pdf`. Not rewritten here.
2. The point-in-time ablation in `runs/pit_ablation/`, which includes later-year scores that this study does not use.
3. The latency-constrained experiment in `runs/pit_verified/`. This is the definitive run. The audit in `runs/final_audit/` reads it and adds the LEAR-style fit, the block-length recomputation, and the target reconciliation.

## Commands

From `Ercot_Forecaster/ercot-grid-forecaster`, with `/net/home/rfozdar/venvs/ercot/bin/python`:

    python models/run_pit_verified.py all

That command rebuilds Chronos caches only when the cache metadata is missing, then fits the trees. It was already run. Do not point it at a new output directory after the fact. The caches are large.

    python models/run_final_audit.py

That command does not refit the primary trees. It recomputes the primary metrics, checks the 2024 index, checks the Chronos caches, refits the complete-four sensitivity because one 2024 hour is incomplete, refits the LEAR-style lasso, and writes this report directory.

Tests that guard the delay:

    python -m pytest tests/test_pit_latency.py tests/test_pit_sets.py tests/test_feature_availability.py

## Environment

Python {env['python']}. Package versions: {json.dumps(env['versions'])}.

SHA-256 prefixes of the files this reproduction reads:

| File | Prefix |
| --- | --- |
| comparisons.json | `{env['sha256_16']['comparisons']}` |
| delay_1h Chronos cache | `{env['sha256_16']['delay_1h_cache']}` |
| exogenous seed 42 predictions | `{env['sha256_16']['prediction_exog_seed42']}` |
| Chronos seed 42 predictions | `{env['sha256_16']['prediction_both_seed42']}` |

## What a table in the paper should cite

| Paper number | Source |
| --- | --- |
| Primary MAE, gap, and 24-hour interval | `runs/pit_verified/comparisons.json`, name `delay_1h__mean_seed__both_minus_exog`, confirmed in `runs/final_audit/reproduction.json` |
| Block lengths including 12 hours | `runs/final_audit/block_lengths.csv` |
| Latency table | `runs/pit_verified/comparisons.json` |
| February 2021 | `runs/pit_verified/crisis_2021.json`, alignment check in `runs/final_audit/crisis_diagnostics.json` |
| LEAR-style MAE | `runs/final_audit/lear_style.json` |
| HB_NORTH and hourly mean | `runs/pit_verified/hub_north.json`, `hourly_mean.json` |
| Feature availability | `reports/final_research_audit/feature_availability.csv` |

Do not cite `runs/pit_ablation/later_years_metrics.csv` or any 2025 or 2026 row.
""")

    _write(REPORTS / "10_claims_register.md", f"""# Claims register

The primary configuration is the one-hour delay, HB_HUBAVG hourly maximum, seed-averaged forecasts, 2024 frozen index, 24-hour block bootstrap.

| Claim | Result | Role | Supported? | Wording that stays inside the evidence |
| --- | --- | --- | --- | --- |
| A. Chronos lowers absolute error | Gap {_money(repro['gap'])}, interval [{_money(repro['ci_low'])}, {_money(repro['ci_high'])}], p {_num(repro['bootstrap_p'])} | Primary | Partially. The point estimate is negative. The interval covers zero. | Chronos statistics change 2024 MAE by about −$0.15/MWh. The prespecified interval includes zero. |
| B. The evaluation is point-in-time verified | No first-publication timestamps. Prices are the weekly archive NP6-785-ER. | Limitation | Unsupported as a full verification. | The study is a retrospective evaluation under an assumed one-hour delay. |
| C. The model is competitive with established benchmarks | Tree MAE {_money(exog['mae'])} versus LEAR-style MAE {_money(lear['metrics']['mae'])} and persistence MAE {_money(lear['persistence']['mae'])}. LEAR-style RMSE {_money(lear['metrics']['rmse'])} is lower than the tree RMSE {_money(exog['rmse'])}. | Secondary | Partially, against the benchmarks that can be built from this archive. | On MAE the tree is below a latency-valid LEAR-style lasso and below latency-valid persistence. On RMSE the lasso is lower. It has not been compared with a LEAR that uses load and renewable forecasts. |
| D. Chronos improves spike forecasting | Recall {_num(both['spike_recall'])} versus {_num(exog['spike_recall'])}. Precision {_num(both['spike_precision'])} versus {_num(exog['spike_precision'])}. | Descriptive | Unsupported as a separate inferential claim. | Spike counts move in the same direction as the MAE gap. No prespecified spike test excludes zero as a primary result. |
| E. The result generalizes across markets | HB_NORTH gap {_money(hub['mae_gap_a_minus_b'])}, interval [{_money(hub['bootstrap_ci_low'])}, {_money(hub['bootstrap_ci_high'])}] | Secondary | Unsupported. | HB_NORTH is another ERCOT hub. |
| F. The model is robust in extreme regimes | Crisis MAE {_money(c_ex['mae'])} versus persistence {_money(c_pers['mae'])}. Forecast max {_money(c_ex['pred_max'])} versus price {_money(c_ex['y_max'])}. | Diagnostic | Unsupported. The evidence goes the other way. | During 10–20 February 2021 the trees miss the price level that persistence, using the last legal hour, already reflects. |

The two MAE estimands must not be swapped. The primary gap is {_money(repro['gap'])}. The difference of the five seed-level MAEs is {_money(repro['mean_of_seed_maes_gap'])}.
""")

    _write(REPORTS / "FINAL_RESEARCH_READINESS.md", f"""# Final research readiness

Recommendation: ready with explicitly documented limitations.

The primary numbers reproduced from the saved predictions. The 17-hour gap is the end of the raw file, not a silent row filter. One retained hour has a single 15-minute price. Refitting both trees without that hour does not move the gap's interval off zero, and it does not replace the 8,767-hour result. The February 2021 gap is a real difference between persistence and the trees on the same hours, not an alignment or clipping error. No publication-time archive was found, so the study cannot be called fully point-in-time verified.

## Issues resolved

- The zero-delay label on real-time prices was withdrawn. The executed model uses an explicit one-hour delay, and the lag at t is hour t−2. Code: `src/features/pit_sets.py`. Tests: `tests/test_pit_latency.py`.
- Chronos was rebuilt per delay from that legal history. Cache metadata matches horizon 1+delay, and a sample row's latest input matches t−(1+delay).
- The 8,767 versus 8,784 difference is enumerated in `03_target_integrity.md`. All 17 absent hours are after the last raw timestamp. One retained hour has a single 15-minute price. Both trees were refit on the other {sensitivity['n']} hours. That sensitivity does not replace the primary result. Gap {_money(sensitivity['gap'])}, interval [{_money(sensitivity['ci_low'])}, {_money(sensitivity['ci_high'])}].
- The primary MAE, interval, and Diebold–Mariano statistic were recomputed from the prediction files and match `runs/pit_verified/comparisons.json`.
- A 12-hour block was added. It does not exclude zero.
- The LEAR-style lasso is fit on the legal lags and is not labeled as Lago et al.
- February 2021 uses the same target and the same t−2 persistence cutoff. The asinh clip does not bind.

## Issues remaining

- No first-publication timestamps for NP6-905-CD or for the load display during 2019–2024.
- The weekly price archive may be a revised settlement value. A delay does not undo that.
- Load's one-hour delay is an assumption about a daily archive.
- Day-ahead corrections before 10:00 on the next business day are not marked.
- The storage holiday calendar for 2019–2024 was not applied row by row.
- A full LEAR with load and renewable forecasts cannot be built from the publication-timed series on disk.
- The last 17 UTC hours of 2024 are not in the raw file.

## Results changed

| Quantity | Earlier conservative ablation, delay 0 | This study, delay 1 hour |
| --- | ---: | ---: |
| MAE with Chronos | $11.33 | {_money(both['mae'])} |
| MAE without Chronos | $11.42 | {_money(exog['mae'])} |
| Gap | −$0.09 | {_money(repro['gap'])} |
| 95% interval | [−$0.27, $0.11] | [{_money(repro['ci_low'])}, {_money(repro['ci_high'])}] |

The level rose because the most recent hour of prices is no longer a feature, and because Chronos was redrawn from that shorter history. The conclusion did not change. Both intervals cover zero. The earlier $11.56 tree and $13.39 three-lag model are a third information set and are not in this table.

## Publication blockers

- Do not write that Chronos significantly improves the forecast.
- Do not write that the study is fully point-in-time verified.
- Do not write that the model is robust in February 2021.
- Do not write that HB_NORTH is another market or that the result generalizes across markets.
- Do not put 2025 or 2026 numbers in the paper.
- Do not call the three-lag historical model, or the new lasso, a full LEAR.

## Nonblocking limitations

- The evaluation index ends at 2024-12-31 06:00 UTC. Say so.
- The one-hour price delay and the one-hour load delay are assumptions. The latency table shows the gap still covers zero at 2, 3, and 6 hours.
- Spike precision and recall are descriptive.
- The hourly-mean interval excludes zero and stays a secondary target.
- The HB_NORTH interval excludes zero and stays a second hub in the same market.

## Definitive results for the paper

- With Chronos, MAE {_money(both['mae'])}, RMSE {_money(both['rmse'])}, median AE {_money(both['medae'])}, spike recall {_num(both['spike_recall'])}, spike precision {_num(both['spike_precision'])}.
- Without Chronos, MAE {_money(exog['mae'])}, RMSE {_money(exog['rmse'])}, median AE {_money(exog['medae'])}, spike recall {_num(exog['spike_recall'])}, spike precision {_num(exog['spike_precision'])}.
- Gap {_money(repro['gap'])}, 95 percent interval [{_money(repro['ci_low'])}, {_money(repro['ci_high'])}], bootstrap p {_num(repro['bootstrap_p'])}, Diebold–Mariano p {_num(repro['dm_p'])}.
- Hours: {repro['n']}. Hash `{repro['test_hash']}`.

## Contribution

Under an explicit one-hour publication delay, frozen Chronos-T5-small forecasts of the ERCOT hub-average hourly maximum change a matched XGBoost model's 2024 MAE by about −$0.15/MWh. The prespecified uncertainty interval includes zero. The same trees miss the February 2021 price level that a latency-valid persistence forecast already reflects. The study's contribution is that measurement, not a claim that the foundation-model features improve the forecast.

## Reproduction

See `09_reproducibility.md`. Definitive command: `python models/run_pit_verified.py all`. Audit command: `python models/run_final_audit.py`.
""")
    print("audit reports written", flush=True)


def reports_only() -> None:
    recon = json.loads((OUT / "target_reconciliation.json").read_text())
    repro = json.loads((OUT / "reproduction.json").read_text())
    caches = json.loads((OUT / "chronos_cache_audit.json").read_text())
    crisis = json.loads((OUT / "crisis_diagnostics.json").read_text())
    lear = json.loads((OUT / "lear_style.json").read_text())
    env = json.loads((OUT / "environment.json").read_text())
    sensitivity = json.loads((OUT / "complete_four_sensitivity.json").read_text())
    write_reports(recon, repro, caches, crisis, lear, env, sensitivity)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "reports":
        reports_only()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    index, y, both, both_maes = _load_pair("xgb_both")
    _, y2, exog, exog_maes = _load_pair("xgb_exog")
    if not np.allclose(y, y2):
        raise AssertionError("the two models do not share a target")
    rt = pd.read_parquet(ROOT / "data" / "raw" / "ercot_rt_prices.parquet", columns=[TARGET])[TARGET]
    if rt.index.max() >= pd.Timestamp("2025-01-01", tz="UTC"):
        raise AssertionError("raw prices enter 2025")
    recon = reconcile(rt, index)
    if recon["complete_four_equals_frozen_index"]:
        sensitivity = {"refit": False, "n": int(len(index))}
        _json(OUT / "complete_four_sensitivity.json", sensitivity)
    else:
        sensitivity = refit_complete_hours(rt, index, y, both, exog)
    repro = reproduce(index, y, both, exog, both_maes, exog_maes)
    caches = audit_caches()
    crisis = diagnose_crisis(rt)
    if not crisis["alignment_ok"]:
        raise AssertionError("crisis target does not match the hourly maximum")
    cutoff = pd.Timestamp("2025-01-01", tz="UTC")
    da = pd.read_parquet(ROOT / "data" / "raw" / "ercot_da_prices.parquet")
    load = pd.read_parquet(ROOT / "data" / "raw" / "ercot_load.parquet")["load_actual_mw"]
    da = da.loc[da.index < cutoff]
    load = load.loc[load.index < cutoff]
    hourly = hourly_max_from_intervals(rt)
    lear = fit_lear(hourly, da, load, index)
    env = environment()
    write_reports(recon, repro, caches, crisis, lear, env, sensitivity)


if __name__ == "__main__":
    main()
