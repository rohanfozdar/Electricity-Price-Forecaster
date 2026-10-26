"""Point-in-time Chronos ablation for one-hour-ahead HB_HUBAVG.

The 2024 index is scored because the comparison was specified before this run.
It is not an untouched test year. No setting in this script is chosen from 2024 error.

Run from the repository root:

    python models/run_pit_ablation.py all

Outputs land in runs/pit_ablation/. The write-up is reports/pit_ablation.md.
Saved honest-tree artifacts are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import file_hash  # noqa: E402
from src.evaluation.statistical_tests import diebold_mariano, paired_bootstrap  # noqa: E402
from src.features.pit_sets import (  # noqa: E402
    CHRONOS_STATS,
    apply_asinh,
    assemble_assumed_realtime,
    attach_split_flags,
    catalog_rows,
    conservative_columns,
    feature_subsets,
    gfs_target_hour,
    hourly_interval_counts,
    reported_column_note,
    signed_asinh_columns,
    train_median_impute,
)
from models.run_benchmark import (  # noqa: E402
    SPIKE_PRICE,
    STRESS_PRICE,
    index_hash,
    regression_metrics,
)

PROTOCOL = "2026-10-12-pit-v1"
OUT = ROOT / "runs" / "pit_ablation"
SEEDS = (0, 1, 2, 3, 42)
PRIMARY_SEED = 42
ASINH_CLIP = 10.0
BOOT_DRAWS = 2000
BOOT_BLOCK = 24
FROZEN_DEPTH = 6
FROZEN_LR = 0.05
TRAIN_START = pd.Timestamp("2019-01-01", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
TARGET = "HB_HUBAVG"


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))


def _read_json(path: Path):
    return json.loads(path.read_text())


def provenance() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception:
            return None

    versions = {}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "scipy"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            versions[name] = getattr(mod, "__version__", "?")
        except Exception:
            versions[name] = None
    gpu = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        gpu = None
    return {
        "protocol": PROTOCOL,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.version.split()[0],
        "versions": versions,
        "gpu": gpu,
        "xgb_device": os.environ.get("ERCOT_XGB_DEVICE", "cuda"),
        "command": "python models/run_pit_ablation.py all",
    }


def slice_index(index: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    mask = np.ones(len(index), dtype=bool)
    if start is not None:
        mask &= index >= start
    if end is not None:
        mask &= index < end
    return index[mask]


def encode_target(y: np.ndarray) -> np.ndarray:
    return np.arcsinh(np.asarray(y, dtype=float))


def decode_target(pred: np.ndarray) -> np.ndarray:
    return np.sinh(np.clip(np.asarray(pred, dtype=float), -ASINH_CLIP, ASINH_CLIP))


def fit_xgb(X_train, y_train, X_eval, y_eval, seed: int, depth: int, lr: float):
    import models.improve_hybrid as hybrid

    cfg = {"learner": "xgb", "objective": "abs", "depth": depth, "lr": lr, "mcw": 1}
    device = os.environ.get("ERCOT_XGB_DEVICE", "cuda")
    # make_model reads this module global, which was set at import time.
    hybrid.XGB_DEVICE = device

    def _fit():
        model = hybrid.make_model(cfg, seed)
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
        return model, int(model.best_iteration)

    try:
        model, best = _fit()
        return model, best, device
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"  cuda fit failed ({type(exc).__name__}: {exc}); switching to cpu", flush=True)
        os.environ["ERCOT_XGB_DEVICE"] = "cpu"
        hybrid.XGB_DEVICE = "cpu"
        model, best = _fit()
        return model, best, "cpu"


def predict_xgb(model, X, best: int) -> np.ndarray:
    raw = model.predict(X, iteration_range=(0, best + 1))
    return decode_target(raw)


def design_matrices(frame: pd.DataFrame, columns: list[str], train_idx, eval_idx, test_idx, kind: str):
    """Return imputed train/eval/test frames and the columns actually used.

    `reported` matches the honest tree: forward-fill inside each split, then zero.
    `pit` applies asinh to signed price columns and fills with the training median.
    The gas-spike flag, when requested, is fit on `train_idx` only.
    """
    work = frame
    p90 = None
    if kind == "pit":
        work, p90 = attach_split_flags(frame.copy(), train_idx)
    cols = [c for c in columns if c in work.columns]
    missing = [c for c in columns if c not in work.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    keep = cols + [TARGET]
    parts = [work.loc[ix, keep].copy() for ix in (train_idx, eval_idx, test_idx)]
    if kind == "reported":
        filled = [part[cols].ffill().fillna(0.0) for part in parts]
        return filled, cols, p90
    transformed = [apply_asinh(part, signed_asinh_columns(cols)) for part in parts]
    usable = [c for c in cols if transformed[0][c].notna().any()]
    filled, _med = train_median_impute(transformed[0], transformed, usable)
    filled = [part[usable] for part in filled]
    return filled, usable, p90


def score_pair(y, pred) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise AssertionError("non-finite target or prediction")
    metrics = regression_metrics(y, pred)
    metrics["medae"] = float(np.median(np.abs(pred - y)))
    neg = y < 0
    metrics["n_negative"] = int(neg.sum())
    metrics["negative_mae"] = float(np.mean(np.abs(pred[neg] - y[neg]))) if neg.any() else None
    return metrics


def prediction_frame(index, y, pred, split: str) -> pd.DataFrame:
    return pd.DataFrame({"split": split, "y_true": np.asarray(y, float), "y_pred": np.asarray(pred, float)}, index=index)


def run_key(info: str, model: str, seed) -> str:
    seed_bit = "na" if seed is None else str(seed)
    return f"{info}__{model}__seed{seed_bit}"


def pred_path(key: str) -> Path:
    return OUT / "predictions" / f"{key}.parquet"


def meta_path(key: str) -> Path:
    return OUT / "predictions" / f"{key}.json"


def save_prediction(key: str, frame: pd.DataFrame, meta: dict) -> None:
    pred_path(key).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(pred_path(key))
    meta = {"protocol": PROTOCOL, **meta}
    _write_json(meta_path(key), meta)


def cached(key: str) -> pd.DataFrame | None:
    path, meta = pred_path(key), meta_path(key)
    if not path.exists() or not meta.exists():
        return None
    if _read_json(meta).get("protocol") != PROTOCOL:
        return None
    return pd.read_parquet(path)


def upsert_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    frame = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size:
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame.drop_duplicates(keys, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def append_metrics(rows: list[dict]) -> None:
    path = OUT / "ablation_metrics.csv"
    frame = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame.drop_duplicates(["info_set", "model", "seed", "split", "role"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def metric_row(info, model, seed, split, role, metrics, extra=None) -> dict:
    row = {
        "info_set": info, "model": model, "seed": seed, "split": split, "role": role,
        "mae": metrics["mae"], "rmse": metrics["rmse"], "medae": metrics.get("medae"),
        "spike_recall": metrics["spike_recall"], "spike_precision": metrics["spike_precision"],
        "spike_f1": metrics["spike_f1"], "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
        "n": metrics["n"], "n_negative": metrics.get("n_negative"),
        "normal_mae": metrics.get("normal_mae"), "stress_mae": metrics.get("stress_mae"),
        "negative_mae": metrics.get("negative_mae"),
    }
    if extra:
        row.update(extra)
    return row


def contiguous_sizes(index: pd.DatetimeIndex) -> list[int]:
    if len(index) == 0:
        return []
    gaps = index.to_series().diff().gt(pd.Timedelta(hours=1, minutes=30)).to_numpy()
    sizes, run = [], 1
    for gap in gaps[1:]:
        if gap:
            sizes.append(run)
            run = 1
        else:
            run += 1
    sizes.append(run)
    return sizes


def load_sources() -> dict:
    raw = ROOT / "data" / "raw"
    matrix = pd.read_parquet(ROOT / "data" / "features" / "feature_matrix_engineered_v2.parquet")
    chronos = pd.read_parquet(ROOT / "data" / "features" / "chronos_hourly" / "t5-small_gap-aware_ctx512.parquet")
    rt = pd.read_parquet(raw / "ercot_rt_prices.parquet")
    da = pd.read_parquet(raw / "ercot_da_prices.parquet")
    load = pd.read_parquet(raw / "ercot_load.parquet")["load_actual_mw"]
    weather = pd.read_parquet(raw / "weather.parquet")
    gas = pd.read_parquet(raw / "eia_gas.parquet")
    storage = pd.read_parquet(raw / "eia_storage.parquet")
    nrc = pd.read_parquet(raw / "nrc_reactors.parquet")
    gfs_path = raw / "weather_forecast_gfs.parquet"
    gfs = pd.read_parquet(gfs_path) if gfs_path.exists() else None
    return {
        "matrix": matrix, "chronos": chronos, "rt": rt, "da": da, "load": load,
        "weather": weather, "gas": gas, "storage": storage, "nrc": nrc, "gfs": gfs,
    }


def data_audit(sources: dict) -> dict:
    matrix = sources["matrix"]
    rt = sources["rt"][TARGET]
    counts = hourly_interval_counts(rt.index)
    aligned = counts.reindex(matrix.index)
    local = rt.index.tz_convert("US/Central")
    dst = {}
    for day in ("2024-03-10", "2024-11-03"):
        stamp = pd.Timestamp(day, tz="US/Central")
        dst[day] = int((local.floor("D") == stamp).sum())
    test = matrix.index[matrix.index >= TEST_START]
    gaps = int(test.to_series().diff().gt(pd.Timedelta(hours=1, minutes=30)).sum())
    payload = {
        "matrix_rows": int(len(matrix)),
        "matrix_start": str(matrix.index.min()),
        "matrix_end": str(matrix.index.max()),
        "test_rows": int(len(test)),
        "test_hash": index_hash(test),
        "test_internal_gaps": gaps,
        "rt_start": str(rt.index.min()),
        "rt_end": str(rt.index.max()),
        "hours_in_matrix_with_not_4_intervals": int((aligned.fillna(0) != 4).sum()),
        "hours_in_matrix_missing_from_rt_counts": int(aligned.isna().sum()),
        "dst_15min_counts_central_day": dst,
        "note": (
            "Incomplete UTC hours are counted and kept. The frozen index is not rebuilt. "
            "A Central spring-forward day has 92 fifteen-minute intervals and a fall-back day has 100 "
            "when the archive is complete; UTC hours still contain four interval starts."
        ),
    }
    _write_json(OUT / "data_audit.json", payload)
    return payload


def chronos_context_check(matrix: pd.DataFrame, cache: pd.DataFrame) -> dict:
    from models.chronos_hourly_features import build_contexts, context_bounds

    series = matrix[TARGET].dropna().sort_index()
    hours, _values, _positions = build_contexts(series, "gap-aware", 512)
    hours = pd.DatetimeIndex(hours)
    grid = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC"))
    test = matrix.index[matrix.index >= TEST_START]
    missing = bad = 0
    for ts in test:
        if ts not in cache.index:
            missing += 1
            continue
        pos = grid.index.get_loc(ts)
        if isinstance(pos, slice) or not isinstance(pos, (int, np.integer)):
            bad += 1
            continue
        start, end = context_bounds(int(pos), 512, len(grid))
        if end != int(pos) or grid.index[end - 1] != ts - pd.Timedelta(hours=1) or start != int(pos) - 512:
            bad += 1
    payload = {
        "cache_rows": int(len(cache)),
        "builder_hours": int(len(hours)),
        "cache_equals_builder_index": bool(cache.index.equals(hours)),
        "test_hours": int(len(test)),
        "test_hours_missing_from_cache": int(missing),
        "test_hours_with_bad_bounds": int(bad),
        "context": 512,
        "mode": "gap-aware",
        "checkpoint": "amazon/chronos-t5-small",
        "note": "Bounds are checked on the cached index. Forecasts were not recomputed.",
    }
    if bad or missing:
        raise AssertionError(f"Chronos context check failed: {payload}")
    _write_json(OUT / "chronos_context_check.json", payload)
    return payload


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "provenance.json", provenance())
    sources = load_sources()
    audit = data_audit(sources)
    context = chronos_context_check(sources["matrix"], sources["chronos"])
    index = sources["matrix"].index
    base = assemble_assumed_realtime(
        index,
        sources["matrix"][TARGET],
        sources["rt"][TARGET],
        sources["da"],
        sources["load"],
        sources["weather"],
        sources["gas"],
        sources["storage"],
        sources["nrc"],
        sources["chronos"],
    )
    if "load_actual_mw" in base.columns:
        raise AssertionError("same-hour load is in the assumed-realtime matrix")
    matrix_dir = OUT / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    base.to_parquet(matrix_dir / "assumed_base.parquet")

    from models.honest_mae import make_honest

    reported, reported_cols, _p90, _val, test_index = make_honest()
    reported_out = reported[reported_cols + [TARGET]].copy()
    reported_out.to_parquet(matrix_dir / "reported.parquet")
    _write_json(matrix_dir / "reported_columns.json", reported_cols)

    assumed_cols = [c for c in base.columns if c != TARGET] + [
        "stress_gas_spike", "stress_reactor_outage", "stress_score",
    ]
    cons_cols = conservative_columns(assumed_cols)
    _write_json(matrix_dir / "assumed_columns.json", assumed_cols)
    _write_json(matrix_dir / "conservative_columns.json", cons_cols)

    catalog = pd.concat([
        catalog_rows(assumed_cols, "assumed_realtime"),
        catalog_rows(cons_cols, "conservative"),
        pd.DataFrame([
            {"information_set": "reported", "feature": name, **reported_column_note(name)}
            for name in reported_cols
        ]),
    ], ignore_index=True)
    catalog.to_csv(OUT / "feature_catalog.csv", index=False)

    hashes = {}
    for label, path in {
        "matrix": ROOT / "data/features/feature_matrix_engineered_v2.parquet",
        "chronos": ROOT / "data/features/chronos_hourly/t5-small_gap-aware_ctx512.parquet",
        "rt": ROOT / "data/raw/ercot_rt_prices.parquet",
        "da": ROOT / "data/raw/ercot_da_prices.parquet",
        "load": ROOT / "data/raw/ercot_load.parquet",
        "weather": ROOT / "data/raw/weather.parquet",
        "gas": ROOT / "data/raw/eia_gas.parquet",
        "storage": ROOT / "data/raw/eia_storage.parquet",
        "nrc": ROOT / "data/raw/nrc_reactors.parquet",
        "gfs": ROOT / "data/raw/weather_forecast_gfs.parquet",
    }.items():
        hashes[label] = file_hash(path, 16) if path.exists() else None
    manifest = {
        "protocol": PROTOCOL,
        "test_hash": audit["test_hash"],
        "reported_test_hash": index_hash(test_index),
        "hashes": hashes,
        "n_assumed_columns": len(assumed_cols),
        "n_conservative_columns": len(cons_cols),
        "n_reported_columns": len(reported_cols),
        "chronos_context": context,
        "data_audit": audit,
    }
    if manifest["test_hash"] != manifest["reported_test_hash"]:
        raise AssertionError("assumed matrix test index does not match the honest-tree test index")
    _write_json(OUT / "build_manifest.json", manifest)
    print(
        f"  built matrices: reported {len(reported_cols)} cols, "
        f"assumed {len(assumed_cols)} cols, conservative {len(cons_cols)} cols",
        flush=True,
    )
    return {"base": base, "reported": reported_out, "reported_cols": reported_cols,
            "assumed_cols": assumed_cols, "cons_cols": cons_cols, "sources": sources,
            "test_index": test_index}


def primary_splits(index: pd.DatetimeIndex):
    train = slice_index(index, TRAIN_START, VAL_START)
    eval_idx = slice_index(index, VAL_START, TEST_START)
    test = slice_index(index, TEST_START, None)
    if min(len(train), len(eval_idx), len(test)) < 500:
        raise AssertionError("primary split is too small")
    return train, eval_idx, test


def fit_one(frame, columns, train_idx, eval_idx, test_idx, kind, info, model, seed, depth, lr, role) -> dict:
    key = run_key(info, model, seed)

    def _finite(idx):
        values = frame.loc[idx, TARGET].to_numpy(dtype=float)
        return idx[np.isfinite(values)]

    train_idx, eval_idx, test_idx = _finite(train_idx), _finite(eval_idx), _finite(test_idx)
    if min(len(train_idx), len(eval_idx), len(test_idx)) < 24:
        raise AssertionError(f"{info} {model} split has fewer than 24 finite targets")
    if not (train_idx.max() < eval_idx.min() and eval_idx.max() < test_idx.min()):
        raise AssertionError(
            f"split overlap for {info} {model}: train {train_idx.max()}, "
            f"eval {eval_idx.min()}..{eval_idx.max()}, test {test_idx.min()}"
        )
    existing = cached(key)
    if existing is not None:
        rows = []
        metrics = None
        for split in ("val", "test"):
            part = existing.loc[existing["split"] == split]
            if part.empty:
                continue
            scored = score_pair(part["y_true"], part["y_pred"])
            rows.append(metric_row(info, model, seed, split, role, scored))
            if split == "test":
                metrics = scored
        if rows:
            append_metrics(rows)
        print(f"  reuse {key} MAE {metrics['mae']:.2f}", flush=True)
        return {"key": key, "metrics": metrics, "frame": existing, "columns": columns, "p90": None}
    X_parts, cols, p90 = design_matrices(frame, columns, train_idx, eval_idx, test_idx, kind)
    y_train = frame.loc[train_idx, TARGET].to_numpy()
    y_eval = frame.loc[eval_idx, TARGET].to_numpy()
    y_test = frame.loc[test_idx, TARGET].to_numpy()

    t0 = time.time()
    model_obj, best, device = fit_xgb(
        X_parts[0], encode_target(y_train), X_parts[1], encode_target(y_eval), seed, depth, lr,
    )
    pred_eval = predict_xgb(model_obj, X_parts[1], best)
    pred_test = predict_xgb(model_obj, X_parts[2], best)
    elapsed = time.time() - t0
    stored = pd.concat([
        prediction_frame(eval_idx, y_eval, pred_eval, "val"),
        prediction_frame(test_idx, y_test, pred_test, "test"),
    ])
    test_metrics = score_pair(y_test, pred_test)
    val_metrics = score_pair(y_eval, pred_eval)
    save_prediction(key, stored, {
        "info_set": info, "model": model, "seed": seed, "role": role, "kind": kind,
        "depth": depth, "lr": lr, "best_iteration": best, "device": device,
        "n_features": len(cols), "features": cols, "seconds": elapsed,
        "gas_p90": p90, "train_start": str(train_idx.min()), "train_end": str(train_idx.max()),
        "eval_start": str(eval_idx.min()), "eval_end": str(eval_idx.max()),
        "test_start": str(test_idx.min()), "test_end": str(test_idx.max()),
        "test_hash": index_hash(pd.DatetimeIndex(test_idx)),
    })
    if seed == PRIMARY_SEED and role == "primary":
        (OUT / "models").mkdir(parents=True, exist_ok=True)
        model_obj.save_model(OUT / "models" / f"{key}.json")
    append_metrics([
        metric_row(info, model, seed, "val", role, val_metrics, {"best_iteration": best, "seconds": elapsed, "n_features": len(cols)}),
        metric_row(info, model, seed, "test", role, test_metrics, {"best_iteration": best, "seconds": elapsed, "n_features": len(cols)}),
    ])
    print(
        f"  fit {key} test MAE {test_metrics['mae']:.2f}  val MAE {val_metrics['mae']:.2f}  "
        f"trees {best}  {elapsed:.1f}s",
        flush=True,
    )
    return {"key": key, "metrics": test_metrics, "frame": stored, "columns": cols, "p90": p90}


def historical_rows(test_index: pd.DatetimeIndex) -> None:
    """Copy already published forecasts onto this protocol's metric table. No refit."""
    art = ROOT / "artifacts" / "honest" / "predictions"
    rows = []
    files = {
        "persistence": art / "persistence.parquet",
        "lear": art / "lear.parquet",
        "stack": art / "ensemble.parquet",
        "chronos_zero_shot": art / "chronos_zero_shot.parquet",
        "honest_tree": art / "honest_tree.parquet",
        "dam": ROOT / "runs" / "benchmark" / "dam" / "predictions.parquet",
        "seasonal_naive_24": ROOT / "runs" / "benchmark" / "seasonal_naive_24" / "predictions.parquet",
        "seasonal_naive_168": ROOT / "runs" / "benchmark" / "seasonal_naive_168" / "predictions.parquet",
    }
    for name, path in files.items():
        key = run_key("historical", name, None)
        if cached(key) is not None:
            continue
        if not path.exists():
            print(f"  historical file missing: {path}", flush=True)
            continue
        frame = pd.read_parquet(path)
        test = frame.loc[frame["split"] == "test"].reindex(test_index)
        if test["y_pred"].isna().any():
            raise AssertionError(f"{name} does not cover the frozen test index")
        stored = prediction_frame(test_index, test["y_true"], test["y_pred"], "test")
        save_prediction(key, stored, {"info_set": "historical", "model": name, "source": str(path), "role": "historical"})
        metrics = score_pair(test["y_true"], test["y_pred"])
        rows.append(metric_row("historical", name, None, "test", "historical", metrics))
    if rows:
        append_metrics(rows)


def replay_saved_tree(reported: pd.DataFrame, test_index: pd.DatetimeIndex) -> None:
    import xgboost as xgb
    from models.honest_mae import predict_tree

    key = run_key("reported", "saved_booster", None)
    if cached(key) is not None:
        return
    meta = _read_json(ROOT / "artifacts" / "honest" / "honest_tree_fit.json")
    model = xgb.XGBRegressor()
    model.load_model(ROOT / "artifacts" / "honest" / "honest_tree_model.json")
    pred = predict_tree(model, meta["cfg"], reported.loc[test_index], meta["features"], int(meta["best_iteration"]))
    y = reported.loc[test_index, TARGET].to_numpy()
    stored = prediction_frame(test_index, y, pred, "test")
    save_prediction(key, stored, {"role": "saved_booster_replay", "features": meta["features"]})
    saved = pd.read_parquet(ROOT / "artifacts" / "honest" / "predictions" / "honest_tree.parquet")
    saved = saved.loc[saved["split"] == "test", "y_pred"].reindex(test_index)
    err = float(np.max(np.abs(pred - saved.to_numpy(dtype=float))))
    metrics = score_pair(y, pred)
    append_metrics([metric_row("reported", "saved_booster", None, "test", "replay", metrics, {"replay_max_abs_error": err})])
    _write_json(OUT / "saved_booster_replay.json", {"max_abs_error": err, "mae": metrics["mae"]})
    if err > 1e-3:
        raise AssertionError(f"saved booster replay drifted by {err}")
    print(f"  saved booster replay MAE {metrics['mae']:.4f} max abs error {err:.6f}", flush=True)


def ablation(built: dict) -> None:
    base, reported = built["base"], built["reported"]
    test_index = built["test_index"]
    historical_rows(test_index)
    replay_saved_tree(reported, test_index)
    specs = [
        ("reported", reported, built["reported_cols"], "reported"),
        ("assumed_realtime", base, built["assumed_cols"], "pit"),
        ("conservative", base, built["cons_cols"], "pit"),
    ]
    for info, frame, cols, kind in specs:
        train, eval_idx, test = primary_splits(frame.index)
        subsets = feature_subsets(cols)
        for model, model_cols in subsets.items():
            for seed in SEEDS:
                fit_one(
                    frame, model_cols, train, eval_idx, test, kind, info, model, seed,
                    FROZEN_DEPTH, FROZEN_LR, "primary",
                )
    write_comparisons()


def test_vectors(info: str, model: str, seed) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frame = cached(run_key(info, model, seed))
    if frame is None:
        raise FileNotFoundError(run_key(info, model, seed))
    test = frame.loc[frame["split"] == "test"]
    return test.index, test["y_true"].to_numpy(), test["y_pred"].to_numpy()


def mean_test_vectors(info: str, model: str, seeds=SEEDS):
    indexes, ys, preds = [], [], []
    for seed in seeds:
        index, y, pred = test_vectors(info, model, seed)
        indexes.append(index)
        ys.append(y)
        preds.append(pred)
    if any(not indexes[0].equals(ix) for ix in indexes[1:]):
        raise AssertionError(f"{info} {model} seeds are not on the same index")
    return indexes[0], ys[0], np.mean(preds, axis=0)


def compare_pair(name: str, info: str, y, pred_a, pred_b, index, label_a: str, label_b: str) -> dict:
    """MAE(a) - MAE(b). Negative means a has lower absolute error than b."""
    def mae(yy, pp):
        return float(np.mean(np.abs(np.asarray(pp) - np.asarray(yy))))

    sizes = contiguous_sizes(index)
    boot = paired_bootstrap(mae, y, pred_a, pred_b, sizes, BOOT_BLOCK, BOOT_DRAWS, seed=0)
    dm = diebold_mariano(np.abs(pred_a - y), np.abs(pred_b - y), BOOT_BLOCK)
    return {
        "name": name,
        "info_set": info,
        "model_a": label_a,
        "model_b": label_b,
        "mae_a": mae(y, pred_a),
        "mae_b": mae(y, pred_b),
        "mae_gap_a_minus_b": boot["diff"],
        "bootstrap_ci_low": boot["ci_low"],
        "bootstrap_ci_high": boot["ci_high"],
        "bootstrap_p": boot["p_value"],
        "n_boot": boot["n_boot"],
        "block_hours": BOOT_BLOCK,
        "group_sizes": sizes,
        "dm_stat": dm["stat"],
        "dm_p": dm["p_value"],
        "dm_mean_loss_gap": dm["mean_diff"],
        "interval_excludes_zero": bool(boot["ci_high"] < 0 or boot["ci_low"] > 0),
        "a_better_on_mae": bool(boot["ci_high"] < 0),
    }


def write_comparisons() -> list[dict]:
    rows = []
    for info in ("reported", "assumed_realtime", "conservative"):
        index, y, both = mean_test_vectors(info, "xgb_both")
        _, _, exog = mean_test_vectors(info, "xgb_exog")
        _, _, chronos_xgb = mean_test_vectors(info, "xgb_chronos")
        rows.append(compare_pair(
            f"{info}__mean_seed__both_minus_exog", info, y, both, exog, index,
            "xgb_both_mean_seed", "xgb_exog_mean_seed",
        ))
        rows.append(compare_pair(
            f"{info}__mean_seed__both_minus_chronos_xgb", info, y, both, chronos_xgb, index,
            "xgb_both_mean_seed", "xgb_chronos_mean_seed",
        ))
        index42, y42, both42 = test_vectors(info, "xgb_both", PRIMARY_SEED)
        _, _, exog42 = test_vectors(info, "xgb_exog", PRIMARY_SEED)
        rows.append(compare_pair(
            f"{info}__seed42__both_minus_exog", info, y42, both42, exog42, index42,
            "xgb_both_seed42", "xgb_exog_seed42",
        ))
        _, _, zero = test_vectors("historical", "chronos_zero_shot", None)
        _, _, persist = test_vectors("historical", "persistence", None)
        # Historical files share the frozen index. Align to this info set's test index.
        zero = pd.Series(zero, index=test_vectors("historical", "chronos_zero_shot", None)[0]).reindex(index).to_numpy()
        persist = pd.Series(persist, index=test_vectors("historical", "persistence", None)[0]).reindex(index).to_numpy()
        rows.append(compare_pair(
            f"{info}__mean_seed__both_minus_chronos_zeroshot", info, y, both, zero, index,
            "xgb_both_mean_seed", "chronos_zero_shot",
        ))
        rows.append(compare_pair(
            f"{info}__mean_seed__exog_minus_persistence", info, y, exog, persist, index,
            "xgb_exog_mean_seed", "persistence",
        ))
        seed_gaps = []
        for seed in SEEDS:
            _, ys, a = test_vectors(info, "xgb_both", seed)
            _, _, b = test_vectors(info, "xgb_exog", seed)
            seed_gaps.append(float(np.mean(np.abs(a - ys)) - np.mean(np.abs(b - ys))))
        rows[-5]["per_seed_mae_gap_both_minus_exog"] = seed_gaps
    _write_json(OUT / "comparisons.json", rows)
    return rows


def factorial(built: dict) -> None:
    from models.honest_mae import (
        apply_gas_flag,
        attach_new_features,
        base_parts,
        clock_lag_frame,
        feature_list,
        recompute_stress_score,
        weather_columns,
    )

    aug, base_feats, raw, _tr, _va, _te = base_parts()
    path = OUT / "factorial_2023.csv"
    done = set()
    if path.exists():
        done = set(pd.read_csv(path)["name"])
    rows = []
    es_end = VAL_START
    es_start = es_end - pd.Timedelta(days=90)
    score_end = TEST_START
    for weather_lag, drop_gdelt, gas_p90, plus, window in product(
        (False, True), (False, True), (False, True), (False, True), ("full", "y2019"),
    ):
        name = (
            f"wx{int(weather_lag)}_gdelt{int(not drop_gdelt)}_gas{int(gas_p90)}"
            f"_plus{int(plus)}_{window}"
        )
        if name in done:
            continue
        frame = aug
        if weather_lag:
            frame = recompute_stress_score(clock_lag_frame(frame, weather_columns(frame), 1))
        if gas_p90:
            frame, _p90 = apply_gas_flag(frame, raw)
        if plus:
            frame = attach_new_features(frame, raw)
        feats = feature_list(base_feats, ["rt15_last", "rt15_max", "rt15_min", "rt15_mean",
                                          "da_shape_ramp", "da_shape_day_max", "da_shape_day_min",
                                          "da_shape_spread_24h"] if plus else None,
                             drop_gdelt=drop_gdelt)
        train_start = TRAIN_START if window == "y2019" else None
        train = slice_index(frame.index, train_start, es_start)
        eval_idx = slice_index(frame.index, es_start, es_end)
        score = slice_index(frame.index, es_end, score_end)
        result = fit_one(
            frame, feats, train, eval_idx, score, "reported", "factorial", name, PRIMARY_SEED,
            FROZEN_DEPTH, FROZEN_LR, "factorial_2023",
        )
        # fit_one stores the score slice as "test". That slice is calendar 2023.
        test = result["frame"].loc[result["frame"]["split"] == "test"]
        metrics = score_pair(test["y_true"], test["y_pred"])
        rows.append({
            "name": name, "weather_lag": weather_lag, "gdelt": not drop_gdelt,
            "gas_train_p90": gas_p90, "plus_features": plus, "window": window,
            "mae_2023": metrics["mae"], "rmse_2023": metrics["rmse"], "medae_2023": metrics["medae"],
            "spike_f1_2023": metrics["spike_f1"], "n": metrics["n"],
            "note": "Scored on 2023. Early stopping uses the last 90 days of 2022. 2024 is not used.",
        })
        upsert_csv(path, rows, ["name"])
        rows.clear()
        print(f"  factorial {name} 2023 MAE {metrics['mae']:.2f}", flush=True)


def expanding(built: dict) -> None:
    path = OUT / "expanding_window.csv"
    done = set(pd.read_csv(path)["name"]) if path.exists() else set()
    rows = []
    specs = [
        ("assumed_realtime", built["base"], built["assumed_cols"]),
        ("conservative", built["base"], built["cons_cols"]),
    ]
    labels = {
        2021: "backtest_column_set_fixed_after_later_inspection",
        2022: "backtest_column_set_fixed_after_later_inspection",
        2023: "development_year",
        2024: "previously_inspected",
    }
    for year in (2021, 2022, 2023, 2024):
        test_start = _ts(f"{year}-01-01")
        test_end = _ts(f"{year + 1}-01-01")
        es_end = test_start
        es_start = es_end - pd.Timedelta(days=90)
        for info, frame, cols in specs:
            subsets = feature_subsets(cols)
            for model in ("xgb_exog", "xgb_both"):
                name = f"{info}__{model}__{year}"
                if name in done:
                    continue
                train = slice_index(frame.index, None, es_start)
                eval_idx = slice_index(frame.index, es_start, es_end)
                test = slice_index(frame.index, test_start, test_end)
                result = fit_one(
                    frame, subsets[model], train, eval_idx, test, "pit",
                    f"expanding_{info}", f"{model}_{year}", PRIMARY_SEED,
                    FROZEN_DEPTH, FROZEN_LR, "expanding",
                )
                test_frame = result["frame"].loc[result["frame"]["split"] == "test"]
                metrics = score_pair(test_frame["y_true"], test_frame["y_pred"])
                rows.append({
                    "name": name, "info_set": info, "model": model, "year": year,
                    "role": labels[year], "seed": PRIMARY_SEED,
                    "train_end_exclusive": str(es_start), "early_stop_start": str(es_start),
                    "test_start": str(test_start), "test_end_exclusive": str(test_end),
                    "n_train": int(len(train)), "n_eval": int(len(eval_idx)), "n_test": int(len(test)),
                    "n_spikes": int((test_frame["y_true"] > SPIKE_PRICE).sum()),
                    "mae": metrics["mae"], "rmse": metrics["rmse"], "medae": metrics["medae"],
                    "spike_recall": metrics["spike_recall"], "spike_precision": metrics["spike_precision"],
                    "spike_f1": metrics["spike_f1"], "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
                    "normal_mae": metrics["normal_mae"], "stress_mae": metrics["stress_mae"],
                })
                upsert_csv(path, rows, ["name"])
                rows.clear()
                print(f"  expanding {name} MAE {metrics['mae']:.2f} n={metrics['n']}", flush=True)


def hyperparameter_check(built: dict) -> None:
    """2023 MAE under a stopping rule that ends in 2022. Not used to rescore 2024."""
    path = OUT / "hyperparameter_check_2023.csv"
    done = set(zip(
        pd.read_csv(path)["model"], pd.read_csv(path)["depth"], pd.read_csv(path)["lr"],
    )) if path.exists() else set()
    frame = built["base"]
    cols = feature_subsets(built["cons_cols"])
    es_end = VAL_START
    es_start = es_end - pd.Timedelta(days=90)
    train = slice_index(frame.index, TRAIN_START, es_start)
    eval_idx = slice_index(frame.index, es_start, es_end)
    score = slice_index(frame.index, es_end, TEST_START)
    rows = []
    grid = [(4, 0.03), (4, 0.05), (6, 0.03), (6, 0.05)]
    for model, model_cols in cols.items():
        for depth, lr in grid:
            if (model, depth, lr) in done:
                continue
            result = fit_one(
                frame, model_cols, train, eval_idx, score, "pit",
                "hp_conservative", f"{model}_d{depth}_lr{lr}", PRIMARY_SEED,
                depth, lr, "hyperparameter_check",
            )
            test = result["frame"].loc[result["frame"]["split"] == "test"]
            metrics = score_pair(test["y_true"], test["y_pred"])
            rows.append({
                "model": model, "depth": depth, "lr": lr, "mae_2023": metrics["mae"],
                "rmse_2023": metrics["rmse"], "spike_f1_2023": metrics["spike_f1"],
                "frozen_primary": depth == FROZEN_DEPTH and lr == FROZEN_LR,
                "note": "Held-out 2023 after early stopping on the last 90 days of 2022. Not applied to 2024.",
            })
            upsert_csv(path, rows, ["model", "depth", "lr"])
            rows.clear()


def gfs_and_hub(built: dict) -> None:
    sources = built["sources"]
    base = built["base"]
    if sources["gfs"] is not None:
        gfs = gfs_target_hour(sources["gfs"], base.index, lead_day=1)
        gfs_frame = base.copy()
        for col in gfs.columns:
            gfs_frame[col] = gfs[col]
        gfs_frame["miss_gfs"] = gfs.isna().all(axis=1).astype(float)
        gfs_cols = conservative_columns(built["assumed_cols"]) + list(gfs.columns) + ["miss_gfs"]
        start = _ts("2021-04-01")
        train = slice_index(base.index, start, VAL_START)
        eval_idx = slice_index(base.index, VAL_START, TEST_START)
        test = slice_index(base.index, TEST_START, None)
        for label, frame, cols in (
            ("gfs_temp", gfs_frame, gfs_cols),
            ("conservative_from_2021_04", base, built["cons_cols"]),
        ):
            subsets = feature_subsets(cols)
            for model in ("xgb_exog", "xgb_both"):
                fit_one(
                    frame, subsets[model], train, eval_idx, test, "pit",
                    label, model, PRIMARY_SEED, FROZEN_DEPTH, FROZEN_LR, "gfs_sensitivity",
                )
        catalog = catalog_rows(list(gfs.columns) + ["miss_gfs"], "gfs_temp")
        catalog.to_csv(OUT / "gfs_catalog.csv", index=False)
    else:
        _write_json(OUT / "gfs_sensitivity.json", {"status": "blocked", "reason": "weather_forecast_gfs.parquet missing"})

    north_price = built["sources"]["matrix"]["HB_NORTH"]
    north = assemble_assumed_realtime(
        base.index, north_price, sources["rt"]["HB_NORTH"], sources["da"], sources["load"],
        sources["weather"], sources["gas"], sources["storage"], sources["nrc"], sources["chronos"],
    )
    # Shape and spread follow HB_NORTH. assemble uses da_HB_HUBAVG by default, so rebuild DA features.
    from src.features.pit_sets import day_ahead_features

    da_north = day_ahead_features(north_price.reindex(base.index), sources["da"].reindex(base.index), "da_HB_NORTH")
    for col in da_north.columns:
        if col.startswith("da_shape"):
            north[col] = da_north[col]
    north[TARGET] = north_price.reindex(base.index)
    train, eval_idx, test = primary_splits(north.index)
    cons = conservative_columns(built["assumed_cols"])
    subsets = feature_subsets(cons)
    for model in ("xgb_exog", "xgb_both"):
        for seed in SEEDS:
            fit_one(
                north, subsets[model], train, eval_idx, test, "pit",
                "hub_HB_NORTH", model, seed, FROZEN_DEPTH, FROZEN_LR, "hub_transfer",
            )
    _write_json(OUT / "hub_transfer.json", {
        "hub": "HB_NORTH",
        "target": "hourly max of HB_NORTH, same UTC-hour definition as HB_HUBAVG",
        "chronos": "still the HB_HUBAVG zero-shot cache; not refit",
        "tuning": "none; depth 6, learning rate 0.05, conservative columns",
        "seeds": list(SEEDS),
    })


def regime_tables() -> None:
    rows = []
    jobs = []
    for info in ("assumed_realtime", "conservative"):
        for model in ("xgb_exog", "xgb_both"):
            index, y, pred = mean_test_vectors(info, model)
            jobs.append((info, model + "_mean_seed", index, y, pred))
    for name in ("persistence", "chronos_zero_shot", "lear", "honest_tree", "stack"):
        index, y, pred = test_vectors("historical", name, None)
        jobs.append(("historical", name, index, y, pred))
    for info, model, index, y, pred in jobs:
        local = pd.DatetimeIndex(index).tz_convert("US/Central")
        season = local.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                  6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
        masks = {
            "all": np.ones(len(y), dtype=bool),
            "price_le_100": y <= 100,
            "price_gt_100": y > STRESS_PRICE,
            "price_gt_200": y > SPIKE_PRICE,
            "negative_price": y < 0,
        }
        for label, months in (("DJF", "DJF"), ("MAM", "MAM"), ("JJA", "JJA"), ("SON", "SON")):
            masks[f"season_{label}"] = season.to_numpy() == months
        for hour in range(24):
            masks[f"hour_local_{hour:02d}"] = local.hour == hour
        for regime, mask in masks.items():
            if mask.sum() == 0:
                continue
            metrics = score_pair(y[mask], pred[mask])
            rows.append({
                "info_set": info, "model": model, "regime": regime, "n": int(mask.sum()),
                "mae": metrics["mae"], "rmse": metrics["rmse"], "medae": metrics["medae"],
                "spike_recall": metrics["spike_recall"], "spike_precision": metrics["spike_precision"],
                "spike_f1": metrics["spike_f1"], "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
            })
    pd.DataFrame(rows).to_csv(OUT / "regimes.csv", index=False)


HUBS = ["HB_HUBAVG", "HB_NORTH", "HB_HOUSTON", "HB_SOUTH", "HB_WEST"]


def _rtm_spp_nonempty_sheets(ercot, year: int) -> pd.DataFrame:
    """Same download as gridstatus, without concatenating empty future months.

    The 2026 workbook contains empty November and December sheets. Concatenating
    those object columns with the integer hour columns makes a later timedelta
    cast fail. Dropping the empty sheets leaves the published rows unchanged.
    """
    from gridstatus import utils
    from gridstatus.base import Markets
    from gridstatus.ercot import HISTORICAL_RTM_LOAD_ZONE_AND_HUB_PRICES_RTID

    doc_info = ercot._get_document(
        report_type_id=HISTORICAL_RTM_LOAD_ZONE_AND_HUB_PRICES_RTID,
        constructed_name_contains=f"{year}.zip",
        verbose=False,
    )
    raw = utils.get_zip_file(doc_info.url, verbose=False)
    sheets = pd.read_excel(raw, sheet_name=None)
    frames = []
    for frame in sheets.values():
        if frame.empty or "Delivery Hour" not in frame.columns:
            continue
        if int(frame["Delivery Hour"].notna().sum()) == 0:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"no non-empty real-time sheets for {year}")
    df = pd.concat(frames, ignore_index=True)
    count = int(df[["Delivery Hour", "Delivery Interval"]].isnull().all(axis=1).sum())
    if count == 1:
        df = df.dropna(subset=["Delivery Hour", "Delivery Interval"], how="all")
    elif count > 1:
        raise ValueError("more than one all-null delivery row in the real-time workbook")
    df["Delivery Interval"] = df["Delivery Interval"].astype("Int64")
    df = ercot.parse_doc(df, verbose=False)
    return ercot._finalize_spp_df(df, market=Markets.REAL_TIME_15_MIN, verbose=False)


def _pivot_spp(raw: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    frame = raw.copy()
    frame["Interval Start"] = pd.to_datetime(frame["Interval Start"], utc=True)
    frame = frame[frame["Location"].isin(HUBS)]
    out = frame.pivot_table(index="Interval Start", columns="Location", values="SPP", aggfunc="first").sort_index()
    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC")
    if prefix:
        out = out.rename(columns={c: f"{prefix}{c}" for c in out.columns})
    return out


def fetch_later() -> dict:
    """Download 2025 and 2026 hub prices into runs/pit_ablation/external. Do not touch data/raw.

    A year that the archive does not serve is recorded and skipped. Scoring is separate,
    because the frozen model may still be fitting while this download runs.
    """
    note = {
        "attempted": True,
        "status": "not_run",
        "years": {},
        "chronos": "not run; the frozen cache ends in 2024",
    }
    dest = OUT / "external"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import gridstatus
        ercot = gridstatus.Ercot()
        rt_parts, da_parts = [], []
        for year in (2025, 2026):
            year_note = {}
            try:
                print(f"  fetching ERCOT RT SPP {year}", flush=True)
                try:
                    raw_rt = ercot.get_rtm_spp(year, verbose=False)
                except ValueError as exc:
                    if "timedelta64" not in str(exc):
                        raise
                    print(f"  RT {year} retry without empty month sheets", flush=True)
                    raw_rt = _rtm_spp_nonempty_sheets(ercot, year)
                    year_note["rt_fallback"] = "empty trailing month sheets were dropped before parsing"
                rt_parts.append(_pivot_spp(raw_rt))
                year_note["rt"] = "fetched"
            except Exception as exc:
                year_note["rt"] = f"{type(exc).__name__}: {exc}"
                print(f"  RT {year} blocked: {year_note['rt']}", flush=True)
            try:
                print(f"  fetching ERCOT DAM SPP {year}", flush=True)
                da_parts.append(_pivot_spp(ercot.get_dam_spp(year, verbose=False), prefix="da_"))
                year_note["da"] = "fetched"
            except Exception as exc:
                year_note["da"] = f"{type(exc).__name__}: {exc}"
                print(f"  DAM {year} blocked: {year_note['da']}", flush=True)
            note["years"][str(year)] = year_note
        if not rt_parts or not da_parts:
            note["status"] = "blocked"
            note["error"] = "no real-time and day-ahead year could be fetched"
        else:
            rt = pd.concat(rt_parts).sort_index()
            rt = rt[~rt.index.duplicated(keep="last")]
            da = pd.concat(da_parts).sort_index()
            da = da[~da.index.duplicated(keep="last")]
            rt.to_parquet(dest / "ercot_rt_prices_2025_2026.parquet")
            da.to_parquet(dest / "ercot_da_prices_2025_2026.parquet")
            note["status"] = "fetched"
            note["rt_start"] = str(rt.index.min())
            note["rt_end"] = str(rt.index.max())
            note["da_start"] = str(da.index.min())
            note["da_end"] = str(da.index.max())
            note["rt_rows"] = int(len(rt))
            note["da_rows"] = int(len(da))
    except Exception as exc:
        note["status"] = "blocked"
        note["error"] = f"{type(exc).__name__}: {exc}"
        note["traceback"] = traceback.format_exc(limit=8)
        print(f"  later-year fetch blocked: {note['error']}", flush=True)
    _write_json(OUT / "later_years.json", note)
    return note


def try_later_years(built: dict) -> dict:
    """Fetch, then score, when this command is run on its own."""
    note = fetch_later()
    if note.get("status") != "fetched":
        return note
    dest = OUT / "external"
    rt = pd.read_parquet(dest / "ercot_rt_prices_2025_2026.parquet")
    da = pd.read_parquet(dest / "ercot_da_prices_2025_2026.parquet")
    score_later_years(built, rt, da, note)
    _write_json(OUT / "later_years.json", note)
    return note


def _complete_month_end(index: pd.DatetimeIndex) -> pd.Timestamp:
    """Exclusive UTC end at the start of the month after the last complete Central month.

    The month that contains today is never complete. A month that does not include its last
    hour is dropped as well.
    """
    observed = pd.DatetimeIndex(index).tz_convert("US/Central")
    last = observed.max()
    month_start = last.tz_localize(None).to_period("M").to_timestamp().tz_localize("US/Central")
    next_month = (month_start.tz_localize(None) + pd.DateOffset(months=1)).tz_localize("US/Central")
    if last < next_month - pd.Timedelta(hours=1):
        exclusive = month_start
    else:
        exclusive = next_month
    current_month = pd.Timestamp.now(tz="US/Central").tz_localize(None).to_period("M").to_timestamp().tz_localize("US/Central")
    if exclusive > current_month:
        exclusive = current_month
    return exclusive.tz_convert("UTC")


def score_later_years(built: dict, rt: pd.DataFrame, da: pd.DataFrame, note: dict) -> None:
    """Score the frozen conservative XGBoost-only model on hours after 2024.

    Chronos features are not available, so the both-model is not scored.
    Load is missing and stays missing. Storage uses the file already on disk.
    """
    from src.features.pit_sets import (
        calendar_central,
        day_ahead_features,
        lagged_load,
        previous_hour_intervals,
        price_history,
    )

    hourly = rt[TARGET].resample("1h").max()
    counts = rt[TARGET].resample("1h").count()
    hourly = hourly.where(counts == 4).dropna()
    if hourly.empty:
        note["score_status"] = "blocked"
        note["score_reason"] = "no hours with four 15-minute prices"
        return
    cutoff = _complete_month_end(hourly.index)
    note["scored_through_exclusive_utc"] = str(cutoff)
    note["cutoff_rule"] = (
        "Latest complete US/Central month. The in-progress month is excluded. "
        "A month whose last hour is missing is excluded. Hours without four 15-minute prices are dropped, "
        "which is stricter than the historical v2 matrix."
    )
    hourly = hourly[hourly.index < cutoff]
    if len(hourly) < 24:
        note["score_status"] = "blocked"
        note["score_reason"] = "fewer than 24 complete hours remain after the month cutoff"
        return
    # Context for lags comes from the historical hourly price plus the new hours.
    hist = built["sources"]["matrix"][TARGET]
    price = pd.concat([hist, hourly[hourly.index > hist.index.max()]]).sort_index()
    price = price[~price.index.duplicated(keep="last")]
    da_hist = built["sources"]["da"]
    da_all = pd.concat([da_hist, da[da.index > da_hist.index.max()]]).sort_index()
    da_all = da_all[~da_all.index.duplicated(keep="last")]
    index = hourly.index[hourly.index >= TEST_START]
    index = index[hourly.reindex(index).notna()]
    # Issue-time features. Load is absent.
    load = pd.Series(np.nan, index=price.index, dtype=float)
    frame = pd.concat([
        price_history(price, index),
        previous_hour_intervals(rt[TARGET], index),
        day_ahead_features(price, da_all.reindex(da_all.index.union(index)).sort_index(), "da_HB_HUBAVG").reindex(index),
        lagged_load(load, index),
        calendar_central(index),
    ], axis=1)
    frame[TARGET] = hourly.reindex(index)
    # Storage as-of from the existing file, which extends into 2026.
    from src.features.pit_sets import fundamental_frames

    storage = fundamental_frames(
        built["sources"]["gas"], built["sources"]["storage"], built["sources"]["nrc"], index,
    )["storage"]
    for col in storage.columns:
        frame[col] = storage[col]
    frame["miss_load"] = 1.0
    frame["miss_storage"] = frame["storage_bcf"].isna().astype(float)
    frame["miss_da"] = frame["da_HB_HUBAVG"].isna().astype(float)
    # The published conservative model was trained with Chronos columns removed in xgb_exog.
    # Refit is not allowed to see 2025. Reuse the primary conservative xgb_exog seed-42 model
    # by applying its training medians. Those medians were not saved, so refit on 2019-2022
    # only, which does not include 2025, and then predict. This is the same frozen specification.
    hist_index = built["base"].index
    train = slice_index(hist_index, TRAIN_START, VAL_START)
    eval_idx = slice_index(hist_index, VAL_START, TEST_START)
    # Build a historical conservative frame slice is already the base. Fit exog without chronos
    # was already done. Predicting a new index requires the saved model and the same columns.
    model_path = OUT / "models" / f"{run_key('conservative', 'xgb_exog', PRIMARY_SEED)}.json"
    meta = _read_json(meta_path(run_key("conservative", "xgb_exog", PRIMARY_SEED)))
    if not model_path.exists():
        note["score_status"] = "blocked"
        note["score_reason"] = "conservative xgb_exog seed 42 model was not saved"
        return
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    cols = meta["features"]
    note["columns_absent_before_fill"] = [c for c in cols if c not in frame.columns]
    # Reconstruct training medians on the asinh scale from the historical base, then apply them.
    hist_frame = built["base"]
    X_parts, usable, _p90 = design_matrices(hist_frame, cols, train, eval_idx, slice_index(hist_index, TEST_START, None), "pit")
    if usable != cols:
        note["score_status"] = "blocked"
        note["score_reason"] = "column set drifted relative to the saved model"
        return
    # Medians are whatever filled the training matrix. Recover them as the fill value of an all-NaN row
    # by imputing a dummy frame through the same routine.
    dummy = hist_frame.loc[train[:1]].copy()
    for col in cols:
        if col in dummy.columns:
            dummy[col] = np.nan
    # Easier: the training matrix after impute has no NaNs; the median is the imputed value
    # only where the raw value was missing. Compute medians directly.
    flagged, _ = attach_split_flags(hist_frame.copy(), train)
    raw_train = apply_asinh(flagged.loc[train], signed_asinh_columns(cols))
    med = raw_train[cols].median(numeric_only=True)
    later = frame.copy()
    # Columns that exist only after flags.
    for col in cols:
        if col not in later.columns:
            later[col] = 1.0 if col.startswith("miss_") else np.nan
    later = apply_asinh(later, signed_asinh_columns(cols))
    for col in cols:
        later[col] = later[col].fillna(med[col])
    pred = predict_xgb(model, later[cols], int(meta["best_iteration"]))
    y = frame[TARGET].to_numpy()
    mask = np.isfinite(y) & np.isfinite(pred)
    stored = prediction_frame(frame.index[mask], y[mask], pred[mask], "later")
    save_prediction(run_key("later", "xgb_exog_conservative", PRIMARY_SEED), stored, {
        "role": "later_years", "features": cols, "load": "missing; miss_load = 1",
        "chronos": "not used",
    })
    # Also persistence and day-ahead on the same hours.
    persist = price.shift(freq=pd.Timedelta(hours=1)).reindex(frame.index)
    # clock persistence
    from src.features.pit_sets import clock_shift
    persist = clock_shift(price, 1, frame.index)
    da_pred = frame["da_HB_HUBAVG"]
    rows = []
    for name, pred_s in (
        ("xgb_exog_conservative", pd.Series(pred, index=frame.index)),
        ("persistence", persist),
        ("day_ahead", da_pred),
    ):
        ok = y
        p = pred_s.to_numpy(dtype=float)
        good = np.isfinite(ok) & np.isfinite(p)
        local_year = frame.index.tz_convert("US/Central").year
        for year in (2025, 2026):
            # Calendar year in Central time, matching the month cutoff. A UTC year
            # would put the last six hours of December into the next year.
            year_mask = good & (local_year == year)
            if year_mask.sum() < 24:
                continue
            metrics = score_pair(ok[year_mask], p[year_mask])
            rows.append({"year": year, "model": name, **{k: metrics[k] for k in (
                "n", "mae", "rmse", "medae", "spike_recall", "spike_precision", "spike_f1", "tp", "fp", "fn",
                "normal_mae", "stress_mae",
            )}})
    pd.DataFrame(rows).to_csv(OUT / "later_years_metrics.csv", index=False)
    note["score_status"] = "scored"
    note["models"] = "conservative XGBoost without Chronos, persistence, day-ahead price"
    note["not_scored"] = "XGBoost with Chronos statistics, because no Chronos cache exists after 2024"
    note["load"] = "not fetched; miss_load is 1 and the training median is used"


def make_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(OUT / "ablation_metrics.csv")
    test = metrics[(metrics["role"] == "primary") & (metrics["split"] == "test")]
    summary = test.groupby(["info_set", "model"])["mae"].agg(mean="mean", std="std").reset_index()
    order = ["xgb_exog", "xgb_chronos", "xgb_both"]
    infos = ["reported", "assumed_realtime", "conservative"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    for i, info in enumerate(infos):
        part = summary[summary["info_set"] == info].set_index("model").reindex(order)
        ax.bar(np.arange(len(order)) + i * width, part["mean"], width=width, yerr=part["std"],
               label=info, capsize=3)
    ax.set_xticks(np.arange(len(order)) + width)
    ax.set_xticklabels(["XGBoost features", "XGBoost Chronos only", "XGBoost both"])
    ax.set_ylabel("2024 MAE ($/MWh)")
    ax.set_title("Mean absolute error across five seeds")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "mae_by_model.png", dpi=140)
    plt.close(fig)

    regimes = pd.read_csv(OUT / "regimes.csv")
    keep = regimes[regimes["regime"].isin(["all", "price_le_100", "price_gt_100", "price_gt_200", "negative_price"])]
    keep = keep[keep["info_set"] == "conservative"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    regimes_order = ["all", "price_le_100", "price_gt_100", "price_gt_200", "negative_price"]
    for model, label in (("xgb_exog_mean_seed", "without Chronos"), ("xgb_both_mean_seed", "with Chronos")):
        part = keep[keep["model"] == model].set_index("regime").reindex(regimes_order)
        ax.plot(regimes_order, part["mae"], marker="o", label=label)
    ax.set_ylabel("MAE ($/MWh)")
    ax.set_title("Conservative information set, 2024, seed-averaged forecast")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "mae_by_regime.png", dpi=140)
    plt.close(fig)

    index, y, pred = mean_test_vectors("conservative", "xgb_both")
    err = np.abs(pred - y)
    worst = np.argsort(err)[-12:][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(y[worst], pred[worst], s=30)
    lim = max(y[worst].max(), pred[worst].max())
    ax.plot([0, lim], [0, lim], color="0.5", lw=1)
    ax.set_xlabel("Realized hourly maximum ($/MWh)")
    ax.set_ylabel("Forecast ($/MWh)")
    ax.set_title("Largest absolute errors, conservative XGBoost with Chronos")
    fig.tight_layout()
    fig.savefig(fig_dir / "worst_spikes.png", dpi=140)
    plt.close(fig)


def _fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def write_report() -> None:
    from models.pit_manuscript import write_manuscript

    metrics = pd.read_csv(OUT / "ablation_metrics.csv")
    comparisons = _read_json(OUT / "comparisons.json")
    manifest = _read_json(OUT / "build_manifest.json")
    lines = [
        "# Point-in-time Chronos ablation",
        "",
        f"Protocol `{PROTOCOL}`. Command: `python models/run_pit_ablation.py all`.",
        "The frozen 2024 index was scored after the comparison was specified. It is a previously inspected year, not an untouched test set.",
        "",
        "## Build",
        "",
        f"- Test index hash `{manifest['test_hash']}`, {manifest['data_audit']['test_rows']} hours.",
        f"- Reported columns: {manifest['n_reported_columns']}. Assumed-realtime columns: {manifest['n_assumed_columns']}. Conservative columns: {manifest['n_conservative_columns']}.",
        f"- Hours in the study index without exactly four 15-minute prices: {manifest['data_audit']['hours_in_matrix_with_not_4_intervals']}. They were counted and kept.",
        f"- Chronos cache matches the gap-aware builder index: {manifest['chronos_context']['cache_equals_builder_index']}. Bad test-hour bounds: {manifest['chronos_context']['test_hours_with_bad_bounds']}.",
        f"- File hashes: `{json.dumps(manifest['hashes'])}`.",
        "",
        "Feature availability is `runs/pit_ablation/feature_catalog.csv`. `verified` is `documented`, `assumed`, or `model_output`. Assumed rules are not treated as measured publication times.",
        "",
        "## Primary 2024 scores",
        "",
        "Train 2019-01-01 through 2022, early stopping on 2023, depth 6, learning rate 0.05, absolute error on an asinh target, seeds 0, 1, 2, 3, and 42. The table is the mean across seeds. The saved honest-tree booster is a separate replay, not a new fit.",
        "",
        "| Information set | Model | MAE | RMSE | Median AE | Spike recall | Spike precision | Spike F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    primary = metrics[(metrics["role"] == "primary") & (metrics["split"] == "test")]
    for info in ("reported", "assumed_realtime", "conservative"):
        for model in ("xgb_exog", "xgb_chronos", "xgb_both"):
            part = primary[(primary["info_set"] == info) & (primary["model"] == model)]
            if part.empty:
                continue
            lines.append(
                f"| {info} | {model} | {_fmt(part['mae'].mean())} | {_fmt(part['rmse'].mean())} | "
                f"{_fmt(part['medae'].mean())} | {_fmt(100 * part['spike_recall'].mean())}% | "
                f"{_fmt(100 * part['spike_precision'].mean())}% | {_fmt(part['spike_f1'].mean(), 3)} |"
            )
    hist = metrics[(metrics["role"].isin(["historical", "replay"])) & (metrics["split"] == "test")]
    lines += ["", "Historical forecasts, unchanged:", "",
              "| Model | MAE | RMSE | Spike recall | Spike precision |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for _, row in hist.iterrows():
        lines.append(
            f"| {row['model']} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} | "
            f"{_fmt(100 * row['spike_recall'])}% | {_fmt(100 * row['spike_precision'])}% |"
        )
    lines += ["", "## Does Chronos add anything?", "",
              "The gap is MAE(with Chronos statistics) minus MAE(without them), in $/MWh. A negative gap means the Chronos columns have lower absolute error. The interval is a 24-hour moving-block bootstrap of that paired difference, 2,000 draws. The Diebold-Mariano statistic uses absolute error, HAC lag 24, and the Harvey-Leybourne-Newbold correction. A negative statistic means the same thing.",
              ""]
    lines.append("| Comparison | MAE gap | 95% interval | Excludes zero | DM stat | DM p |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: |")
    for row in comparisons:
        if "both_minus_exog" not in row["name"] and "exog_minus_persistence" not in row["name"]:
            continue
        lines.append(
            f"| {row['name']} | {_fmt(row['mae_gap_a_minus_b'])} | "
            f"[{_fmt(row['bootstrap_ci_low'])}, {_fmt(row['bootstrap_ci_high'])}] | "
            f"{row['interval_excludes_zero']} | {_fmt(row['dm_stat'])} | {_fmt(row['dm_p'], 4)} |"
        )
    cons = next(r for r in comparisons if r["name"] == "conservative__mean_seed__both_minus_exog")
    lines += ["",
              f"On the conservative set, the seed-averaged MAE gap is {_fmt(cons['mae_gap_a_minus_b'])} "
              f"with interval [{_fmt(cons['bootstrap_ci_low'])}, {_fmt(cons['bootstrap_ci_high'])}]. "
              f"Per-seed gaps (both minus exogenous): {', '.join(_fmt(v) for v in cons.get('per_seed_mae_gap_both_minus_exog', []))}.",
              ""]
    if cons["a_better_on_mae"]:
        lines.append("The conservative interval lies below zero, so the Chronos statistics reduce MAE relative to the matched XGBoost model under this information set.")
    elif cons["bootstrap_ci_low"] > 0:
        lines.append("The conservative interval lies above zero, so the Chronos statistics increase MAE relative to the matched XGBoost model under this information set.")
    else:
        lines.append("The conservative interval covers zero. The Chronos statistics are not distinguishable from noise on MAE under this information set.")
    lines.append("This is one market and one previously inspected test year. It is not a general claim about foundation-model features.")
    for extra, title in (
        ("factorial_2023.csv", "Factorial on 2023"),
        ("expanding_window.csv", "Expanding-window years"),
        ("hyperparameter_check_2023.csv", "Hyperparameter check on 2023"),
        ("regimes.csv", "Regimes"),
        ("later_years_metrics.csv", "Hours after 2024"),
    ):
        path = OUT / extra
        lines += ["", f"## {title}", ""]
        if not path.exists():
            lines.append("Not produced.")
            continue
        frame = pd.read_csv(path)
        lines.append(f"`runs/pit_ablation/{extra}` has {len(frame)} rows.")
        if extra == "expanding_window.csv":
            lines.append("")
            lines.append("| Year | Set | Model | Role | Hours | Spikes | MAE | RMSE |")
            lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |")
            for _, row in frame.iterrows():
                lines.append(
                    f"| {int(row['year'])} | {row['info_set']} | {row['model']} | {row['role']} | "
                    f"{int(row['n_test'])} | {int(row['n_spikes'])} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} |"
                )
        if extra == "later_years_metrics.csv":
            lines.append("")
            lines.append("| Year | Model | Hours | MAE | RMSE | Spike F1 |")
            lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
            for _, row in frame.iterrows():
                lines.append(
                    f"| {int(row['year'])} | {row['model']} | {int(row['n'])} | {_fmt(row['mae'])} | "
                    f"{_fmt(row['rmse'])} | {_fmt(row['spike_f1'], 3)} |"
                )
    later = OUT / "later_years.json"
    if later.exists():
        payload = _read_json(later)
        lines += ["", "## Later-year fetch", "",
                  f"Status: {payload.get('status')}. Score: {payload.get('score_status')}.",
                  f"Scored through (exclusive UTC): {payload.get('scored_through_exclusive_utc')}.",
                  ]
        for year, info in (payload.get("years") or {}).items():
            lines.append(f"- {year}: real-time `{info.get('rt')}`; day-ahead `{info.get('da')}`.")
            if info.get("rt_fallback"):
                lines.append(f"  Fallback: {info['rt_fallback']}.")
        if payload.get("error"):
            lines.append(payload["error"])
        if payload.get("not_scored"):
            lines.append(f"Not scored: {payload['not_scored']}.")
        if payload.get("load"):
            lines.append(f"Load: {payload['load']}.")
    lines += ["", "Figures, when the report step finishes, are under `runs/pit_ablation/figures/`.", ""]
    text = "\n".join(lines)
    (ROOT / "reports" / "pit_ablation.md").write_text(text)
    write_manuscript(ROOT, OUT, comparisons, metrics)
    print(f"  wrote {ROOT / 'reports' / 'pit_ablation.md'}", flush=True)


def _score_fetched(built: dict) -> None:
    dest = OUT / "external"
    rt_path = dest / "ercot_rt_prices_2025_2026.parquet"
    da_path = dest / "ercot_da_prices_2025_2026.parquet"
    note_path = OUT / "later_years.json"
    if not note_path.exists() or not rt_path.exists() or not da_path.exists():
        return
    note = _read_json(note_path)
    if note.get("status") != "fetched" or note.get("score_status") == "scored":
        return
    rt = pd.read_parquet(rt_path)
    da = pd.read_parquet(da_path)
    score_later_years(built, rt, da, note)
    _write_json(note_path, note)


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "build", "ablation", "generalize", "later", "fetch", "report", "figures", "all",
    ])
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_later()
        return
    if args.command in {"build", "ablation", "generalize", "later", "all"}:
        built = build()
    else:
        built = None
    fetch_proc = None
    if args.command == "all":
        fetch_proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "fetch"],
            cwd=ROOT,
        )
    if args.command in {"ablation", "all"}:
        ablation(built)
    if args.command in {"generalize", "all"}:
        factorial(built)
        expanding(built)
        hyperparameter_check(built)
        gfs_and_hub(built)
        regime_tables()
    if args.command == "later":
        try_later_years(built)
    if args.command == "all" and fetch_proc is not None:
        try:
            fetch_proc.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            fetch_proc.kill()
            fetch_proc.wait()
            if not (OUT / "later_years.json").exists():
                _write_json(OUT / "later_years.json", {
                    "attempted": True,
                    "status": "blocked",
                    "error": "The 2025-2026 archive download was still running 20 minutes after the ablation finished, so it was stopped. No later-year score is claimed.",
                    "chronos": "not run; the frozen cache ends in 2024",
                })
        if fetch_proc.returncode not in (0, None) and not (OUT / "later_years.json").exists():
            _write_json(OUT / "later_years.json", {
                "attempted": True,
                "status": "blocked",
                "error": f"fetch process exited {fetch_proc.returncode}",
                "chronos": "not run; the frozen cache ends in 2024",
            })
        try:
            _score_fetched(built)
        except Exception as exc:
            note = _read_json(OUT / "later_years.json") if (OUT / "later_years.json").exists() else {"attempted": True}
            note["score_status"] = "blocked"
            note["score_reason"] = f"{type(exc).__name__}: {exc}"
            _write_json(OUT / "later_years.json", note)
            print(f"  later-year score blocked: {note['score_reason']}", flush=True)
    if args.command in {"figures", "all"}:
        make_figures()
    if args.command in {"report", "all"}:
        write_report()


if __name__ == "__main__":
    main()
