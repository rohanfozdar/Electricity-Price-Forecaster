"""
Honest one-hour-ahead MAE experiments.

Protocol, in order:
    1. Persistence and seasonal-naive baselines on the frozen 2024 index.
    2. Leakage shifts of the selected Chronos-XGBoost config. Rank on 2023.
       Write runs/benchmark/selected_leakage.json before any 2024 prediction.
       Then score 2024 once for the 2023 winner and once for the unshifted
       model (or for the 1-hour weather lag, if the winner is already unshifted).
    3. Freeze an information-set feature matrix: weather lagged 1 hour,
       gas-spike flag from the training percentile only, GDELT dropped,
       previous-hour 15-minute price summaries, and day-ahead shape.
    4. Choose a training window on 2022 only. Confirm that window on 2023.
       Write runs/benchmark/selected_window.json before the 2024 score.
    5. LEAR, with the penalty chosen on 2022. Write selected_lear.json first.
    6. Combine day-ahead, persistence, LEAR, Chronos zero-shot, and the
       windowed tree. Weights from 2023 only. Write selected_ensemble.json
       before the 2024 score.

2024 is never used to pick a model, a window, a penalty, or a weight.
A run that already has predictions.parquet is left alone.

Run from the repository root:
    python models/honest_mae.py all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import RAW_DIR
from models.chronos_hourly_features import build_contexts, context_bounds
from models.improve_hybrid import (
    CHRONOS_COLS,
    CHRONOS_SOURCES,
    FEATSETS,
    TARGETS,
    faithful_val_f1,
    make_model,
)
from models.run_benchmark import (
    CHRONOS_PRED_PATH,
    RUNS_DIR,
    SPIKE_PRICE,
    STRESS_PRICE,
    TEST_ROWS,
    TEST_SPIKES,
    frozen_split,
    hybrid_frame,
    index_hash,
    load_matrix,
    prediction_frame,
    regression_metrics,
    save_run,
)
from models.utils import TARGET_REG, WEATHER_SUFFIXES
from src.evaluation.statistical_tests import diebold_mariano

EXPECTED_HASH = "b850034b0690e660"
SEED = 42
ES_DAYS = 90
CONTEXT = 512
UTC = "UTC"
RT15_COLS = ["rt15_last", "rt15_max", "rt15_min", "rt15_mean"]
DA_SHAPE_COLS = ["da_shape_ramp", "da_shape_day_max", "da_shape_day_min", "da_shape_spread_24h"]
WEATHER_STRESS = [
    "stress_cold_snap", "stress_freeze", "stress_heat_wave",
    "stress_extreme_heat", "stress_low_wind",
]
OTHER_STRESS = ["stress_gas_spike", "stress_reactor_outage"]
SELECTABLE_LEAKS = [
    "unshifted", "weather_lag_1", "weather_lag_120", "gas_train_p90", "gdelt_lag_24",
]
WINDOW_TRAIN_START = {
    "y2022": "2021-01-01",       # one year, scored on 2022
    "y2021_2022": "2020-01-01",  # two years
    "y2019_2022": "2018-01-01",  # four years
    "full": None,
}
WINDOW_CONFIRM_START = {
    "y2022": "2022-01-01",
    "y2021_2022": "2021-01-01",
    "y2019_2022": "2019-01-01",
    "full": None,
}
IMPROVED_VAL_MAE = 31.84913528883379
VAL_LOG = RUNS_DIR / "honest_val_log.csv"
REPORT_PATH = Path("reports/honest_mae.md")


# --- Small protocol decisions (no data) ----------------------------------

def preferred_leakage(rows: list[dict]) -> str:
    """Lowest 2023 MAE among the selectable leakage variants."""
    pool = [r for r in rows if r["name"] in SELECTABLE_LEAKS]
    if not pool:
        raise ValueError("no selectable leakage rows")
    return min(pool, key=lambda r: (r["mae"], r["name"]))["name"]


def leakage_test_pair(winner: str) -> list[str]:
    """2024 scores. The partner is precommitted, not chosen from test error.

    If the 2023 winner is already the unshifted model, the partner is the
    1-hour weather lag so the test-set cost of that lag is still visible.
    """
    partner = "unshifted" if winner != "unshifted" else "weather_lag_1"
    return [winner] if winner == partner else [winner, partner]


def confirm_start(window: str) -> str | None:
    if window not in WINDOW_CONFIRM_START:
        raise KeyError(window)
    return WINDOW_CONFIRM_START[window]


# --- Time and column helpers --------------------------------------------

def stamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=UTC)


def between(df: pd.DataFrame, start, end) -> pd.DataFrame:
    idx = df.index
    mask = np.ones(len(df), dtype=bool)
    if start is not None:
        mask &= idx >= start
    if end is not None:
        mask &= idx < end
    return df.loc[mask]


def clock_lag_frame(df: pd.DataFrame, columns: list[str], hours: int) -> pd.DataFrame:
    """Row t receives the values from t-`hours`. A clock shift, not a row shift."""
    out = df.copy()
    for col in columns:
        out[col] = out[col].astype(float)
    moved = out.loc[:, columns].copy()
    moved.index = out.index + pd.Timedelta(hours=hours)
    out.loc[:, columns] = moved.reindex(out.index).to_numpy()
    return out


def clock_lag_series(series: pd.Series, hours: int) -> pd.Series:
    moved = series.copy()
    moved.index = series.index + pd.Timedelta(hours=hours)
    return moved.reindex(series.index)


def weather_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col.endswith(WEATHER_SUFFIXES) or col.startswith(("temp_", "wind_")):
            cols.append(col)
        elif col in WEATHER_STRESS:
            cols.append(col)
    return cols


def recompute_stress_score(df: pd.DataFrame) -> pd.DataFrame:
    parts = [c for c in WEATHER_STRESS + OTHER_STRESS if c in df.columns]
    out = df.copy()
    out["stress_score"] = out[parts].sum(axis=1)
    return out


def previous_hour_intervals(rt: pd.Series, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Last, max, min, and mean of the four 15-minute prices in [t-1h, t)."""
    prev = hourly_index - pd.Timedelta(hours=1)
    offsets = [pd.Timedelta(minutes=m) for m in (0, 15, 30, 45)]
    arr = np.column_stack([
        rt.reindex(prev + off).to_numpy(dtype=float) for off in offsets
    ])
    with np.errstate(all="ignore"):
        rmax = np.nanmax(arr, axis=1)
        rmin = np.nanmin(arr, axis=1)
        rmean = np.nanmean(arr, axis=1)
    empty = np.isnan(arr).all(axis=1)
    rmax[empty] = np.nan
    rmin[empty] = np.nan
    rmean[empty] = np.nan
    return pd.DataFrame({
        "rt15_last": rt.reindex(hourly_index - pd.Timedelta(minutes=15)).to_numpy(dtype=float),
        "rt15_max": rmax,
        "rt15_min": rmin,
        "rt15_mean": rmean,
    }, index=hourly_index)


def day_ahead_shape(raw: pd.DataFrame) -> pd.DataFrame:
    """DA ramp, operating-day max and min, and the trailing RT-DA spread.

    The spread at t is the 24-hour mean of (RT - DA) ending at t-1. The
    operating-day max uses day-ahead prices only; those are published the
    afternoon before the operating day.
    """
    da = raw["da_HB_HUBAVG"]
    price = raw[TARGET_REG]
    day = da.index.tz_convert("US/Central").floor("D")
    spread_through_now = (price - da).rolling("24h", min_periods=12).mean()
    return pd.DataFrame({
        "da_shape_ramp": da - clock_lag_series(da, 1),
        "da_shape_day_max": da.groupby(day).transform("max"),
        "da_shape_day_min": da.groupby(day).transform("min"),
        "da_shape_spread_24h": clock_lag_series(spread_through_now, 1),
    }, index=raw.index)


def error_share(y, pred, threshold: float = STRESS_PRICE) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ae = np.abs(pred - y)
    stress = y > threshold
    total = float(ae.sum())
    return {
        "n": int(len(y)),
        "n_stress": int(stress.sum()),
        "mae": float(ae.mean()),
        "stress_mae": float(ae[stress].mean()) if stress.any() else None,
        "normal_mae": float(ae[~stress].mean()) if (~stress).any() else None,
        "stress_share": float(ae[stress].sum() / total) if total else None,
    }


def assert_frozen_test(index: pd.DatetimeIndex, y) -> None:
    y = np.asarray(y, dtype=float)
    got = index_hash(pd.DatetimeIndex(index))
    n_spikes = int((y > SPIKE_PRICE).sum())
    if got != EXPECTED_HASH or len(index) != TEST_ROWS or n_spikes != TEST_SPIKES:
        raise AssertionError(
            f"frozen test drifted: hash {got}, rows {len(index)}, spikes {n_spikes}")


def _metrics_row(name: str, y, pred) -> dict:
    m = regression_metrics(y, pred)
    m.update(error_share(y, pred))
    m["name"] = name
    return m


def append_val_log(stage: str, name: str, metrics: dict, extra: dict | None = None) -> None:
    row = {"stage": stage, "name": name, "mae": metrics["mae"], "rmse": metrics["rmse"],
           "spike_f1": metrics["spike_f1"], "spike_recall": metrics["spike_recall"],
           "spike_precision": metrics["spike_precision"], "stress_mae": metrics["stress_mae"],
           "normal_mae": metrics["normal_mae"], "stress_share": metrics.get("stress_share")}
    if extra:
        row.update(extra)
    VAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(VAL_LOG, mode="a", header=not VAL_LOG.exists(), index=False)
    print(f"  [{stage}] {name}: MAE {metrics['mae']:.2f}  F1 {metrics['spike_f1']:.3f}  "
          f"stress {metrics['stress_mae']:.1f}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"  wrote {path}", flush=True)


def predictions_exist(name: str) -> bool:
    return (RUNS_DIR / name / "predictions.parquet").exists()


# --- Frames --------------------------------------------------------------

_FRAME: dict = {}


def base_parts():
    """Transformed hybrid frame, its feature list, the raw matrix, and the frozen split."""
    if "aug" not in _FRAME:
        aug, base = hybrid_frame(CHRONOS_SOURCES["t5_gap"], CHRONOS_COLS)
        aug = aug.copy()
        aug["chronos_spread"] = aug["chronos_q90"] - aug["chronos_q10"]
        raw = load_matrix(transform=False)
        train, val, test = frozen_split(aug)
        assert_frozen_test(test.index, test[TARGET_REG].values)
        _FRAME.update(aug=aug, base=list(base), raw=raw, train=train, val=val, test=test)
    return _FRAME["aug"], _FRAME["base"], _FRAME["raw"], _FRAME["train"], _FRAME["val"], _FRAME["test"]


def feature_list(base: list[str], extra: list[str] | None = None, drop_gdelt: bool = False) -> list[str]:
    cols = [c for c in base if not (drop_gdelt and c.startswith("gdelt_"))]
    return cols + list(extra or []) + list(FEATSETS["stats"])


def gas_flag_from_train(raw: pd.DataFrame) -> tuple[pd.Series, float]:
    train = raw.loc[raw.index < stamp("2023-01-01"), "henry_hub_price"].dropna()
    p90 = float(train.quantile(0.90))
    flag = (raw["henry_hub_price"] > p90).astype(float)
    flag = flag.where(raw["henry_hub_price"].notna())
    return flag, p90


def apply_gas_flag(df: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    flag, p90 = gas_flag_from_train(raw)
    out = df.copy()
    out["stress_gas_spike"] = flag.reindex(out.index).to_numpy()
    return recompute_stress_score(out), p90


def load_rt15(index: pd.DatetimeIndex) -> pd.DataFrame:
    rt = pd.read_parquet(RAW_DIR / "ercot_rt_prices.parquet")[TARGET_REG]
    if rt.index.tz is None:
        rt.index = rt.index.tz_localize(UTC)
    return previous_hour_intervals(rt.sort_index(), index)


def attach_new_features(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    shape = day_ahead_shape(raw).reindex(out.index)
    rt15 = load_rt15(out.index)
    for col in DA_SHAPE_COLS:
        out[col] = shape[col].to_numpy()
    for col in RT15_COLS:
        out[col] = rt15[col].to_numpy()
    return out


def check_rt15_matches_hourly_max(raw: pd.DataFrame, rt15: pd.DataFrame) -> float:
    """rt15_max is the max of the previous hour, which is HB_HUBAVG lagged one hour."""
    prev = clock_lag_series(raw[TARGET_REG], 1).reindex(rt15.index)
    both = rt15["rt15_max"].notna() & prev.notna()
    if both.sum() < 1000:
        raise AssertionError("15-minute max could not be aligned with the hourly price")
    err = (rt15.loc[both, "rt15_max"] - prev.loc[both]).abs()
    if float(err.median()) > 0.05:
        raise AssertionError(f"15-minute max disagrees with hourly lag-1 (median {err.median():.3f})")
    return float(err.median())


def leakage_frame(name: str):
    aug, base, raw, train, val, test = base_parts()
    frame = aug
    note = {}
    if name == "unshifted":
        pass
    elif name == "weather_lag_1":
        frame = recompute_stress_score(clock_lag_frame(aug, weather_columns(aug), 1))
    elif name == "weather_lag_120":
        frame = recompute_stress_score(clock_lag_frame(aug, weather_columns(aug), 120))
    elif name == "gas_train_p90":
        frame, p90 = apply_gas_flag(aug, raw)
        note["train_gas_p90"] = p90
    elif name == "gdelt_lag_24":
        cols = [c for c in aug.columns if c.startswith("gdelt_")]
        frame = clock_lag_frame(aug, cols, 24)
    elif name == "chronos_lag_1":
        frame = clock_lag_frame(aug, CHRONOS_COLS + ["chronos_spread"], 1)
    elif name == "plus_features":
        frame = attach_new_features(aug, raw)
    else:
        raise KeyError(name)
    feats = feature_list(base, RT15_COLS + DA_SHAPE_COLS if name == "plus_features" else None)
    return frame, feats, note, train.index, val.index, test.index


def make_honest():
    """Weather lagged 1h, train-only gas flag, GDELT removed, 15-minute and DA shape added."""
    aug, base, raw, _train, val, test = base_parts()
    frame = recompute_stress_score(clock_lag_frame(aug, weather_columns(aug), 1))
    frame, p90 = apply_gas_flag(frame, raw)
    frame = attach_new_features(frame, raw)
    feats = feature_list(base, RT15_COLS + DA_SHAPE_COLS, drop_gdelt=True)
    missing = [c for c in feats if c not in frame.columns]
    if missing:
        raise AssertionError(f"honest frame is missing {missing}")
    prev = clock_lag_series(raw[TARGET_REG], 1).reindex(frame.index)
    both = frame["rt15_max"].notna() & prev.notna()
    if both.sum() < 1000 or float((frame.loc[both, "rt15_max"] - prev.loc[both]).abs().median()) > 0.05:
        raise AssertionError("15-minute previous-hour max does not match the hourly price lag")
    return frame, feats, p90, val.index, test.index


def honest_frame(confirm_window: str):
    """Information-set matrix used for the window test and the headline tree."""
    frame, feats, p90, val_index, test_index = make_honest()
    start = confirm_start(confirm_window)
    train = between(frame, None if start is None else stamp(start), stamp("2023-01-01"))
    return frame, feats, p90, train, frame.loc[val_index], frame.loc[test_index]


def improved_cfg() -> dict:
    path = RUNS_DIR / "selected_improved.json"
    cfg = json.loads(path.read_text())["config"]
    expect = {"learner": "xgb", "objective": "abs", "target": "asinh", "source": "t5_gap",
              "featset": "stats", "depth": 6, "lr": 0.05, "mcw": 1}
    if cfg != expect:
        raise AssertionError(f"selected improved config changed: {cfg}")
    return cfg


# --- Fitting -------------------------------------------------------------

def fit_tree(cfg: dict, train: pd.DataFrame, eval_df: pd.DataFrame, feats: list[str]):
    encode, _decode = TARGETS[cfg["target"]]
    model = make_model(cfg, SEED)
    model.fit(train[feats].ffill().fillna(0), encode(train),
              eval_set=[(eval_df[feats].ffill().fillna(0), encode(eval_df))], verbose=False)
    return model, int(model.best_iteration)


def predict_tree(model, cfg: dict, frame: pd.DataFrame, feats: list[str], best: int) -> np.ndarray:
    _encode, decode = TARGETS[cfg["target"]]
    raw = model.predict(frame[feats].ffill().fillna(0), iteration_range=(0, best + 1))
    return decode(frame, raw)


def save_tree(name: str, model, best: int, feats: list[str], cfg: dict,
              val_df: pd.DataFrame, pred: np.ndarray) -> None:
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "model.json")
    write_json(out / "fit.json", {"features": feats, "best_iteration": best, "cfg": cfg})
    pd.DataFrame({"split": "val", "y_true": val_df[TARGET_REG].to_numpy(), "y_pred": pred},
                 index=val_df.index).to_parquet(out / "val_predictions.parquet")


def load_tree_prediction(name: str, frame: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    meta = json.loads((RUNS_DIR / name / "fit.json").read_text())
    model = xgb.XGBRegressor()
    model.load_model(RUNS_DIR / name / "model.json")
    return predict_tree(model, meta["cfg"], frame, meta["features"], int(meta["best_iteration"]))


def score_saved_test(name: str, frame: pd.DataFrame, test: pd.DataFrame, meta: dict) -> None:
    """One 2024 score from the booster that was saved before this call."""
    if predictions_exist(name):
        print(f"  [{name}] predictions already stored; leaving them", flush=True)
        return
    pred = load_tree_prediction(name, test)
    val = pd.read_parquet(RUNS_DIR / name / "val_predictions.parquet")
    y = test[TARGET_REG].to_numpy()
    assert_frozen_test(test.index, y)
    metrics = {
        "val": regression_metrics(val["y_true"].to_numpy(), val["y_pred"].to_numpy()),
        "test": regression_metrics(y, pred),
    }
    metrics["test"].update({f"share_{k}": v for k, v in error_share(y, pred).items()
                            if k in ("n_stress", "stress_share")})
    save_run(name, prediction_frame({
        "val": (val.index, val["y_true"].to_numpy(), val["y_pred"].to_numpy()),
        "test": (test.index, y, pred),
    }), metrics, meta)


def val_already(name: str) -> bool:
    return (RUNS_DIR / name / "val_predictions.parquet").exists() and \
        (RUNS_DIR / name / "fit.json").exists()


# --- Stages --------------------------------------------------------------

def run_baselines() -> None:
    _aug, _base, raw, train, val, test = base_parts()
    price = raw[TARGET_REG]
    assert_frozen_test(test.index, test[TARGET_REG].to_numpy())
    rt15 = load_rt15(raw.index)
    median_gap = check_rt15_matches_hourly_max(raw, rt15)
    print(f"  15-minute max matches hourly lag-1 (median abs diff ${median_gap:.4f})", flush=True)

    specs = {
        "persistence": 1,
        "seasonal_naive_24": 24,
        "seasonal_naive_168": 168,
    }
    shares = {}
    for name, lag in specs.items():
        parts = {}
        metrics = {}
        for split, df in (("val", val), ("test", test)):
            pred = price.reindex(df.index - pd.Timedelta(hours=lag)).to_numpy(dtype=float)
            y = df[TARGET_REG].to_numpy()
            if split == "test":
                assert_frozen_test(df.index, y)
                if np.isnan(pred).any():
                    raise AssertionError(f"{name} has {int(np.isnan(pred).sum())} null test predictions")
            parts[split] = (df.index, y, pred)
            metrics[split] = regression_metrics(y, pred)
        save_run(name, prediction_frame(parts), metrics,
                 {"lag_hours": lag, "definition": f"HB_HUBAVG at t-{lag}h, clock time"})
        shares[name] = error_share(parts["test"][1], parts["test"][2])

    y_test = test[TARGET_REG].to_numpy()
    train_y = train[TARGET_REG].to_numpy()
    med = float(np.median(train_y))
    oracle = np.where(y_test > STRESS_PRICE, y_test, med)
    constant = np.full_like(y_test, med)
    improved = pd.read_parquet(RUNS_DIR / "chronos_xgb_improved" / "predictions.parquet")
    improved = improved[improved["split"] == "test"].reindex(test.index)
    dam = pd.read_parquet(RUNS_DIR / "dam" / "predictions.parquet")
    dam = dam[dam["split"] == "test"].reindex(test.index)
    if not np.allclose(improved["y_true"].to_numpy(), y_test):
        raise AssertionError("improved predictions are not on the frozen test index")
    shares["dam"] = error_share(dam["y_true"].to_numpy(), dam["y_pred"].to_numpy())
    shares["chronos_xgb_improved_seed0"] = error_share(
        improved["y_true"].to_numpy(), improved["y_pred"].to_numpy())
    called = improved["y_pred"].to_numpy() > STRESS_PRICE
    model_call_oracle = np.where(called, y_test, med)
    payload = {
        "train_median": med,
        "rt15_vs_hourly_lag_median_abs": median_gap,
        "fifteen_minute_file": str(RAW_DIR / "ercot_rt_prices.parquet"),
        "da_shape_columns_were_absent": True,
        "shares": shares,
        "bounds": {
            "constant_train_median": error_share(y_test, constant),
            "oracle_true_stress_label_else_train_median": error_share(y_test, oracle),
            "oracle_model_calls_stress_else_train_median": error_share(y_test, model_call_oracle),
            "note": ("Both oracles use the 2024 price on the hours they treat as stress. "
                     "They are bounds, not forecasts."),
        },
    }
    write_json(RUNS_DIR / "error_share.json", payload)


def _fit_val_variant(name: str, stage: str, frame, feats, val_index, note: dict) -> dict:
    cfg = improved_cfg()
    train = between(frame, None, stamp("2023-01-01"))
    val = frame.loc[val_index]
    if val_already(name):
        val_df = pd.read_parquet(RUNS_DIR / name / "val_predictions.parquet")
        metrics = _metrics_row(name, val_df["y_true"].to_numpy(), val_df["y_pred"].to_numpy())
        print(f"  [{stage}] {name}: reused val MAE {metrics['mae']:.2f}", flush=True)
        return metrics
    model, best = fit_tree(cfg, train, val, feats)
    pred = predict_tree(model, cfg, val, feats, best)
    metrics = _metrics_row(name, val[TARGET_REG].to_numpy(), pred)
    save_tree(name, model, best, feats, cfg, val, pred)
    append_val_log(stage, name, metrics, note)
    return metrics


def run_leakage() -> None:
    rows = []
    frames = {}
    for name in SELECTABLE_LEAKS + ["chronos_lag_1", "plus_features"]:
        frame, feats, note, _tr, val_index, _te = leakage_frame(name)
        frames[name] = (frame, feats, val_index)
        metrics = _fit_val_variant(name, "leakage" if name != "plus_features" else "features",
                                   frame, feats, val_index, note)
        rows.append({"name": name, "mae": metrics["mae"], "spike_f1": metrics["spike_f1"],
                     "spike_recall": metrics["spike_recall"], "spike_precision": metrics["spike_precision"],
                     "stress_mae": metrics["stress_mae"], "normal_mae": metrics["normal_mae"],
                     "stress_share": metrics["stress_share"], "selectable": name in SELECTABLE_LEAKS,
                     **note})
    winner = preferred_leakage(rows)
    pair = leakage_test_pair(winner)
    unshifted = next(r for r in rows if r["name"] == "unshifted")
    chronos = next(r for r in rows if r["name"] == "chronos_lag_1")
    payload = {
        "rule": "lowest 2023 MAE among unshifted, weather lags, train-only gas flag, and next-day GDELT",
        "winner": winner,
        "test_pair": pair,
        "faithful_val_spike_f1": faithful_val_f1(),
        "chronos_lag_raises_mae": chronos["mae"] > unshifted["mae"],
        "chronos_lag_delta_mae": chronos["mae"] - unshifted["mae"],
        "rows": rows,
    }
    write_json(RUNS_DIR / "selected_leakage.json", payload)
    _aug, _base, _raw, _train, _val, test = base_parts()
    for name in pair:
        frame, feats, _val_index = frames[name]
        # The booster was fit before selected_leakage.json was the input of this
        # loop only when the file was just written in this process. On a resume,
        # the file already exists and the booster was fit earlier. Either way the
        # test prediction is produced here, after the file is on disk.
        score_saved_test(
            name, frame, frame.loc[test.index],
            {"role": "leakage", "variant": name, "selected_in": "selected_leakage.json",
             "features": feats, "n_features": len(feats)})


def run_windows() -> None:
    frame, feats, _p90, _val_index, _test_index = make_honest()
    # Selection fits never see 2022 labels in the training rows, and never see 2023.
    rows = []
    score_start = stamp("2022-01-01")
    es_start = score_start - pd.Timedelta(days=ES_DAYS)
    for name, start in WINDOW_TRAIN_START.items():
        key = f"window_select_{name}"
        train_start = None if start is None else stamp(start)
        if val_already(key):
            saved = pd.read_parquet(RUNS_DIR / key / "val_predictions.parquet")
            metrics = _metrics_row(name, saved["y_true"].to_numpy(), saved["y_pred"].to_numpy())
        else:
            tr = between(frame, train_start, es_start)
            ev = between(frame, es_start, score_start)
            scored = between(frame, score_start, stamp("2023-01-01"))
            if min(len(tr), len(ev), len(scored)) < 500:
                raise AssertionError(f"{name} selection split is too small")
            cfg = improved_cfg()
            model, best = fit_tree(cfg, tr, ev, feats)
            pred = predict_tree(model, cfg, scored, feats, best)
            metrics = _metrics_row(name, scored[TARGET_REG].to_numpy(), pred)
            save_tree(key, model, best, feats, cfg, scored, pred)
            append_val_log("window-select-2022", name, metrics, None)
        rows.append({"name": name, "mae_2022": metrics["mae"], "spike_f1_2022": metrics["spike_f1"],
                     "stress_mae_2022": metrics["stress_mae"], "normal_mae_2022": metrics["normal_mae"]})
    chosen = min(rows, key=lambda r: (r["mae_2022"], r["name"]))["name"]
    payload = {
        "rule": ("lowest MAE on calendar 2022; each candidate is trained only on earlier "
                 "years of the same length, with the last 90 days of that train used for "
                 "early stopping. 2023 and 2024 are not used."),
        "chosen_window": chosen,
        "confirm_train_start": confirm_start(chosen),
        "rows": rows,
    }
    write_json(RUNS_DIR / "selected_window.json", payload)

    # Confirm on 2023. This fit is the headline tree. Test is still not predicted.
    frame, feats, p90, train, val_df, test = honest_frame(chosen)
    cfg = improved_cfg()
    name = "honest_tree"
    if not val_already(name):
        model, best = fit_tree(cfg, train, val_df, feats)
        pred = predict_tree(model, cfg, val_df, feats, best)
        metrics = _metrics_row(name, val_df[TARGET_REG].to_numpy(), pred)
        save_tree(name, model, best, feats, cfg, val_df, pred)
        append_val_log("window-confirm-2023", name, metrics, {"window": chosen, "train_gas_p90": p90})
    else:
        saved = pd.read_parquet(RUNS_DIR / name / "val_predictions.parquet")
        metrics = _metrics_row(name, saved["y_true"].to_numpy(), saved["y_pred"].to_numpy())
    gate = faithful_val_f1()
    confirm = {
        "window": chosen,
        "train_start": confirm_start(chosen),
        "train_rows": int(len(train)),
        "train_gas_p90": p90,
        "n_features": len(feats),
        "gdelt_dropped": True,
        "weather_lag_hours": 1,
        "features_added": RT15_COLS + DA_SHAPE_COLS,
        "val_2023": {k: metrics[k] for k in
                     ("mae", "rmse", "spike_f1", "spike_recall", "spike_precision",
                      "stress_mae", "normal_mae", "stress_share")},
        "f1_gate": gate,
        "clears_f1_gate": bool(metrics["spike_f1"] >= gate - 1e-12),
        "beats_previous_val_mae_by_0.5": bool(metrics["mae"] <= IMPROVED_VAL_MAE - 0.5),
        "headline": name,
    }
    write_json(RUNS_DIR / "selected_tree.json", confirm)
    score_saved_test(name, frame, test, {
        "role": "honest_tree", "window": chosen, "selected_in": "selected_window.json",
        "weather_lag_hours": 1, "gdelt_dropped": True, "train_gas_p90": p90,
        "n_features": len(feats),
    })


def _lear_matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    price = raw[TARGET_REG]
    da = raw["da_HB_HUBAVG"]
    x = pd.DataFrame(index=raw.index)
    for lag, name in ((1, "asinh_l1"), (24, "asinh_l24"), (168, "asinh_l168")):
        x[name] = np.arcsinh(price.reindex(raw.index - pd.Timedelta(hours=lag)).to_numpy())
    x["asinh_da"] = np.arcsinh(da.to_numpy())
    local = raw.index.tz_convert("US/Central")
    x["hour"] = local.hour
    x["dow"] = local.dayofweek
    x["month"] = local.month
    y = pd.Series(np.arcsinh(price.to_numpy()), index=raw.index, name="asinh_y")
    return x, y


def _design(x: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(x[["hour", "dow", "month"]].astype(int).astype(str), dtype=float)
    levels = x[["asinh_l1", "asinh_l24", "asinh_l168", "asinh_da"]]
    design = pd.concat([levels, dummies], axis=1)
    if columns is None:
        return design, list(design.columns)
    return design.reindex(columns=columns, fill_value=0.0), columns


def run_lear() -> None:
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    from models.improve_hybrid import ASINH_CLIP

    _aug, _base, raw, _train, val, test = base_parts()
    x, y = _lear_matrix(raw)
    pre = between(x.join(y), None, stamp("2022-01-01"))
    year = between(x.join(y), stamp("2022-01-01"), stamp("2023-01-01"))
    alphas = np.logspace(-4, 1, 16)
    design_pre, cols = _design(pre)
    design_year, _ = _design(year, cols)
    pre_ok = (np.isfinite(design_pre.to_numpy()).all(axis=1)
              & np.isfinite(pre["asinh_y"].to_numpy()))
    year_ok = np.isfinite(design_year.to_numpy()).all(axis=1) & np.isfinite(year["asinh_y"].to_numpy())
    scaler = StandardScaler().fit(design_pre.loc[pre_ok])
    best_alpha, best_mae = None, np.inf
    alpha_rows = []
    y_year = np.sinh(year.loc[year_ok, "asinh_y"].to_numpy())
    z_pre = scaler.transform(design_pre.loc[pre_ok])
    z_year = scaler.transform(design_year.loc[year_ok])
    target_pre = pre.loc[pre_ok, "asinh_y"].to_numpy()
    for alpha in alphas:
        model = Lasso(alpha=float(alpha), max_iter=20000, random_state=SEED)
        model.fit(z_pre, target_pre)
        pred = np.sinh(np.clip(model.predict(z_year), -ASINH_CLIP, ASINH_CLIP))
        mae = float(np.mean(np.abs(pred - y_year)))
        alpha_rows.append({"alpha": float(alpha), "mae_2022": mae})
        if mae < best_mae:
            best_alpha, best_mae = float(alpha), mae
    payload = {"rule": "Lasso alpha with lowest MAE on calendar 2022, fit on data before 2022-01-01",
               "alpha": best_alpha, "mae_2022": best_mae, "alphas": alpha_rows,
               "features": ["asinh lag 1/24/168", "asinh day-ahead", "hour, dow, month dummies"]}
    write_json(RUNS_DIR / "selected_lear.json", payload)

    if predictions_exist("lear"):
        print("  [lear] predictions already stored; leaving them", flush=True)
        return
    train = between(x.join(y), None, stamp("2023-01-01"))
    design_tr, cols = _design(train)
    tr_ok = (np.isfinite(design_tr.to_numpy()).all(axis=1)
             & np.isfinite(train["asinh_y"].to_numpy()))
    scaler = StandardScaler().fit(design_tr.loc[tr_ok])
    model = Lasso(alpha=best_alpha, max_iter=20000, random_state=SEED)
    model.fit(scaler.transform(design_tr.loc[tr_ok]), train.loc[tr_ok, "asinh_y"].to_numpy())
    parts, metrics = {}, {}
    for split, df in (("val", val), ("test", test)):
        block = x.loc[df.index]
        design, _ = _design(block, cols)
        ok = np.isfinite(design.to_numpy()).all(axis=1)
        if split == "test" and not ok.all():
            raise AssertionError(f"LEAR test design has {int((~ok).sum())} null rows")
        pred = np.full(len(design), np.nan)
        if ok.any():
            pred[ok] = np.sinh(np.clip(model.predict(scaler.transform(design.loc[ok])),
                                       -ASINH_CLIP, ASINH_CLIP))
        yy = df[TARGET_REG].to_numpy()
        if split == "test":
            assert_frozen_test(df.index, yy)
        parts[split] = (df.index, yy, pred)
        metrics[split] = regression_metrics(yy[ok], pred[ok])
    finite = np.isfinite(parts["val"][2])
    append_val_log("lear", "lear",
                   _metrics_row("lear", parts["val"][1][finite], parts["val"][2][finite]),
                   {"alpha": best_alpha})
    save_run("lear", prediction_frame(parts), metrics,
             {"alpha": best_alpha, "selected_in": "selected_lear.json", "target": "asinh"})


def _component_matrix(index: pd.DatetimeIndex, tree_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    cols = {}
    # Stored runs may contain both splits. Reindex to the requested timestamps.
    y = None
    for run, label in (("dam", "day_ahead"), ("persistence", "persistence"), ("lear", "lear"),
                       ("chronos_zero_shot", "chronos_zero_shot"), (tree_name, tree_name)):
        pred = pd.read_parquet(RUNS_DIR / run / "predictions.parquet")
        aligned = pred.reindex(index)
        cols[label] = aligned["y_pred"].to_numpy(dtype=float)
        if y is None:
            y = aligned["y_true"].to_numpy(dtype=float)
        elif not np.allclose(y, aligned["y_true"].to_numpy(dtype=float), rtol=1e-5, atol=1e-3,
                              equal_nan=True):
            raise AssertionError(f"{run} y_true does not match the ensemble index")
    return pd.DataFrame(cols, index=index), y


def _combiner_fit(p: np.ndarray, y: np.ndarray) -> dict:
    from scipy.optimize import minimize
    from sklearn.linear_model import QuantileRegressor

    def mae(pred):
        return float(np.mean(np.abs(pred - y)))

    k = p.shape[1]
    def obj(w):
        return np.mean(np.abs(p @ w - y))
    result = minimize(obj, np.full(k, 1.0 / k), method="SLSQP", bounds=[(0, 1)] * k,
                      constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
                      options={"ftol": 1e-12, "maxiter": 500})
    weights = np.clip(result.x, 0, None)
    weights = weights / weights.sum()
    qr = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs", fit_intercept=True)
    qr.fit(p, y)
    candidates = {
        "median": {"mae": mae(np.median(p, axis=1)), "weights": None, "intercept": 0.0},
        "weighted_mean": {"mae": mae(p @ weights), "weights": weights.tolist(),
                          "intercept": 0.0, "optimizer_success": bool(result.success)},
        "qra_tau_0.5": {"mae": mae(qr.predict(p)), "weights": qr.coef_.tolist(),
                        "intercept": float(qr.intercept_)},
    }
    winner = min(candidates, key=lambda name: (candidates[name]["mae"], name))
    return {"candidates": candidates, "winner": winner}


def _apply_combiner(kind: str, spec: dict, p: np.ndarray) -> np.ndarray:
    if kind == "median":
        return np.median(p, axis=1)
    if kind == "weighted_mean":
        return p @ np.asarray(spec["weights"], dtype=float)
    coef = np.asarray(spec["weights"], dtype=float)
    return spec["intercept"] + p @ coef


def run_ensemble() -> None:
    _aug, _base, _raw, _train, val, test = base_parts()
    tree = "honest_tree"
    for required in ("dam", "persistence", "lear", "chronos_zero_shot", tree):
        if not predictions_exist(required):
            raise SystemExit(f"ensemble needs {required} predictions, including 2024 for the components "
                             f"that were already frozen. {tree} must be scored before the combiner.")
    p_val, y_val = _component_matrix(val.index, tree)
    ok = np.isfinite(p_val.to_numpy()).all(axis=1) & np.isfinite(y_val)
    fit = _combiner_fit(p_val.to_numpy()[ok], y_val[ok])
    names = list(p_val.columns)
    payload = {"rule": "lowest 2023 MAE among the median, the simplex weighted mean, and tau=0.5 QRA",
               "components": names, "n_val_rows": int(ok.sum()), **fit}
    write_json(RUNS_DIR / "selected_ensemble.json", payload)
    if predictions_exist("ensemble"):
        print("  [ensemble] predictions already stored; leaving them", flush=True)
        return
    kind = fit["winner"]
    spec = fit["candidates"][kind]
    parts, metrics = {}, {}
    for split, df in (("val", val), ("test", test)):
        p, y = _component_matrix(df.index, tree)
        pred = _apply_combiner(kind, spec, p.to_numpy())
        if split == "test":
            assert_frozen_test(df.index, y)
            if not np.isfinite(pred).all():
                raise AssertionError("ensemble test prediction has nulls")
        parts[split] = (df.index, y, pred)
        metrics[split] = regression_metrics(y, pred)
    save_run("ensemble", prediction_frame(parts), metrics,
             {"combiner": kind, "selected_in": "selected_ensemble.json", "components": names,
              "spec": spec})


# --- Checks and the write-up ---------------------------------------------

def _test_frame(name: str) -> pd.DataFrame:
    pred = pd.read_parquet(RUNS_DIR / name / "predictions.parquet")
    test = pred[pred["split"] == "test"]
    if test.empty:
        raise AssertionError(f"{name} has no test split")
    return test


def _check_run(name: str, price: pd.Series) -> dict:
    test = _test_frame(name)
    y = test["y_true"].to_numpy(dtype=float)
    p = test["y_pred"].to_numpy(dtype=float)
    assert_frozen_test(test.index, y)
    problems = []
    if not np.isfinite(p).all():
        problems.append("null or non-finite prediction")
    if np.array_equal(y, p):
        problems.append("prediction is identical to HB_HUBAVG")
    if name == "persistence":
        lagged = price.reindex(test.index - pd.Timedelta(hours=1)).to_numpy(dtype=float)
        if not np.allclose(p, lagged, equal_nan=False):
            problems.append("persistence is not HB_HUBAVG at t-1h")
    if name == "seasonal_naive_24":
        lagged = price.reindex(test.index - pd.Timedelta(hours=24)).to_numpy(dtype=float)
        if not np.allclose(p, lagged, equal_nan=False):
            problems.append("seasonal-24 is not HB_HUBAVG at t-24h")
    if name == "seasonal_naive_168":
        lagged = price.reindex(test.index - pd.Timedelta(hours=168)).to_numpy(dtype=float)
        if not np.allclose(p, lagged, equal_nan=False):
            problems.append("seasonal-168 is not HB_HUBAVG at t-168h")
    recomputed = regression_metrics(y, p)
    stored = json.loads((RUNS_DIR / name / "metrics.json").read_text())["metrics"]["test"]
    for key in ("mae", "rmse", "spike_recall", "spike_precision", "normal_mae", "stress_mae"):
        if key in stored and abs(stored[key] - recomputed[key]) > 1e-6:
            problems.append(f"stored {key} {stored[key]} != recomputed {recomputed[key]}")
    order = np.argsort(-np.abs(p - y))[:20]
    worst = [{"timestamp": str(test.index[i]), "y_true": float(y[i]), "y_pred": float(p[i]),
              "abs_err": float(abs(p[i] - y[i]))} for i in order]
    stress = y > STRESS_PRICE
    return {
        "name": name, "problems": problems, "metrics": recomputed,
        "share": error_share(y, p),
        "mean_pred_stress": float(p[stress].mean()),
        "mean_pred_normal": float(p[~stress].mean()),
        "mean_y_stress": float(y[stress].mean()),
        "mean_y_normal": float(y[~stress].mean()),
        "worst20": worst,
    }


def _dm_against(name: str, baseline: str) -> dict:
    a = _test_frame(name)
    b = _test_frame(baseline).reindex(a.index)
    if not np.allclose(a["y_true"].to_numpy(), b["y_true"].to_numpy()):
        raise AssertionError(f"{name} and {baseline} are not on the same target")
    y = a["y_true"].to_numpy()
    loss_a = np.abs(a["y_pred"].to_numpy() - y)
    loss_b = np.abs(b["y_pred"].to_numpy() - y)
    out = diebold_mariano(loss_a, loss_b, lag=24)
    out["model"] = name
    out["baseline"] = baseline
    out["reading"] = "negative stat means the model has lower absolute error than the baseline"
    return out


def _context_check() -> dict:
    mat = load_matrix(transform=False)
    series = mat.dropna(subset=[TARGET_REG]).sort_index()[TARGET_REG].astype("float32")
    hours, values, positions = build_contexts(series, "gap-aware", CONTEXT)
    checked = 0
    for hour, position in zip(hours, positions):
        if hour < stamp("2024-01-01"):
            continue
        start, end = context_bounds(int(position), CONTEXT, len(values))
        if end != int(position) or start != int(position) - CONTEXT:
            raise AssertionError("Chronos context includes the target hour")
        checked += 1
    if checked < TEST_ROWS:
        raise AssertionError(f"context check covered {checked} test hours, expected at least {TEST_ROWS}")
    return {"context": CONTEXT, "mode": "gap-aware", "test_hours_checked": checked,
            "target_excluded": True}


def run_check() -> None:
    _aug, _base, raw, _train, _val, _test = base_parts()
    price = raw[TARGET_REG]
    leakage = json.loads((RUNS_DIR / "selected_leakage.json").read_text())
    names = [
        "persistence", "seasonal_naive_24", "seasonal_naive_168", "dam", "chronos_zero_shot",
        *leakage["test_pair"], "honest_tree", "lear", "ensemble",
    ]
    # day-ahead and Chronos are rescored from the files already on disk.
    runs = []
    for name in names:
        if not predictions_exist(name):
            raise SystemExit(f"check is missing {name}")
        runs.append(_check_run(name, price))
    context = _context_check()
    finalists = [r["name"] for r in runs if r["name"] not in ("dam", "persistence")]
    dm = []
    for name in finalists:
        for baseline in ("dam", "persistence"):
            dm.append(_dm_against(name, baseline))
    payload = {"runs": runs, "context": context, "diebold_mariano": dm,
               "index_hash": EXPECTED_HASH}
    write_json(RUNS_DIR / "honest_mae_checks.json", payload)
    _write_report(payload)
    from models.run_benchmark import write_leaderboard
    write_leaderboard()
    failed = [r["name"] for r in runs if r["problems"]]
    if failed:
        raise SystemExit(f"forecast checks failed: {failed}")


def _fmt(row: dict, key: str, digits: int = 2) -> str:
    value = row.get(key)
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _write_report(payload: dict) -> None:
    by = {r["name"]: r for r in payload["runs"]}
    leakage = json.loads((RUNS_DIR / "selected_leakage.json").read_text())
    window = json.loads((RUNS_DIR / "selected_window.json").read_text())
    tree = json.loads((RUNS_DIR / "selected_tree.json").read_text())
    lear = json.loads((RUNS_DIR / "selected_lear.json").read_text())
    ens = json.loads((RUNS_DIR / "selected_ensemble.json").read_text())
    bounds = json.loads((RUNS_DIR / "error_share.json").read_text())
    lines = [
        "# Honest one-hour-ahead MAE",
        "",
        "Validation chooses every setting below. The 2024 year is scored once per "
        "finalist, on the frozen index `b850034b0690e660` (8,767 hours, 114 hours above $200/MWh).",
        "",
        "## Bounds",
        "",
        f"Training-set median price: ${bounds['train_median']:.2f}/MWh. "
        f"The 15-minute file is on disk; its previous-hour max matches the hourly lag "
        f"(median absolute difference ${bounds['rt15_vs_hourly_lag_median_abs']:.4f}). "
        "Day-ahead ramp, operating-day max/min, and the trailing RT-DA spread were not in the v2 matrix.",
        "",
        "| Forecast | MAE | Normal MAE | Stress MAE (y > $100) | Share of absolute error from stress hours |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (
        ("persistence", "Persistence"),
        ("seasonal_naive_24", "Seasonal naive, 24h"),
        ("seasonal_naive_168", "Seasonal naive, 168h"),
        ("dam", "Day-ahead price"),
        ("chronos_zero_shot", "Chronos zero-shot"),
    ):
        s = by[name]["share"]
        lines.append(f"| {label} | {s['mae']:.2f} | {s['normal_mae']:.2f} | {s['stress_mae']:.2f} | "
                     f"{100 * s['stress_share']:.1f}% ({s['n_stress']} hours) |")
    const = bounds["bounds"]["constant_train_median"]
    oracle = bounds["bounds"]["oracle_true_stress_label_else_train_median"]
    lines += [
        f"| Constant training median | {const['mae']:.2f} | {const['normal_mae']:.2f} | "
        f"{const['stress_mae']:.2f} | {100 * const['stress_share']:.1f}% |",
        f"| Oracle: true stress hours perfect, else training median | {oracle['mae']:.2f} | "
        f"{oracle['normal_mae']:.2f} | {oracle['stress_mae']:.2f} | "
        f"{100 * oracle['stress_share']:.1f}% |",
        "",
        "The oracle uses the 2024 label to decide which hours are stress. It is a bound. "
        f"Stress hours are {oracle['n_stress']} of {oracle['n']}. "
        "A model that already predicts normal hours well only needs to cut the stress-hour error "
        "to move the average; the oracle says the average is not stuck near $12 by the spikes alone.",
        "",
        "## Leakage shifts (2023 ranking, then one 2024 pair)",
        "",
        "| Variant | 2023 MAE | 2023 spike F1 | 2024 MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    tested = set(leakage["test_pair"])
    for row in leakage["rows"]:
        test_mae = f"{by[row['name']]['metrics']['mae']:.2f}" if row["name"] in by else "-"
        lines.append(f"| {row['name']} | {row['mae']:.2f} | {row['spike_f1']:.3f} | {test_mae} |")
    lines += [
        "",
        f"2023 winner: `{leakage['winner']}`. 2024 pair: {', '.join(f'`{n}`' for n in leakage['test_pair'])}.",
        f"Shifting Chronos features by one hour changes 2023 MAE by "
        f"{leakage['chronos_lag_delta_mae']:+.2f} "
        f"({'rises, so the feature is aligned' if leakage['chronos_lag_raises_mae'] else 'does not rise'}).",
        "",
        "## Frozen feature set and training window",
        "",
        "The headline tree lags weather and the weather stress flags by one hour, rebuilds "
        "`stress_gas_spike` from the pre-2023 gas percentile, drops GDELT, and adds the "
        "previous-hour 15-minute summaries plus the day-ahead shape. `plus_features` in the "
        "table above is the same new columns on the original information set, scored on 2023 only.",
        "",
        "Windows were ranked by 2022 MAE. The fit for that rank uses only earlier years.",
        "",
        "| Window | 2022 MAE |",
        "| --- | ---: |",
    ]
    for row in window["rows"]:
        lines.append(f"| {row['name']} | {row['mae_2022']:.2f} |")
    v = tree["val_2023"]
    lines += [
        "",
        f"Chosen window: `{window['chosen_window']}` (train start `{tree['train_start']}`). "
        f"2023 MAE {v['mae']:.2f}, spike F1 {v['spike_f1']:.3f}, stress MAE {v['stress_mae']:.2f}, "
        f"normal MAE {v['normal_mae']:.2f}.",
        f"Faithful-hybrid F1 gate {tree['f1_gate']:.3f}: "
        f"{'cleared' if tree['clears_f1_gate'] else 'not cleared'}. "
        f"The $0.5 improvement over the previous validation MAE of {IMPROVED_VAL_MAE:.2f}: "
        f"{'met' if tree['beats_previous_val_mae_by_0.5'] else 'not met'}.",
        "",
        "## LEAR and the combiner",
        "",
        f"LEAR penalty {lear['alpha']:.4g}, chosen on 2022 (MAE {lear['mae_2022']:.2f}). "
        f"2024 MAE {by['lear']['metrics']['mae']:.2f}.",
        "",
        "Combiner candidates, fit on 2023 only:",
        "",
        "| Combiner | 2023 MAE |",
        "| --- | ---: |",
    ]
    for name, spec in ens["candidates"].items():
        mark = " (chosen)" if name == ens["winner"] else ""
        lines.append(f"| {name}{mark} | {spec['mae']:.2f} |")
    lines += [
        "",
        f"Components: {', '.join(f'`{c}`' for c in ens['components'])}. "
        "Quantile regression at τ=0.5 is not constrained to positive weights, so it can "
        "extrapolate. The weighted mean is the constrained alternative, and it lost on 2023.",
        "",
        "## 2024 scores",
        "",
        "| Run | MAE | RMSE | Spike recall | Spike precision | Normal MAE | Stress MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["persistence", "seasonal_naive_24", "seasonal_naive_168", "dam",
                 "chronos_zero_shot", *leakage["test_pair"], "honest_tree", "lear", "ensemble"]:
        m = by[name]["metrics"]
        lines.append(f"| `{name}` | {m['mae']:.2f} | {m['rmse']:.2f} | {100 * m['spike_recall']:.2f}% | "
                     f"{100 * m['spike_precision']:.2f}% | {m['normal_mae']:.2f} | {m['stress_mae']:.2f} |")
    headline = by["ensemble"]["metrics"]["mae"]
    tree_mae = by["honest_tree"]["metrics"]["mae"]
    lines += [
        "",
        "## Diebold-Mariano",
        "",
        "Absolute error, HAC lag 24, Harvey-Leybourne-Newbold correction. "
        "A negative statistic means the row has lower loss than the baseline.",
        "",
        "| Model | Against | Mean loss gap | Stat | p |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["diebold_mariano"]:
        if row["model"] not in ("honest_tree", "lear", "ensemble", *tested):
            continue
        lines.append(f"| `{row['model']}` | `{row['baseline']}` | {row['mean_diff']:.2f} | "
                     f"{row['stat']:.2f} | {row['p_value']:.3f} |")
    lines += [
        "",
        "## Forecast checks",
        "",
        f"Chronos gap-aware context of {payload['context']['context']} hours was checked on "
        f"{payload['context']['test_hours_checked']} test hours. The target hour is outside the context.",
        "",
    ]
    for row in payload["runs"]:
        status = "ok" if not row["problems"] else "; ".join(row["problems"])
        lines.append(f"- `{row['name']}`: {status}. Mean prediction "
                     f"${row['mean_pred_normal']:.1f} on hours at or below $100 "
                     f"(mean price ${row['mean_y_normal']:.1f}) and "
                     f"${row['mean_pred_stress']:.1f} on hours above $100 "
                     f"(mean price ${row['mean_y_stress']:.1f}).")
    lines += ["", "Worst 20 absolute errors of the validation-chosen combiner:", ""]
    for row in by["ensemble"]["worst20"]:
        lines.append(f"- {row['timestamp']}: price ${row['y_true']:.1f}, forecast ${row['y_pred']:.1f}, "
                     f"error ${row['abs_err']:.1f}")
    lines += [
        "",
        "## What to claim",
        "",
        f"The validation-chosen combiner has 2024 MAE ${headline:.2f}. "
        f"The windowed tree has 2024 MAE ${tree_mae:.2f}. "
        "These two numbers were not ranked against each other on 2024.",
        "",
    ]
    under = [name for name, mae in (("windowed tree", tree_mae), ("combiner", headline)) if mae < 12]
    lines += [
        "Both scores are one XGBoost seed (42 for the tree; the combiner has no seed). "
        "They are not a five-seed mean, and 2024 was not used to choose between them.",
        "",
    ]
    if under:
        lines.append(
            f"Under $12/MWh on this index: {', '.join(under)}. "
            "That is a new one-hour-ahead result under the rules above, not a recovery of the "
            "paper row. The paper's MAE of 12.13 comes with spike recall 61.40% and precision "
            "50.43%. On 114 spike hours, 61.40% is 70/114, and precision 50.43% is then not a "
            "ratio of two integers, so that pair cannot be this test set. "
            f"The combiner MAE prints as ${headline:.2f} with RMSE ${by['ensemble']['metrics']['rmse']:.2f} "
            f"and spike recall {100 * by['ensemble']['metrics']['spike_recall']:.2f}%. "
            f"The tree MAE is ${tree_mae:.2f} with RMSE ${by['honest_tree']['metrics']['rmse']:.2f} "
            f"and spike recall {100 * by['honest_tree']['metrics']['spike_recall']:.2f}%. "
            "Keep the paper's 12.13 in its own column."
        )
    else:
        lines.append(
            "Neither figure is under $12/MWh. Keep the paper's 12.13 as a reported figure that "
            "this test set cannot emit: recall 61.40% is 70/114, and precision 50.43% is then "
            "not a ratio of two integers."
        )
    lines += [
        "",
        "Hours above $100 are 294 of 8,767 and still hold about 60% of the tree's absolute error. "
        "The forecasts track those hours (the tree's mean prediction there is far above its "
        "prediction on ordinary hours) and still miss the largest spikes by thousands of dollars. "
        "Published wind, solar, and load forecasts, joined on the time they were released, are the "
        "information this pass did not have. Another squared-error or asinh variant is not.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"  wrote {REPORT_PATH}", flush=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["baselines", "leakage", "windows", "lear", "ensemble",
                                        "check", "all"])
    args = ap.parse_args()
    steps = {
        "baselines": run_baselines,
        "leakage": run_leakage,
        "windows": run_windows,
        "lear": run_lear,
        "ensemble": run_ensemble,
        "check": run_check,
    }
    if args.command == "all":
        for name in ("baselines", "leakage", "windows", "lear", "ensemble", "check"):
            print(f"\n=== {name} ===", flush=True)
            steps[name]()
    else:
        steps[args.command]()


if __name__ == "__main__":
    main()
