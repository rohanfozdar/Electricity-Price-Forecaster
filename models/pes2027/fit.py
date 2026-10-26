"""Shared tree fit used by E1–E8. Same hyperparameters as the original primary pair."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models.pes2027.common import SEEDS, TARGET, write_json, write_parquet
from models.pes2027.protocol import require_frozen
from models.run_pit_ablation import FROZEN_DEPTH, FROZEN_LR, decode_target, fit_xgb
from models.run_pit_verified import prepare
from src.features.pit_sets import feature_subsets


def split_index(index: pd.DatetimeIndex, train_end, val_end):
    train = index[index < train_end]
    val = index[(index >= train_end) & (index < val_end)]
    test = index[index >= val_end]
    return train, val, test


def baseline_columns(frame: pd.DataFrame) -> list[str]:
    return feature_subsets(list(frame.columns))["xgb_exog"]


def fit_one(frame: pd.DataFrame, columns: list[str], train, val, test, seed: int) -> dict:
    t0 = time.time()
    scored, best, device, usable, _ = _fit(frame, columns, train, val, test, seed)
    return {
        "seed": seed, "best_iteration": best, "device": device, "n_columns": len(usable),
        "seconds": time.time() - t0, "usable": usable,
        "val": scored["val"], "test": scored["test"],
    }


def _fit(frame, columns, train, val, test, seed):
    return __import__("models.run_pit_verified", fromlist=["fit_predict"]).fit_predict(
        frame, columns, train, val, test, seed, "asinh",
    )


def save_pred(path, index, y, pred, meta: dict) -> None:
    write_parquet(path, pd.DataFrame({"y_true": y, "y_pred": pred}, index=index))
    write_json(path.with_suffix(".json"), meta)


def seed_mean(preds: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(preds), axis=0)


def require() -> dict:
    return require_frozen()
