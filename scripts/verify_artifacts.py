"""Recompute the published 2024 scores from artifacts/honest.

Run from the repository root:

    python scripts/verify_artifacts.py

This reads the committed prediction files. It does not fit a model.
`--replay-tree` also loads the saved booster and checks it against those
predictions. That path needs the local feature matrix and Chronos cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.run_benchmark import SPIKE_PRICE, regression_metrics  # noqa: E402

ART = ROOT / "artifacts" / "honest"
PRED = ART / "predictions"

# Two-decimal figures as printed in reports/honest_mae.md.
EXPECTED = {
    "honest_tree": {"mae": 11.56, "rmse": 99.96, "recall": 50.00, "precision": 73.08},
    "lear": {"mae": 13.39, "rmse": 97.92, "recall": 34.21, "precision": 61.90},
    "ensemble": {"mae": 12.13, "rmse": 96.23, "recall": 53.51, "precision": 64.21},
    "persistence": {"mae": 17.04, "rmse": 125.44, "recall": 41.23, "precision": 41.23},
    "dam": {"mae": 17.51, "rmse": 112.85, "recall": 28.95, "precision": 48.53},
    "chronos_zero_shot": {"mae": 14.54, "rmse": 113.62, "recall": 41.23, "precision": 61.04},
    "unshifted": {"mae": 12.95, "rmse": 104.13, "recall": 42.11, "precision": 69.57},
    "weather_lag_1": {"mae": 12.96, "rmse": 104.38, "recall": 43.86, "precision": 69.44},
}


def test_rows(name: str) -> pd.DataFrame:
    path = PRED / f"{name}.parquet"
    frame = pd.read_parquet(path)
    extra = [c for c in frame.columns if c not in ("split", "y_true", "y_pred")]
    if extra:
        raise SystemExit(f"{path} has unexpected columns: {extra}")
    test = frame.loc[frame["split"] == "test"].copy()
    if len(test) != 8767:
        raise SystemExit(f"{name} test rows: {len(test)}, expected 8767")
    spikes = int((test["y_true"].to_numpy() > SPIKE_PRICE).sum())
    if spikes != 114:
        raise SystemExit(f"{name} spike hours: {spikes}, expected 114")
    return test


def printed(metrics: dict) -> dict:
    return {
        "mae": round(metrics["mae"], 2),
        "rmse": round(metrics["rmse"], 2),
        "recall": round(100 * metrics["spike_recall"], 2),
        "precision": round(100 * metrics["spike_precision"], 2),
    }


def check_metrics() -> None:
    print(f"Spike threshold: HB_HUBAVG > ${SPIKE_PRICE:.0f}/MWh")
    failed = False
    for name, expect in EXPECTED.items():
        metrics = regression_metrics(
            test_rows(name)["y_true"].to_numpy(),
            test_rows(name)["y_pred"].to_numpy(),
        )
        got = printed(metrics)
        ok = got == expect
        flag = "ok" if ok else "MISMATCH"
        print(
            f"  {name:22} MAE {got['mae']:7.2f}  RMSE {got['rmse']:7.2f}  "
            f"recall {got['recall']:6.2f}%  precision {got['precision']:6.2f}%  {flag}"
        )
        if not ok:
            print(f"    expected {expect}")
            print(f"    exact    mae {metrics['mae']:.6f} rmse {metrics['rmse']:.6f}")
            failed = True
    if failed:
        raise SystemExit("a printed metric does not match reports/honest_mae.md")


def check_stack() -> None:
    spec = json.loads((ART / "stack_coefficients.json").read_text())
    saved = test_rows("ensemble")
    columns = []
    for name in spec["components"]:
        filename = Path(spec["component_files"][name]).name.replace(".parquet", "")
        part = test_rows(filename)["y_pred"].reindex(saved.index)
        if part.isna().any():
            raise SystemExit(f"{name} does not cover the stack index")
        columns.append(part.to_numpy(dtype=float))
    rebuilt = spec["intercept"] + np.column_stack(columns) @ np.asarray(spec["weights"], dtype=float)
    err = np.max(np.abs(rebuilt - saved["y_pred"].to_numpy(dtype=float)))
    print(f"  stack rebuild max abs error ${err:.6f}/MWh")
    if err > 1e-4:
        raise SystemExit("rebuilt stack forecast does not match predictions/ensemble.parquet")


def replay_tree() -> None:
    import xgboost as xgb

    from models.honest_mae import make_honest, predict_tree

    frame, _feats, _p90, _val_index, test_index = make_honest()
    meta = json.loads((ART / "honest_tree_fit.json").read_text())
    model = xgb.XGBRegressor()
    model.load_model(ART / "honest_tree_model.json")
    pred = predict_tree(model, meta["cfg"], frame.loc[test_index], meta["features"], int(meta["best_iteration"]))
    saved = test_rows("honest_tree")["y_pred"].reindex(test_index)
    err = np.max(np.abs(pred - saved.to_numpy(dtype=float)))
    print(f"  booster replay max abs error ${err:.6f}/MWh on {len(test_index)} hours")
    if not np.isfinite(err) or err > 1e-3:
        raise SystemExit("reloaded booster does not match predictions/honest_tree.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-tree", action="store_true",
                        help="load the booster and compare it with the saved 2024 forecasts")
    args = parser.parse_args()
    check_metrics()
    check_stack()
    if args.replay_tree:
        replay_tree()
    print("verified")


if __name__ == "__main__":
    main()
