"""
Chronos + XGBoost hybrid - the best-performing model in this project.

Amazon's pretrained Chronos foundation model produces a one-hour-ahead
forecast from the raw HB_HUBAVG history alone. That single forecast is then
appended as one extra column to the 87 engineered features, and XGBoost is
trained on the augmented matrix:

    Raw price series --> Chronos (512h context) --> chronos_pred
                                                        |
    87 engineered features -----------------------------+
                                                        v
                        XGBoost([87 features + chronos_pred] -> price)

This is a script form of notebooks/10_chronos (1).ipynb, so the headline
result is reproducible without opening a notebook.

RUNTIME WARNING
---------------
The Chronos backfill is the single slowest step in this repository: one
forward pass per hourly observation, roughly 77,400 of them. Expect
~30-60 min on an NVIDIA GPU and *many hours* on CPU. A GPU is strongly
recommended.

The backfill checkpoints to data/features/chronos_pred.parquet every
--checkpoint-every batches, so an interrupted run resumes where it stopped
instead of starting over. Re-running the script is always safe.

Usage:
    python models/train_chronos_hybrid.py                  # full run, auto device
    python models/train_chronos_hybrid.py --limit 200      # smoke test
    python models/train_chronos_hybrid.py --device cpu     # force device
    python models/train_chronos_hybrid.py --skip-backfill  # reuse checkpoint, train only

Requires the `foundation` extra:
    pip install -e ".[foundation]"

Outputs:
    data/features/chronos_pred.parquet                     (resumable checkpoint)
    models/artifacts/chronos_hybrid_regressor.json
    models/artifacts/chronos_hybrid_report.json
    models/artifacts/chronos_hybrid_feature_importance.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable when run as `python models/train_chronos_hybrid.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import FEATURES_DIR
from models.utils import (
    TARGET_REG,
    apply_feature_transforms,
    regression_report,
    select_enhanced_features,
)

MODELS_DIR = Path("models/artifacts")
V2_MATRIX = FEATURES_DIR / "feature_matrix_engineered_v2.parquet"
CHRONOS_PRED_PATH = FEATURES_DIR / "chronos_pred.parquet"

# Match notebooks/10_chronos (1).ipynb exactly.
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 1
MODEL_NAME = "amazon/chronos-t5-small"


def resolve_device(requested: str = "auto") -> tuple[str, "object"]:
    """Pick cuda -> mps -> cpu. bfloat16 is only reliable on CUDA."""
    import torch

    if requested != "auto":
        dtype = torch.bfloat16 if requested == "cuda" else torch.float32
        return requested, dtype
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _release(device: str) -> None:
    """Free cached accelerator memory between batches.

    Chronos builds a KV cache proportional to batch_size x context_length.
    Without this, MPS in particular accumulates allocations across batches and
    OOMs partway through a long backfill.
    """
    import torch

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def backfill_chronos(target_series: pd.Series, device: str, batch_size: int,
                     checkpoint_every: int, limit: int | None) -> pd.Series:
    """One-hour-ahead Chronos forecast for every position with 512h of history.

    Resumes from CHRONOS_PRED_PATH if it exists, so a killed run costs only
    the batches that were in flight.
    """
    import torch
    from chronos import ChronosPipeline

    try:
        from tqdm.auto import tqdm
    except ImportError:  # tqdm is optional
        def tqdm(x, **kwargs):
            return x

    done = pd.Series(dtype="float64")
    if CHRONOS_PRED_PATH.exists():
        done = pd.read_parquet(CHRONOS_PRED_PATH)["chronos_pred"]
        print(f"  Resuming: {len(done):,} predictions already checkpointed")

    positions = np.arange(CONTEXT_LENGTH, len(target_series))
    wanted_index = target_series.index[positions]
    todo_mask = ~wanted_index.isin(done.index)
    positions = positions[todo_mask]
    if limit is not None:
        positions = positions[:limit]

    if len(positions) == 0:
        print("  Nothing to do - checkpoint already complete.")
        return done.sort_index()

    print(f"  {len(positions):,} forward passes remaining on {device}")
    if device == "cpu":
        print("  WARNING: CPU backfill is very slow. Consider a GPU machine.")

    _, dtype = resolve_device(device)
    pipe = ChronosPipeline.from_pretrained(
        MODEL_NAME, device_map=device, torch_dtype=dtype,
    )

    values, index = [], []
    n_batches = 0
    for start in tqdm(range(0, len(positions), batch_size), desc="Chronos backfill"):
        batch_positions = positions[start:start + batch_size]
        contexts = [
            torch.tensor(
                target_series.iloc[p - CONTEXT_LENGTH:p].to_numpy(),
                dtype=torch.float32,
            )
            for p in batch_positions
        ]
        try:
            with torch.inference_mode():
                samples = pipe.predict(contexts, prediction_length=PREDICTION_LENGTH)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _checkpoint(done, values, index)
                raise SystemExit(
                    f"Out of memory at batch size {batch_size}. Progress was checkpointed to\n"
                    f"  {CHRONOS_PRED_PATH}\n"
                    f"Re-run with a smaller batch, e.g. --batch-size {max(8, batch_size // 2)}; "
                    "it will resume from the checkpoint."
                ) from e
            raise
        if samples.ndim != 3:
            raise ValueError(
                f"Expected Chronos shape (batch, samples, horizon), got {tuple(samples.shape)}"
            )
        # Median over sample paths at the 1-hour horizon.
        median = samples[:, :, 0].median(dim=1).values.detach().cpu().numpy()
        values.extend(median.tolist())
        index.extend(target_series.index[batch_positions])
        del samples, contexts
        _release(device)

        n_batches += 1
        if n_batches % checkpoint_every == 0:
            _checkpoint(done, values, index)

    return _checkpoint(done, values, index)


def _checkpoint(done: pd.Series, values: list, index: list) -> pd.Series:
    """Merge new predictions into the checkpoint file and return the union."""
    fresh = pd.Series(values, index=pd.DatetimeIndex(index), name="chronos_pred")
    merged = pd.concat([done, fresh]) if len(done) else fresh
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.name = "chronos_pred"
    CHRONOS_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_frame().to_parquet(CHRONOS_PRED_PATH)
    return merged


def train_hybrid(mat: pd.DataFrame, features: list[str],
                 chronos_pred: pd.Series) -> dict:
    """XGBoost over the 87 engineered features + chronos_pred.

    Hyperparameters are identical to notebook cell 12 so the reported
    numbers stay comparable.
    """
    import xgboost as xgb

    augmented = mat.loc[mat.index.intersection(chronos_pred.index)].copy()
    augmented["chronos_pred"] = chronos_pred.reindex(augmented.index).values
    features_aug = features + ["chronos_pred"]

    train = augmented[augmented.index < "2023-01-01"]
    val = augmented[(augmented.index >= "2023-01-01") & (augmented.index < "2024-01-01")]
    test = augmented[augmented.index >= "2024-01-01"]
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            raise SystemExit(
                f"The {name} split is empty ({len(augmented):,} rows with Chronos "
                "predictions). This usually means the backfill only partially ran - "
                "re-run without --limit to complete it."
            )
    print(f"  {len(features_aug)} augmented features  "
          f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

    X_train = train[features_aug].ffill().fillna(0)
    X_val = val[features_aug].ffill().fillna(0)
    X_test = test[features_aug].ffill().fillna(0)

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", tree_method="hist",
        early_stopping_rounds=30, eval_metric="rmse",
        n_jobs=-1, random_state=42,
    )
    model.fit(X_train, train[TARGET_REG],
              eval_set=[(X_val, val[TARGET_REG])], verbose=False)

    reports = {
        "train": regression_report(train[TARGET_REG].values, model.predict(X_train),
                                   "chronos-hybrid-train"),
        "val": regression_report(val[TARGET_REG].values, model.predict(X_val),
                                 "chronos-hybrid-val"),
        "test": regression_report(test[TARGET_REG].values, model.predict(X_test),
                                  "chronos-hybrid-test"),
    }

    importance = pd.Series(model.feature_importances_, index=features_aug)\
                   .sort_values(ascending=False)
    rank = list(importance.index).index("chronos_pred") + 1
    print(f"\n  chronos_pred importance rank: {rank} / {len(features_aug)}")
    print("  Top 10 features:")
    for feat, imp in importance.head(10).items():
        marker = "  <-- Chronos forecast" if feat == "chronos_pred" else ""
        print(f"    {feat:45s} {imp:.4f}{marker}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / "chronos_hybrid_regressor.json"))
    importance.to_csv(MODELS_DIR / "chronos_hybrid_feature_importance.csv")
    reports["chronos_pred_importance_rank"] = rank
    reports["n_features"] = len(features_aug)
    with open(MODELS_DIR / "chronos_hybrid_report.json", "w") as f:
        json.dump(reports, f, indent=2)
    return reports


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Lower to 8/16 if memory is tight (default: 32). "
                         "Apple MPS OOMs above ~32 at this context length.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only backfill this many rows - for smoke tests")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="Write the checkpoint every N batches (default: 20)")
    ap.add_argument("--skip-backfill", action="store_true",
                    help="Reuse the existing checkpoint and only train XGBoost")
    args = ap.parse_args()

    if not V2_MATRIX.exists():
        raise SystemExit(
            f"Missing {V2_MATRIX}\n"
            "This file ships with the repo. Regenerate with:\n"
            "  python run_full_pipeline.py --stage pipelines\n"
            "  python run_full_pipeline.py --stage features"
        )

    print("\n" + "=" * 70)
    print("  CHRONOS + XGBOOST HYBRID")
    print("=" * 70)

    mat = apply_feature_transforms(pd.read_parquet(V2_MATRIX))
    features = select_enhanced_features(mat)
    df_reg = mat.dropna(subset=[TARGET_REG]).sort_index()
    target_series = df_reg[TARGET_REG].astype("float32")
    print(f"  Matrix: {mat.shape[0]:,} rows x {mat.shape[1]} cols  "
          f"| {len(features)} engineered features")

    if args.skip_backfill:
        if not CHRONOS_PRED_PATH.exists():
            raise SystemExit(f"--skip-backfill needs {CHRONOS_PRED_PATH}, which does not exist.")
        chronos_pred = pd.read_parquet(CHRONOS_PRED_PATH)["chronos_pred"]
        print(f"\n--- Using checkpoint: {len(chronos_pred):,} predictions ---")
    else:
        device, _ = resolve_device(args.device)
        print(f"\n--- Chronos backfill ({MODEL_NAME}, device={device}) ---")
        chronos_pred = backfill_chronos(
            target_series, device, args.batch_size, args.checkpoint_every, args.limit,
        )
        print(f"  Checkpoint written: {CHRONOS_PRED_PATH} ({len(chronos_pred):,} rows)")

    expected = len(target_series) - CONTEXT_LENGTH
    if len(chronos_pred) < expected:
        print(f"\n  NOTE: {len(chronos_pred):,} / {expected:,} predictions present. "
              "Re-run without --limit to finish the backfill.")

    print("\n--- Training hybrid XGBoost ---")
    reports = train_hybrid(mat, features, chronos_pred)
    print(f"\n  Test MAE:  ${reports['test']['mae']:.2f}/MWh")
    print(f"  Test RMSE: ${reports['test']['rmse']:.2f}/MWh")
    print(f"\n  Artifacts written to {MODELS_DIR}/\n")


if __name__ == "__main__":
    main()
