"""
Hourly one-step Chronos forecasts for the hybrid search.

Writes data/features/chronos_hourly/<model>_<context-mode>_ctx<N>.parquet with one
row per hour that has N hours of history and these columns:
    chronos_pred     median forecast
    chronos_q10      10% quantile
    chronos_q90      90% quantile
    chronos_mean     mean forecast
    chronos_p_gt200  probability that the hour exceeds $200/MWh

Context modes:
    positional  the prior N rows of HB_HUBAVG, as in models/train_chronos_hybrid.py
    gap-aware   the prior N clock hours, with NaN for missing hours, so a context
                never silently straddles a data gap (audit item 13)

T5 checkpoints draw 20 sample paths; Bolt checkpoints return quantiles directly,
so their exceedance probability is interpolated from the 0.1-0.9 quantiles.

--horizon 24 (Bolt only) instead writes <model>_<mode>_ctx<N>_h24.parquet for the
24h alert classifier. Row t holds the label window of future_spike_24h, hours
t+1..t+24, forecast from the same history as the other features (up to t-1):
    chronos24_q50_max    max over the window of the median
    chronos24_q50_mean   mean over the window of the median
    chronos24_q90_max    max over the window of the 90% quantile
    chronos24_p_max      max over the window of P(hour > $200)
    chronos24_p_any      1 - prod(1 - P(hour > $200)), hours treated as independent

Usage:
    python models/chronos_hourly_features.py --model t5-small --context-mode gap-aware
    python models/chronos_hourly_features.py --model bolt-small --context-mode positional
    python models/chronos_hourly_features.py --model bolt-small --context-mode positional --horizon 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import FEATURES_DIR
from models.utils import TARGET_REG

CHECKPOINTS = {
    "t5-small": "amazon/chronos-t5-small",
    "bolt-small": "amazon/chronos-bolt-small",
    "bolt-base": "amazon/chronos-bolt-base",
}
OUT_DIR = FEATURES_DIR / "chronos_hourly"
SPIKE_PRICE = 200.0
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
COLUMNS = ["chronos_pred", "chronos_q10", "chronos_q90", "chronos_mean", "chronos_p_gt200"]
WINDOW_COLUMNS = ["chronos24_q50_max", "chronos24_q50_mean", "chronos24_q90_max",
                  "chronos24_p_max", "chronos24_p_any"]


def context_bounds(position: int, context: int, n: int) -> tuple[int, int]:
    """Half-open [start, end) slice of history for the forecast stored at `position`.

    The target index is `position`, so it is excluded: the last context hour is
    position - 1. A zero-shot Chronos model is not fit on these labels, but the
    context must still stop before the hour being forecast.
    """
    if context <= 0 or position < context or position >= n:
        raise AssertionError(
            f"Chronos context [{position - context}, {position}) does not lie "
            f"strictly before the target (n={n}, context={context})")
    start = position - context
    if not (0 <= start <= position - 1 < position <= n):
        raise AssertionError("target hour is inside the Chronos context")
    return start, position


def build_contexts(series: pd.Series, mode: str, context: int):
    """Return the forecast hours and, for each, the slice bounds into `values`."""
    if mode == "positional":
        values = series.to_numpy(np.float32)
        positions = np.arange(context, len(series))
        return series.index[positions], values, positions
    grid = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="h"))
    values = grid.to_numpy(np.float32)
    positions = grid.index.get_indexer(series.index)
    keep = positions >= context
    return series.index[keep], values, positions[keep]


def exceedance_from_quantiles(q: np.ndarray) -> np.ndarray:
    """P(y > 200) by linear interpolation of the CDF through the quantile levels."""
    levels = np.asarray(QUANTILE_LEVELS)
    cdf = np.array([np.interp(SPIKE_PRICE, row, levels, left=0.05, right=0.95) for row in np.sort(q, axis=1)])
    return 1.0 - cdf


def window_summary(q: np.ndarray) -> np.ndarray:
    """(batch, steps, levels) quantiles of hours t+1..t+24 -> WINDOW_COLUMNS."""
    b, steps, n_levels = q.shape
    p = exceedance_from_quantiles(q.reshape(-1, n_levels)).reshape(b, steps)
    return np.column_stack([q[:, :, 4].max(axis=1), q[:, :, 4].mean(axis=1), q[:, :, 8].max(axis=1),
                            p.max(axis=1), 1.0 - np.prod(1.0 - p, axis=1)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(CHECKPOINTS), default="t5-small")
    ap.add_argument("--context-mode", choices=["positional", "gap-aware"], default="gap-aware")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, choices=[1, 24], default=1)
    args = ap.parse_args()
    if args.horizon > 1 and args.model.startswith("t5"):
        ap.error("--horizon 24 needs a Bolt checkpoint")

    import torch
    from chronos import BaseChronosPipeline

    torch.manual_seed(args.seed)
    mat = pd.read_parquet(FEATURES_DIR / "feature_matrix_engineered_v2.parquet")
    series = mat.dropna(subset=[TARGET_REG]).sort_index()[TARGET_REG].astype("float32")
    hours, values, positions = build_contexts(series, args.context_mode, args.context)

    is_t5 = args.model.startswith("t5")
    batch = args.batch_size or (256 if is_t5 else 1024)
    pipe = BaseChronosPipeline.from_pretrained(
        CHECKPOINTS[args.model], device_map="cuda",
        torch_dtype=torch.bfloat16 if is_t5 else torch.float32)

    rows = []
    t0 = time.time()
    for start in range(0, len(positions), batch):
        ctx = []
        for p in positions[start:start + batch]:
            a, b = context_bounds(int(p), args.context, len(values))
            ctx.append(torch.tensor(values[a:b]))
        with torch.inference_mode():
            if args.horizon > 1:
                q, _ = pipe.predict_quantiles(ctx, prediction_length=args.horizon + 1,
                                              quantile_levels=QUANTILE_LEVELS)
                block = window_summary(q[:, 1:, :].float().cpu().numpy())
            elif is_t5:
                paths = pipe.predict(ctx, prediction_length=1)[:, :, 0].float().cpu()
                q10, q90 = torch.quantile(paths, torch.tensor([0.1, 0.9]), dim=1)
                block = np.column_stack([paths.median(dim=1).values, q10, q90, paths.mean(dim=1),
                                         (paths > SPIKE_PRICE).float().mean(dim=1)])
            else:
                q, mean = pipe.predict_quantiles(ctx, prediction_length=1, quantile_levels=QUANTILE_LEVELS)
                q = q[:, 0, :].float().cpu().numpy()
                block = np.column_stack([q[:, 4], q[:, 0], q[:, 8], mean[:, 0].float().cpu().numpy(),
                                         exceedance_from_quantiles(q)])
        rows.append(block)
        if (start // batch) % 50 == 0:
            print(f"  {start + len(ctx):,} / {len(positions):,}  ({time.time() - t0:.0f}s)", flush=True)

    columns = WINDOW_COLUMNS if args.horizon > 1 else COLUMNS
    out = pd.DataFrame(np.vstack(rows), index=hours, columns=columns)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_h{args.horizon}" if args.horizon > 1 else ""
    path = OUT_DIR / f"{args.model}_{args.context_mode}_ctx{args.context}{suffix}.parquet"
    out.to_parquet(path)
    meta = {"checkpoint": CHECKPOINTS[args.model], "context_mode": args.context_mode,
            "context": args.context, "horizon": args.horizon, "seed": args.seed, "rows": len(out),
            "seconds": round(time.time() - t0, 1), "torch": torch.__version__}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {path} ({len(out):,} rows, {meta['seconds']}s)")


if __name__ == "__main__":
    main()
