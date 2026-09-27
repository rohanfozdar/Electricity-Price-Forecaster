"""Zero-shot Chronos forecast cache for the early-warning experiment.

Standalone on purpose: it needs only numpy, pandas, torch and chronos-forecasting,
plus one input file, so it can be copied to a GPU server on its own.

Input : hourly price parquet with a UTC DatetimeIndex and one column `y`
        (written by `python scripts/run_early_warning_experiment.py --export-chronos-input`,
        default data/features/chronos_input_y.parquet).
Output: .npz with
          origins_ns  int64   issue times, ns since epoch (UTC)
          values      float32 (n, 144, 9) quantiles (bolt) or (n, S, 144) samples (t5)
          levels      float32 quantile levels (bolt only)
          meta        json string: checkpoint, kind, context, seed, versions, timing

For each origin t the context is y[t-context .. t-1] on the complete hourly grid;
missing hours stay NaN (Chronos masks them). Nothing at or after t is read.

Examples
    # local, Chronos-Bolt (quantiles), minutes on a laptop
    python scripts/run_chronos_cache.py --variant bolt-small
    # GPU, Chronos-T5 small with 20 sample paths (the capstone checkpoint)
    python scripts/run_chronos_cache.py --variant t5-small --device cuda --batch-size 64
    # smoke test
    python scripts/run_chronos_cache.py --variant t5-small --limit 64
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = {
    "bolt-small": ("amazon/chronos-bolt-small", "quantile"),
    "bolt-base": ("amazon/chronos-bolt-base", "quantile"),
    "t5-small": ("amazon/chronos-t5-small", "samples"),
}
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def origin_grid(index: pd.DatetimeIndex, every_hours: int, start: str) -> pd.DatetimeIndex:
    first = max(pd.Timestamp(start), index.min()).ceil(f"{every_hours}h")
    return pd.date_range(first, index.max(), freq=f"{every_hours}h")


def contexts(y: pd.Series, origins: pd.DatetimeIndex, length: int) -> np.ndarray:
    pos = y.index.get_indexer(origins)
    if (pos < 0).any():
        raise ValueError("origins must lie on the hourly grid of y")
    vals = np.append(y.to_numpy(dtype=np.float32), np.float32(np.nan))
    idx = pos[:, None] - length + np.arange(length)[None, :]
    idx = np.where(idx >= 0, idx, len(y))
    return vals[idx]


def resolve_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(input_path: Path, out_path: Path, variant: str, context: int, horizon: int,
        every_hours: int, start: str, device: str, batch_size: int, num_samples: int,
        seed: int, limit: int | None) -> Path:
    import torch
    from chronos import BaseChronosPipeline

    checkpoint, kind = VARIANTS[variant]
    y = pd.read_parquet(input_path)["y"].astype("float32")
    y.index = pd.DatetimeIndex(y.index)
    if y.index.tz is None:
        y.index = y.index.tz_localize("UTC")
    full = pd.date_range(y.index.min(), y.index.max(), freq="h")
    y = y.reindex(full)
    origins = origin_grid(full, every_hours, start)
    if limit:
        origins = origins[:limit]
    ctx = contexts(y, origins, context)

    device = resolve_device(device)
    dtype = torch.bfloat16 if (device == "cuda" and kind == "samples") else torch.float32
    pipe = BaseChronosPipeline.from_pretrained(checkpoint, device_map=device, dtype=dtype)

    out_shape = (len(origins), horizon, len(LEVELS)) if kind == "quantile" else (len(origins), num_samples, horizon)
    values = np.full(out_shape, np.nan, dtype=np.float32)
    t0 = time.time()
    for b, s in enumerate(range(0, len(origins), batch_size)):
        batch = [torch.tensor(c) for c in ctx[s:s + batch_size]]
        torch.manual_seed(seed + b)
        with torch.inference_mode():
            if kind == "quantile":
                q, _ = pipe.predict_quantiles(batch, prediction_length=horizon, quantile_levels=LEVELS)
                values[s:s + len(batch)] = q.float().cpu().numpy()
            else:
                smp = pipe.predict(batch, prediction_length=horizon, num_samples=num_samples,
                                   limit_prediction_length=False)
                values[s:s + len(batch)] = smp.float().cpu().numpy()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
        if b % 20 == 0:
            done = s + len(batch)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done:,}/{len(origins):,} origins  {rate:.1f}/s  eta {(len(origins) - done) / rate / 60:.1f} min",
                  flush=True)

    import chronos
    meta = {
        "checkpoint": checkpoint, "kind": kind, "variant": variant, "context_hours": context,
        "prediction_length": horizon, "every_hours": every_hours, "start": start,
        "num_samples": num_samples if kind == "samples" else None, "seed": seed,
        "device": device, "dtype": str(dtype), "seconds": round(time.time() - t0, 1),
        "torch": torch.__version__, "chronos": getattr(chronos, "__version__", "unknown"),
        "python": platform.python_version(), "input_rows": int(len(y)),
        "input_first": str(y.index.min()), "input_last": str(y.index.max()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, origins_ns=origins.as_unit("ns").asi8, values=values,
                        levels=np.array(LEVELS, dtype=np.float32), meta=json.dumps(meta))
    print(f"wrote {out_path}  shape={values.shape}  {meta['seconds']}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="bolt-small")
    ap.add_argument("--input", type=Path, default=Path("data/features/chronos_input_y.parquet"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=144)
    ap.add_argument("--issue-every-hours", type=int, default=6)
    ap.add_argument("--start", default="2016-02-01T00:00Z")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N origins (smoke test)")
    a = ap.parse_args()
    kind = VARIANTS[a.variant][1]
    bs = a.batch_size or (256 if kind == "quantile" else 32)
    out = a.out or Path(f"data/features/chronos/{a.variant}_ctx{a.context}"
                        f"{'_limit' + str(a.limit) if a.limit else ''}.npz")
    run(a.input, out, a.variant, a.context, a.horizon, a.issue_every_hours, a.start,
        a.device, bs, a.num_samples, a.seed, a.limit)


if __name__ == "__main__":
    main()
