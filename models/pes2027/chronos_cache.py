"""Delayed Chronos caches with an explicit sampling seed and output path (E4, E5).

The context builder is the original `models.chronos_delayed_cache._contexts`. Batch
order and batch size match the original single-pass run, and `torch.manual_seed(seed)`
is called once before the pipeline is loaded, as in the original. There is no resume:
an interrupted run starts again from the first batch, so the random stream is the same.

    python -m models.pes2027.chronos_cache --seed 1 --delay 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models.chronos_delayed_cache import BATCH, COLUMNS, CONTEXT, SPIKE_PRICE, _contexts
from models.pes2027.common import RUNS, TARGET, git_revision, hourly_max, utc_now, write_json, write_parquet
from models.pes2027.protocol import require_frozen
from src.features.pit_latency import context_hash

CHECKPOINT = "amazon/chronos-t5-small"
REVISION = "a971ba21945c4f1796b17a91fe69214b5f4ad472"


def summarize(paths, step: int):
    """Five stored summaries of the sampled value at `step` (0-based) for each origin.

    `paths` has shape (batch, num_samples, horizon). The median is torch.median, which
    returns the lower of the two middle order statistics for an even sample count.
    """
    import torch

    at = paths[:, :, step].float().cpu()
    q10, q90 = torch.quantile(at, torch.tensor([0.1, 0.9]), dim=1)
    return np.column_stack([
        at.median(dim=1).values.numpy(), q10.numpy(), q90.numpy(),
        at.mean(dim=1).numpy(), (at > SPIKE_PRICE).float().mean(dim=1).numpy(),
    ])


def load_pipeline(device: str = "cuda"):
    import torch
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(
        CHECKPOINT, revision=REVISION, device_map=device, torch_dtype=torch.bfloat16,
    )


def generate(series: pd.Series, delay_hours: int, seed: int, out_path: Path,
             batch: int = BATCH, device: str = "cuda", pipe=None, limit: int | None = None) -> dict:
    import torch

    hourly = hourly_max(series)
    grid, values, positions, end_pos = _contexts(hourly, delay_hours)
    if limit is not None:
        positions = positions[:limit]
    t0 = time.time()
    torch.manual_seed(seed)
    if pipe is None:
        pipe = load_pipeline(device)
    if device == "cuda":
        torch.cuda.synchronize()
    load_s = time.time() - t0
    horizon = 1 + delay_hours
    blocks, index, latest, hashes, batch_s = [], [], [], [], []
    t1 = time.time()
    for start in range(0, len(positions), batch):
        chunk = positions[start:start + batch]
        ctx = []
        for p in chunk:
            end = int(end_pos[p])
            window = values[end - CONTEXT + 1:end + 1]
            ctx.append(torch.tensor(window))
            hashes.append(context_hash(window))
            latest.append(grid[end])
            index.append(grid[p])
        tb = time.time()
        with torch.inference_mode():
            paths = pipe.predict(ctx, prediction_length=horizon)
        blocks.append(summarize(paths, delay_hours))
        if device == "cuda":
            torch.cuda.synchronize()
        batch_s.append(time.time() - tb)
        if (start // batch) % 50 == 0:
            print(f"  seed {seed}: {start + len(chunk):,} / {len(positions):,}", flush=True)
    gen_s = time.time() - t1
    frame = pd.DataFrame(np.vstack(blocks), index=pd.DatetimeIndex(index), columns=COLUMNS)
    frame["latest_input_timestamp"] = pd.DatetimeIndex(latest)
    frame["context_hash"] = hashes
    write_parquet(out_path, frame)
    import chronos
    import transformers
    meta = {
        "created_utc": utc_now(), "code_revision": git_revision(), "seed": seed, "delay_hours": delay_hours,
        "rows": int(len(frame)), "context": CONTEXT, "horizon": horizon, "step_index": delay_hours,
        "batch": batch, "checkpoint": CHECKPOINT, "revision": REVISION, "dtype": "bfloat16", "device": device,
        "num_samples": "library default from model config", "single_pass": True,
        "torch": torch.__version__, "chronos_forecasting": getattr(chronos, "__version__", "unknown"),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "seconds_load": load_s, "seconds_generate": gen_s, "n_batches": len(batch_s),
        "seconds_per_batch_median": float(np.median(batch_s)) if batch_s else None,
        "ms_per_origin": 1000 * gen_s / max(len(frame), 1),
    }
    write_json(out_path.with_suffix(".json"), meta)
    return meta


def latency_probe(series: pd.Series, delay_hours: int = 1, n_single: int = 50, n_batches: int = 3,
                  device: str = "cuda", seed: int = 12345) -> dict:
    """Per-origin latency at batch 1 and batch 256 on the last origins of the series."""
    import torch

    hourly = hourly_max(series)
    grid, values, positions, end_pos = _contexts(hourly, delay_hours)
    torch.manual_seed(seed)
    t0 = time.time()
    pipe = load_pipeline(device)
    torch.cuda.synchronize()
    load_s = time.time() - t0

    def ctx_for(pos):
        return [torch.tensor(values[int(end_pos[p]) - CONTEXT + 1:int(end_pos[p]) + 1]) for p in pos]

    tail = positions[-(BATCH * n_batches):]
    with torch.inference_mode():
        pipe.predict(ctx_for(tail[:8]), prediction_length=1 + delay_hours)
        torch.cuda.synchronize()
        single = []
        for p in tail[:n_single]:
            t = time.time()
            pipe.predict(ctx_for([p]), prediction_length=1 + delay_hours)
            torch.cuda.synchronize()
            single.append(time.time() - t)
        batched = []
        for b in range(n_batches):
            t = time.time()
            pipe.predict(ctx_for(tail[b * BATCH:(b + 1) * BATCH]), prediction_length=1 + delay_hours)
            torch.cuda.synchronize()
            batched.append(time.time() - t)
    return {
        "seconds_load": load_s, "batch1_ms_median": 1000 * float(np.median(single)),
        "batch1_ms_p90": 1000 * float(np.quantile(single, 0.9)), "n_single": n_single,
        "batch256_seconds_median": float(np.median(batched)),
        "batch256_ms_per_origin": 1000 * float(np.median(batched)) / BATCH, "n_batches": n_batches,
        "gpu": torch.cuda.get_device_name(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--delay", type=int, default=1)
    args = parser.parse_args()
    require_frozen()
    from models.run_pit_ablation import load_sources
    from models.run_pit_verified import study_sources

    rt = study_sources(load_sources())["rt"][TARGET]
    out = RUNS / "E4-CSEED" / "caches" / f"delay_{args.delay}h_seed{args.seed}.parquet"
    if out.exists():
        raise SystemExit(f"{out} exists; remove it to regenerate")
    meta = generate(rt, args.delay, args.seed, out)
    print(meta)


if __name__ == "__main__":
    main()
