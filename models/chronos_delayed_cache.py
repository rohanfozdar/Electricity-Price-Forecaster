"""Chronos caches whose context ends on the last legal hour.

Writes runs/pit_verified/chronos/ only. Does not touch data/features/chronos_hourly/.
The selected step is the forecast of the target hour, not a lag of an older forecast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.pit_latency import context_hash, hourly_max_from_intervals

SPIKE_PRICE = 200.0
CONTEXT = 512
BATCH = 256
COLUMNS = ["chronos_pred", "chronos_q10", "chronos_q90", "chronos_mean", "chronos_p_gt200"]


def cache_path(root: Path, name: str) -> Path:
    return root / "runs" / "pit_verified" / "chronos" / f"{name}.parquet"


def _contexts(hourly: pd.Series, delay_hours: int):
    grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", tz=hourly.index.tz)
    values = hourly.reindex(grid).to_numpy(np.float32)
    end_pos = np.arange(len(grid)) - (1 + delay_hours)
    keep = end_pos >= (CONTEXT - 1)
    positions = np.flatnonzero(keep)
    return grid, values, positions, end_pos


def ensure_cache(rt_price: pd.Series, delay_hours: int, path: Path, protocol: str) -> pd.DataFrame:
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("protocol") == protocol and meta.get("delay_hours") == delay_hours and meta.get("complete"):
            return pd.read_parquet(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hourly = hourly_max_from_intervals(rt_price)
    grid, values, positions, end_pos = _contexts(hourly, delay_hours)
    partial = path.with_name(path.stem + ".partial.parquet")
    done = pd.read_parquet(partial) if partial.exists() else None
    if done is not None and len(done):
        have = set(pd.DatetimeIndex(done.index).tz_convert("UTC"))
    else:
        have = set()
    pending = [int(p) for p in positions if grid[p].tz_convert("UTC") not in have]
    print(f"chronos {path.name}: {len(pending):,} hours to score, delay {delay_hours}h", flush=True)
    if pending:
        import torch
        from chronos import BaseChronosPipeline

        torch.manual_seed(0)
        pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small", device_map="cuda", torch_dtype=torch.bfloat16,
        )
        rows = []
        indexes = []
        latest = []
        hashes = []
        horizon = 1 + delay_hours
        for start in range(0, len(pending), BATCH):
            batch_pos = pending[start:start + BATCH]
            ctx = []
            for p in batch_pos:
                end = int(end_pos[p])
                window = values[end - CONTEXT + 1:end + 1]
                ctx.append(torch.tensor(window))
                hashes.append(context_hash(window))
                latest.append(grid[end])
                indexes.append(grid[p])
            with torch.inference_mode():
                paths = pipe.predict(ctx, prediction_length=horizon)[:, :, delay_hours].float().cpu()
            q10, q90 = torch.quantile(paths, torch.tensor([0.1, 0.9]), dim=1)
            block = np.column_stack([
                paths.median(dim=1).values.numpy(),
                q10.numpy(), q90.numpy(),
                paths.mean(dim=1).numpy(),
                (paths > SPIKE_PRICE).float().mean(dim=1).numpy(),
            ])
            rows.append(block)
            if (start // BATCH) % 20 == 0:
                print(f"  {start + len(batch_pos):,} / {len(pending):,}", flush=True)
                chunk = pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(indexes), columns=COLUMNS)
                chunk["latest_input_timestamp"] = pd.DatetimeIndex(latest)
                chunk["context_hash"] = hashes
                if done is not None and len(done):
                    chunk = pd.concat([done, chunk])
                chunk.to_parquet(partial)
        fresh = pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(indexes), columns=COLUMNS)
        fresh["latest_input_timestamp"] = pd.DatetimeIndex(latest)
        fresh["context_hash"] = hashes
        frame = pd.concat([done, fresh]) if done is not None and len(done) else fresh
    else:
        frame = done
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.to_parquet(path)
    if partial.exists():
        partial.unlink()
    meta = {
        "protocol": protocol, "delay_hours": delay_hours, "complete": True,
        "rows": int(len(frame)), "context": CONTEXT, "horizon": 1 + delay_hours,
        "step_index": delay_hours, "checkpoint": "amazon/chronos-t5-small",
        "note": "Forecast of hour t from the 512 clock hours ending at the last legal hour. Not a shifted copy of the old cache.",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {path} ({len(frame):,} rows)", flush=True)
    return frame
