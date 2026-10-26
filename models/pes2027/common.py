"""Paths, loaders, and write guards shared by the PES 2027 scripts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERIFIED = ROOT / "runs" / "pit_verified"
ABLATION = ROOT / "runs" / "pit_ablation"
FINAL_AUDIT = ROOT / "runs" / "final_audit"
RUNS = ROOT / "runs" / "pes2027"
EVIDENCE = ROOT / "evidence" / "pes_gm_2027"
PROTOCOL_DIR = EVIDENCE / "protocol"

SEEDS = (0, 1, 2, 3, 42)
TARGET = "HB_HUBAVG"
TRAIN_START = pd.Timestamp("2019-01-01", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
STUDY_END = pd.Timestamp("2025-01-01", tz="UTC")
CRISIS_START = pd.Timestamp("2021-02-10", tz="US/Central").tz_convert("UTC")
CRISIS_END = pd.Timestamp("2021-02-21", tz="US/Central").tz_convert("UTC")
FROZEN_HASH = "b850034b0690e660"

_PROTECTED = [VERIFIED, ABLATION, FINAL_AUDIT, ROOT / "reports" / "final_research_audit"]


def guard(path: Path) -> Path:
    path = Path(path).resolve()
    for root in _PROTECTED:
        if path == root.resolve() or root.resolve() in path.parents:
            raise PermissionError(f"refusing to write inside an original run directory: {path}")
    return path


def write_json(path: Path, payload) -> None:
    path = guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default))


def write_csv(path: Path, frame: pd.DataFrame, index: bool = False) -> None:
    path = guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, float_format="%.17g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path = guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def git_revision() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked_files": [line for line in run("status", "--porcelain", "--untracked-files=no").splitlines()],
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def index_hash(index: pd.DatetimeIndex) -> str:
    return hashlib.sha256(np.asarray(pd.DatetimeIndex(index).asi8).tobytes()).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def saved_seed_frames(prefix: str, directory: Path = VERIFIED / "predictions",
                      seeds=SEEDS) -> list[pd.DataFrame]:
    frames = [pd.read_parquet(directory / f"{prefix}__seed{s}.parquet") for s in seeds]
    for frame in frames[1:]:
        if not frame.index.equals(frames[0].index):
            raise AssertionError(f"{prefix}: seed indexes differ")
        if not np.array_equal(frame["y_true"].to_numpy(), frames[0]["y_true"].to_numpy()):
            raise AssertionError(f"{prefix}: seed targets differ")
    return frames


def seed_average(frames: list[pd.DataFrame]) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Index, target, dollar-scale mean forecast, and the per-seed forecast matrix."""
    stack = np.vstack([f["y_pred"].to_numpy(np.float64) for f in frames])
    return frames[0].index, frames[0]["y_true"].to_numpy(np.float64), stack.mean(axis=0), stack


def raw_rt() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "raw" / "ercot_rt_prices.parquet")


def hourly_max(series: pd.Series) -> pd.Series:
    s = series.sort_index().astype(float)
    return s.groupby(pd.DatetimeIndex(s.index).floor("h")).max()


def interval_counts(series: pd.Series) -> pd.Series:
    s = series.dropna()
    return s.groupby(pd.DatetimeIndex(s.index).floor("h")).size()
