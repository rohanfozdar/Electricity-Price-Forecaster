"""Build every Chronos forecast cache from Jupyter. No terminal.

This file lives in scripts/, next to run_chronos_cache.py. From the
repository root:

    python scripts/run_chronos_gpu.py

It also works as a flat upload: put this file in the same folder as
run_chronos_cache.py and chronos_input_y.parquet, then run it (or, in a
notebook cell, `%run run_chronos_gpu.py`). Set SMOKE = True first for a
one-minute check.

Chronos is zero-shot: this file runs inference and writes the .npz caches.
It does not train Chronos. XGBoost training and the evaluation need the full
repository and data/raw, so they run on your Mac after you copy the .npz files
back into data/features/chronos/:

    python scripts/run_early_warning_experiment.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# --- edit these ---------------------------------------------------------------
SMOKE = False          # True: 64 origins per cache, written to ./chronos_smoke/
BATCH_SIZE = None      # None uses the cache script's default (256 Bolt, 32 T5)
DEVICE = "cuda"
# ------------------------------------------------------------------------------

JOBS = [
    ("bolt-small", 512),   # primary
    ("bolt-small", 256),   # context-length sensitivity
    ("bolt-small", 1024),
    ("bolt-base", 512),    # model-size sensitivity
    ("t5-small", 512),     # capstone checkpoint, 20 sample paths
]


def here() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def pip_install(packages: list[str], index_url: str | None = None) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    if index_url:
        cmd += ["--index-url", index_url]
    print("installing", " ".join(packages), flush=True)
    subprocess.check_call(cmd)


def ensure_packages() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        pip_install(["torch"], index_url="https://download.pytorch.org/whl/cu121")
    pip_install(["chronos-forecasting>=2.0", "pandas", "pyarrow", "numpy"])


def find_input(folder: Path) -> Path:
    for candidate in (folder / "chronos_input_y.parquet",
                      folder / "data" / "features" / "chronos_input_y.parquet",
                      folder.parent / "data" / "features" / "chronos_input_y.parquet"):
        if candidate.exists():
            return candidate
    raise SystemExit(f"chronos_input_y.parquet not found in {folder}")


def main() -> None:
    folder = here()
    sys.path.insert(0, str(folder))
    ensure_packages()

    import torch
    from run_chronos_cache import run

    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__}  cuda={cuda}"
          + (f"  {torch.cuda.get_device_name(0)}" if cuda else ""), flush=True)

    src = find_input(folder)
    out_dir = folder / ("chronos_smoke" if SMOKE else "chronos")
    out_dir.mkdir(exist_ok=True)
    limit = 64 if SMOKE else None

    for variant, context in JOBS:
        name = f"{variant}_ctx{context}.npz"
        dest = out_dir / name
        if dest.exists():
            print(f"skip {dest.name} (already exists)", flush=True)
            continue
        print(f"=== {variant} context={context} -> {dest}", flush=True)
        run(src, dest, variant, context, horizon=144, every_hours=6,
            start="2016-02-01T00:00Z", device=DEVICE,
            batch_size=BATCH_SIZE or (32 if variant.startswith("t5") else 256),
            num_samples=20, seed=0, limit=limit)

    print("\nfinished. Download these files:", flush=True)
    for p in sorted(out_dir.glob("*.npz")):
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
    print("\nPut them in data/features/chronos/ on your Mac, then run\n"
          "  python scripts/run_early_warning_experiment.py", flush=True)


if __name__ == "__main__":
    main()
