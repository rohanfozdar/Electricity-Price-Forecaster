# Chronos-T5 forecast cache on a GPU (early-warning robustness run)

The early-warning experiment's primary Chronos variant is Chronos-Bolt-small, which runs on a laptop in about 5 minutes. This page covers the **robustness variant**, `amazon/chronos-t5-small` with 20 sample paths: the checkpoint the capstone used. It needs a GPU.

The job produces one file, which you copy back into the repository. You don't need to run anything else on the server.

## What to upload

Two files, about 1 MB in total:

| File | Where it comes from |
|---|---|
| `scripts/run_chronos_cache.py` | this repository; standalone, no other repo imports |
| `data/features/chronos_input_y.parquet` | on your Mac: `python scripts/run_early_warning_experiment.py --export-chronos-input` |

## Setup on the server (JupyterHub terminal or SSH)

```bash
python3 -m venv ~/chronos-env && source ~/chronos-env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # skip if torch+CUDA is already installed
pip install "chronos-forecasting>=2.0" pandas pyarrow numpy
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If you are inside a Jupyter notebook where `pip install --user` fails (as happened in `notebooks/10_chronos (1).ipynb`), use a target folder instead:

```python
import sys; !{sys.executable} -m pip install chronos-forecasting --target=/tmp/chronos_pkg
sys.path.insert(0, "/tmp/chronos_pkg")
```

## Run

Put both files in one folder, keeping the `data/features/` path for the parquet, then:

```bash
# 1) smoke test: 64 origins, under a minute
python scripts/run_chronos_cache.py --variant t5-small --device cuda --batch-size 64 --limit 64

# 2) full run: 13,026 origins (every 6 h, 2016-02-01 to 2024-12-31)
python scripts/run_chronos_cache.py --variant t5-small --device cuda --batch-size 64
```

The script prints its throughput and an ETA every 20 batches. On this project's Apple M5 laptop, T5 sampling ran at about 0.2–1 origin/s. A datacentre GPU should be one to two orders of magnitude faster; if it runs out of memory, lower `--batch-size` to 32 or 16.

**Offline compute node?** On a login node, run `python -c "from chronos import BaseChronosPipeline as P; P.from_pretrained('amazon/chronos-t5-small')"` once, then set `HF_HOME` to the same cache directory on the compute node.

**SLURM:**

```bash
#!/bin/bash
#SBATCH --job-name=chronos-t5
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
source ~/chronos-env/bin/activate
python scripts/run_chronos_cache.py --variant t5-small --device cuda --batch-size 64
```

## What to send back

One file: `data/features/chronos/t5-small_ctx512.npz` (about 60 MB compressed). Put it at the same path in the repository and rerun:

```bash
python scripts/run_early_warning_experiment.py
```

The runner detects the cache automatically and adds a `chronos_t5-small_ctx512` robustness row. The primary results do not change, because the primary variant was frozen before any results were seen (`reports/experiment_design.md` §8).

The `meta` field inside the `.npz` records the checkpoint, seed, device, library versions and runtime, so the result stays traceable.
