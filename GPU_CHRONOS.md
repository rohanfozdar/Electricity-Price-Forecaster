# Chronos forecast caches on a GPU (early-warning experiment)

The early-warning experiment runs in two steps:

1. **GPU:** generate the zero-shot Chronos forecast caches. This page covers that step.
2. **CPU:** run `scripts/run_early_warning_experiment.py`, which trains XGBoost and the baselines on the caches and writes all metrics, tests and figures. That script never runs Chronos. It stops with an error if the primary cache is missing, and skips any sensitivity cache that is absent.

## What to upload

| File | Where it comes from |
|---|---|
| `scripts/run_chronos_cache.py` | this repository; standalone, no other repo imports |
| `scripts/run_chronos_caches.sh` | this repository; runs all five caches in order |
| `data/features/chronos_input_y.parquet` | on your Mac: `python scripts/run_early_warning_experiment.py --export-chronos-input` (about 1 MB) |

Keep the relative paths (`scripts/`, `data/features/`) on the server.

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

```bash
bash scripts/run_chronos_caches.sh --smoke   # 64 origins per cache, writes to /tmp/chronos_smoke, about a minute
bash scripts/run_chronos_caches.sh           # full run: 13,026 origins per cache
```

The full run produces these files in `data/features/chronos/`:

| Cache | Role (frozen in `reports/experiment_design.md` §8) |
|---|---|
| `bolt-small_ctx512.npz` | primary |
| `bolt-small_ctx256.npz`, `bolt-small_ctx1024.npz` | context-length sensitivity |
| `bolt-base_ctx512.npz` | model-size sensitivity |
| `t5-small_ctx512.npz` | robustness: the capstone's checkpoint, 20 sample paths |

The four Bolt caches take minutes each. T5 sampling is the slow one. The script prints throughput and an ETA every 20 batches. Existing files are skipped, so after an interruption you can simply rerun it. Set `DEVICE` or `BATCH` if needed, for example `BATCH=16 bash scripts/run_chronos_caches.sh` after running out of memory.

**Offline compute node?** On a login node, download the checkpoints once:

```bash
python -c "from chronos import BaseChronosPipeline as P; [P.from_pretrained(m) for m in ['amazon/chronos-bolt-small','amazon/chronos-bolt-base','amazon/chronos-t5-small']]"
```

Then set `HF_HOME` to the same cache directory on the compute node.

**SLURM:**

```bash
#!/bin/bash
#SBATCH --job-name=chronos-caches
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
source ~/chronos-env/bin/activate
bash scripts/run_chronos_caches.sh
```

## Afterwards (CPU)

Copy the five `.npz` files (about 60 MB each) into `data/features/chronos/` in the repository. Then:

```bash
pip install -e .                                   # CPU dependencies only; no torch needed
pytest -q                                          # 37 leakage/target/split/alignment tests
python scripts/run_early_warning_experiment.py --no-sensitivity --horizons 24 48   # quick check, writes runs/<hash>/summary
python scripts/run_early_warning_experiment.py     # full frozen run; design estimate about 1 hour on a 10-core laptop
```

The full run writes `runs/<hash>/` (predictions, per-model metadata, metrics, CIs, tests, figures, manifest) and copies the summaries to `reports/results/`. The `meta` field inside each `.npz` records the checkpoint, seed, device, library versions and runtime, and the run manifest hashes every cache, so the results stay traceable.
