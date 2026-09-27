# Running the Chronos hybrid on a GPU machine

> **Historical.** This page reproduces the capstone's *next-hour* hybrid, which is trained on the v2 matrix. The audit (`reports/repository_audit.md`) found that matrix is not point-in-time safe. For the current early-warning experiment's GPU job, see [GPU_CHRONOS_T5.md](GPU_CHRONOS_T5.md).

The Chronos + XGBoost hybrid is the headline result of this project and the only step that really
wants a GPU. This page is the copy-paste version for a lab workstation or a cluster node.

**What it does:** runs `amazon/chronos-t5-small` once per hourly observation (about **77,400 forward
passes** over a 512-hour context), stores those forecasts as a column named `chronos_pred`, and
trains XGBoost on the 87 engineered features plus that column.

**What it costs:** roughly **30-60 minutes on an NVIDIA GPU**. Many hours on a CPU. The run
checkpoints and resumes, so an interrupted job is never wasted.

---

## 1. Before you start

You need, on the GPU machine:

- **Python 3.9 or newer** and `pip`
- **An NVIDIA GPU with 8 GB or more**, and a CUDA-capable PyTorch build
- **About 3 GB of disk**: 2 GB for PyTorch and friends, 10 MB for the model weights, 10 MB for the
  feature matrix
- **Outbound internet**, at least once, to install packages and download the Chronos weights from
  HuggingFace. See [step 5](#5-if-the-node-has-no-internet) if the compute node is offline.

Check the GPU is visible first. If this prints nothing useful, stop here and sort that out:

```bash
nvidia-smi
```

---

## 2. Set up

```bash
git clone https://github.com/rohanfozdar/Electricity-Spot-Price-Forecaster.git
cd Electricity-Spot-Price-Forecaster

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[foundation]"
```

The `foundation` extra pulls `chronos-forecasting`, `transformers` and `accelerate`. PyTorch arrives
as a dependency; if `pip` installs a CPU-only build, install the CUDA build explicitly first and
then re-run the line above:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Confirm PyTorch actually sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want `True` and a device name. No clone-time data download is needed: the feature matrix
`data/features/feature_matrix_engineered_v2.parquet` ships in the repository.

---

## 3. Smoke test first, in about a minute

Do not launch the full backfill blind. This runs 200 forward passes and then stops:

```bash
python models/train_chronos_hybrid.py --device cuda --limit 200
```

Expected: a Chronos download on first use, a short progress bar, then a complaint that the train,
validation or test split is empty. **That error is the correct outcome here** — 200 rows out of
77,400 cannot fill a 2016-2024 split. It proves the model loads, the GPU works, and predictions are
being written.

Then clear the smoke-test checkpoint so the real run starts clean:

```bash
rm data/features/chronos_pred.parquet
```

---

## 4. The real run

```bash
python models/train_chronos_hybrid.py --device cuda --batch-size 64
```

If you have to leave, or your session might drop, detach it:

```bash
nohup python models/train_chronos_hybrid.py --device cuda --batch-size 64 > chronos.log 2>&1 &
tail -f chronos.log
```

Useful flags:

| Flag | Default | Why you would change it |
| --- | --- | --- |
| `--device cuda` | `auto` | `auto` picks cuda, then Apple MPS, then CPU. Being explicit fails loudly instead of quietly running on CPU for nine hours. |
| `--batch-size 64` | `32` | 64 suits a 16 GB or larger card. **Halve it on an out-of-memory error.** |
| `--checkpoint-every 20` | `20` | Batches between checkpoint writes. Lower it on a flaky or preemptible node. |
| `--skip-backfill` | off | Reuse an existing checkpoint and only retrain XGBoost. Seconds, not an hour. |
| `--limit N` | off | Smoke tests only. |

**If it runs out of memory**, the script writes its checkpoint, exits, and tells you the batch size
to retry with. Nothing is lost:

```bash
python models/train_chronos_hybrid.py --device cuda --batch-size 32
```

**If the job dies for any other reason**, re-run the same command. It reads
`data/features/chronos_pred.parquet`, reports how many predictions already exist, and continues from
there.

### On a SLURM cluster

Save as `chronos.sbatch`, adjusting the partition and account to match your cluster, then submit
with `sbatch chronos.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=chronos-hybrid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=chronos-%j.log

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# Keep model weights on shared storage rather than in a wiped home directory.
export HF_HOME="$SLURM_SUBMIT_DIR/.hf_cache"

python models/train_chronos_hybrid.py --device cuda --batch-size 64
```

The three-hour wall clock is deliberate headroom over the expected 30-60 minutes. If the job is
killed at the limit anyway, resubmit it: the checkpoint means the second job finishes the remainder.

---

## 5. If the node has no internet

Compute nodes often cannot reach HuggingFace. Download the weights somewhere that can, then move the
cache across:

```bash
# On a machine with internet, inside the same venv:
python -c "from chronos import ChronosPipeline; ChronosPipeline.from_pretrained('amazon/chronos-t5-small')"

# The weights land in ~/.cache/huggingface. Copy that to the GPU machine, then:
export HF_HOME=/path/to/copied/huggingface
export HF_HUB_OFFLINE=1
```

---

## 6. What success looks like

The script finishes with something close to this, the exact numbers being what we are after:

```text
  chronos_pred importance rank: 1 / 88
  Top 10 features:
    chronos_pred     0.xxxx  <-- Chronos forecast
    ...

  Test MAE:  $12.xx/MWh
  Test RMSE: $9x.xx/MWh

  Artifacts written to models/artifacts/
```

Three files are produced. **Send back the first one at minimum:**

| File | Size | Contents |
| --- | --- | --- |
| `models/artifacts/chronos_hybrid_report.json` | a few KB | train, validation and test MAE, RMSE, spike recall and precision — **this is the one that matters** |
| `models/artifacts/chronos_hybrid_feature_importance.csv` | a few KB | all 88 importances, including where `chronos_pred` ranks |
| `models/artifacts/chronos_hybrid_regressor.json` | a few MB | the trained model itself |

Optionally also keep `data/features/chronos_pred.parquet` (about 1 MB), the 77,400 Chronos
predictions, so the hybrid can be retrained in seconds with `--skip-backfill`.

Those paths are gitignored, so nothing is committed by accident. Paste the contents of
`chronos_hybrid_report.json` into the chat and the reproduced numbers will be added to the
Reproducibility section of the README, beside the paper's $12.13/MWh.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Missing .../feature_matrix_engineered_v2.parquet` | The clone is incomplete, or you are not in the repository root. The file ships in git; check with `git status`. |
| `CUDA out of memory` | Re-run with `--batch-size 32`, then `16`. The checkpoint is already saved. |
| Runs, but `nvidia-smi` shows no activity | PyTorch is a CPU-only build. Reinstall it from the CUDA index, per step 2. |
| `OSError: ... amazon/chronos-t5-small` | No route to HuggingFace. Use the offline cache in step 5. |
| `The train split is empty` | Only from `--limit` runs. Delete `data/features/chronos_pred.parquet` and run without `--limit`. |
| Very slow, hours not minutes | It is on the CPU. Pass `--device cuda` explicitly and check `torch.cuda.is_available()`. |

For background on where this model sits among the other twelve, see the
[README](README.md#results) and the
[project page](https://rohanfozdar.github.io/Electricity-Spot-Price-Forecaster/).
