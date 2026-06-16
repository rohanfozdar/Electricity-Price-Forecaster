# Dead-File Audit — Cleanup Candidates

> **READ-ONLY REPORT.** Nothing has been deleted. Review each section and run a
> follow-up deletion task for items you approve.

Generated: 2026-05-04

---

## 1. `data/raw/backup/` — stale snapshots (10 files, ~5.5 MB total)

All files in this directory are pre-Task-1 snapshots of the corresponding
files now living in `data/raw/`. No Python code references `backup/`
anywhere in the codebase (grep found zero hits).

| File | Backup size (Apr 21) | Current raw/ size (latest) | Notes |
|---|---|---|---|
| `eia_gas.parquet` | 55 KB | 79 KB | Current is larger — refreshed by Task 1 |
| `eia_storage.parquet` | 6.4 KB | 17 KB | Current is larger |
| `ercot_da_prices.parquet` | 838 KB | 1.6 MB | Current is larger |
| `ercot_load.parquet` | 486 KB | 702 KB | Current is larger |
| `ercot_rt_prices.parquet` | 2.8 MB | 5.7 MB | Current is larger |
| `gdelt.parquet` | 26 KB | 52 KB | Current is larger |
| `gdelt_preswap.parquet` | 52 KB | _(no matching file in raw/)_ | Pre-Task-1 GDELT before column swap; no importer |
| `google_trends.parquet` | 9.1 KB | 11 KB (also in raw/) | Both copies unused — see §3 |
| `nrc_reactors.parquet` | 17 KB | 36 KB | Current is larger |
| `weather.parquet` | 1.0 MB | 2.3 MB | Current is larger |

**Confidence: HIGH** — the backup/ directory is never referenced, all live
copies in `data/raw/` are newer and larger, and `gdelt_preswap.parquet`
is an intermediate artifact with no consumer.

---

## 2. `data/raw/google_trends.parquet` — orphaned data file

**Path:** `data/raw/google_trends.parquet` (11 KB, Apr 21 2025)

No Python file anywhere in `features/`, `models/`, `evaluation/`,
`pipelines/`, `dashboard/`, or either pipeline orchestrator loads or
references `google_trends`. The word "trends" appears only as a comment in
`models/utils.py` ("baseline + sentiment + trends"). Neither `build_matrix.py`
nor `build_matrix_v2.py` reads it. Google Trends was removed as a data
source in Task 1.

**Confidence: HIGH** — zero importers or file-path references.

---

## 3. v1 vs v2 training scripts — import trace

### What `run_full_pipeline.py` imports (v1 path)

```
run_full_pipeline.py
  → features.build_matrix      (features/build_matrix.py)
  → features.engineer          (features/engineer.py)
  → models.train_baseline      (models/train_baseline.py)
  → models.train_enhanced      (models/train_enhanced.py)
  → models.train_classifier    (models/train_classifier.py)
```

### What references the v2 scripts

`features/build_matrix_v2.py` — imported by **nothing**; run directly as a
script. Referenced only in error-string comments in `engineer_v2.py`,
`train_all_v2.py`, and `evaluation/granger.py`.

`features/engineer_v2.py` — imported by **nothing**; standalone script.

`models/train_all_v2.py` — imported by **nothing**; standalone script.
`evaluation/benchmark_dam.py` requires its output
(`enhanced_v2_regressor.json`) at runtime but does not import it.

`models/print_final_comparison.py` — reads both v1 and v2 JSON reports from
`models/artifacts/` for a side-by-side comparison table. Not imported
anywhere; run directly.

### Conclusion

The system is split: `run_full_pipeline.py` runs the v1 feature + training
scripts; the v2 scripts are standalone. The v2 artifacts are **newer** (Apr
25 vs Apr 21) and the evaluation/ scripts (`granger.py`, `benchmark_dam.py`)
require v2 output files. The dashboard (`dashboard/app.py`) still loads v1
artifacts (`enhanced_regressor.json`, `enhanced_classifier.json`,
`enhanced_feature_importance.csv`).

#### Candidates for deletion once `run_full_pipeline.py` is updated to call v2

| File | Size | Status |
|---|---|---|
| `features/build_matrix.py` | 5.2 KB | Only caller: `run_full_pipeline.py` |
| `features/engineer.py` | 4.9 KB | Only caller: `run_full_pipeline.py` |
| `models/train_baseline.py` | 3.0 KB | Only caller: `run_full_pipeline.py` |
| `models/train_enhanced.py` | 3.1 KB | Only caller: `run_full_pipeline.py` |
| `models/train_classifier.py` | 3.5 KB | Only caller: `run_full_pipeline.py` |

**Confidence: MEDIUM** — these are live code called by `run_full_pipeline.py`
today. They become dead only after `run_full_pipeline.py` is migrated to
call the `_v2` equivalents AND the dashboard is updated to load v2
artifacts.

---

## 4. v1 model artifacts — superseded by v2 runs

The v2 training run (Apr 25) produced `*_v2_*.json` equivalents for every v1
artifact. The v1 artifacts below are only loaded by
`models/print_final_comparison.py` (for the side-by-side table); the
dashboard uses `enhanced_regressor.json` and `enhanced_classifier.json`.

| File | Size | Dashboard dependency? | Notes |
|---|---|---|---|
| `baseline_regressor.json` | 627 KB | No | v2: `baseline_v2_regressor.json` (578 KB) |
| `baseline_report.json` | 574 B | No | Read by `print_final_comparison.py` |
| `baseline_feature_importance.csv` | 1.2 KB | No | No reader outside train script |
| `baseline_classifier.json` | 386 KB | No | v2: `baseline_v2_classifier.json` (519 KB) |
| `baseline_classifier_report.json` | 741 B | No | Read by `print_final_comparison.py` |
| `baseline_classifier_importance.csv` | 1.2 KB | No | No reader |
| `enhanced_regressor.json` | 933 KB | **YES** (`dashboard/app.py:39`) | Do not delete without dashboard update |
| `enhanced_classifier.json` | 493 KB | **YES** (`dashboard/app.py:40`) | Do not delete without dashboard update |
| `enhanced_feature_importance.csv` | 1.7 KB | **YES** (`dashboard/app.py:152`) | Do not delete without dashboard update |
| `enhanced_report.json` | 571 B | No | Read by `print_final_comparison.py` |
| `enhanced_classifier_report.json` | 741 B | No | Read by `print_final_comparison.py` |
| `enhanced_classifier_importance.csv` | 1.7 KB | No | No reader |

**Safe to delete now (no importers):** the 8 rows marked "No" above (≈1.6 MB
total), provided you no longer need `print_final_comparison.py`'s v1 column.

**Confidence: HIGH** for the 8 non-dashboard files. **LOW** for the 3
dashboard files — they are still active.

---

## 5. `run_pipeline.py` vs `run_full_pipeline.py`

`run_pipeline.py` (2.5 KB, Apr 6) is a **targeted script** for pulling ERCOT
RT and DA price data with configurable date ranges and a `--full` flag.
`run_full_pipeline.py` even references it in warning messages:

```python
print("[WARN] ERCOT RT prices missing - run: python run_pipeline.py --pipeline rt --full")
```

It is **not superseded** — it serves a distinct purpose (targeted re-pull of
RT/DA data) that `run_full_pipeline.py` deliberately delegates to it.

**Confidence: LOW** — keep unless the ERCOT price pipelines are folded into
`run_full_pipeline.py` directly.

---

## 6. Stale tooling / build artifacts

| Path | Size | Why unused |
|---|---|---|
| `INTEGRATION_GUIDE.md` | 5.6 KB | V1 guide ("first iteration zip"); superseded by README.md |
| `.DS_Store` (root) | — | macOS metadata; should be in `.gitignore` |
| `ercot-grid-forecaster/.DS_Store` | — | macOS metadata |
| `data/.DS_Store` | — | macOS metadata |
| `build/` (directory) | 12 KB | Stale `setuptools` build output; regenerated by `pip install -e .` |
| `ercot_grid_forecaster.egg-info/` | 20 KB | Editable-install metadata; regenerated on next `pip install -e .`; usually gitignored |

**Confidence: HIGH** for all six — none are imported or referenced by
production code, and `build/` + `egg-info/` are standard gitignore targets.

---

## Summary Table

| Candidate | Size | Confidence | Blocker before deleting |
|---|---|---|---|
| `data/raw/backup/` (all 10 files) | ~5.5 MB | HIGH | None |
| `data/raw/google_trends.parquet` | 11 KB | HIGH | None |
| `features/build_matrix.py` | 5.2 KB | MEDIUM | Update `run_full_pipeline.py` to use v2 |
| `features/engineer.py` | 4.9 KB | MEDIUM | Same |
| `models/train_baseline.py` | 3.0 KB | MEDIUM | Same |
| `models/train_enhanced.py` | 3.1 KB | MEDIUM | Same |
| `models/train_classifier.py` | 3.5 KB | MEDIUM | Same |
| v1 model artifacts (8 non-dashboard files) | ~1.6 MB | HIGH | None (or accept loss of `print_final_comparison` v1 column) |
| v1 dashboard artifacts (3 files) | ~1.4 MB | LOW | Update `dashboard/app.py` to load v2 artifacts |
| `INTEGRATION_GUIDE.md` | 5.6 KB | HIGH | None |
| `.DS_Store` files (3) | negligible | HIGH | None |
| `build/` directory | 12 KB | HIGH | None |
| `ercot_grid_forecaster.egg-info/` | 20 KB | HIGH | None |
| `run_pipeline.py` | 2.5 KB | LOW | Keep — still referenced by `run_full_pipeline.py` warnings |
