# Reproducibility

Three layers stay separate.

1. Original capstone artifacts, including `artifacts/honest/` and `docs/static/pdfs/paper.pdf`. Not rewritten here.
2. The point-in-time ablation in `runs/pit_ablation/`, which includes later-year scores that this study does not use.
3. The latency-constrained experiment in `runs/pit_verified/`. This is the definitive run. The audit in `runs/final_audit/` reads it and adds the LEAR-style fit, the block-length recomputation, and the target reconciliation.

## Commands

From `Ercot_Forecaster/ercot-grid-forecaster`, with `/net/home/rfozdar/venvs/ercot/bin/python`:

    python models/run_pit_verified.py all

That command rebuilds Chronos caches only when the cache metadata is missing, then fits the trees. It was already run. Do not point it at a new output directory after the fact. The caches are large.

    python models/run_final_audit.py

That command does not refit the primary trees. It recomputes the primary metrics, checks the 2024 index, checks the Chronos caches, refits the complete-four sensitivity because one 2024 hour is incomplete, refits the LEAR-style lasso, and writes this report directory.

Tests that guard the delay:

    python -m pytest tests/test_pit_latency.py tests/test_pit_sets.py tests/test_feature_availability.py

## Environment

Python 3.12.3. Package versions: {"numpy": "2.2.6", "pandas": "2.3.3", "sklearn": "1.6.1", "xgboost": "2.1.4", "scipy": "1.18.1", "torch": "2.6.0+cu124"}.

SHA-256 prefixes of the files this reproduction reads:

| File | Prefix |
| --- | --- |
| comparisons.json | `26609436b52b3f13` |
| delay_1h Chronos cache | `c21b72a189648831` |
| exogenous seed 42 predictions | `4485d0fdc4609609` |
| Chronos seed 42 predictions | `5aaa59a7677f7b57` |

## What a table in the paper should cite

| Paper number | Source |
| --- | --- |
| Primary MAE, gap, and 24-hour interval | `runs/pit_verified/comparisons.json`, name `delay_1h__mean_seed__both_minus_exog`, confirmed in `runs/final_audit/reproduction.json` |
| Block lengths including 12 hours | `runs/final_audit/block_lengths.csv` |
| Latency table | `runs/pit_verified/comparisons.json` |
| February 2021 | `runs/pit_verified/crisis_2021.json`, alignment check in `runs/final_audit/crisis_diagnostics.json` |
| LEAR-style MAE | `runs/final_audit/lear_style.json` |
| HB_NORTH and hourly mean | `runs/pit_verified/hub_north.json`, `hourly_mean.json` |
| Feature availability | `reports/final_research_audit/feature_availability.csv` |

Do not cite `runs/pit_ablation/later_years_metrics.csv` or any 2025 or 2026 row.
