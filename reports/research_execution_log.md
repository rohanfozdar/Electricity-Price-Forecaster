# Research execution log

This log is the record for a later rewrite. Commands were run from `Ercot_Forecaster/ercot-grid-forecaster` with `/net/home/rfozdar/venvs/ercot/bin/python`. The study does not score 2025 or 2026.

## 1. Confirm the saved ablation

Task: reproduce the conservative 2024 gap without refitting. Evidence: `runs/pit_ablation/comparisons.json` and `ablation_metrics.csv`. Command: `python models/run_pit_verified.py` writes the summary before fitting. Outcome: seed-averaged gap $-0.09 with an interval that covers zero. The mean of the seed MAEs is a different estimand. Artifact: `reports/pit_verified_reproduction.md`.

## 2. Source register

Task: retrieve ERCOT and EIA publication evidence. Pages retrieved on 2026-10-13 are listed in `reports/source_evidence_register.md`. Outcome: the price file is weekly archive NP6-785-ER. The live 15-minute product NP6-905-CD is not in the repository. Load is daily archive NP6-345-CD. Day-ahead posting is documented at no later than 13:30 Central, with a later correction window. EIA storage is Thursday 10:30 a.m. Eastern, with a code buffer of the following Friday 18:00 UTC. No historical first-publication timestamps were found.

## 3. Failed timestamp check

Task: look for a publication column in the price file. Outcome: no publication timestamp column in ercot_rt_prices.parquet. Columns: ['HB_HOUSTON', 'HB_NORTH', 'HB_SOUTH', 'HB_WEST', 'HB_HUBAVG', 'price_spike_flag']. Artifact: `runs/pit_verified/timestamp_check.json`.

## 4. Availability code

Task: add an explicit delay and tests that fail if the interval ending at the issue time is used. Command: `pytest tests/test_pit_latency.py tests/test_pit_sets.py tests/test_feature_availability.py`. The manifest is written by code to `reports/feature_availability_manifest.csv`. Primary delay: 1 hour, chosen as the shortest pre-specified assumption, before the corrected MAE was known.

## 5. Series check

The 15-minute hourly maximum and the matrix target differ by more than one cent on 0 hours. Maximum absolute difference $0.0000. Artifact: `runs/pit_verified/series_check.json`.

## 6. Chronos caches

One batch of 256 was timed before the rebuild. Load seconds: 5.26. Batch seconds: 5.15. Estimated full-cache minutes: 26.01. That estimate was about half an hour, so the primary delay and the other whole-hour delays were rebuilt. The old cache in `data/features/chronos_hourly/` was not overwritten. Each new row forecasts hour t from a context that ends at the last legal hour. Artifacts: `runs/pit_verified/chronos_probe.json` and `runs/pit_verified/chronos/delay_*h.parquet`.

## 7. Refit

Command: `python models/run_pit_verified.py all`. Both trees, five seeds, delays of 1, 2, 3, and 6 hours, plus a 24-hour load lag and a dropped-load case. Primary seed-averaged gap: $-0.15, interval [$-0.32, $0.06]. Artifacts: `runs/pit_verified/metrics.csv`, `comparisons.json`, `master_metrics.csv`.

## 8. Remaining checks

February 2021 expanding window, regime bins, block lengths 24/48/72/168, spike intervals with the undefined-draw rule, latency-valid persistence and seasonal naive, and a 2023-tuned lasso on legal lags and calendar. Reports in this directory cite those files. The hourly-mean target is secondary. HB_NORTH, when scored, uses that hub's own delayed prices and its own Chronos cache. It is another ERCOT hub, not another market.

## 9. Verdict

Partially verified, with unresolved concerns. The MAE was not used to choose the label or the primary delay.
