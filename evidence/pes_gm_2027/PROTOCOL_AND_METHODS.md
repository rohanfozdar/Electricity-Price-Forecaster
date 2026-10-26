# Protocol and methods

Machine-readable companion: `config/methods_spec.yaml`. The study is an archived-data evaluation under assumed delays. This file records the original experiment as it was run and the frozen plan for E0–E8. It does not replace `CORRECTIONS_LOG.md`.

## Target and information cutoff

The target at issue time t (UTC hour start) is the maximum of the four HB_HUBAVG 15-minute settlement prices whose interval starts are t, t+15, t+30, and t+45 minutes. Incomplete hours are kept. The 2024 evaluation index has 8,767 hours, hash `b850034b0690e660`, from 2024-01-01 00:00 UTC through 2024-12-31 06:00 UTC. The last hour has one interval.

The primary information cutoff is t − (1 + d) with d = 1 hour after interval end. That delay is an assumption (see `FEATURE_AVAILABILITY_REGISTER.md`). Day-ahead prices unlock at 14:00 Central on D−1. Storage unlocks at week-ending Friday + 7 days 18 hours. Calendar fields are functions of t.

## Features and transforms

The baseline (`xgb_exog`) has 37 columns: 10 delayed price-history features, 4 delayed 15-minute summaries, 9 day-ahead columns, 2 load, 3 storage, 5 Central calendar, 4 missingness flags. The flag `miss_chronos` is in the baseline and is constant on 2019–2024 rows. The augmented model (`xgb_both`) adds the six Chronos summaries, 43 columns.

Price-like columns are asinh-transformed. The tree target is asinh(y). Predictions are decoded as sinh of a clip to ±10. Imputation uses the training-split median.

The inherited v2 matrix uses row-based lag/roll then `dropna` on `hubavg_lag_720h` and `hubavg_rollmean_168h`. That, not a missing raw day, removes 191 hours after the 2020-12-31 fetch gap.

## Trees

XGBoost 2.1.4 through `models.improve_hybrid.make_model`: 2,000 trees, depth 6, learning rate 0.05, min_child_weight 1, subsample 0.8, colsample_bytree 0.8, `reg:absoluteerror`, hist, eval_metric mae, early_stopping_rounds 50 on 2023, `random_state = seed`, device cuda. Seeds {0, 1, 2, 3, 42}. Headline forecasts are the arithmetic mean of the five dollar-scale predictions.

Primary `best_iteration` values, from the saved sidecar JSON:

| Seed | xgb_exog | xgb_both |
| --- | --- | --- |
| 0 | 624 | 412 |
| 1 | 685 | 749 |
| 2 | 410 | 533 |
| 3 | 559 | 706 |
| 42 | 584 | 440 |

Depth 6 and learning rate 0.05 come from `models/honest_mae.py`. They were not re-tuned on 2024.

## Chronos

Checkpoint `amazon/chronos-t5-small`, revision `a971ba21945c4f1796b17a91fe69214b5f4ad472` for new caches. Original caches were loaded without an explicit revision; the on-disk snapshot hash is in `env/`. Context is 512 clock hours ending at t − (1 + d). Missing hours stay NaN. Horizon is 1 + d; the stored step is index d. Twenty paths, temperature 1.0, top_k 50, top_p 1.0, batch 256, bf16. `chronos_pred` is the torch lower median.

On the inherited matrix index, 345 of 34,851 training rows (2021-01-08 06:00 through 2021-01-22 14:00 UTC) have up to 23 NaNs in the 512-hour window, all from the 2020-12-31 fetch gap. The 2023 validation and 2024 evaluation contexts have none. The 2021 scored crisis index (8,563 hours) contains those same 345 rows. “512-hour context” therefore means 512 clock stamps, not 512 observed prices, on those rows.

ERCOT Load appears only in Chronos Benchmark II (zero-shot). ERCOT prices are not on the published corpus list. A pre-2021 tree split does not date the foundation-model weights. See `CHRONOS_PROVENANCE.md`.

## Inference

The paired estimand is the mean of δ_t = |y_t − ȳ¹_t| − |y_t − ȳ⁰_t| on shared origins. The moving-block bootstrap uses 24-hour blocks, 2,000 draws, seed 0, segments split at gaps over 1.5 hours, a percentile interval, and the interval-inversion p-value, labeled as such. Diebold–Mariano uses Bartlett lag 24 and a t(n − 1) reference. The original code applies HLN with h = lag + 1 = 25. The corrected primary convention is h = 1. Both are reported. Tests are conditional on fixed forecasts. Holm adjustment is used only inside predeclared secondary families. There is no equivalence test.

## Original baselines

Persistence is the latest legal hourly max (`hubavg_lag_1h`). Seasonal naive uses 24 h and 168 h. The LEAR-style lasso uses lags 1–24, 48, 168, gated DA, lagged load, and Central dummies, with α on a 15-point geometric grid.

## Selection chronology (Concern E)

These choices were made after 2024, or after 2021, had already been scored. They are not treated as pre-registered.

- 2024 prices were in the working tree from the June 2026 capstone onward.
- The honest tree scored 2024 on 2026-10-08.
- `pit_ablation` (2026-10-12) scored delay-0 2024 and expanding windows 2021–2024, plus 2025–26 for three models.
- `pit_verified` (2026-10-13) chose the 1-hour delay after seeing delay-0 2024 results and before the delay-1 2024 results.
- The February 2021 crisis window was fixed on 2026-10-13 after 2021 had already been scored in the ablation.
- LEAR-style, HB_NORTH, hourly-mean, and complete-four analyses were defined after those inspections.
- The $200 threshold is an inherited constant. Its scientific rationale is [MISSING].

2024 is therefore a retrospective evaluation year, not an untouched holdout. E7 is labeled retrospective robustness for that reason.

## Frozen new experiments

Definitions, multiplicity families, and compute caps are in `protocol/protocol_v1.yaml`. Runners call `models.pes2027.protocol.require_frozen()` before any new scoring. E1 representation selection uses 2023 five-seed MAE only; the selection record is committed before 2024 is scored. E6 remains secondary. E7 windows train and validate only on earlier years. E8 keeps the original seed-42 crisis numbers and adds a five-seed refit; official sub-windows are declared before those metrics are computed.
