# Point-in-time Chronos ablation

Protocol `2026-10-12-pit-v1`. Command: `python models/run_pit_ablation.py all`.
The frozen 2024 index was scored after the comparison was specified. It is a previously inspected year, not an untouched test set.

## Build

- Test index hash `b850034b0690e660`, 8767 hours.
- Reported columns: 87. Assumed-realtime columns: 93. Conservative columns: 43.
- Hours in the study index without exactly four 15-minute prices: 3. They were counted and kept.
- Chronos cache matches the gap-aware builder index: True. Bad test-hour bounds: 0.
- File hashes: `{"matrix": "e4778a1147d8ed24", "chronos": "aebe403acbee07f1", "rt": "f93cbe66f50307b5", "da": "755300038c333024", "load": "332569c6a56fe193", "weather": "e228634605f41bf2", "gas": "bb230181613f07b6", "storage": "69ba4872a58310dc", "nrc": "a748342fa40bc990", "gfs": "a113561d7ef372d4"}`.

Feature availability is `runs/pit_ablation/feature_catalog.csv`. `verified` is `documented`, `assumed`, or `model_output`. Assumed rules are not treated as measured publication times.

## Primary 2024 scores

Train 2019-01-01 through 2022, early stopping on 2023, depth 6, learning rate 0.05, absolute error on an asinh target, seeds 0, 1, 2, 3, and 42. The table is the mean across seeds. The saved honest-tree booster is a separate replay, not a new fit.

| Information set | Model | MAE | RMSE | Median AE | Spike recall | Spike precision | Spike F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reported | xgb_exog | 11.69 | 99.22 | 1.89 | 43.33% | 74.45% | 0.547 |
| reported | xgb_chronos | 14.15 | 111.76 | 2.47 | 44.91% | 59.85% | 0.513 |
| reported | xgb_both | 11.62 | 102.07 | 1.85 | 49.65% | 73.61% | 0.592 |
| assumed_realtime | xgb_exog | 12.24 | 101.62 | 1.88 | 42.46% | 71.53% | 0.533 |
| assumed_realtime | xgb_chronos | 14.13 | 111.26 | 2.47 | 46.32% | 60.66% | 0.525 |
| assumed_realtime | xgb_both | 11.42 | 100.59 | 1.84 | 44.04% | 80.59% | 0.568 |
| conservative | xgb_exog | 11.52 | 100.70 | 1.85 | 42.46% | 80.32% | 0.555 |
| conservative | xgb_chronos | 14.13 | 111.26 | 2.47 | 46.32% | 60.66% | 0.525 |
| conservative | xgb_both | 11.46 | 99.81 | 1.85 | 44.56% | 81.38% | 0.575 |

Historical forecasts, unchanged:

| Model | MAE | RMSE | Spike recall | Spike precision |
| --- | ---: | ---: | ---: | ---: |
| persistence | 17.04 | 125.44 | 41.23% | 41.23% |
| lear | 13.39 | 97.92 | 34.21% | 61.90% |
| stack | 12.13 | 96.23 | 53.51% | 64.21% |
| chronos_zero_shot | 14.54 | 113.62 | 41.23% | 61.04% |
| honest_tree | 11.56 | 99.96 | 50.00% | 73.08% |
| dam | 17.51 | 112.85 | 28.95% | 48.53% |
| seasonal_naive_24 | 29.58 | 176.89 | 11.40% | 11.40% |
| seasonal_naive_168 | 32.35 | 185.21 | 7.02% | 6.96% |
| saved_booster | 11.56 | 99.96 | 50.00% | 73.08% |

## Does Chronos add anything?

The gap is MAE(with Chronos statistics) minus MAE(without them), in $/MWh. A negative gap means the Chronos columns have lower absolute error. The interval is a 24-hour moving-block bootstrap of that paired difference, 2,000 draws. The Diebold-Mariano statistic uses absolute error, HAC lag 24, and the Harvey-Leybourne-Newbold correction. A negative statistic means the same thing.

| Comparison | MAE gap | 95% interval | Excludes zero | DM stat | DM p |
| --- | ---: | --- | --- | ---: | ---: |
| reported__mean_seed__both_minus_exog | -0.11 | [-0.32, 0.14] | False | -0.91 | 0.3654 |
| reported__seed42__both_minus_exog | -0.21 | [-0.54, 0.11] | False | -1.28 | 0.1997 |
| reported__mean_seed__exog_minus_persistence | -5.46 | [-6.95, -4.15] | True | -7.75 | 0.0000 |
| assumed_realtime__mean_seed__both_minus_exog | -0.85 | [-1.60, -0.29] | True | -2.48 | 0.0130 |
| assumed_realtime__seed42__both_minus_exog | -0.29 | [-0.81, 0.13] | False | -1.22 | 0.2235 |
| assumed_realtime__mean_seed__exog_minus_persistence | -4.89 | [-6.49, -3.39] | True | -6.47 | 0.0000 |
| conservative__mean_seed__both_minus_exog | -0.09 | [-0.27, 0.11] | False | -0.87 | 0.3829 |
| conservative__seed42__both_minus_exog | -0.33 | [-0.64, -0.07] | True | -2.20 | 0.0279 |
| conservative__mean_seed__exog_minus_persistence | -5.62 | [-7.11, -4.30] | True | -8.05 | 0.0000 |

On the conservative set, the seed-averaged MAE gap is -0.09 with interval [-0.27, 0.11]. Per-seed gaps (both minus exogenous): 0.06, -0.01, -0.17, 0.18, -0.33.

The conservative interval covers zero. The Chronos statistics are not distinguishable from noise on MAE under this information set.
This is one market and one previously inspected test year. It is not a general claim about foundation-model features.

## Factorial on 2023

`runs/pit_ablation/factorial_2023.csv` has 32 rows.

## Expanding-window years

`runs/pit_ablation/expanding_window.csv` has 16 rows.

| Year | Set | Model | Role | Hours | Spikes | MAE | RMSE |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 2021 | assumed_realtime | xgb_exog | backtest_column_set_fixed_after_later_inspection | 8563 | 363 | 125.95 | 937.81 |
| 2021 | assumed_realtime | xgb_both | backtest_column_set_fixed_after_later_inspection | 8563 | 363 | 125.11 | 932.38 |
| 2021 | conservative | xgb_exog | backtest_column_set_fixed_after_later_inspection | 8563 | 363 | 124.76 | 928.02 |
| 2021 | conservative | xgb_both | backtest_column_set_fixed_after_later_inspection | 8563 | 363 | 124.43 | 928.64 |
| 2022 | assumed_realtime | xgb_exog | backtest_column_set_fixed_after_later_inspection | 8760 | 326 | 24.81 | 159.51 |
| 2022 | assumed_realtime | xgb_both | backtest_column_set_fixed_after_later_inspection | 8760 | 326 | 24.62 | 159.14 |
| 2022 | conservative | xgb_exog | backtest_column_set_fixed_after_later_inspection | 8760 | 326 | 24.87 | 157.04 |
| 2022 | conservative | xgb_both | backtest_column_set_fixed_after_later_inspection | 8760 | 326 | 24.17 | 147.65 |
| 2023 | assumed_realtime | xgb_exog | development_year | 8760 | 291 | 34.05 | 240.30 |
| 2023 | assumed_realtime | xgb_both | development_year | 8760 | 291 | 32.93 | 228.58 |
| 2023 | conservative | xgb_exog | development_year | 8760 | 291 | 30.79 | 206.24 |
| 2023 | conservative | xgb_both | development_year | 8760 | 291 | 31.86 | 215.13 |
| 2024 | assumed_realtime | xgb_exog | previously_inspected | 8767 | 114 | 11.70 | 102.26 |
| 2024 | assumed_realtime | xgb_both | previously_inspected | 8767 | 114 | 11.03 | 98.59 |
| 2024 | conservative | xgb_exog | previously_inspected | 8767 | 114 | 11.42 | 102.12 |
| 2024 | conservative | xgb_both | previously_inspected | 8767 | 114 | 11.12 | 99.96 |

## Hyperparameter check on 2023

`runs/pit_ablation/hyperparameter_check_2023.csv` has 12 rows.

## Regimes

`runs/pit_ablation/regimes.csv` has 297 rows.

## Hours after 2024

`runs/pit_ablation/later_years_metrics.csv` has 6 rows.

| Year | Model | Hours | MAE | RMSE | Spike F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2025 | xgb_exog_conservative | 8760 | 8.39 | 51.06 | 0.422 |
| 2026 | xgb_exog_conservative | 6551 | 7.08 | 27.24 | 0.642 |
| 2025 | persistence | 8759 | 11.53 | 67.09 | 0.440 |
| 2026 | persistence | 6551 | 8.95 | 33.31 | 0.675 |
| 2025 | day_ahead | 8760 | 12.87 | 57.74 | 0.143 |
| 2026 | day_ahead | 6551 | 13.77 | 60.50 | 0.359 |

## Later-year fetch

Status: fetched. Score: scored.
Scored through (exclusive UTC): 2026-10-01 05:00:00+00:00.
- 2025: real-time `fetched`; day-ahead `fetched`.
- 2026: real-time `fetched`; day-ahead `fetched`.
  Fallback: empty trailing month sheets were dropped before parsing.
Not scored: XGBoost with Chronos statistics, because no Chronos cache exists after 2024.
Load: not fetched; miss_load is 1 and the training median is used.

Figures, when the report step finishes, are under `runs/pit_ablation/figures/`.
