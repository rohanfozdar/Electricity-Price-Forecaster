# Claim-to-code audit

Audit date: 2026-10-12. Branch `pit-chronos-ablation`. This file records what the PDF, the faithful rerun, and the later one-hour-ahead protocol each actually are. It does not replace saved forecasts in `artifacts/honest/` or the PDF.

Three objects are in circulation. They are not interchangeable.

| Object | Where it lives | What it is |
| --- | --- | --- |
| Capstone PDF | `docs/static/pdfs/paper.pdf` | Nine-page manuscript, *A Multimodal Framework for Electricity Price Forecasting Using Sentiment-Augmented Time-Series Signals* |
| Faithful rerun | `reports/reproduction.md`, `runs/benchmark/` | Re-execution of the notebook hybrid on the frozen 2024 index |
| Honest protocol | `models/honest_mae.py`, `reports/honest_mae.md`, `artifacts/honest/` | Later one-hour-ahead tree, thin lasso, and 2023-fit quantile stack |

The frozen 2024 index used by the rerun and the honest protocol is 8,767 hours, 114 hours with `HB_HUBAVG` above $200/MWh, hash `b850034b0690e660` (`models/honest_mae.py`, `EXPECTED_HASH`).

## What is being forecast

| Item | Implementation | Status |
| --- | --- | --- |
| Hub and market | `HB_HUBAVG`, ERCOT real-time settlement-point price. Day-ahead prices are features. | Defined in code |
| Hourly target | Maximum of the 15-minute prices in the UTC hour. `features/build_matrix_v2.py` uses `resample("1h").max()` and does not require four intervals. `src/data/sources.py` `hourly_price` does require four. | Aggregation rule exists; incomplete hours are counted by the point-in-time builder and are not dropped from this index |
| Issue time | Row `t` is a forecast of hour `t`. Price lags are shifted off `t`. | One-hour-ahead, not a day-ahead bid |
| When the target is observable | The last 15-minute interval of hour `t` ends at `t+1`. A posting delay of minutes is ERCOT practice and is not measured in this repository. | Delay unverified |
| Train / validation / test in the PDF | 2016–2022 / 2023 / 2024, stated as 60,428 / 8,760 / 8,767 | The 8,767 test count matches the frozen index. The honest tree does not use that training window |
| Train window of the $11.56 tree | 2019-01-01 through 2022, 34,850 hours, early stopping on 2023, 494 trees. `reports/honest_mae.md` | Chosen by 2022 MAE among four windows. 2024 was not used for that choice |

Central time is a whole-hour offset from UTC, so a UTC hour contains the same four interval starts as the corresponding Central hour. Calendar features in `features/engineer_v2.py` are still computed on the UTC clock.

## PDF claims

| Paper claim | Paper location | Code implementation | Evidence | Reproduction result | Status |
| --- | --- | --- | --- | --- | --- |
| Chronos–XGBoost MAE $12.13, RMSE $93.98, spike recall 61.40%, precision 50.43% | PDF §3.2 table and §4.3 | Hybrid cells in `notebooks/07_chronos_and_hybrid.ipynb` have no saved output. `models/train_chronos_hybrid.py` is the faithful script: Chronos-T5-small, 512 prior rows, median of 20 paths, 87 engineered columns, squared-error XGBoost | `reports/repository_audit.md` finding 17; `reports/reproduction.md` | Faithful 2024 MAE $21.96, RMSE $112.83, recall 58.77%, precision 30.45%. Ten XGBoost seeds span MAE $18.60–$24.64 and precision 27.5–41.1% | Not reproduced. On 114 spikes, 61.40% is 70/114. Precision 50.43% is then not a ratio of two integers, so that pair cannot be this test set |
| Twelve models, one 87-feature matrix, identical splits | PDF §3.1–§3.2 | Chronos and TimesFM notebooks are univariate. `notebooks/03_catboost.ipynb` reports 63 features. Prophet and SARIMAX notebooks were not executed | `reports/repository_audit.md` §6 | Only some rows were rerun. Day-ahead price matches the PDF exactly: MAE $17.51, RMSE $112.85, recall 28.95%, precision 48.53% | The shared-matrix claim is false |
| GDELT sentiment reduces XGBoost MAE by $10.10, 29.3% ($34.53 to $24.43) | PDF §5.1 | Matched notebook ablation is a different pair: with GDELT MAE $22.21, without $24.64 (`notebooks/01_xgboost.ipynb`) | `reports/reproduction.md` | Ten-seed paired gap averages −$0.10 (standard deviation $2.13); with-GDELT is better on 4 of 10 seeds | The 29.3% figure is not a matched ablation, and the matched effect is not stable |
| GDELT is known at the forecast hour | PDF §2.1 | `gdelt_tone` at every hour equals that UTC day's mean, so hour 00:00 sees the rest of the day | `reports/repository_audit.md` finding 2 | Honest protocol drops `gdelt_` columns | Same-day sentiment is not a one-hour-ahead feature |
| 720-hour price lag was considered and not implemented | PDF §2.2 | `hubavg_lag_720h` is built in `features/engineer_v2.py` and is in `artifacts/honest/honest_tree_fit.json` | Feature list, 87 names | Present in the $11.56 model | PDF is wrong |
| Peak hours are weekday 16:00–22:00 local | PDF §2.2 | `is_peak_hours` is UTC hours 14 through 20 (`features/engineer_v2.py`) | Code | Not the stated local window | PDF does not match the code |
| Day-ahead prices are lagged by one day | PDF §2.2 | The matrix joins day-ahead prices on the operating hour. They are published the previous afternoon, so the same-hour value can be legal, but the honest path does not apply a publication gate | `features/build_matrix_v2.py`; `src/data/asof.py` `dam_publication` | Gate exists only on the early-warning path | Description and code differ |
| CatBoost wins because ordered boosting denies future prices | PDF §4.1 | Ordered boosting changes how gradients are computed inside the training sample. It is not a temporal embargo | CatBoost documentation; the notebook still trains on the same matrix as the other trees | Not a leakage control | Conceptual error |
| TimesFM citation arXiv:2304.08424 | PDF references | That preprint is TiDE (*Long-term Forecasting with TiDE*), not TimesFM | Title on the preprint | Citation is mislabeled | Wrong reference |
| Weather, load, gas, storage, and reactor status are joined on a shared hourly timestamp without leakage | PDF §2.1 | Thirteen same-hour features; gas, NRC, and storage joined on the observation date; storage often before the Thursday release | `reports/repository_audit.md` findings 1–4 and 12 | Honest path lags weather one hour, drops GDELT, and rebuilds only the gas-spike threshold. It does not fix load, storage, NRC, or the reanalysis delay | Forecasting claim is not supported for the PDF |

## Faithful and improved hybrids

| Claim | Location | Code | Evidence | Result | Status |
| --- | --- | --- | --- | --- | --- |
| Zero-shot Chronos MAE $14.72, RMSE $115.59, recall 42.11%, precision 60.76% | PDF table | Gap-aware cache `data/features/chronos_hourly/t5-small_gap-aware_ctx512.parquet`, median path | `reports/reproduction.md` | This machine, sampling seed 0: MAE $14.54, RMSE $113.62, recall 41.23%, precision 61.04%. Four sampling seeds span MAE $14.54–$14.89 | Reproduced within sampling noise |
| Improved hybrid, five-seed 2024 mean MAE $13.06 ± $0.17 | `reports/reproduction.md` | `models/improve_hybrid.py`, chosen on 2023 by MAE among configs with spike F1 at least 0.617 | `runs/benchmark/selected_improved.json` | MAE $13.06, RMSE $107.18, recall 38.1%, precision 70.9% | Traceable. Still uses the leaky v2 matrix. Not the PDF row |
| Unshifted improved hybrid, seed 42, MAE $12.95 | `reports/honest_mae.md` | Same config on the original feature timing | `runs/benchmark/unshifted/metrics.json` | RMSE $104.13, recall 42.11%, precision 69.57% | Traceable. One seed |
| Weather lag alone | `reports/honest_mae.md` | `weather_lag_1` | Same report | 2024 MAE $12.96 against unshifted $12.95 | The lag does not explain the later gain |

## Honest protocol ($11.56 and the stack's $12.13)

| Claim | Location | Code | Evidence | Result | Status |
| --- | --- | --- | --- | --- | --- |
| Gradient-boosted tree MAE $11.56, RMSE $99.96, recall 50% (57/114), precision 73.08% (57/78) | `reports/honest_mae.md`, README | `models/honest_mae.py` `make_honest`, absolute error, `asinh` target, depth 6, learning rate 0.05, seed 42, 494 trees | `artifacts/honest/honest_tree_fit.json`; `scripts/verify_artifacts.py` | Recomputed from saved forecasts on the frozen index | Traceable. One seed. Not a Chronos-versus-XGBoost comparison: every honest feature list appends the six Chronos statistics (`feature_list`) |
| Quantile stack MAE $12.13, RMSE $96.23, recall 53.51% (61/114), precision 64.21% (61/95) | Same | Median regression on 2023 of day-ahead, persistence, LEAR, Chronos median, and the tree. Weights are not constrained to be positive | `artifacts/honest/stack_coefficients.json` | Same index. This $12.13 is not the PDF's $12.13 (RMSE $93.98, recall 61.40%) | Traceable. Not ranked against the tree before the 2024 score |
| LEAR MAE $13.39 | Same | Three `asinh` price lags, same-hour day-ahead price, Central-time calendar dummies, alpha 0.01 chosen on 2022 | `models/honest_mae.py` `_lear_matrix` | Beats persistence ($17.04) and the day-ahead price ($17.51) on this index | A thin lasso, not the Lago–Marcjasz–De Schutter–Weron LEAR benchmark |
| Chronos context is the previous 512 hours and excludes hour `t` | README | `models/chronos_hourly_features.py` `context_bounds`; gap-aware cache | Shifting the block one hour raises 2023 MAE from $31.85 to $35.78 | Holds for the cache the honest tree uses. The older positional cache could straddle the 2021 gap | Supported for the honest cache only |
| Every feature is known at the start of hour `t` | README, "Information set" | Honest list still contains `load_actual_mw`, same-trade-date Henry Hub, week-ending storage, same-report-date NRC, and Open-Meteo archive weather lagged one hour | `artifacts/honest/honest_tree_fit.json`; `configs/feature_availability.yaml` | Not established. See `reports/pit_ablation.md` after the new runs | False as a blanket statement |
| The drop from $12.95 to $11.56 has a single cause | Not claimed in `reports/honest_mae.md` | Joint change of window, GDELT removal, gas threshold, eight columns, and the weather lag | No factorial on 2024 in the original protocol | Weather lag alone is $12.96 | Do not assign the dollar gap to one change |

## Point-in-time status of the $11.56 inputs

| Feature family | In the $11.56 list | Publication status used by that model | Verdict |
| --- | --- | --- | --- |
| Price lags, rolling stats, previous-hour 15-minute summaries | Yes | Hour `t-1` is complete at `t`, assuming real-time prices are posted by the top of the hour. The posting delay is not measured | Usable, with that assumption stated |
| Chronos-T5-small median, q10, q90, mean, share of paths above $200, and q90−q10 | Yes | Computed from a context that ends at `t-1` in the gap-aware cache | Usable if the context check passes |
| Day-ahead hub prices and day-ahead shape | Yes | DAM publication is about 13:30 Central on the previous day. The honest path does not enforce the 14:00 gate in `src/data/asof.py` | Usually known; gate not applied |
| Zone weather and weather stress flags | Yes, clock-lagged one hour | Open-Meteo archive reanalysis, not a forecast vintage. One-hour lag is an unverified proxy for station observations. ERA5's multi-day release delay is not applied | Not established |
| `load_actual_mw` | Yes, same hour | Realized load. Even a one-hour lag would be an assumption. The same-hour column is not a lag | Not usable at hour `t` |
| Henry Hub level, lags, changes, gas-spike flag | Yes | Joined on the trade date. Weekly EIA release in the YAML file is marked `verified: assumed`. Only the $5.69 threshold is restricted to pre-2023 prices | Release time not verified |
| Storage level, lag, week-over-week change | Yes | Joined on the week-ending Friday. Thursday 10:30 ET release is documented and not applied | Premature relative to the documented release |
| NRC outage columns | Yes | Joined on the report date. Next-day availability is assumed | Not verified |
| `hour`, `day_of_week`, `month`, `is_weekend`, `is_peak_hours` | Yes | Functions of the timestamp, but on the UTC clock | Known, and misaligned with Central time |
| GDELT | No | Dropped | Not in this model |
| GFS previous-run temperatures | On disk, 2021-03-23 through 2024-12-31 | Used by `src/` only | Not in the $11.56 model |
| ERCOT wind or solar generation, actual or forecast | Absent | Not in the raw files used here | Absent |

`ffill().fillna(0)` in `fit_tree` fills remaining gaps with zero, including load before 2019. `log1p` after clipping at zero maps negative price features to zero. The target uses `asinh`.

## What this audit does not claim

Reproducing $11.56 does not show that Chronos adds information, that the inputs were published by the forecast origin, or that the result holds in another year. Those questions are the point-in-time ablation under `runs/pit_ablation/`. The early-warning stack in `src/` is a different target (spike probability at horizons of 24 to 144 hours) and did not produce these prices.
