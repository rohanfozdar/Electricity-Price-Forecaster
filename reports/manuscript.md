# A point-in-time evaluation of foundation-model features for ERCOT real-time price forecasting

## Other titles that were not used

- Do frozen Chronos forecast statistics improve point-in-time ERCOT real-time price forecasts?
- When do zero-shot Chronos statistics help a gradient-boosted ERCOT price forecast?
- A point-in-time evaluation of foundation-model features for ERCOT real-time price forecasting

The title above is the one that matches the conservative-set interval. The other two are recorded so the choice is visible.

## Abstract

This paper asks whether six forecast statistics from a frozen, zero-shot Chronos-T5-small model improve a one-hour-ahead forecast of the ERCOT real-time hub-average price. The target at hour t is HB_HUBAVG, the maximum of the four 15-minute settlement prices inside that hour. The forecast is issued at the start of hour t. A feature may be used only when its availability time is at or before that instant. Where a release time could not be verified, the feature is either lagged under an explicit assumption or removed. The conservative information set keeps price history, publication-gated day-ahead prices, a one-hour lag of realized load under an unverified posting assumption, weekly storage under a documented release offset, Central-time calendar variables, and the Chronos statistics. It removes archive weather, Henry Hub, and nuclear-outage status. Across five seeds, conservative XGBoost without Chronos has 2024 MAE $11.52/MWh and the same model with the six Chronos statistics has MAE $11.46/MWh. On the conservative information set, the MAE gap from adding the six Chronos statistics is -$0.09/MWh, with a 95% moving-block interval [-$0.27, $0.11] that covers zero. The statistics are not distinguishable from noise on this comparison. The 2024 evaluation year was inspected in earlier development and is not an untouched test set. An expanding-window backtest covers 2021–2024 under a rule fixed before those scores were read, but the column set itself was specified after that inspection. The study is one hub in one market. It does not establish a new forecasting architecture.

## 1. Contributions

1. A column-level availability audit of the existing ERCOT hourly-maximum forecast, separating documented release rules, assumptions, and inputs that are not usable at the issue time.
2. A matched comparison, on identical origins, of persistence, a Chronos zero-shot median, XGBoost without Chronos statistics, XGBoost on those statistics alone, and XGBoost on both, under three labeled information sets.
3. Expanding-window scores for 2021–2024 and regime errors, with a block-bootstrap interval for the Chronos MAE gap.

These statements are limited to the runs in `runs/pit_ablation/`. They do not say that an earlier capstone leaderboard was reproduced, that news sentiment improves the forecast, or that the hybrid beats deep-learning models under a shared feature set.

## 2. What the earlier manuscript claimed

The capstone PDF, `docs/static/pdfs/paper.pdf`, reports a Chronos–XGBoost hybrid at MAE $12.13/MWh, RMSE $93.98, spike recall 61.40%, and spike precision 50.43%, and a 29.3% MAE reduction from GDELT sentiment. The hybrid cells were not executed. On the frozen 2024 index of 8,767 hours and 114 spikes, 61.40% is 70/114, and precision 50.43% is then not a ratio of two integers. A faithful rerun has MAE $21.96. A later honest-tree protocol, with a different loss, target transform, window, and feature timing, has MAE $11.56 and a quantile stack has MAE $12.13 with RMSE $96.23. That stack is not the PDF row. The $11.56 fit always included the Chronos statistics, so it did not measure their incremental value. The audit is `reports/claim_audit.md`.

## 3. Forecast definition

The market is ERCOT real-time settlement. The series is the hub-average settlement-point price. Hour t is the UTC hour beginning at t, which is the same four interval starts as the corresponding Central hour because Central time differs from UTC by a whole number of hours. The target is the maximum of those four prices. The v2 matrix does not require all four intervals to be present. Those hours were counted and retained so the index would stay the frozen 2024 index. The early-warning code drops incomplete hours; this study does not silently switch to that rule.

The issue time is the start of hour t. The target hour has not occurred. It becomes observable when the last 15-minute interval is posted, nominally at t plus one hour, plus a posting delay that this repository does not measure. This is a one-hour-ahead forecast. It is not a day-ahead bid submitted after the day-ahead market clears.

## 4. Information sets

Three sets are scored. `reported` is the honest-tree column list, including same-hour realized load, trade-date gas, week-ending storage, report-date reactor status, UTC calendar features, and archive weather lagged one hour by the clock. It is a retrospective set with known gaps. Missing values are forward-filled inside each split and then set to zero, matching that protocol.

`assumed_realtime` drops same-hour load. Weather is still archive reanalysis lagged one hour and is labeled an unverified station proxy. Gas uses the YAML weekly rule, which the file marks as assumed. Storage uses the documented Thursday release plus a buffer, implemented as the following Friday 18:00 UTC. Reactor status uses an assumed next-UTC-day rule. Day-ahead prices are masked unless their publication time, 14:00 Central on the previous local day, is at or before the issue time. Calendar features use US/Central time. Peak hours are weekday 16:00–22:00 local, which is what the PDF stated and not what the v2 matrix computed. Signed price features and the Chronos dollar statistics use `asinh`. Medians are fit on the training rows of that fit. A missingness indicator is kept for each family. Nothing is filled with zero.

`conservative` starts from that set and removes archive weather, gas, and reactor status. Storage remains. Lagged realized load remains, under the assumption that the previous hour's load has been posted. That assumption is not a measured timestamp.

The Chronos block is the existing gap-aware cache from `amazon/chronos-t5-small`: median, 10th and 90th percentiles, mean, the share of 20 paths above $200/MWh, and the gap between the 90th and 10th percentiles. The context check requires the 512-hour window to end at t−1 for every 2024 test hour. The cache was not recomputed. The check result is `runs/pit_ablation/chronos_context_check.json`.

## 5. Models and uncertainty

The primary tree is XGBoost with histogram splits, absolute error, depth 6, learning rate 0.05, at most 2,000 trees, and early stopping after 50 rounds on 2023. The target inside the model is `asinh` of price, clipped to ±10 before `sinh`. Seeds are 0, 1, 2, 3, and 42. The same settings are used for exogenous features only, Chronos statistics only, and both. A depth and learning-rate grid is scored on 2023 after early stopping on the last 90 days of 2022. It is not used to replace the 2024 models.

Persistence, seasonal persistence, the day-ahead price, the thin LEAR, the zero-shot Chronos median, the saved honest tree, and the saved quantile stack are copied from earlier artifacts. They are not refit on the new information sets. The thin LEAR is three price lags, the day-ahead price, and calendar dummies. It is not the LEAR benchmark of Lago, Marcjasz, De Schutter, and Weron, Applied Energy 293 (2021) 116983.

The pre-registered contrast is MAE with the Chronos statistics minus MAE without them, on the same hours. Uncertainty is a moving-block bootstrap with 24-hour blocks and 2,000 resamples, inside contiguous stretches of the test index, and a Diebold–Mariano test on absolute error with HAC lag 24 and the Harvey–Leybourne–Newbold correction. The primary interval uses the seed-averaged forecast. Per-seed gaps are reported beside it. Spike precision and recall use the pre-specified threshold of $200/MWh. That threshold is not tuned on 2024.

## 6. Results

Means across the five primary seeds, 2024. Spike rates are means of the seed-level rates.

| Model | MAE | RMSE | Median AE | Spike recall | Spike precision | Spike F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservative, XGBoost without Chronos | 11.52 | 100.70 | 1.85 | 42.46% | 80.32% | 0.555 |
| Conservative, XGBoost on Chronos statistics only | 14.13 | 111.26 | 2.47 | 46.32% | 60.66% | 0.525 |
| Conservative, XGBoost with Chronos statistics | 11.46 | 99.81 | 1.85 | 44.56% | 81.38% | 0.575 |
| Assumed real-time, XGBoost without Chronos | 12.24 | 101.62 | 1.88 | 42.46% | 71.53% | 0.533 |
| Assumed real-time, XGBoost with Chronos statistics | 11.42 | 100.59 | 1.84 | 44.04% | 80.59% | 0.568 |
| Reported honest-tree features, XGBoost without Chronos | 11.69 | 99.22 | 1.89 | 43.33% | 74.45% | 0.547 |
| Reported honest-tree features, XGBoost with Chronos statistics | 11.62 | 102.07 | 1.85 | 49.65% | 73.61% | 0.592 |

Unchanged earlier forecasts on the same index:

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

On the conservative information set, the MAE gap from adding the six Chronos statistics is -$0.09/MWh, with a 95% moving-block interval [-$0.27, $0.11] that covers zero. The statistics are not distinguishable from noise on this comparison.
 Per-seed gaps on that set, in seed order 0, 1, 2, 3, 42, are $0.06, -$0.01, -$0.17, $0.18, -$0.33.
 The seed-42 pair alone has gap -$0.33 [-$0.64, -$0.07]. That pair is not the pre-registered summary. The summary is the seed-averaged forecast.
 On the assumed-real-time set the seed-averaged MAE gap is -$0.85 [-$1.60, -$0.29].
 That interval lies below zero. The set still includes archive weather, gas, and reactor status under rules that are not measured publication times.
 Its seed-42 interval is [-$0.81, $0.13].
 On the reported honest-tree features the gap is -$0.11 [-$0.32, $0.14]. That set is not a clean publication-time comparison.


A negative gap means the model with Chronos statistics has lower MAE. An interval that covers zero means the study does not support a claim that Chronos helps. The pre-registered contrast is the conservative seed-averaged forecast. Large spike errors can dominate RMSE while leaving MAE almost unchanged.

Conservative information set, seed-averaged forecast, 2024. The $200 threshold is the pre-specified spike rule.

| Regime | Hours | MAE without Chronos | MAE with Chronos |
| --- | ---: | ---: | ---: |
| All hours | 8767 | $11.42 | $11.33 |
| Price at or below $100 | 8473 | $4.60 | $4.55 |
| Price above $100 | 294 | $208.00 | $206.79 |
| Price above $200 | 114 | $446.21 | $446.27 |
| Negative price | 145 | $3.66 | $3.62 |

The all-hours row is the MAE of the seed-averaged forecast. It is not the average of the five seed-level MAEs in the table above. On hours above $200 the MAE gap, with Chronos minus without, is $0.06/MWh. That slice was not used to choose the model.

### Expanding window

Each test year is scored by a model trained only on earlier hours, with early stopping on the last 90 days before that year. Seed 42 only. The column set was specified after 2024 had already been inspected, so these years are not an untouched confirmation of that specification. 2023 is the development year of the earlier protocol. 2024 is previously inspected.

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

The 2021 rows use the 8,563 hours present in the frozen matrix. 197 UTC hours in calendar 2021 are absent from that matrix, in 2021-01-01 00:00:00+00:00 through 2021-01-08 05:00:00+00:00 and 2021-01-30 07:00:00+00:00 through 2021-01-31 05:00:00+00:00. Clock-hour persistence on the scored hours has MAE $28.52/MWh. The conservative tree without Chronos has MAE $3608.20/MWh on 2021-02-10 through 2021-02-20, when the hourly maximum reaches $9051.55/MWh and the forecast stays at or below $2153.73/MWh, and MAE $13.95/MWh on the other scored hours of 2021. Almost all of the year-level absolute error comes from that February window. It is not evidence that the tree fails in ordinary hours of 2021.

### Hours after 2024

These hours were fetched after the 2024 comparison was frozen. The scored model is the conservative XGBoost fit without Chronos statistics, trained on 2019–2022. Chronos statistics were not scored, because the cache ends in 2024 and was not rerun. Load was not refetched; the load missingness indicator is on and the training median is used. Incomplete hours, those without four 15-minute prices, are dropped in this extension only.

| Year | Model | Hours | Spikes | MAE | RMSE | Spike F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2025 | xgb_exog_conservative | 8760 | 75 | 8.39 | 51.06 | 0.422 |
| 2026 | xgb_exog_conservative | 6551 | 80 | 7.08 | 27.24 | 0.642 |
| 2025 | persistence | 8759 | 75 | 11.53 | 67.09 | 0.440 |
| 2026 | persistence | 6551 | 80 | 8.95 | 33.31 | 0.675 |
| 2025 | day_ahead | 8760 | 75 | 12.87 | 57.74 | 0.143 |
| 2026 | day_ahead | 6551 | 80 | 13.77 | 60.50 | 0.359 |

Fetch record: 2025: real-time fetched; day-ahead fetched. 2026: real-time fetched; day-ahead fetched. Fallback: empty trailing month sheets were dropped before parsing. Scores use Central calendar years and stop before 2026-10-01 05:00:00+00:00. Latest complete US/Central month. The in-progress month is excluded. A month whose last hour is missing is excluded. Hours without four 15-minute prices are dropped, which is stricter than the historical v2 matrix. Not scored: XGBoost with Chronos statistics, because no Chronos cache exists after 2024. Load: not fetched; miss_load is 1 and the training median is used. The 2024 conservative model without Chronos has RMSE $100.70/MWh on the earlier test year, so a lower MAE after 2024 is not, by itself, evidence that the model improved. This block does not score Chronos statistics.

### Other checks

The 2023 factorial varies weather lag, whether GDELT is kept, the training-only gas threshold, the eight extra columns, and the training window, on the old feature matrix. Early stopping ends in 2022. 2024 is not used, and none of these contrasts is a share of the $11.56 result. Across the matched pairs, clock-lagging weather changes 2023 MAE by $1.43/MWh, keeping GDELT changes it by $0.82/MWh, the training-only gas threshold changes it by $0.16/MWh, and the eight extra columns change it by -$3.94/MWh. A positive number means the named choice has higher MAE. The 2019-start window averages $31.88/MWh and the full window averages $32.92/MWh. The primary fits were not switched to whichever cell won on 2023.

The depth and learning-rate grid is scored on 2023 only, after early stopping on the last 90 days of 2022. The lowest 2023 MAE in the grid is $29.09/MWh at depth 4, learning rate 0.03, model xgb_exog. The frozen primary setting, depth 6 and learning rate 0.05, has 2023 MAE $30.94/MWh for exogenous features without Chronos. 2024 was not rescored with the 2023 winner.

HB_NORTH uses that hub's hourly maximum as the target, the same conservative columns, and the HB_HUBAVG Chronos cache without refitting it. Across five seeds, XGBoost without Chronos has 2024 MAE $11.50/MWh and XGBoost with the Chronos statistics has $11.32/MWh. The seed-averaged MAE gap is -$0.19/MWh with interval [-$0.47, $0.01], which covers zero.

GFS previous-run temperature at a one-day lead is scored from a training window that starts 2021-04-01, because that archive starts in March 2021. It is temperature, not a wind or solar generation forecast. On 2024, seed 42, the temperature-augmented exogenous model has MAE $11.79/MWh and the same model with Chronos statistics has $11.69/MWh. The conservative model trained on that same shorter window has $11.62/MWh without Chronos and $11.57/MWh with Chronos. Adding the temperature forecasts does not lower MAE relative to that matched window.

## 7. Related work and what is not novel

Appending a frozen forecast to a gradient-boosted model is stacking. Lago, Marcjasz, De Schutter, and Weron, Applied Energy 293 (2021) 116983, already require ex-ante features, a LEAR baseline, multi-year evaluation, and dependence-aware comparison for day-ahead prices. Their target is a day-ahead auction price, not the real-time hourly maximum used here.

The FutureBoosting preprint, arXiv:2603.06726, uses a frozen time-series foundation model to forecast variables that are not yet known and passes those forecasts to a gradient-boosted regressor for electricity prices. The present study instead feeds a univariate Chronos-T5-small forecast of the price itself into XGBoost, and its useful content is the publication-time audit and the matched ablation, not a new hybrid architecture.

Chronos is Ansari and coauthors, arXiv:2403.07815. The capstone PDF's TimesFM reference, arXiv:2304.08424, is the TiDE paper, not TimesFM.

Four framings were considered. Foundation-model feature augmentation is not a sufficient claim of novelty. Extreme-event skill is not the lead claim unless the regime table shows a gain on spikes that survives the same interval; the point metrics above are the ones that were pre-registered. Robustness across years is only as strong as the expanding-window table. The framing that matches the evidence is a point-in-time benchmark of when these Chronos statistics help or fail.

## 8. Limitations

Real-time price posting is assumed to have happened by the top of the next hour. The delay was not measured. Lagged load uses the same kind of assumption. The gas and reactor rules that appear only in the assumed-real-time set are marked assumed in the YAML file. Archive weather is not a forecast vintage. The GFS sensitivity is temperature at a one-day lead from 2021 onward, not the operational ERCOT wind and solar forecasts.

2024 was used in earlier development. Scoring a pre-specified comparison on it is not the same as confirming a model that was chosen on it, and it is also not a fresh test. The expanding window does not repair that, because the columns were chosen with knowledge of this history. A later period is a confirmation only to the extent the fetch section reports one.

The bootstrap treats dependence inside 24-hour blocks. It is not a claim that 24 is the true dependence length. Five seeds describe fitting noise for this tree. They are not a posterior. The study is one hub-average series. The HB_NORTH transfer is the only geographic check, and it reuses the hub-average Chronos cache.

Spike recall near one half, if that is what the table shows, still leaves a large share of $200 hours undetected, and absolute errors on the highest prices remain large. Predicting the hourly maximum is a different decision from predicting the hourly average or the day-ahead settlement. This paper does not claim the maximum is the right operational target. It keeps it because it is the target the earlier study defined.

## 9. Reproducibility

From the repository root, with the local raw files and the Chronos cache present:

```bash
python models/run_pit_ablation.py all
python -m pytest tests/test_pit_sets.py tests/test_feature_availability.py tests/test_honest_mae.py -q
```

The command writes `runs/pit_ablation/` and this manuscript's numerical sections. `artifacts/honest/` and `docs/static/pdfs/paper.pdf` are not modified. Data hashes, the git commit, package versions, and the GPU string are in `runs/pit_ablation/provenance.json` and `runs/pit_ablation/build_manifest.json`. Chronos weights are the public checkpoint `amazon/chronos-t5-small` and are not vendored. A 2025–2026 extension also needs network access to ERCOT's public archive and does not retrain Chronos.

## Appendix A. Capstone PDF table, not a result of this protocol

These figures are what the PDF prints. They were not produced by one executed information set. They are not compared with the tables above.

| Model | MAE | RMSE | Spike recall | Spike precision |
| --- | ---: | ---: | ---: | ---: |
| Chronos hybrid (PDF) | 12.13 | 93.98 | 61.40% | 50.43% |
| CatBoost (PDF) | 13.75 | 95.48 | 44.35% | 56.94% |
| NGBoost (PDF) | 14.59 | 106.44 | 25.44% | 59.18% |
| Chronos (PDF) | 14.72 | 115.59 | 42.11% | 60.76% |
| SARIMAX (PDF) | 18.45 | 124.31 | 45.61% | 38.81% |
| LightGBM (PDF) | 19.76 | 100.41 | 34.21% | 45.35% |
| LSTM (PDF) | 24.42 | 148.72 | 16.67% | 23.46% |
| XGBoost (PDF) | 24.43 | 112.46 | 57.02% | 25.10% |
| TimesFM (PDF) | 32.51 | 136.94 | 1.75% | 11.76% |
| XGBoost, no sentiment (PDF) | 34.53 | 122.57 | 50.53% | 19.54% |
| TFT (PDF) | 45.75 | 134.60 | 0.00% | 0.00% |
| Prophet hybrid (PDF) | 48.36 | 108.96 | 71.05% | 14.16% |
| Prophet (PDF) | 119.14 | 168.75 | 57.14% | 17.70% |

The honest-tree MAE of $11.56 and the stack MAE of $12.13 with RMSE $96.23 belong to `reports/honest_mae.md`. They are traceable on the frozen index and are not rows of the PDF table.
