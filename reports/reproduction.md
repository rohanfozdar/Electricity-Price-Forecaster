# Reproduction of the headline scores

Frozen 2024 test year: 8,767 hours, 114 hours with `HB_HUBAVG` above $200/MWh, index hash `b850034b0690e660`. Every setting below was chosen on 2023 validation. The 2024 year was scored once per finalist. Prediction vectors and the side-by-side leaderboard are in `runs/benchmark/` (gitignored). Regenerate the leaderboard with `python models/run_benchmark.py leaderboard`.

## Headline price forecast

| Metric | Paper | Faithful Chronos–XGBoost | Improved (5-seed mean) | Alternative (5-seed mean) |
| --- | ---: | ---: | ---: | ---: |
| MAE | 12.13 | 21.96 | 13.06 ± 0.17 | 13.90 ± 0.13 |
| RMSE | 93.98 | 112.83 | 107.18 ± 1.49 | 109.11 ± 0.38 |
| Spike recall | 61.40% | 58.77% | 38.1% ± 3.4 | 43.7% ± 1.3 |
| Spike precision | 50.43% | 30.45% | 70.9% ± 3.3 | 60.3% ± 0.9 |

The improved model misses the paper on MAE (13.06 against 12.13), RMSE (107.18 against 93.98), and spike recall (38.1% against 61.40%). Its spike precision, 70.9%, is above the paper's 50.43%. Across its five seeds the best MAE is 12.81 and the best RMSE is 105.44, so no seed reaches 12.13 or 93.98. The best seed recall is 42.1%.

The faithful model is the notebook 07 hybrid: `amazon/chronos-t5-small`, 512 hours of prior rows, median of 20 paths (torch seed 0), plus the 87 engineered features, XGBoost squared error, 500 trees, depth 6, early stopping 25, `random_state` 42. A 10-seed sweep of that XGBoost stage stays in MAE 18.60–24.64 and precision 27.5–41.1%. The paper's MAE and precision sit outside that range. Recall does reach 61.4% at one seed, with precision 41.1% at best.

**Improved** (lowest 2023 MAE among 68 Phase 2 configurations whose 2023 spike F1 was at least the faithful hybrid's 0.617): gap-aware Chronos-T5 context, the five Chronos statistics plus the q90–q10 spread, an `asinh` target, absolute-error XGBoost, depth 6, learning rate 0.05. 2023 MAE 31.85, spike F1 0.653.

**Alternative** (same rule over the 62 caveat configurations): Chronos-Bolt base, gap-aware context, residual target `y − chronos_pred`, pseudo-Huber loss, depth 4, learning rate 0.03. 2023 MAE 31.58, spike F1 0.657. That is the lowest eligible 2023 MAE in the whole search, and on 2024 it is farther from the paper than the Phase 2 model. The search also tried Chronos-Bolt small, LightGBM, CatBoost, a residual on the day-ahead price, blends of Chronos and the tree model, and a spike-gated two-stage model. Logged in `runs/benchmark/search_log.csv`.

## Other price rows

| Result | Paper | Reproduced (seed 42) | 10-seed MAE range |
| --- | ---: | ---: | ---: |
| Chronos zero-shot MAE / RMSE / recall / precision | 14.72 / 115.59 / 42.11% / 60.76% | 14.54 / 113.62 / 41.23% / 61.04% | 14.54–14.89 over 4 sampling seeds |
| XGBoost consolidated MAE | 24.43 | 24.89 | 20.31–26.37 |
| XGBoost with GDELT MAE | 22.21 | 24.89 | 20.31–26.37 |
| XGBoost without GDELT MAE | 24.64 | 25.35 | 20.46–25.81 |
| Enhanced model vs day-ahead MAE | 19.56 | 25.03 | 20.82–25.47 |
| Day-ahead price MAE / RMSE / recall / precision | 17.51 / 112.85 / 28.95% / 48.53% | 17.51 / 112.85 / 28.95% / 48.53% | exact |

Chronos zero-shot matches the paper inside sampling noise. One extra true positive (48 of 114) is the whole gap between 41.23% and 42.11% recall.

The day-ahead price row matches the paper on every reported figure. That checks the test index and the metric code. The enhanced-model MAE of 19.56 is below every seed on this machine (minimum 20.82). Its spike recall of 51.75% is the minimum of the seed sweep.

The GDELT ablation does not reproduce as a stable effect. Pairing the with-GDELT and without-GDELT models on the same ten seeds, the MAE difference averages −0.10 with a standard deviation of 2.13, and the with-GDELT model is better on 4 of 10 seeds. The paper's 9.9% MAE drop (24.64 to 22.21) is larger than that spread. Spike PR-AUC is higher with GDELT by 0.011 on average.

## Why two table rows cannot be matched exactly

Spike recall and precision are `TP / 114` and `TP / (TP + FP)` on this test set, so both percentages have to be integer ratios.

- Chronos–XGBoost, 61.40% and 50.43%. The only true-positive count that prints as 61.40% is 70 (`70/114 = 0.614035`). Precision 50.43% would then require `70 / n = 0.5043`, so `n = 138.81`. The neighbours print as 50.72% (`70/138`) and 50.36% (`70/139`).
- CatBoost, 44.35% and 56.94%. `0.4435 × 114 = 50.559`, which is not an integer. `50/114` prints as 43.86% and `51/114` as 44.74%.

No spike count from 90 to 139 makes either pair exact at two decimal places. Prophet's 57.14% recall is `60/105`, so that row was scored on a different spike count. The success rule used here was to meet or beat each target separately. Precision is met. MAE, RMSE, and recall are not, and the gap is reported as it stands.

## 24-hour alerts

| Setting | Threshold | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| Paper | 0.30 | 34.63% | 24.22% | 28.50% |
| Paper protocol, this machine | 0.30 | 32.72% | 24.92% | 28.29% |
| Validation-chosen, 5-seed mean | 0.28 (mean of the five cuts) | 43.10% ± 8.3 | 24.63% ± 1.0 | 31.01% ± 1.3 |

The paper protocol refits the untransformed `train_all_v2` classifier, picks Platt because it has the lower 2024 Brier score, and alerts at 0.3. Over ten classifier seeds that protocol spans recall 32.4–47.6%, precision 22.4–29.3%, and F1 29.6–34.2%. The paper's recall and precision sit inside those ranges. Its F1 of 28.50% sits just below the lowest seed.

The validation-chosen classifier adds the five one-step Chronos-T5 statistics to the same untransformed features. The calibrator is the one with the lower out-of-fold 2023 Brier score (7-day blocks, 5 folds), and the threshold is the cut that maximises out-of-fold 2023 F1. That choice, among 42 candidates, had 2023 F1 0.505. Each of the five test seeds repeats the choice. The mean recall, precision, and F1 are all above the paper's. PR-AUC is 0.250 against 0.237 for the paper protocol.

## Lead times

Hours from the first alert to the first spike above $200/MWh, using `evaluation/backtest.py`. The validation-chosen column is seed 0, with the range over five seeds in parentheses.

| Event | Paper | Paper protocol | Validation-chosen |
| --- | ---: | ---: | ---: |
| Jan 2018 cold snap | 18 | 18 | 18 (18–90) |
| Aug 2019 heatwave | 47 | 65 | 48 (45–48) |
| Winter Storm Uri | 48 | 48 | 48 (48–48) |
| June 2022 heat stress | 25 | 13 | 25 (25–25) |
| Aug 2023 heatwave | 48 | 48 | 48 (48–48) |
| Jan 2024 cold snap | 83 | 83 | 76 (75–83) |
| Aug 2024 heat | 143 | 72 | 144 (97–144) |

Five of the seven paper-protocol leads match the May 4 backtest exactly. June 2022 and August 2024 do not, and August 2019 is 65 against 47. The validation-chosen threshold restores June 2022 to 25 hours and brings August 2019, January 2024, and August 2024 inside an hour of the paper on at least part of the seed range. Hurricane Harvey has no hour above $200 in the window, in the paper's file and here.

Event-level detection (`src/evaluation/events.py`, 24-hour horizon, spike hours merged across gaps of up to 6 hours) is a different count from those seven named events. In 2024 the paper-protocol alert detects 50 of 63 events, median lead 23 hours, 21.0 false alert episodes per 30 days. The validation-chosen alert (seed 0, threshold 0.29) detects 51 of 63, median lead 22 hours, 24.2 false episodes per 30 days.

## What the numbers come from

- The day-ahead price, the alert, and the lead times trace to the untransformed `train_all_v2` pipeline that produced `evaluation/dam_benchmark.json` and `evaluation/backtest_results.json` on 4 May 2026. Commit `a8a4a57` later added a `log1p` transform to those scripts. Rerunning today's scripts on the saved artifacts does not regenerate the May scores; the harness refits the untransformed pipeline into `runs/benchmark/` and leaves the tracked evaluation JSON files as they were.
- Standalone Chronos and the XGBoost rows are reproducible on this machine up to sampling noise and the seed ranges above.
- The Chronos–XGBoost row was never produced by the saved notebooks. The hybrid cells in `notebooks/07_chronos_and_hybrid.ipynb` have no executed output, and the old MPS checkpoint covered 12,928 of about 77,400 hours. The faithful number here is a full CUDA backfill.

XGBoost with row subsampling does not match a Mac run bit for bit: the histogram builder draws different sequences under libc++ and libstdc++ for the same seed. Repeating a seed on this Linux host is stable, including across `n_jobs`. The seed sweep is the uncertainty that remains.

## Provenance

- Branch `reproduce-paper`, commit `3fbb56637cb27ca6baf85f89f76be17651579cc9`, dirty working tree (the reproduction scripts are uncommitted, on top of earlier uncommitted edits that this work left in place).
- Host: Linux, 2× NVIDIA RTX A6000, driver 595.84, CUDA 12.4. Environment `~/venvs/ercot`, Python 3.12.3.
- Packages: torch 2.6.0+cu124, xgboost 2.1.4, chronos-forecasting 2.3.2, transformers 4.57.6, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.6.1, lightgbm 4.7.0, catboost 1.2.10.
- Test index hash `b850034b0690e660` on all regression runs.
- Prediction-file SHA-256 prefixes: faithful hybrid `4395d0a4e15547d1`, improved `5bc7a3ca91debd93`, alternative `f815c10dd5159031`, Chronos zero-shot `238378ca55cc1b71`, XGBoost with GDELT `e98b52f35d874266`, XGBoost without GDELT `f4c61dfdbf70c5d2`, enhanced `9cf1f118eede68b8`, day-ahead `95788769529f7636`, alert paper protocol `9aa787a4ac3cf2af`, alert validation-chosen `3d84a8f862d79920`.
