# Honest one-hour-ahead MAE

Validation chooses every setting below. The 2024 year is scored once per finalist, on the frozen index `b850034b0690e660` (8,767 hours, 114 hours above $200/MWh).

## Bounds

Training-set median price: $21.30/MWh. The 15-minute file is on disk; its previous-hour max matches the hourly lag (median absolute difference $0.0000). Day-ahead ramp, operating-day max/min, and the trailing RT-DA spread were not in the v2 matrix.

| Forecast | MAE | Normal MAE | Stress MAE (y > $100) | Share of absolute error from stress hours |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 17.04 | 8.90 | 251.63 | 49.5% (294 hours) |
| Seasonal naive, 24h | 29.58 | 19.39 | 323.37 | 36.7% (294 hours) |
| Seasonal naive, 168h | 32.35 | 23.13 | 298.07 | 30.9% (294 hours) |
| Day-ahead price | 17.51 | 8.65 | 273.09 | 52.3% (294 hours) |
| Chronos zero-shot | 14.54 | 6.13 | 256.85 | 59.3% (294 hours) |
| Constant training median | 21.93 | 11.46 | 323.58 | 49.5% |
| Oracle: true stress hours perfect, else training median | 11.07 | 11.46 | 0.00 | 0.0% |

The oracle uses the 2024 label to decide which hours are stress. It is a bound. Stress hours are 294 of 8767. A model that already predicts normal hours well only needs to cut the stress-hour error to move the average; the oracle says the average is not stuck near $12 by the spikes alone.

## Leakage shifts (2023 ranking, then one 2024 pair)

| Variant | 2023 MAE | 2023 spike F1 | 2024 MAE |
| --- | ---: | ---: | ---: |
| unshifted | 31.85 | 0.653 | 12.95 |
| weather_lag_1 | 33.76 | 0.649 | 12.96 |
| weather_lag_120 | 33.58 | 0.628 | - |
| gas_train_p90 | 32.25 | 0.646 | - |
| gdelt_lag_24 | 32.41 | 0.651 | - |
| chronos_lag_1 | 35.78 | 0.610 | - |
| plus_features | 29.88 | 0.688 | - |

2023 winner: `unshifted`. 2024 pair: `unshifted`, `weather_lag_1`.
Shifting Chronos features by one hour changes 2023 MAE by +3.93 (rises, so the feature is aligned).

## Frozen feature set and training window

The headline tree lags weather and the weather stress flags by one hour, rebuilds `stress_gas_spike` from the pre-2023 gas percentile, drops GDELT, and adds the previous-hour 15-minute summaries plus the day-ahead shape. `plus_features` in the table above is the same new columns on the original information set, scored on 2023 only.

Windows were ranked by 2022 MAE. The fit for that rank uses only earlier years.

| Window | 2022 MAE |
| --- | ---: |
| y2022 | 25.70 |
| y2021_2022 | 23.75 |
| y2019_2022 | 22.98 |
| full | 24.97 |

Chosen window: `y2019_2022` (train start `2019-01-01`). 2023 MAE 29.47, spike F1 0.683, stress MAE 416.89, normal MAE 5.77.
Faithful-hybrid F1 gate 0.617: cleared. The $0.5 improvement over the previous validation MAE of 31.85: met.

## LEAR and the combiner

LEAR penalty 0.01, chosen on 2022 (MAE 26.34). 2024 MAE 13.39.

Combiner candidates, fit on 2023 only:

| Combiner | 2023 MAE |
| --- | ---: |
| median | 29.85 |
| weighted_mean | 29.05 |
| qra_tau_0.5 (chosen) | 27.90 |

Components: `day_ahead`, `persistence`, `lear`, `chronos_zero_shot`, `honest_tree`. Quantile regression at τ=0.5 is not constrained to positive weights, so it can extrapolate. The weighted mean is the constrained alternative, and it lost on 2023.

## 2024 scores

| Run | MAE | RMSE | Spike recall | Spike precision | Normal MAE | Stress MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `persistence` | 17.04 | 125.44 | 41.23% | 41.23% | 8.90 | 251.63 |
| `seasonal_naive_24` | 29.58 | 176.89 | 11.40% | 11.40% | 19.39 | 323.37 |
| `seasonal_naive_168` | 32.35 | 185.21 | 7.02% | 6.96% | 23.13 | 298.07 |
| `dam` | 17.51 | 112.85 | 28.95% | 48.53% | 8.65 | 273.09 |
| `chronos_zero_shot` | 14.54 | 113.62 | 41.23% | 61.04% | 6.13 | 256.85 |
| `unshifted` | 12.95 | 104.13 | 42.11% | 69.57% | 5.50 | 227.55 |
| `weather_lag_1` | 12.96 | 104.38 | 43.86% | 69.44% | 5.59 | 225.36 |
| `honest_tree` | 11.56 | 99.96 | 50.00% | 73.08% | 4.72 | 208.58 |
| `lear` | 13.39 | 97.92 | 34.21% | 61.90% | 6.37 | 215.91 |
| `ensemble` | 12.13 | 96.23 | 53.51% | 64.21% | 5.38 | 206.78 |

## Diebold-Mariano

Absolute error, HAC lag 24, Harvey-Leybourne-Newbold correction. A negative statistic means the row has lower loss than the baseline.

| Model | Against | Mean loss gap | Stat | p |
| --- | --- | ---: | ---: | ---: |
| `unshifted` | `dam` | -4.57 | -6.11 | 0.000 |
| `unshifted` | `persistence` | -4.09 | -6.61 | 0.000 |
| `weather_lag_1` | `dam` | -4.56 | -5.92 | 0.000 |
| `weather_lag_1` | `persistence` | -4.08 | -6.74 | 0.000 |
| `honest_tree` | `dam` | -5.95 | -7.28 | 0.000 |
| `honest_tree` | `persistence` | -5.48 | -7.69 | 0.000 |
| `lear` | `dam` | -4.12 | -5.69 | 0.000 |
| `lear` | `persistence` | -3.65 | -5.87 | 0.000 |
| `ensemble` | `dam` | -5.38 | -7.31 | 0.000 |
| `ensemble` | `persistence` | -4.91 | -6.53 | 0.000 |

## Forecast checks

Chronos gap-aware context of 512 hours was checked on 8767 test hours. The target hour is outside the context.

- `persistence`: ok. Mean prediction $27.1 on hours at or below $100 (mean price $24.4) and $267.7 on hours above $100 (mean price $344.9).
- `seasonal_naive_24`: ok. Mean prediction $32.1 on hours at or below $100 (mean price $24.4) and $123.7 on hours above $100 (mean price $344.9).
- `seasonal_naive_168`: ok. Mean prediction $34.2 on hours at or below $100 (mean price $24.4) and $62.3 on hours above $100 (mean price $344.9).
- `dam`: ok. Mean prediction $23.7 on hours at or below $100 (mean price $24.4) and $154.9 on hours above $100 (mean price $344.9).
- `chronos_zero_shot`: ok. Mean prediction $23.0 on hours at or below $100 (mean price $24.4) and $159.6 on hours above $100 (mean price $344.9).
- `unshifted`: ok. Mean prediction $23.0 on hours at or below $100 (mean price $24.4) and $169.4 on hours above $100 (mean price $344.9).
- `weather_lag_1`: ok. Mean prediction $23.0 on hours at or below $100 (mean price $24.4) and $177.1 on hours above $100 (mean price $344.9).
- `honest_tree`: ok. Mean prediction $23.1 on hours at or below $100 (mean price $24.4) and $191.4 on hours above $100 (mean price $344.9).
- `lear`: ok. Mean prediction $24.0 on hours at or below $100 (mean price $24.4) and $178.0 on hours above $100 (mean price $344.9).
- `ensemble`: ok. Mean prediction $24.0 on hours at or below $100 (mean price $24.4) and $209.7 on hours above $100 (mean price $344.9).

Worst 20 absolute errors of the validation-chosen combiner:

- 2024-08-21 00:00:00+00:00: price $4852.8, forecast $375.7, error $4477.1
- 2024-11-17 21:00:00+00:00: price $3680.1, forecast $562.0, error $3118.1
- 2024-08-21 01:00:00+00:00: price $4222.2, forecast $1273.6, error $2948.6
- 2024-11-11 01:00:00+00:00: price $3353.7, forecast $547.5, error $2806.2
- 2024-05-09 01:00:00+00:00: price $4981.3, forecast $2264.3, error $2717.0
- 2024-04-29 00:00:00+00:00: price $2388.4, forecast $237.3, error $2151.1
- 2024-04-17 00:00:00+00:00: price $2395.8, forecast $322.5, error $2073.3
- 2024-05-09 00:00:00+00:00: price $4109.3, forecast $2236.3, error $1872.9
- 2024-12-14 23:00:00+00:00: price $1555.5, forecast $107.2, error $1448.3
- 2024-08-20 00:00:00+00:00: price $1575.6, forecast $289.8, error $1285.8
- 2024-05-08 23:00:00+00:00: price $1920.5, forecast $927.3, error $993.2
- 2024-04-27 01:00:00+00:00: price $1087.9, forecast $312.6, error $775.3
- 2024-01-16 23:00:00+00:00: price $846.0, forecast $123.0, error $723.0
- 2024-04-17 01:00:00+00:00: price $1398.1, forecast $704.9, error $693.2
- 2024-11-17 23:00:00+00:00: price $218.4, forecast $872.3, error $653.9
- 2024-11-17 22:00:00+00:00: price $1428.5, forecast $817.3, error $611.3
- 2024-04-29 01:00:00+00:00: price $2009.3, forecast $1430.0, error $579.3
- 2024-01-17 01:00:00+00:00: price $313.0, forecast $848.5, error $535.5
- 2024-08-08 00:00:00+00:00: price $880.8, forecast $356.7, error $524.1
- 2024-11-11 02:00:00+00:00: price $553.4, forecast $67.6, error $485.8

## What to claim

The validation-chosen combiner has 2024 MAE $12.13. The windowed tree has 2024 MAE $11.56. These two numbers were not ranked against each other on 2024.

Both scores are one XGBoost seed (42 for the tree; the combiner has no seed). They are not a five-seed mean, and 2024 was not used to choose between them.

Under $12/MWh on this index: windowed tree. That is a new one-hour-ahead result under the rules above, not a recovery of the paper row. The paper's MAE of 12.13 comes with spike recall 61.40% and precision 50.43%. On 114 spike hours, 61.40% is 70/114, and precision 50.43% is then not a ratio of two integers, so that pair cannot be this test set. The combiner MAE prints as $12.13 with RMSE $96.23 and spike recall 53.51%. The tree MAE is $11.56 with RMSE $99.96 and spike recall 50.00%. Keep the paper's 12.13 in its own column.

Hours above $100 are 294 of 8,767 and still hold about 60% of the tree's absolute error. The forecasts track those hours (the tree's mean prediction there is far above its prediction on ordinary hours) and still miss the largest spikes by thousands of dollars. Published wind, solar, and load forecasts, joined on the time they were released, are the information this pass did not have. Another squared-error or asinh variant is not.
