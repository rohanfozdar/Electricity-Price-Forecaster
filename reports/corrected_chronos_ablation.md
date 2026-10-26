# Corrected Chronos ablation

The forecast is the one-hour-ahead hourly maximum of HB_HUBAVG. The issue time is the start of that hour. The primary information set uses a one-hour publication delay on real-time prices and on realized load. That delay was fixed from the source register before these errors were computed. Seeds are 0, 1, 2, 3, and 42. Trees use depth 6, learning rate 0.05, and absolute error on the asinh target, with early stopping on 2023.

The headline gap is the MAE of the seed-averaged forecast with Chronos minus the MAE of the seed-averaged forecast without Chronos. A negative number favors Chronos.

| Estimand | With Chronos | Without Chronos | Difference |
| --- | --- | --- | --- |
| MAE of the averaged forecast | $14.92 | $15.07 | $-0.15 |
| Mean of the five seed MAEs | $15.03 | $15.17 | $-0.14 |

The 24-hour block bootstrap interval for the averaged-forecast gap is [$-0.32, $0.06]. Bootstrap p is 0.131. The Diebold-Mariano statistic is -1.54 with p 0.124. The interval excludes zero: False.

Other scores of the averaged forecast, with Chronos and without:

| | With Chronos | Without |
| --- | --- | --- |
| RMSE | $120.73 | $121.60 |
| Bias (forecast minus price) | $-10.38 | $-10.78 |
| Median absolute error | $2.93 | $3.00 |
| Spike precision | 0.622 | 0.559 |
| Spike recall | 0.202 | 0.167 |
| Spike F1 | 0.305 | 0.257 |
| MAE on hours above $200 | $562.28 | $565.56 |
| MAE on the other hours | $7.71 | $7.82 |
| Hours | 8767 | 8767 |
| Spikes | 114 | 114 |

Per-seed gaps (with Chronos minus without), seeds 0, 1, 2, 3, 42: $0.13, $-0.15, $-0.18, $-0.30, $-0.20.

The saved pre-correction gap on the same task was about $-0.09, with an interval that covered zero. This file does not treat a similar number as confirmation that the old information set was valid. 2024 was already inspected. Chronos rows are a new T5-small sample. The context ends at the last legal hour, and the selected step is the forecast of hour t. See `runs/pit_verified/comparisons.json`.

Hourly maxima from the 15-minute file and from the modeling matrix differ by more than one cent on 0 overlapping hours. The maximum absolute difference is $0.0000. Lag features and the new Chronos context both use the 15-minute hourly maximum.
