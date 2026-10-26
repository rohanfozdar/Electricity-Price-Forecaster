# Benchmark validity

The matched tree without Chronos, MAE $15.07 on the seed-averaged forecast, is the strong tree benchmark. No second tree family was added. Adding one would change the question.

The historical three-lag lasso at about $13.39 and the honest tree at about $11.56 used a different information set. They are not latency-valid competitors. The earlier one-hour persistence at about $17.04 used hour t−1. Under the one-hour publication delay the last legal hour is t−2. That persistence MAE, recomputed on the frozen index, is $25.84.

## Latency-valid LEAR-style lasso

This is not the Lago, Marcjasz, De Schutter, and Weron LEAR. The full specification uses day-ahead forecasts of load and renewables that this repository does not have as publication-timed series. The model that was fit uses only information that is already in the primary set:

- Price lags 1 through 24, plus 48 and 168, after the one-hour delay. The shortest lag is hour t−2.
- The publication-gated day-ahead HB_HUBAVG price.
- Realized load lagged by the same one-hour assumption.
- US/Central hour, weekday, and month dummies.

The target is asinh, decoded with sinh clipped at ±10, matching the trees. The penalty is the 2023 MAE minimum among penalties whose fit converged, 1.0e-06, validation MAE $36.94. Penalties whose 2023 MAE is within one cent of that minimum run from 1.0e-06 to 3.2e-04. The selected penalty is interior to that search. The fit uses 2019–2022 only. 2024 was not used to choose the penalty. Feature count: 71. On the evaluation hours, the day-ahead price is finite for 100.0 percent of hours and lagged load is finite for 100.0 percent. Missing values use the training median.

| | MAE | RMSE | Median AE | n |
| --- | ---: | ---: | ---: | ---: |
| LEAR-style lasso | $16.32 | $110.17 | $3.93 | 8767 |
| XGBoost without Chronos | $15.07 | $121.60 | $3.00 | 8767 |
| Latency-valid persistence | $25.84 | $169.15 | $5.33 | 8767 |

Predictions: `runs/final_audit/lear_style_predictions.parquet`.

On MAE, the tree is lower than this linear model by about $1.25. On RMSE, the linear model is lower, $110.17 against $121.60. Both are more accurate on MAE than latency-valid persistence. That supports competitiveness against a LEAR-style linear model built from the available series. It does not establish competitiveness against a LEAR that also sees a real load forecast and renewable forecasts. The comparison is not a reason to replace the primary Chronos test.
