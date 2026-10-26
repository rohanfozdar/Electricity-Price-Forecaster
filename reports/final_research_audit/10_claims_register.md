# Claims register

The primary configuration is the one-hour delay, HB_HUBAVG hourly maximum, seed-averaged forecasts, 2024 frozen index, 24-hour block bootstrap.

| Claim | Result | Role | Supported? | Wording that stays inside the evidence |
| --- | --- | --- | --- | --- |
| A. Chronos lowers absolute error | Gap −$0.15, interval [−$0.32, $0.06], p 0.131 | Primary | Partially. The point estimate is negative. The interval covers zero. | Chronos statistics change 2024 MAE by about −$0.15/MWh. The prespecified interval includes zero. |
| B. The evaluation is point-in-time verified | No first-publication timestamps. Prices are the weekly archive NP6-785-ER. | Limitation | Unsupported as a full verification. | The study is a retrospective evaluation under an assumed one-hour delay. |
| C. The model is competitive with established benchmarks | Tree MAE $15.07 versus LEAR-style MAE $16.32 and persistence MAE $25.84. LEAR-style RMSE $110.17 is lower than the tree RMSE $121.60. | Secondary | Partially, against the benchmarks that can be built from this archive. | On MAE the tree is below a latency-valid LEAR-style lasso and below latency-valid persistence. On RMSE the lasso is lower. It has not been compared with a LEAR that uses load and renewable forecasts. |
| D. Chronos improves spike forecasting | Recall 0.202 versus 0.167. Precision 0.622 versus 0.559. | Descriptive | Unsupported as a separate inferential claim. | Spike counts move in the same direction as the MAE gap. No prespecified spike test excludes zero as a primary result. |
| E. The result generalizes across markets | HB_NORTH gap −$0.15, interval [−$0.29, −$0.02] | Secondary | Unsupported. | HB_NORTH is another ERCOT hub. |
| F. The model is robust in extreme regimes | Crisis MAE $3836.73 versus persistence $540.13. Forecast max $697.21 versus price $9051.55. | Diagnostic | Unsupported. The evidence goes the other way. | During 10–20 February 2021 the trees miss the price level that persistence, using the last legal hour, already reflects. |

The two MAE estimands must not be swapped. The primary gap is −$0.15. The difference of the five seed-level MAEs is −$0.14.
