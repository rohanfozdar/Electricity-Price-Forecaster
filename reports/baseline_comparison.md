# Baselines

Persistence is the last legal hourly maximum under the one-hour delay, which is hour t−2. Seasonal naive forecasts use the price 24 and 168 hours earlier. Those hours have already ended under a delay of one to six hours. The new lasso uses the legal price lags and the Central calendar, standardized with 2019–2022 moments. Its penalty is the value with the lowest 2023 MAE. It is fit on 2019–2022 only.

The historical three-lag lasso is the simplified model already in the repository. It is not the Lago, Marcjasz, De Schutter, and Weron LEAR specification, and it was not rebuilt on the delayed prices.

The latency-valid exogenous tree, for reference, has MAE $15.07 on the seed-averaged forecast.

| Model | Information set | n | MAE | RMSE | Median AE |
| --- | --- | --- | --- | --- | --- |
| seasonal_naive_24 | delay_1h | 8767 | $29.58 | $176.89 | $7.57 |
| seasonal_naive_168 | delay_1h | 8767 | $32.35 | $185.21 | $9.11 |
| persistence_latency | delay_1h | 8767 | $25.84 | $169.15 | $5.33 |
| lasso_lags_calendar | delay_1h | 8767 | $26.20 | $151.32 | $8.31 |
| simplified_three_lag_lasso_historical | historical_not_latency_matched | 8767 | $13.39 | $97.92 | $2.71 |
| honest_tree_historical | historical_not_latency_matched | 8767 | $11.56 | $99.96 | $1.85 |
| persistence_historical | historical_not_latency_matched | 8767 | $17.04 | $125.44 | $3.02 |

Source: `runs/pit_verified/baselines.csv`.
