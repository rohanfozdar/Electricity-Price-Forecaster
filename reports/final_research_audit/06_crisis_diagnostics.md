# February 2021 diagnostics

The window is 2021-02-10 00:00 through 2021-02-21 00:00 US/Central. The fit uses only hours before 2021, with early stopping on the last 90 days of 2020. No hyperparameter was changed for this audit.

## Alignment

The exogenous and Chronos models share one index of 8563 scored 2021 hours. The saved target matches the hourly maximum of the raw 15-minute prices. Maximum absolute difference: 0.0. Persistence is the hourly maximum at t−2, and it is finite on every scored hour. It predicts the same hour t. The information cutoff is the same one-hour delay.

The asinh decode is clipped at ±$11013.23. Exogenous forecasts at the clip: 0. Chronos-model forecasts at the clip: 0. The exogenous maximum of $697.21 is far below the clip. The inverse transform is not what caps the forecast near $700.

Early stopping selected iteration 135 for the exogenous tree and 363 for the Chronos tree, out of a 2000-tree cap and a patience of 50. The stopping rule ran. It was not a silent one-tree fit. It was also not retuned.

## Same hours, two regimes

| Slice | Model | n | MAE | Pred max | Price max | Share of absolute error in the 10 largest hours |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Crisis | XGBoost without Chronos | 264 | $3836.73 | $697.21 | $9051.55 | 0.09 |
| Crisis | XGBoost with Chronos | 264 | $3604.11 | $2366.94 | $9051.55 | 0.09 |
| Crisis | Persistence at t−2 | 264 | $540.13 | $9051.55 |  |  |
| Outside | XGBoost without Chronos | 8299 | $16.93 | $666.24 | $2001.00 | 0.12 |
| Outside | Persistence at t−2 | 8299 | $25.83 |  |  |  |

Persistence is better inside the window because the last legal price is already a storm price. The trees forecast a few hundred dollars while the realized maximum is about $9,052. Outside the window the trees are better than persistence, MAE $16.93 versus $25.83.

The fitted model's training rows reach $9005.29 at 2019-08-13 20:00:00+00:00, and their 99th percentile is $334.40. On the hourly price series before the same early-stopping cutoff, 10 hours exceed $5,000 and 5 exceed $8,000, including 13 and 15 August 2019. The series had already reached this price level. It had not contained an eleven-day stretch like February 2021. The failure is consistent with a tree that does not carry a rare spike forward, not with a misaligned target or a clipped inverse transform. A raw-target exogenous fit on the same split has crisis MAE $3824.17 and maximum prediction $669.52, so changing only the target transform does not repair the window.

No remedy was fit. Changing depth, loss, or the training window to chase February 2021 would be a different study.
