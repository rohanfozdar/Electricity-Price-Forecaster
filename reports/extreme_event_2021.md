# February 10–20, 2021

The window is midnight US/Central on February 10 through midnight on February 21, exclusive. It was fixed before the corrected predictions were scored. The fit is an expanding window: training uses only hours before the last 90 days of 2020, those 90 days are the early-stopping sample, and 2021 is scored once. This is not a model trained through 2022.

Training prices before that early-stopping window reach a maximum of $9005.29. Their 99th percentile is $334.40. The asinh decode is clipped at sinh(10), about $11013.23. Calendar hours in 2021 missing from the matrix index: 197. Scored hours: 8563.

| Slice | Model | n | MAE | RMSE | Bias | Pred max | Price max | Forecasts above $10,000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crisis_hours | xgb_exog | 264 | $3836.73 | $5523.16 | $-3825.32 | $697.21 | $9051.55 | 0 |
| crisis_hours | xgb_both | 264 | $3604.11 | $5239.81 | $-3574.59 | $2366.94 | $9051.55 | 0 |
| crisis_hours | persistence | 264 | $540.13 | $1446.69 | $0.78 | NA | NA | NA |
| crisis_hours | xgb_exog raw target | 264 | $3824.17 | $5512.65 | $-3805.67 | $669.52 | NA | NA |
| all_scored_2021 | xgb_exog | 8563 | $134.70 | $973.82 | $-129.65 | $697.21 | $9051.55 | 0 |
| all_scored_2021 | xgb_both | 8563 | $127.78 | $924.19 | $-120.27 | $2366.94 | $9051.55 | 0 |
| all_scored_2021 | persistence | 8563 | $41.69 | $276.73 | $0.00 | NA | NA | NA |
| outside_crisis | xgb_exog | 8299 | $16.93 | $89.95 | $-12.09 | $666.24 | $2001.00 | 0 |
| outside_crisis | xgb_both | 8299 | $17.20 | $88.91 | $-10.38 | $882.85 | $2001.00 | 0 |
| outside_crisis | persistence | 8299 | $25.83 | $111.54 | $-0.02 | NA | NA | NA |

On the saved exogenous forecasts, the largest prediction in the crisis window is $697.21 and the largest price is $9051.55. Bias is $-3825.32. No saved forecast in the exogenous asinh model reached $10,000, so the asinh clip did not bind on this window. Before the early-stopping window, the 99th percentile of prices is $334.40 and the maximum is $9005.29, at 2019-08-13 20:00:00+00:00. That maximum is the same order as the crisis peak, so the miss is not explained by the training sample never having contained a price near $9,000. The forecasts still stayed far below the storm. Persistence in this table is the last legal hourly price under the one-hour delay, not the price from hour t−1. On the crisis hours it has a much lower MAE than either tree. Outside that window the trees have lower MAE than persistence.

Detail is in `runs/pit_verified/crisis_2021.json`.
