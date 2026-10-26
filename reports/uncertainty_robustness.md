# Uncertainty

The primary gap uses 2,000 moving-block bootstrap draws and seed 0. The pre-specified block is 24 hours. The same draw count and seed are repeated at 48, 72, and 168 hours. The gap is MAE(with Chronos) minus MAE(without) for the seed-averaged one-hour-delay forecasts. The Diebold-Mariano lag matches the block length in this table. The 24-hour row is the one paired with the ablation.

| Block hours | Gap | 95% interval | Excludes zero | DM p |
| --- | --- | --- | --- | --- |
| 24 | $-0.15 | [$-0.32, $0.06] | False | 0.124 |
| 48 | $-0.15 | [$-0.33, $0.06] | False | 0.149 |
| 72 | $-0.15 | [$-0.34, $0.08] | False | 0.165 |
| 168 | $-0.15 | [$-0.34, $0.09] | False | 0.188 |

The primary 24-hour interval from the ablation file is [$-0.32, $0.06].

Spike precision, recall, and F1 use the same blocks on the seed-averaged forecasts. A draw with no true spike leaves recall and F1 undefined. A draw with no predicted spike leaves precision and F1 undefined. Those draws are counted below and are left out of the quantile for that metric. They are not replaced with zero. Quantiles are omitted in the CSV when fewer than 100 draws are defined; the point estimates below are the full-sample scores.

| Model | Block | Precision | Recall | F1 | Undefined precision draws | Undefined recall draws | Undefined F1 draws |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_both | 24 | 0.622 | 0.202 | 0.305 | 0 | 0 | 0 |
| xgb_both | 48 | 0.622 | 0.202 | 0.305 | 0 | 0 | 0 |
| xgb_both | 72 | 0.622 | 0.202 | 0.305 | 0 | 0 | 0 |
| xgb_both | 168 | 0.622 | 0.202 | 0.305 | 0 | 0 | 0 |
| xgb_exog | 24 | 0.559 | 0.167 | 0.257 | 0 | 0 | 0 |
| xgb_exog | 48 | 0.559 | 0.167 | 0.257 | 0 | 0 | 0 |
| xgb_exog | 72 | 0.559 | 0.167 | 0.257 | 0 | 0 | 0 |
| xgb_exog | 168 | 0.559 | 0.167 | 0.257 | 0 | 0 | 0 |

An interval that covers zero does not establish a Chronos benefit. Changing the block length does not, by itself, repair an unverified publication time.
