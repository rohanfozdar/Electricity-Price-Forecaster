# Latency robustness

Each delay was fit as its own pair of trees on its own Chronos cache. The target hour does not move. At issue t the model still predicts the maximum inside hour t. Only the latest legal history changes. For delay D the latest legal hour is t−(1+D), and the cache horizon is 1+D so that the selected step is hour t.

| Series | Delay | Horizon | Step | Horizon is 1+delay | Sample latest input matches |
| --- | ---: | ---: | ---: | --- | --- |
| HB_HUBAVG | 1 | 2 | 1 | yes | yes |
| HB_HUBAVG | 2 | 3 | 2 | yes | yes |
| HB_HUBAVG | 3 | 4 | 3 | yes | yes |
| HB_HUBAVG | 6 | 7 | 6 | yes | yes |
| HB_NORTH | 1 | 2 | 1 | yes | yes |

The sample issue time for the hub-average rows is 2024-06-15 12:00 UTC. A match means the stored latest input is exactly t−(1+delay).

The saved gaps, MAE of the seed-averaged forecast, are unchanged by this audit because the models were not refit:

| Delay | With Chronos | Without | Gap | 95% interval | Excludes zero |
| --- | ---: | ---: | ---: | --- | --- |
| 1 hour | $14.92 | $15.07 | −$0.15 | [−$0.32, $0.06] | no |
| 2 hours | $16.23 | $16.39 | −$0.17 | [−$0.34, $0.04] | no |
| 3 hours | $16.55 | $16.65 | −$0.10 | [−$0.32, $0.20] | no |
| 6 hours | $17.03 | $17.12 | −$0.09 | [−$0.18, $0.01] | no |
| 1 hour price, 24 hour load | $15.13 | $15.24 | −$0.11 | [−$0.34, $0.24] | no |
| 1 hour price, load dropped | $14.79 | $14.97 | −$0.18 | [−$0.44, $0.18] | no |

Within each row the two models see the same delay. No row's interval lies entirely below zero. The one-hour row remains the primary result because it was the shortest pre-specified assumption, not because its gap is the largest.
