# Ablation reproduction

Recomputed from the five saved prediction files for `delay_1h`, without refitting. The estimand is the MAE of the seed-averaged forecast. The average of the seed-level MAEs is reported beside it and is not the primary comparison.

| | With Chronos | Without Chronos |
| --- | ---: | ---: |
| MAE of the averaged forecast | $14.92 | $15.07 |
| RMSE | $120.73 | $121.60 |
| Median absolute error | $2.93 | $3.00 |
| Spike recall at $200 | 0.202 | 0.167 |
| Spike precision | 0.622 | 0.559 |

Gap, with Chronos minus without: −$0.1499. Interval [−$0.32, $0.06]. Bootstrap p 0.131. Diebold–Mariano statistic −1.54, p 0.124. Interval excludes zero: no.

Match to the saved comparison file: gap yes, interval yes. Difference of the mean seed MAEs: −$0.14. Per-seed gaps, seeds 0, 1, 2, 3, 42: $0.13, −$0.15, −$0.18, −$0.30, −$0.20.

Hours 8767. Test hash `b850034b0690e660`.

## Block length

The prespecified block is 24 hours. The other lengths were computed from the same predictions and the same 2000 draws. None is selected because it favors Chronos.

| Block hours | Gap | 95% interval | Excludes zero | DM p |
| --- | ---: | --- | --- | ---: |
| 12 | −$0.15 | [−$0.32, $0.05] | no | 0.106 |
| 24 | −$0.15 | [−$0.32, $0.06] | no | 0.124 |
| 48 | −$0.15 | [−$0.33, $0.06] | no | 0.149 |
| 168 | −$0.15 | [−$0.34, $0.09] | no | 0.188 |

A 24-hour block does not cover a weekly cycle. Price errors can remain dependent across several days, especially around a heat wave or an outage. The 168-hour row is the check for that. It still covers zero. A longer block widens the interval. It does not create a significant result, and it does not prove the dependence has been fully captured.

The Diebold–Mariano test here uses absolute error and a Bartlett lag equal to the block length, with the Harvey–Leybourne–Newbold correction. A non-significant result is not a proof that the effect is exactly zero.

## Manifest

- Predictions: `runs/pit_verified/predictions/delay_1h__max__asinh__{xgb_exog,xgb_both}__seed{0,1,2,3,42}.parquet`
- Comparison file this reproduction matches: `runs/pit_verified/comparisons.json`
- Recomputed artifact: `runs/final_audit/reproduction.json`
- Checksums: `runs/final_audit/environment.json`
