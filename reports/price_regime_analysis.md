# Price regimes in 2024

Bins were fixed in `models/run_pit_verified.py` before the corrected predictions were read: negative, 0–50, 50–100, 100–200, 200–500, and above 500 dollars per megawatt-hour. They are subsets of the realized price. They are not model inputs. The gap is MAE(with Chronos) minus MAE(without) on the seed-averaged primary forecasts. A benefit is not claimed when the interval covers zero or the hour count is below 100. The bootstrap resamples the subset as one group because a price bin is not a contiguous block of hours.

| Regime | Hours | MAE with Chronos | MAE without | Gap | 95% interval | Benefit claimed |
| --- | --- | --- | --- | --- | --- | --- |
| negative | 145 | $5.44 | $5.35 | $0.09 | [$-0.44, $0.48] | no |
| 0_to_50 | 7573 | $3.97 | $4.04 | $-0.08 | [$-0.12, $-0.03] | yes |
| 50_to_100 | 755 | $29.50 | $29.69 | $-0.19 | [$-0.87, $0.66] | no |
| 100_to_200 | 180 | $75.41 | $76.74 | $-1.33 | [$-5.56, $1.21] | no |
| 200_to_500 | 89 | $228.15 | $231.64 | $-3.49 | [$-15.47, $3.48] | no |
| above_500 | 25 | $1751.79 | $1754.32 | $-2.54 | [$-2.56, $-2.01] | no |

The 0–50 bin is the only slice with at least 100 hours whose interval lies entirely below zero. The gap there is about eight cents per megawatt-hour. The bin is defined by the realized price, so the comparison describes those hours. It is not a regime the forecast could have selected in advance. The above-500 slice has fewer than 100 hours, and its interval is not treated as a benefit.

Source: `runs/pit_verified/regimes.csv`.
