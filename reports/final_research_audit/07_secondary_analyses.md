# Secondary analyses

The only primary test is the hub-average, hourly-maximum, one-hour-delay, seed-averaged MAE gap, with the 24-hour block bootstrap. Its interval covers zero.

## HB_NORTH

The cache file is `runs/pit_verified/chronos/hb_north_delay_1h.parquet`, built from HB_NORTH prices, horizon 2, step index 1. It is not the HB_HUBAVG cache. The price lags in that fit are HB_NORTH hourly maxima under the same one-hour delay. The protocol is the same five seeds and the same hyperparameters. The paired gap is −$0.15, interval [−$0.29, −$0.02]. The interval excludes zero: yes. MAE with Chronos $14.97, without $15.12.

HB_NORTH is another ERCOT hub. This is not evidence about another market. The interval is a nominal 95 percent interval for one secondary comparison.

## Hourly mean

Same one-hour-delay features. The outcome is the mean of the four 15-minute prices. Chronos features remain forecasts of the hourly maximum. The gap is −$0.18, interval [−$0.28, −$0.07]. The interval excludes zero: yes. This does not replace the hourly-maximum headline.

## What else was looked at

Prespecified alongside the primary test, and not promoted when the interval excludes zero:

- Price and load delays of 2, 3, and 6 hours.
- A 24-hour load lag and a dropped-load case, both at the one-hour price delay.
- Block lengths 12, 24, 48, and 168 hours.
- Six realized-price bins.
- Spike precision, recall, and F1.
- The February 2021 window.
- The LEAR-style lasso.
- HB_NORTH and the hourly mean.

A secondary interval that excludes zero is not a correction of the primary test. The primary interval covers zero. Selecting the hourly-mean or HB_NORTH result because it excludes zero would be a selection effect. Neither is the paper's main claim.
