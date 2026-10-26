# Latency sensitivity

The delays are assumptions. None of them is a measured 2019–2024 posting time. The primary row is the one-hour delay, not the row with the smallest MAE.

A one-hour delay makes hour t−1 unavailable at the start of hour t, because the last interval of hour t−1 ends at that instant. Two, three, and six hours push the last legal hour to t−3, t−4, and t−7. The 24-hour load row keeps the one-hour price delay and lags realized load by 24 hours. The dropped-load row removes realized load.

| Scenario | Price delay | Load | MAE with Chronos | MAE without | Gap | 95% interval | Excludes zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| delay_1h | 1h | 1h | $14.92 | $15.07 | $-0.15 | [$-0.32, $0.06] | False |
| delay_2h | 2h | 2h | $16.23 | $16.39 | $-0.17 | [$-0.34, $0.04] | False |
| delay_3h | 3h | 3h | $16.55 | $16.65 | $-0.10 | [$-0.32, $0.20] | False |
| delay_6h | 6h | 6h | $17.03 | $17.12 | $-0.09 | [$-0.18, $0.01] | False |
| delay_1h_load_24h | 1h | 24h | $15.13 | $15.24 | $-0.11 | [$-0.34, $0.24] | False |
| delay_1h_load_dropped | 1h | dropped | $14.79 | $14.97 | $-0.18 | [$-0.44, $0.18] | False |

Each delay has its own Chronos cache under `runs/pit_verified/chronos/`. The context is the 512 clock hours of the latency-filtered hourly maximum ending at the last legal hour. The model forecasts far enough to reach hour t, and that step is the feature. This is not a shift of the old cache in `data/features/chronos_hourly/`. One timed batch, before the full rebuild, implied about 26.01 minutes per cache. Each saved row stores the latest input timestamp and a hash of the context.

The gap above is MAE(with Chronos) minus MAE(without), using the seed-averaged forecast. An interval that covers zero is not evidence that Chronos helps.
