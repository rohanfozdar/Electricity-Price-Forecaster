# Target definition

The headline target is the maximum of the four 15-minute HB_HUBAVG settlement prices whose interval starts are t, t+15, t+30, and t+45 minutes, with t in UTC. The forecast is issued at t. The target is not observable at t. Under a one-hour posting delay it is not fully observable until t+2 hours, because the last interval ends at t+1 hour and the delay adds another hour.

Central time is a whole-hour offset from UTC, so a UTC hour is a Central clock hour. The feature calendar uses US/Central. The scored 2024 index is the existing matrix index, which runs through 2024-12-31 06:00 UTC. That is Central midnight at the end of December 31, not a full Central-year tail into January 1. No 2025 hour is added to repair that edge.

Incomplete hours are kept when a price exists. The 15-minute hourly maximum and the matrix target differ by more than one cent on 0 overlapping hours.

On the hourly-mean target, the seed-averaged gap is $-0.18, interval [$-0.28, $-0.07], excludes zero: True. The difference of the mean seed MAEs is $-0.16.

The hourly mean is a secondary outcome on the same one-hour-delay features. Chronos statistics in that fit are still forecasts of the hourly maximum. A mean-target interval that excludes zero does not replace the hourly-maximum headline.
