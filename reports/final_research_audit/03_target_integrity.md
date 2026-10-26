# Target integrity

UTC 2024 has 8784 hours. The evaluation index has 8767. The difference is 17 hours. The raw price file runs from 2016-01-01 06:00:00+00:00 to 2024-12-31 06:00:00+00:00. Duplicate 15-minute stamps: 0. Internal gaps inside the evaluation index: 0.

Interval counts inside the evaluation index:

| Intervals in the hour | Hours |
| --- | ---: |
| 4 | 8766 |
| 3 | 0 |
| 2 | 0 |
| 1 | 1 |
| missing from the raw file | 0 |

The complete-four-interval index is a strict subset of the frozen index. The frozen index keeps the hour below. Its target is the maximum of the intervals that exist, which is one interval.

| Hour UTC | Hour Central | Intervals |
| --- | --- | ---: |
| 2024-12-31 06:00:00+00:00 | 2024-12-31 00:00:00-06:00 | 1 |

Both trees were refit on the other 8766 hours. Depth, learning rate, asinh target, seeds, training window, and 2023 early stopping are unchanged. The refit does not replace the 8767-hour primary result. The largest absolute difference between the refit seed-averaged forecast and the saved primary forecast, on the shared hours, is zero for both models. The dropped hour is the last timestamp in the raw file, so the file ends inside that hour. It is a 2024 evaluation hour, so it was outside the training window.

| | With Chronos | Without Chronos | Gap | 95% interval | Excludes zero |
| --- | ---: | ---: | ---: | --- | --- |
| Refit on complete hours | $14.92 | $15.07 | −$0.15 | [−$0.31, $0.06] | no |
| Saved primary predictions, same hours | $14.92 | $15.07 | −$0.15 |  |  |

On the complete hours the target is the maximum of four 15-minute prices. The headline remains the frozen index.


Exclusion reasons: {'after the last raw price timestamp': 17}.

| Hour UTC | Hour Central | Intervals | Reason |
| --- | --- | ---: | --- |
| 2024-12-31 07:00:00+00:00 | 2024-12-31 01:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 08:00:00+00:00 | 2024-12-31 02:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 09:00:00+00:00 | 2024-12-31 03:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 10:00:00+00:00 | 2024-12-31 04:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 11:00:00+00:00 | 2024-12-31 05:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 12:00:00+00:00 | 2024-12-31 06:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 13:00:00+00:00 | 2024-12-31 07:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 14:00:00+00:00 | 2024-12-31 08:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 15:00:00+00:00 | 2024-12-31 09:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 16:00:00+00:00 | 2024-12-31 10:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 17:00:00+00:00 | 2024-12-31 11:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 18:00:00+00:00 | 2024-12-31 12:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 19:00:00+00:00 | 2024-12-31 13:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 20:00:00+00:00 | 2024-12-31 14:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 21:00:00+00:00 | 2024-12-31 15:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 22:00:00+00:00 | 2024-12-31 16:00:00-06:00 | 0 | after the last raw price timestamp |
| 2024-12-31 23:00:00+00:00 | 2024-12-31 17:00:00-06:00 | 0 | after the last raw price timestamp |

These hours are the end of the raw file. They were not dropped by a feature filter, by Chronos warm-up, or by a daylight-saving conversion. Recovering them would require prices that are not in `data/raw/ercot_rt_prices.parquet`. They were not fetched.

The frozen index is therefore not a complete UTC year and not a complete Central year. It is every hour from 2024-01-01 00:00 UTC through the last hour present in the price file, 2024-12-31 06:00 UTC. That last stamp is midnight Central at the end of December 31. The first hours of the index are still the evening of December 31, 2023, in Central time. The paper should say that, rather than "calendar year 2024."

Central daylight-saving days in the raw file have 92 fifteen-minute stamps on 2024-03-10 and 100 on 2024-11-03. Those days are short and long in local time. They do not remove UTC hours from this index.
