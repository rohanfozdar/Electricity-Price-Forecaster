# Final research readiness

Recommendation: ready with explicitly documented limitations.

The primary numbers reproduced from the saved predictions. The 17-hour gap is the end of the raw file, not a silent row filter. One retained hour has a single 15-minute price. Refitting both trees without that hour does not move the gap's interval off zero, and it does not replace the 8,767-hour result. The February 2021 gap is a real difference between persistence and the trees on the same hours, not an alignment or clipping error. No publication-time archive was found, so the study cannot be called fully point-in-time verified.

## Issues resolved

- The zero-delay label on real-time prices was withdrawn. The executed model uses an explicit one-hour delay, and the lag at t is hour t−2. Code: `src/features/pit_sets.py`. Tests: `tests/test_pit_latency.py`.
- Chronos was rebuilt per delay from that legal history. Cache metadata matches horizon 1+delay, and a sample row's latest input matches t−(1+delay).
- The 8,767 versus 8,784 difference is enumerated in `03_target_integrity.md`. All 17 absent hours are after the last raw timestamp. One retained hour has a single 15-minute price. Both trees were refit on the other 8766 hours. That sensitivity does not replace the primary result. Gap −$0.15, interval [−$0.31, $0.06].
- The primary MAE, interval, and Diebold–Mariano statistic were recomputed from the prediction files and match `runs/pit_verified/comparisons.json`.
- A 12-hour block was added. It does not exclude zero.
- The LEAR-style lasso is fit on the legal lags and is not labeled as Lago et al.
- February 2021 uses the same target and the same t−2 persistence cutoff. The asinh clip does not bind.

## Issues remaining

- No first-publication timestamps for NP6-905-CD or for the load display during 2019–2024.
- The weekly price archive may be a revised settlement value. A delay does not undo that.
- Load's one-hour delay is an assumption about a daily archive.
- Day-ahead corrections before 10:00 on the next business day are not marked.
- The storage holiday calendar for 2019–2024 was not applied row by row.
- A full LEAR with load and renewable forecasts cannot be built from the publication-timed series on disk.
- The last 17 UTC hours of 2024 are not in the raw file.

## Results changed

| Quantity | Earlier conservative ablation, delay 0 | This study, delay 1 hour |
| --- | ---: | ---: |
| MAE with Chronos | $11.33 | $14.92 |
| MAE without Chronos | $11.42 | $15.07 |
| Gap | −$0.09 | −$0.15 |
| 95% interval | [−$0.27, $0.11] | [−$0.32, $0.06] |

The level rose because the most recent hour of prices is no longer a feature, and because Chronos was redrawn from that shorter history. The conclusion did not change. Both intervals cover zero. The earlier $11.56 tree and $13.39 three-lag model are a third information set and are not in this table.

## Publication blockers

- Do not write that Chronos significantly improves the forecast.
- Do not write that the study is fully point-in-time verified.
- Do not write that the model is robust in February 2021.
- Do not write that HB_NORTH is another market or that the result generalizes across markets.
- Do not put 2025 or 2026 numbers in the paper.
- Do not call the three-lag historical model, or the new lasso, a full LEAR.

## Nonblocking limitations

- The evaluation index ends at 2024-12-31 06:00 UTC. Say so.
- The one-hour price delay and the one-hour load delay are assumptions. The latency table shows the gap still covers zero at 2, 3, and 6 hours.
- Spike precision and recall are descriptive.
- The hourly-mean interval excludes zero and stays a secondary target.
- The HB_NORTH interval excludes zero and stays a second hub in the same market.

## Definitive results for the paper

- With Chronos, MAE $14.92, RMSE $120.73, median AE $2.93, spike recall 0.202, spike precision 0.622.
- Without Chronos, MAE $15.07, RMSE $121.60, median AE $3.00, spike recall 0.167, spike precision 0.559.
- Gap −$0.15, 95 percent interval [−$0.32, $0.06], bootstrap p 0.131, Diebold–Mariano p 0.124.
- Hours: 8767. Hash `b850034b0690e660`.

## Contribution

Under an explicit one-hour publication delay, frozen Chronos-T5-small forecasts of the ERCOT hub-average hourly maximum change a matched XGBoost model's 2024 MAE by about −$0.15/MWh. The prespecified uncertainty interval includes zero. The same trees miss the February 2021 price level that a latency-valid persistence forecast already reflects. The study's contribution is that measurement, not a claim that the foundation-model features improve the forecast.

## Reproduction

See `09_reproducibility.md`. Definitive command: `python models/run_pit_verified.py all`. Audit command: `python models/run_final_audit.py`.
