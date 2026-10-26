# Paper rewrite handoff

Write from this file and the reports next to it. Do not restore claims from the old PDF. Do not cite 2025 or 2026.

## Title

A point-in-time evaluation of Chronos features for one-hour-ahead ERCOT prices, under an explicit publication delay.

## Question

After removing features that are not known to be public, and after lagging real-time prices and realized load by a stated delay, do Chronos-T5-small statistics from that delayed context lower the absolute error of an XGBoost model that already sees the legal lags, day-ahead prices, storage, and calendar?

## What the study does

The target is the hourly maximum of four 15-minute HB_HUBAVG prices. The issue time is the start of the hour. The primary delay is one hour after interval end. It is an assumption, chosen as the shortest delay in a pre-specified list, not as the best MAE. Trees are matched: same rows, same seeds {0, 1, 2, 3, 42}, depth 6, learning rate 0.05, absolute error on asinh, early stopping on 2023, training on 2019–2022, score on the existing 2024 index. The gap is MAE(with Chronos) minus MAE(without).

## Numbers, and the files that hold them

- Pre-correction seed-averaged gap, audit baseline only: about $-0.09, interval covering zero. `reports/pit_verified_reproduction.md`.
- Corrected primary gap, MAE of the averaged forecast: $-0.15, interval [$-0.32, $0.06], excludes zero: False. With Chronos $14.92, without $15.07. `runs/pit_verified/comparisons.json`.
- Difference of the mean seed-level MAEs: $-0.14. This is not the same estimand.
- Spike MAE with Chronos $562.28, without $565.56. Recall 0.202 versus 0.167.
- February 2021 exogenous MAE on the crisis window: $3836.73. `runs/pit_verified/crisis_2021.json`.
- Hourly-mean secondary gap: $-0.18. Its interval can exclude zero. It does not replace the hourly-maximum headline. `runs/pit_verified/hourly_mean.json`.
- HB_NORTH, same market, own delayed prices and own Chronos cache: see `runs/pit_verified/hub_north.json`. Do not promote it over the hub-average result.
- Regime, block-length, spike-interval, and baseline tables: the reports with those names, each pointing at a CSV.

## Negative and limited findings

- No historical first-publication timestamp was found. The price file is a weekly archive and may be revised.
- The one-hour delay is not a verified ERCOT rule.
- The primary interval has to be read from the comparison file. If it covers zero, Chronos is not shown to help. If it excludes zero, the result is still conditional on the delay assumption and on archived prices.
- Seed-level gaps need not share a sign. Report the averaged-forecast gap and the per-seed list, not a single seed.
- February 2021 is an expanding-window stress check. A large MAE there is a statement about that window, not a tuning target.
- Regime bins are realized-price slices. Do not call a benefit when the interval covers zero or the count is small.
- The simplified three-lag lasso is not Lago et al. LEAR.
- HB_NORTH is another ERCOT hub, not another market. The new result uses that hub's own delayed prices and its own Chronos cache. The older transfer that borrowed the hub-average cache is not this result, and it is not independent.
- The hourly mean does not replace the hourly maximum.

## Claims to remove

- Every feature is known at the forecast origin.
- Real-time prices are documented to be available with zero delay.
- The old PDF hybrid MAE, spike precision, and GDELT percentage.
- A claim that Chronos improves ERCOT forecasting, unless the corrected interval lies entirely below zero and the sentence states the delay assumption and the archive limitation.
- A claim that the result is fully point-in-time verified.
- Any 2025 or 2026 number.
- CatBoost ordered boosting as a temporal embargo, and the TimesFM citation that points at TiDE.

## Claims that can stay, with the stated limit

- The target, the issue time, and the one-hour delay are explicit.
- Day-ahead prices are gated at 14:00 US/Central, a buffer after a documented 13:30 deadline. Corrections are still possible.
- Storage uses a Friday 18:00 UTC buffer after a Thursday 10:30 a.m. Eastern release.
- Matched trees isolate the Chronos columns under that information set.
- The verdict is partially verified, with unresolved concerns. `reports/pit_verification_final.md`.

## Suggested structure

1. Question and target.
2. What "available" means, and the source register.
3. The delay assumption and the tests.
4. Models, seeds, and the two MAE estimands.
5. Primary result and latency table.
6. February 2021, regimes, baselines, uncertainty.
7. What is not claimed.

## Sources actually consulted

See `reports/source_evidence_register.md`. Access date 2026-10-13.

## Outstanding

A first-publication archive of NP6-905-CD is not in this repository. Without it, the verdict should not move to fully verified. A new Chronos sample would measure sampling noise, not publication time.
