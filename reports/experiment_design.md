# Experiment design (Gate B)

Frozen config: `configs/experiment_early_warning.yaml`. Availability rules: `configs/feature_availability.yaml`. This document explains the choices behind them. It was written before any Gate D model was trained.

## 1. Question

At issue time *t*, can we estimate the probability that ERCOT's hub-average real-time price exceeds $200/MWh at least once in the next *H* hours, for *H* ∈ {24, 48, 72, 96, 120, 144}? And do features derived from a zero-shot Chronos forecast improve that estimate over well-specified point-in-time baselines?

The Chronos-augmented model is a hypothesis to test, not an assumed winner. A null or negative result is reported as such.

## 2. Definitions

- **Hourly price.** *y*[*h*] is the maximum of the four 15-minute real-time settlement point prices in the UTC hour [*h*, *h*+1). This is the target used throughout the capstone. With 4 intervals per hour, it amounts to "any 15-minute interval above the threshold". Hours with fewer than 4 intervals are treated as missing.
- **Issue time.** Every 6 hours at 00, 06, 12 and 18 UTC. At *t*, prices are known for hours *h* ≤ *t*−1.
- **Label.** *z*(*t*, *H*) = 1 if max(*y*[*t*], …, *y*[*t*+*H*−1]) > threshold, i.e. over the real-time interval (*t*, *t*+*H*]. If any hour in the window is missing, the label is missing and the row is dropped; it is never set to 0.
- **Events.** Exceedance hours are merged into one event when the gap between them is at most 6 hours (12 hours as a sensitivity). The event onset *s* is its first exceedance hour.
- **Event detection.** An event is detected at horizon *H* if some alerting origin *t* has *s* − *H* + 1 ≤ *t* ≤ *s*. The lead time is *s* − *t* for the earliest such alert. Only events whose whole opportunity window [*s*−*H*+1, *s*] falls inside the test period are counted.
- **Alert episodes.** An episode is a maximal run of consecutive alerting origins. It is *true* if any of its origins has *z* = 1. Event precision is true episodes divided by all episodes. False-alert rate is false episodes per 30 days.

## 3. Splits

The windows expand, and the data is purged. A row belongs to a split only if its whole label window lies inside that split, so no training label ever sees validation or test prices.

| Fold | Train origins | Validation origins | Test origins |
|---|---|---|---|
| test2023 | *t*+*H* ≤ 2022-01-01 | 2022 | 2023 |
| test2024 | *t*+*H* ≤ 2023-01-01 | 2023 | 2024 |

Every quantity fitted on data is fitted on training origins only: the missing-value medians, the scalers, the P99 threshold, and the XGBoost trees. Early stopping, hyperparameter selection, Platt calibration and the alert threshold use validation only. The test set is used only once, to score the frozen models.

**Limitation.** One validation year serves all four roles. This is applied identically to every model, and is recorded as a source of mild optimism in validation, not in test.

## 4. Experiments

| Experiment | Period | Weather information | Claim type |
|---|---|---|---|
| **Point-in-time** (primary) | 2016–2024, both folds | past observations only, lagged 1 hour | operational, but conservative: it omits weather forecasts |
| **Forecast weather** | 2021-03-30 to 2024, fold test2024 | point-in-time + GFS 2 m temperature forecasts issued at least 6 h before *t* | operational |
| **Oracle weather** | 2016–2024, both folds | point-in-time + *realized* weather over the label window | upper bound only |

The weather rows use the Open-Meteo archive, a reanalysis, as a proxy for real-time station observations. A sensitivity run lags them by 120 hours, ERA5's publication delay, which makes them strictly release-safe.

## 5. Feature families (all as-of the issue time)

| Family | Contents |
|---|---|
| price_history | *y*[*t*−1], *y*[*t*−24], *y*[*t*−168]; trailing max and mean over 6, 24, 72, 168 and 720 h; std over 24 and 168 h; min over 24 h; counts above $100 and $200 over 24, 168 and 720 h; hours since the last exceedance of $100 and of $200. All values signed, with no clipping. |
| calendar | local hour, weekday, month, weekend flag, US federal holiday flag, day-of-year sine and cosine (US/Central) |
| day_ahead | DA hub prices known at *t* (by DAM publication time): over the part of the label window already known, the max, the mean, the count above $100 and $200, and the number of known hours; the max and mean of the last published operating day; the mean RT−DA spread over the past 24 h |
| load | load at *t*−1, trailing 24 h max and mean, 24 h and 168 h change. Coverage starts in 2019; earlier values are NaN, not 0. |
| weather_obs | zone temperatures at *t*−1; cross-zone min and max; 24 h extremes; cross-zone mean wind at *t*−1 and its 24 h minimum; cooling and heating degree-hours over the past 24 h |
| fundamentals | Henry Hub (latest released price and 7-day change), storage (latest released level, weekly change, deviation from the prior 5-year same-week mean), NRC (offline sum and 1-day change), all by release time |
| news | GDELT tone and volume for the previous complete UTC day, 3-day and 7-day mean tone, 30-day z-scores |
| weather_forecast | over the window, GFS forecast temperature: max, min, mean, cooling and heating degree-hours (lead day *N* = ⌈(*k*+6)/24⌉ for offset *k*) |
| weather_oracle | the same summaries, from realized reanalysis |
| chronos (per *H*) | Over steps 1..*H* of the 144 h forecast issued at *t* from *y*[*t*−512 … *t*−1]: the max and mean of the median; the max of q90; the mean q90−q10 spread; the per-step exceedance probability (max over steps, and 1−∏(1−*p*)); the median minus DA over known DA hours; the step-1 median; recent skill (Brier score and log path-max error of the forecast issued at *t*−*H*, whose window has resolved by *t*). The T5 sample variant adds the exact P(path max > threshold). |

## 6. Models

| Model | Inputs | Notes |
|---|---|---|
| prevalence | none | training positive rate |
| persistence_max24h | trailing 24 h max | score is monotone in it; threshold set on validation |
| seasonal_naive_168h | max *y* over the same window one week earlier | |
| day_ahead_max | max known DA price in the window | |
| logistic_structured | structured | L2 penalty, C from {0.01, 0.1, 1}; imputation and scaling fitted on training data |
| XGB_price / XGB_structured / XGB_structured_news | feature sets from §5 | 8-config grid, chosen by validation AP |
| Chronos_only | Chronos features | score = max per-step exceedance probability; no training beyond Platt calibration |
| Chronos_XGB_{price, structured, full} | feature set + Chronos | same grid as XGB |
| Blend_full | calibrated Chronos_only and XGB_structured_news | weight *w* ∈ {0, 0.1, …, 1} chosen by validation log loss |

Residual-correction hybrids are out of scope. The ablation matrix, (price, structured, structured_news) × (without, with Chronos), runs at every horizon and in both folds.

## 7. Statistical plan

- **Primary hypothesis.** At *H* = 48 h, point-in-time experiment, pooled 2023+2024 test origins: Chronos_XGB_full beats **both** XGB_structured_news **and** Chronos_only on average precision. This is tested with a paired moving-block bootstrap: 7-day blocks (28 origins) resampled within each test year, 2,000 resamples. The claim holds only if both 95% CIs of ΔAP exclude 0. Because this is an intersection-union test, no multiplicity correction is needed for the primary claim.
- **Secondary.** The same two comparisons at the other five horizons, on AP by bootstrap and on Brier loss by Diebold–Mariano (HAC lag *H*/6 origins, Harvey–Leybourne–Newbold correction). Holm correction across these 10 tests at α = 0.05.
- **Descriptive only**, with no significance language: event metrics, lead times, thresholds of $100/$500/P99, the 12 h merge gap, the forecast-weather and oracle experiments, Chronos context and checkpoint sensitivity, and seed robustness.
- **Causal language** is not used. Feature importance is reported as model behaviour, not as mechanism.

## 8. Chronos variant decision

The capstone used `amazon/chronos-t5-small` with sampled paths. In a timed smoke test on this machine (Apple M5), 144-hour sample forecasts took 31–249 s per 32 origins on MPS and about 180 s on CPU, i.e. more than 3.5 hours for about 13,000 origins even in the best case. `amazon/chronos-bolt-small` produced 144-hour quantile forecasts for 256 origins in about 5 s.

The primary Chronos variant is therefore Chronos-Bolt-small. It is deterministic and was designed for direct multi-step forecasting; Chronos-T5 warns that quality may degrade beyond 64 steps. The T5 sample cache is handed over as a GPU script (`scripts/run_chronos_cache.py --variant t5-small`) and is read by the same feature code. When it is returned, its results are added as a robustness row and do not replace the primary one. This decision was made before any model was trained on the new design.

## 9. Runtime estimate (Apple M5, 10 cores)

- Chronos-Bolt cache: about 5 minutes per checkpoint and context.
- Feature build: about 1 minute.
- XGBoost: about 600 fits per experiment (6 models × 6 horizons × 2 folds × 8 configs) at 1–3 s each, so about 20 minutes. Sensitivities add about 30 minutes.
- Bootstrap and DM tests: a few minutes.

## Deviations

None yet.
