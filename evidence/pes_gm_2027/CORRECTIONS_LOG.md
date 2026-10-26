# Corrections log

Audit target: `inputs/manuscript_copy_B_complete.tex` (copy B). The original record is never replaced. Each entry gives the original claim, the verified finding, the cause, and the evidence. "Recomputed" means recomputed by this package from saved per-origin forecasts, using code written independently of the original statistics module (`models/pes2027/indep_stats.py`).

Status labels:
- **Error**: the manuscript states something the artifacts contradict.
- **Omission**: the statement is true but leaves out a fact that changes interpretation.
- **Convention**: the number is reproducible, but the method label or convention needs correcting.
- **Precision**: rounding or definitional detail only.

Every numeric value in copy B was checked against full precision. The results are 105 consistent with rounding, 4 values omitted, and 0 substantive differences (`runs/pes2027/E0-REPRO/manuscript_precision_check.csv`, copied to `tables/T_manuscript_precision_check.csv`). The saved forecasts reproduce bit for bit: a refit of all 10 primary models gives a maximum absolute difference of 0.0 and identical best iterations (`runs/pes2027/E0-REPRO/rerun/rerun_fits.csv`).

---

## CL-01 (Error, wording) "Exogenous-only" model in the crisis section
- **Original.** Sec. VI-C: "An exogenous-only XGBoost model reaches at most $697.21/MWh … Outside the 264 crisis hours, exogenous XGBoost achieves $16.93/MWh".
- **Verified.** This is the same 37-feature baseline as the primary comparison (code label `xgb_exog`). Those 37 features include 10 delayed hub-average price-history features and 4 delayed 15-minute price summaries. It is not exogenous-only.
- **Cause.** The code label `xgb_exog` means "without Chronos" (`src/features/pit_sets.py: feature_subsets`).
- **Evidence.** `runs/pes2027/E0-CENSUS/census.json` → `raw.features_xgb_exog`.
- **Corrected wording.** "the baseline XGBoost model without Chronos features".

## CL-02 (Omission) Crisis results come from single-seed models
- **Original.** Sec. VI-C and Table VI report crisis results without naming a seed. Table II states that headline forecasts are averaged across five seeds.
- **Verified.** All February 2021 tree results (3,836.73; 3,604.11; 697.21; 2,366.94; 16.93; 17.20; and the raw-target value 3,824.17) come from **seed 42 only**.
- **Evidence.** `runs/pit_verified/predictions/crisis2021__{asinh,raw}__xgb_*__seed42.parquet`; `models/run_pit_verified.py: run_crisis` uses `PRIMARY_SEED`.
- **Action.** E8 adds a five-seed refit on the same index, alongside the original. The seed-42 values remain the original record.

## CL-03 (Omission) Crisis-fit training window and imputed load
- **Original.** "the models train only on pre-2021 history, reserve the last 90 days of 2020 for early stopping".
- **Verified.** That statement is true. In full:
  - Training covers **2016-01-31 06:00 UTC to 2020-10-02 23:00 UTC** (40,962 rows, every row of the inherited matrix index before 2020-10-03), not 2019 onward.
  - Realized load starts on 2019-01-01, so **62.5% of crisis-fit training rows have no load**. They are filled with the training median and carry `miss_load = 1`.
  - Early stopping uses 2,143 hours from 2020-10-03 to 2020-12-31.
- **Evidence.** `runs/pes2027/E0-CENSUS/split_census.csv` (window `crisis_fit_train_pre_2020_10_03`); `feature_missingness.csv` (`load_lag_1h` 0.6245, `load_rollmean_24h` 0.6247).

## CL-04 (Omission; resolves Concern B) The 197 missing 2021 hours and 214 missing training hours
- **Original.** The crisis caption reports 8,299 + 264 = 8,563 scored 2021 hours. The manuscript does not explain the 197 missing hours (8,760 − 8,563), and it does not mention that training rows are incomplete.
- **Verified.**
  - The raw 15-minute file has one gap in 2016–2024. Hour 2020-12-31 06:00 UTC has a single interval, and **2020-12-31 07:00 UTC to 2021-01-01 05:00 UTC (23 hours) has no price**.
  - 2021 lacks 197 hours. **6** (2021-01-01 00:00–05:00 UTC) have no raw price. **191** have raw prices but were dropped from the inherited v2 matrix index. The inherited `features/engineer_v2.py` uses row-based `rolling(168)` and `shift(720)`, then `dropna(subset=["hubavg_lag_720h", "hubavg_rollmean_168h"])`. That removes the 168 hours after the gap (2021-01-01 06:00 to 2021-01-08 05:00 UTC) and the 23 hours exactly 720 hours later (2021-01-30 07:00 to 2021-01-31 05:00 UTC).
  - None of the 197 hours falls in the February 10–20 window, which has all 264 hours.
  - The 2019–2022 training window lacks **214** of 35,064 hours: 22 have no raw price (16 in 2020, 6 in 2021), 191 were dropped by the v2 rule, and 1 (2020-12-31 07:00 UTC) is in the index with a missing target.
  - The 2023 validation year is complete (8,760 of 8,760).
- **Cause.** A gap in the locally fetched raw file, propagated by an inherited row-based rule that determines which hours exist in the index. The rule looks only backward, so it is not leakage. It is an undocumented sample restriction.
- **Evidence.** `runs/pes2027/E0-CENSUS/missing_hours.csv` (every hour with its reason); `split_census.csv`; `census.json` → `raw.raw_empty_runs`.

## CL-05 (Error in explanation; resolves Concern D) Why 17 hours of 2024 are absent and the last hour is incomplete
- **Original.** Sec. III-B: "The source series ends within 2024-12-31 06:00 UTC; only one of its four 15-minute observations is present. The next 17 UTC hours … have no source prices and are therefore absent."
- **Verified.** The local file does end at 2024-12-31 06:00 UTC. The cause is the fetch window in `pipelines/ercot_rt_prices.py`: it keeps `Interval Start <= pd.Timestamp("2024-12-31").tz_localize("US/Central")` because `DATA_END = "2024-12-31"`. That keeps only the 00:00 Central interval of December 31. The truncation is therefore an artifact of the local fetch, not evidence that ERCOT published no prices. The identical signature on 2020-12-31 (one interval at 06:00 UTC, then nothing until 2021-01-01 06:00 UTC) is not explained by that filter. [MISSING: whether ERCOT's NP6-785-ER 2020 annual archive contains 2020-12-31 prices; see WP2 entry CL-12.]
- **Also.** The **training** set contains one incomplete hour (2020-12-31 06:00 UTC, 1 interval). Validation has none. The manuscript discusses only the evaluation hour.
- **Evidence.** `pipelines/ercot_rt_prices.py` lines 52–56; `utils/config.py` `DATA_END`; `runs/pes2027/E0-CENSUS/split_census.csv` (`rows_1_interval`).

## CL-06 (Error, wording; resolves Concern C) The complete-target "refit"
- **Original.** Sec. VI-B: "refits both XGBoost variants on the remaining 8,766 evaluation origins under identical model settings. The refitted predictions agree with the corresponding saved predictions".
- **Verified.** The refit kept the same 34,850 training rows (2019–2022) and 8,760 validation rows (2023). Only the scoring index changed, by dropping the single incomplete hour. Refit forecasts equal the saved forecasts (maximum absolute difference 0.0), so this analysis is a **re-scoring on 8,766 hours**, not a model change.
  - Gap −0.149896, 95% interval [−0.31215, 0.06238], bootstrap p (interval inversion) 0.144, DM p 0.1244.
  - The interval and p-value differ from the primary only because the resampled series lost one hour.
- **Evidence.** `runs/final_audit/complete_four_sensitivity.json` (`train_rows`, `val_rows`, `max_abs_diff_vs_saved_seed_average`); `runs/pes2027/E0-REPRO/paired_comparisons.csv` (rows `complete_four_refit` and `complete_four_saved_subset` are identical).
- **Corrected wording.** "Re-scoring the same forecasts on the 8,766 hours with four observed intervals gives …".

## CL-07 (Error) Baseline RMSE values shown as dashes
- **Original.** Table III shows "—" for the RMSE of four baselines, with the caption "A dash means the corresponding metric was not supplied by the final research audit".
- **Verified.** The values exist in `runs/pit_verified/baselines.csv` and were recomputed:

  | Baseline | RMSE ($/MWh) |
  |---|---|
  | Seasonal naive, 168 h | 185.212169 |
  | Seasonal naive, 24 h | 176.888489 |
  | Latest legal price (t−2) | 169.154951 |
  | Lag/calendar lasso | 151.322088 |

  All are on the same 8,767 hours.
- **Evidence.** `runs/pes2027/E0-REPRO/baselines.csv`.

## CL-08 (Convention) HLN correction in the Diebold–Mariano test
- **Original.** DM p = 0.124 (abstract, Table II). The block-length paragraph reports DM p 0.124, 0.149, 0.165, and 0.188 for 24, 48, 72, and 168 hours.
- **Verified.**
  - `src/evaluation/statistical_tests.py: diebold_mariano` applies the Harvey–Leybourne–Newbold factor with h = lag + 1 = 25, not the forecast horizon. For a one-step-ahead forecast, the HLN factor uses h = 1. With h = 1 (HAC lag 24 unchanged), the primary DM p = **0.123368**, versus 0.124383 originally. That is 0.123 rather than 0.124 at three decimals; the conclusion is unchanged.
  - In the block-length sensitivity, the DM HAC lag is set **equal to the block length** (12, 24, 48, 72, 168). The manuscript does not say this.
  - With h = 1, the block-length DM p-values are 0.1056, 0.1234, 0.1471, 0.1611, and 0.1796.
- **Evidence.** `runs/pes2027/E0-REPRO/block_lengths.csv` (`dm_p_hln_h1`, `dm_p_hln_h_lag_plus_1`, `dm_lag`).

## CL-09 (Convention) What the bootstrap p-value is
- **Original.** "bootstrap p = 0.131".
- **Verified.** The value is 2·min(P*(Δ ≤ 0), P*(Δ ≥ 0)) over 2,000 uncentered moving-block draws, floored at 1/2000. That is the smallest two-sided level at which the percentile interval excludes zero (interval inversion). It is not a resampling test under an imposed null. A null-centered version gives **0.105**, reported as supplementary only.
  - The resampler is a non-circular moving-block bootstrap: block starts are uniform on [0, n − 24], ceil(n/24) blocks are concatenated and truncated to n, and the paired loss differential is resampled jointly.
- **Evidence.** `src/evaluation/statistical_tests.py`; `models/pes2027/indep_stats.py`; `runs/pes2027/E0-REPRO/paired_comparisons.csv` (`p_boot_interval_inversion`, `p_boot_centered`).

## CL-10 (Precision) Definition of the Chronos median
- **Original.** Eq. (5) lists q0.50 among the "empirical sample quantiles".
- **Verified.** `chronos_pred` is `torch.median` over the 20 samples, which returns the **lower median** (the 10th order statistic). q10 and q90 use `torch.quantile` with linear interpolation. The spread is q90 − q10. The exceedance share uses a strict inequality (> 200).
- **Evidence.** `models/chronos_delayed_cache.py` lines 77–84.

## CL-11 (Omission) Feature counts
- **Verified.** The baseline uses **37** features: 10 price-history, 4 delayed 15-minute summaries, 5 day-ahead hub prices, 4 day-ahead shape features, 2 load, 3 storage, 5 Central calendar, and 4 missingness flags (load, storage, day-ahead, Chronos). The augmented model adds 6 Chronos summaries, for **43**. `miss_chronos` is in the baseline set and is constant on 2019–2024 rows. No feature is all-missing in training.
- **Evidence.** `runs/pes2027/E0-CENSUS/census.json`; `runs/pes2027/E0-REPRO/rerun/rerun_summary.json`.

## CL-13 (Omission) Chronology of the February 2021 window
- **Original.** "The stress-window definition predates scoring the corrected forecasts, but it remains a single historically salient episode."
- **Verified.** That statement is true for the corrected delay-1h forecasts, which were scored 2026-10-13. However, on 2026-10-12 the earlier ablation had already scored **all of 2021, including February**, under the same fit protocol (training before 2020-10-03, early stopping on the last 90 days of 2020) for two other information sets.
- **Evidence.** `runs/pit_ablation/expanding_window.csv` (rows with `year = 2021`, role `backtest_column_set_fixed_after_later_inspection`); `reports/extreme_event_2021.md`.

## CL-14 (Precision) "512-hour" Chronos context
- **Verified.** Every cache row's context is exactly the 512 clock hours ending at t − (1 + d). This was checked for all rows of all five caches: endpoints and SHA-256 context hashes match for 78,376 of 78,376 delay-1h rows.
  - Validation and evaluation contexts contain **no** missing hours.
  - 345 training rows (2021-01-01 to about 2021-01-22) contain up to 23 missing hours, which are passed to Chronos as NaN, not compressed.
- **Evidence.** `runs/pes2027/E0-CENSUS/chronos_cache_rows.csv`; `chronos_context_nan_by_window.csv`.

## CL-15 (Minor boundary) First evaluation origin
- **Verified.** Under the one-hour delay, the last early-stopping label (hour 2023-12-31 23:00 UTC) becomes public at 2024-01-01 01:00 UTC. A model frozen after 2023 could therefore not have issued the 2024-01-01 00:00 UTC forecast. One of 8,767 origins is affected. The crisis protocol has the same one-origin issue at 2021-01-01 00:00 UTC.
- **Evidence.** `runs/pes2027/E0-CENSUS/census.json` → `boundary`.

## New measurement, not a correction: the 2023 development year
On 2023, the year used for early stopping, the seed-averaged augmented model has MAE **39.1375** and the baseline **38.5535**, a gap of +0.5840. Augmentation is worse on 2023. This was not used for any selection. Early stopping on the same year biases both values downward.
- **Evidence.** `runs/pes2027/E0-REPRO/rerun/*__val.parquet`.

<!-- WP2 entries (CL-12 load source and timestamp convention; DAM and EIA exceptions; price corrections) are appended below after primary-source verification. -->
