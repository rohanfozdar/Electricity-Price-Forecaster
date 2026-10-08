# Repository audit (Gate A)

Branch: `early-warning`, cut from `main` at `cf06362`. Audit date: 2026-09-26.
Notebook filenames below were updated on 2026-10-07 after the experiment notebooks were renamed. The findings are unchanged.
Evidence: `scripts/audit_matrix.py` writes `reports/audit_evidence.json`. Every number below comes from that file, from a notebook's saved output, or from a committed JSON, and is cited where it appears.

## 1. Summary verdict

The v2 feature matrix is **not point-in-time safe**. Thirteen of its 87 "enhanced" features take the value observed in the same hour as the target, 29 features are joined on observation dates rather than publication dates, and the calendar features run on the UTC clock. So none of the historical results can support a forecasting claim. They are best read as a *retrospective estimation* benchmark, which is how the author's own `.tex` draft already frames them.

The public headline (Chronos+XGBoost hybrid at $12.13/MWh MAE, CatBoost at $13.75) **cannot be traced to any executed artifact in the repository**. The five hybrid cells in `notebooks/07_chronos_and_hybrid.ipynb` (cells 8–12) have `execution_count = None` and no outputs. Neither 12.13 nor 13.75 appears in any notebook output, JSON, or text report. The only saved CatBoost result is $20.90 on 63 features (`03_catboost.ipynb`).

## 2. Repository map

| Layer | Files | Notes |
|---|---|---|
| Ingestion | `pipelines/*.py` (8 sources) | gridstatus (RT, DA), load (ERCOT archive or EIA-930 fallback), Open-Meteo **archive** (reanalysis), GDELT DOC API, EIA XLS (gas, storage), NRC status files |
| Raw data | `data/raw/*.parquet` (local, gitignored) | inventory in `audit_evidence.json → sources` |
| Matrix | `features/build_matrix_v2.py`, `features/engineer_v2.py` | output `data/features/feature_matrix_engineered_v2.parquet` (committed; 77,956 rows) |
| Models | `models/train_all_v2.py`, `train_classifier.py`, `calibrate_classifier.py`, `train_chronos_hybrid.py` | artifacts in `models/artifacts/` (gitignored) |
| Evaluation | `evaluation/{backtest,benchmark_dam,granger,generate_final_report}.py` + JSON | |
| Notebooks | 17 in `notebooks/` (10 tracked, 7 gitignored) | results in the paper come from here |
| Public claims | `README.md`, `docs/index.html`, `docs/static/pdfs/paper.pdf`, `~/Downloads/ERCOT_Capstone_Paper.tex` | three incompatible result versions (§7) |

## 3. Timestamp, DST, and coverage evidence

| Check | Result | Evidence key |
|---|---|---|
| RT prices | 15-minute, tz-aware UTC, 78,864 hours with exactly 4 intervals. DST handled (the 18 transition days have 92 or 100 intervals). | `rt_prices` |
| Missing day | RT **and** DA both lack Central-time **2020-12-31** (UTC gap 2020-12-31 06:00 → 2021-01-01 06:00) | `sources.*.gaps_beyond_modal_step` |
| Test-year end | Data stops at **2024-12-31 06:00 UTC** (local midnight). The last 17 UTC hours of 2024 are missing because the pipeline filters `<= end.tz_localize("US/Central")` | `rt_prices.hours_missing_at_end_of_2024_utc` |
| Matrix gaps | 213 missing hours inside the range: 2020-12-31 → 2021-01-08, where the 168-hour rolling windows are NaN after the gap, and 2021-01-30 → 01-31, where the 720-hour lag falls in the gap | `matrix.gaps` |
| Row loss | 78,889 raw-v2 rows → 77,956 engineered. `dropna` on the 720-hour lag and the 168-hour roll removes 933. The first month of 2016 is lost. | `matrix.row_loss` |
| Load timezone | Correctly UTC. Mean load is lowest at 10 UTC and highest at 22 UTC; summer load/temperature correlation peaks at a 1-hour lag | `diurnal_alignment` |
| Load coverage | **2019-01-01 onward only.** Load features are 100% NaN before 2019 (33% of rows), and training fills them with `ffill().fillna(0)`, so trees can tell the pre-2019 era apart by `load == 0` | `matrix.load_feature_nan_share_before_2019` |
| GDELT coverage | Tone starts 2017-02-01. Volume columns are 69–76% NaN. Both are filled with 0 in training | `matrix.feature_nan_share_top` |
| Target | `HB_HUBAVG` hourly = **max** of four 15-minute prices. 1,854 hours exceed $200 under the max, 997 under the mean | `rt_prices.spike_hours_hourly_*` |
| Price summary | median $21.40, max $9,051.55, 2.37% of hours above $200 | `matrix.price_summary` |

## 4. Integrity failures

Severity: **C** = invalidates a forecasting claim; **M** = biases results or reproducibility; **L** = cosmetic or documentation.

| # | Sev | Failure | Evidence |
|---|---|---|---|
| 1 | C | **Same-hour features.** 13 of 87 features equal their raw source at the *same* timestamp (fraction 1.00 at lag 0, ≈0 at lag 1): all 5 `da_*` prices, `load_actual_mw`, `temp_max_across_zones`, and all per-zone weather columns (the Houston ones sampled; the other zones share the merge). The claimed one-hour shift exists only for `load_lag_1h` and the `hubavg_*` lags. | `matrix.same_hour_match_fractions` |
| 2 | C | **GDELT same-day lookahead.** `gdelt_tone` at every hour equals the same UTC day's daily mean (fraction 1.00), so 00:00 sees the news of 23:59. Every derived GDELT feature inherits this. | `daily_weekly_timing.gdelt_tone_equals_same_utc_day_value` |
| 3 | C | **Gas and NRC are joined on observation date.** Henry Hub equals the same trade date (1.00); NRC equals the same report date (1.00). | `daily_weekly_timing` |
| 4 | C | **Storage is joined on week-ending Friday, not the Thursday release.** 94.7% of hourly rows use a storage report not yet published under a Thursday 10:30 ET release. | `storage_rows_using_unreleased_report_share` |
| 5 | M | **Full-sample threshold.** `stress_gas_spike` uses a full-sample P90 ($5.10) instead of the training-only P90 ($5.69), so it crosses split boundaries. | `matrix.stress_gas_spike` |
| 6 | M | **`log1p(clip(lower=0))` erases negative prices.** About 1% of price-lag values are negative, and all of them are mapped to 0. | `negative_value_share_erased_by_log1p_clip` |
| 7 | M | **Calendar features are on the UTC clock.** `is_peak_hours` = 14–20 UTC = 08–15 Central, which is not ERCOT's peak; the median price peaks at 00 UTC. `hour` and `day_of_week` are also UTC. | `matrix.calendar` |
| 8 | M | **Tail labels are zero instead of missing.** 49 rows with an incomplete 24-hour future window get `future_spike_24h = 0`. | `matrix.labels` |
| 9 | M | **No purge at split boundaries.** 24 training rows and 24 validation rows have label windows reaching into the next split. In this sample none of them is positive, so the realised effect is nil, but the design is wrong. | `boundary_crossing_labels` |
| 10 | C | **Test-set threshold selection.** `train_all_v2.py` and `calibrate_classifier.py` sweep thresholds on the 2024 test set. The operating threshold 0.3 used in `evaluation/backtest.py` and the `.tex` (TP=509, FP=1,593, FN=961) is taken from those sweeps. | code: `_threshold_sweep(y_test, …)` |
| 11 | M | **Calibrator picked on test Brier.** `calibrate_classifier.py` chooses Platt vs isotonic by lowest *test* Brier. | code, line ~180 |
| 12 | M | **Weather is reanalysis.** ERA5 is published with a 5-day delay (Open-Meteo docs), so same-hour and future weather were never available at issue time. | `pipelines/weather.py`, Open-Meteo docs |
| 13 | M | **Positional Chronos context.** `train_chronos_hybrid.py` takes `iloc[p-512:p]` on a series with 213 missing hours, so contexts silently straddle the 2021 gap. The partial cache has 12,928 rows (2016-02-21 → 2017-08-13). | `chronos_cache` |
| 14 | M | **Missing data filled with 0.** Load (pre-2019), GDELT (pre-2017 and volume) and every other NaN get `ffill().fillna(0)`, which silently encodes the data era. | `feature_nan_share_top` |
| 15 | M | **Backtest is in-sample for train events.** `evaluation/backtest.py` scores named events in 2017–2022 with a model trained on those years. | `evaluation/backtest.py` |
| 16 | M | **Stress-hour breakdown conditions on the realised outcome.** The DAM benchmark's "stress hours" are hours with realised price above $100; this is descriptive, not a forecastable subset. | `evaluation/benchmark_dam.py` |
| 17 | C | **Unexecuted headline.** The hybrid cells in notebook 10 were never run, and 12.13 / 13.75 are untraceable (§1). | `notebooks/07_chronos_and_hybrid.ipynb` code cells 8–12 |
| 18 | M | **Stale or contradictory evaluation text.** `granger_results.json` has all p > 0.05 (e.g. `gdelt_tone` min p = 0.308), while the older narrative claims significance. | `evaluation/granger_results.json` |
| 19 | L | **Notebook execution order.** `data_inspection`, `build_engineered_matrix`, `03_catboost`, `04_ngboost`, and `residual_diagnostics` have non-monotone execution counts; `09_prophet_and_hybrid` and `10_sarimax` were never executed. | notebook scan |
| 20 | L | **Missing RT day 2020-12-31 and truncated 2024-12-31** (§3). | `sources` |

**Confirmed risks from the plan (items 1–15):** all confirmed except the load-timezone hypothesis (load is correctly UTC). New findings: 7 (UTC calendar), 13 (positional Chronos context straddling a gap), 14 (zero-fill era leakage), 15–16, and 20.

## 5. Source-by-source verdict: is a one-hour lag enough?

| Source | Native timing | Publication | One-hour lag enough? | Point-in-time rule adopted |
|---|---|---|---|---|
| ERCOT RT SPP (15-minute) | interval start, UTC | posted within minutes of each interval | **Yes.** Hour *t−1* is complete at *t*. | use `y[h]` for `h ≤ t−1` |
| ERCOT DA SPP (hourly) | operating hour | DAM results for day *D* posted ≈13:30 Central on *D−1* | **Not the right question.** DA is known *ahead*: at *t*, all hours of each operating day whose DAM has published are known. Same-hour DA is legitimate **only** if its DAM published before *t*. | as-of by publication time (14:00 Central on *D−1*, with a 30-minute buffer) |
| Load (hourly) | hour start, UTC | real-time system demand is posted continuously; settlement values later | **Yes, as an assumption.** It cannot be verified from the data. | `h ≤ t−1` |
| Weather, Open-Meteo archive | hour, UTC | ERA5: 5-day delay; IFS analysis: no delay | **No** for the same hour (a future observation). Past hours are used as a *proxy for real-time station observations*, with the 1-hour lag declared as an assumption and a 5-day-lag sensitivity run. | `h ≤ t−1` (proxy); future weather only in the oracle experiment |
| Weather forecasts, Open-Meteo Previous Runs (GFS) | valid hour, at a lead of *N* days | GFS from 2021-03-23 | n/a | forecast value for hour *h* from lead day *N* = ⌈(*h*−*t*+6h)/24⌉ |
| GDELT DOC timeline | UTC day | complete after the day ends | **No.** The daily mean is stamped at 00:00 and needs a lag of about 24 hours. | UTC day *D* is usable from *D*+1 01:00 UTC |
| Henry Hub spot (EIA RNGWHHD) | trade date | EIA posts daily spot prices weekly | **No.** | *assumed* weekly Wednesday release covering trade dates through the prior Friday (not verified: the EIA page timed out) |
| EIA storage | week ending Friday | Thursday 10:30 ET the following week | **No.** Needs about 6 days. | usable from the following Friday 18:00 UTC (holiday buffer) |
| NRC reactor status | report date | posted the same morning (ET) | **No.** | date *D* is usable from *D*+1 00:00 UTC |
| Google Trends | weekly | unused by the models | n/a | excluded |

## 6. Notebook audit

| Notebook | Executed | Saved headline numbers | Issues |
|---|---|---|---|
| `01_xgboost` | all 12 cells, in order | GDELT ablation: with 22.21 / 108.88, without 24.64 / 114.20; spike PR-AUC 0.422 vs 0.400 | leaky features (failures 1–4); test threshold 0.5 |
| `02_lightgbm` | 9/10 | 19.76 | test threshold sweep |
| `03_catboost` | 9/12, out of order | **20.90 / 101.54 on 63 features** | test threshold sweep; 63 ≠ 87 features |
| `05_lstm`, `06_tft` | mostly | 32.45, 25.61 | |
| `07_chronos_and_hybrid` | **8/13** | zero-shot Chronos **14.86 / 116.42** (recall 41.2%, precision 59.5%) | **hybrid cells never executed** |
| `08_timesfm` | 7/11 | 32.72 | |
| `09_prophet_and_hybrid`, `10_sarimax` | **0** | none | never run |
| `timesfm_xgboost` (gitignored) | 8/8 | TimesFM+XGB hybrid **23.52 / 120.15**, TimesFM alone 32.67 | an executed foundation hybrid that does *not* beat the XGB baseline in its notebook |
| `residual_diagnostics`, `feature_diagnostics` | yes | transform diagnostics | source of `apply_feature_transforms` |
| `rag_forecaster` | 12/13 | 25.61, 19.56 appear | test-query k sweep |

## 7. Result versions in circulation

1. **Google Docs PDF leaderboard** (now `docs/static/pdfs/paper.pdf`, mirrored in `README.md` and `docs/index.html`): hybrid 12.13 / 93.98, CatBoost 13.75 / 95.48. **Untraceable.**
2. **`.tex` draft**: numbers traced above (GDELT 24.64 → 22.21, 9.9%; CatBoost 20.90; Chronos 14.86; TimesFM 32.72; LSTM 32.45; TFT 25.61; DAM benchmark 19.56 vs 17.51). **Traceable, but computed on the leaky matrix.**
3. **Script re-run** (`train_all_v2.py`): 24.58 → 19.17. It differs from version 2 because of the transform and seed path.

All three are retrospective regression numbers on a leaky matrix. They were previously described as next-hour forecasts. None of them measures multi-horizon early warning.

## 8. Decision needed: public claims

The README and project page currently headline version 1. The recommendation, carried out at Gate E: remove the untraceable leaderboard from the headline, keep the historical tables only as "capstone snapshot (retrospective, not point-in-time)", and headline the frozen early-warning results from Gate D, whatever their direction.

## 9. Draft availability config

`configs/feature_availability.yaml` encodes §5 as machine-readable rules. It is consumed by `src/data/asof.py` and checked by `tests/test_feature_availability.py`.
