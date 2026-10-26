"""Write the corrected-study reports from runs/pit_verified/. No new fitting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "pit_verified"
REPORTS = ROOT / "reports"


def _load(name):
    path = OUT / name
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return pd.read_csv(path)


def _num(value, digits=2):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if value != value:
        return "NA"
    return f"{value:.{digits}f}"


def _money(value, digits=2):
    text = _num(value, digits)
    return text if text == "NA" else f"${text}"


def _primary(comparisons):
    for row in comparisons:
        if row.get("name") == "delay_1h__mean_seed__both_minus_exog":
            return row
    raise KeyError("primary comparison missing")


def _scenario_table(comparisons) -> str:
    lines = [
        "| Scenario | Price delay | Load | MAE with Chronos | MAE without | Gap | 95% interval | Excludes zero |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparisons:
        if row.get("estimand") != "mae_of_seed_averaged_forecast":
            continue
        load = "dropped" if row.get("load_mode") == "dropped" else f"{row.get('load_delay_hours')}h"
        lines.append(
            f"| {row['info_set']} | {row.get('price_delay_hours')}h | {load} | "
            f"{_money(row['mae_a'])} | {_money(row['mae_b'])} | {_money(row['mae_gap_a_minus_b'])} | "
            f"[{_money(row['bootstrap_ci_low'])}, {_money(row['bootstrap_ci_high'])}] | "
            f"{row['interval_excludes_zero']} |"
        )
    return "\n".join(lines)


def write_reports() -> None:
    comparisons = _load("comparisons.json")
    primary = _primary(comparisons)
    both = primary["seed_averaged_metrics_both"]
    exog = primary["seed_averaged_metrics_exog"]
    metrics = _load("metrics.csv")
    regimes = _load("regimes.csv")
    blocks = _load("block_lengths.csv")
    spikes = _load("spike_uncertainty.csv")
    baselines = _load("baselines.csv")
    crisis = _load("crisis_2021.json")
    hourly = _load("hourly_mean.json")
    series = _load("series_check.json") or {}
    stamps = _load("timestamp_check.json") or {}
    probe = _load("chronos_probe.json") or {}
    repro = _load("reproduction_summary.json") or {}

    (REPORTS / "corrected_chronos_ablation.md").write_text(_ablation(primary, both, exog, comparisons, series))
    (REPORTS / "latency_sensitivity.md").write_text(_latency(comparisons, probe))
    (REPORTS / "extreme_event_2021.md").write_text(_crisis(crisis))
    (REPORTS / "price_regime_analysis.md").write_text(_regimes(regimes))
    (REPORTS / "uncertainty_robustness.md").write_text(_uncertainty(primary, blocks, spikes))
    (REPORTS / "baseline_comparison.md").write_text(_baselines(baselines, exog))
    (REPORTS / "external_validity.md").write_text(_external(probe))
    (REPORTS / "target_definition.md").write_text(_target(hourly, series))
    (REPORTS / "pit_verification_final.md").write_text(_verdict(primary, stamps, series))
    (REPORTS / "research_execution_log.md").write_text(_log(primary, stamps, series, probe, repro))
    (REPORTS / "paper_rewrite_handoff.md").write_text(_handoff(primary, both, exog, crisis, baselines, hourly))
    print("reports written", flush=True)


def _ablation(primary, both, exog, comparisons, series) -> str:
    gap = primary["mae_gap_a_minus_b"]
    return f"""# Corrected Chronos ablation

The forecast is the one-hour-ahead hourly maximum of HB_HUBAVG. The issue time is the start of that hour. The primary information set uses a one-hour publication delay on real-time prices and on realized load. That delay was fixed from the source register before these errors were computed. Seeds are 0, 1, 2, 3, and 42. Trees use depth 6, learning rate 0.05, and absolute error on the asinh target, with early stopping on 2023.

The headline gap is the MAE of the seed-averaged forecast with Chronos minus the MAE of the seed-averaged forecast without Chronos. A negative number favors Chronos.

| Estimand | With Chronos | Without Chronos | Difference |
| --- | --- | --- | --- |
| MAE of the averaged forecast | {_money(both['mae'])} | {_money(exog['mae'])} | {_money(gap)} |
| Mean of the five seed MAEs | {_money(primary['mean_of_seed_maes_both'])} | {_money(primary['mean_of_seed_maes_exog'])} | {_money(primary['gap_of_mean_seed_maes'])} |

The 24-hour block bootstrap interval for the averaged-forecast gap is [{_money(primary['bootstrap_ci_low'])}, {_money(primary['bootstrap_ci_high'])}]. Bootstrap p is {_num(primary['bootstrap_p'], 3)}. The Diebold-Mariano statistic is {_num(primary['dm_stat'], 2)} with p {_num(primary['dm_p'], 3)}. The interval excludes zero: {primary['interval_excludes_zero']}.

Other scores of the averaged forecast, with Chronos and without:

| | With Chronos | Without |
| --- | --- | --- |
| RMSE | {_money(both['rmse'])} | {_money(exog['rmse'])} |
| Bias (forecast minus price) | {_money(both['bias'])} | {_money(exog['bias'])} |
| Median absolute error | {_money(both['medae'])} | {_money(exog['medae'])} |
| Spike precision | {_num(both['spike_precision'], 3)} | {_num(exog['spike_precision'], 3)} |
| Spike recall | {_num(both['spike_recall'], 3)} | {_num(exog['spike_recall'], 3)} |
| Spike F1 | {_num(both['spike_f1'], 3)} | {_num(exog['spike_f1'], 3)} |
| MAE on hours above $200 | {_money(both['spike_mae'])} | {_money(exog['spike_mae'])} |
| MAE on the other hours | {_money(both['nonspike_mae'])} | {_money(exog['nonspike_mae'])} |
| Hours | {both['n']} | {exog['n']} |
| Spikes | {both['n_spikes']} | {exog['n_spikes']} |

Per-seed gaps (with Chronos minus without), seeds 0, 1, 2, 3, 42: {', '.join(_money(v) for v in primary['per_seed_gap'])}.

The saved pre-correction gap on the same task was about $-0.09, with an interval that covered zero. This file does not treat a similar number as confirmation that the old information set was valid. 2024 was already inspected. Chronos rows are a new T5-small sample. The context ends at the last legal hour, and the selected step is the forecast of hour t. See `runs/pit_verified/comparisons.json`.

Hourly maxima from the 15-minute file and from the modeling matrix differ by more than one cent on {series.get('hours_abs_diff_gt_0_01', 'NA')} overlapping hours. The maximum absolute difference is {_money(series.get('max_abs_diff'), 4)}. Lag features and the new Chronos context both use the 15-minute hourly maximum.
"""


def _latency(comparisons, probe) -> str:
    minutes = probe.get("estimated_full_cache_minutes", "NA")
    return f"""# Latency sensitivity

The delays are assumptions. None of them is a measured 2019–2024 posting time. The primary row is the one-hour delay, not the row with the smallest MAE.

A one-hour delay makes hour t−1 unavailable at the start of hour t, because the last interval of hour t−1 ends at that instant. Two, three, and six hours push the last legal hour to t−3, t−4, and t−7. The 24-hour load row keeps the one-hour price delay and lags realized load by 24 hours. The dropped-load row removes realized load.

{_scenario_table(comparisons)}

Each delay has its own Chronos cache under `runs/pit_verified/chronos/`. The context is the 512 clock hours of the latency-filtered hourly maximum ending at the last legal hour. The model forecasts far enough to reach hour t, and that step is the feature. This is not a shift of the old cache in `data/features/chronos_hourly/`. One timed batch, before the full rebuild, implied about {minutes} minutes per cache. Each saved row stores the latest input timestamp and a hash of the context.

The gap above is MAE(with Chronos) minus MAE(without), using the seed-averaged forecast. An interval that covers zero is not evidence that Chronos helps.
"""


def _crisis(crisis) -> str:
    if not crisis:
        return "# February 2021\n\nNot computed.\n"
    parts = [
        "# February 10–20, 2021",
        "",
        "The window is midnight US/Central on February 10 through midnight on February 21, exclusive. It was fixed before the corrected predictions were scored. The fit is an expanding window: training uses only hours before the last 90 days of 2020, those 90 days are the early-stopping sample, and 2021 is scored once. This is not a model trained through 2022.",
        "",
        f"Training prices before that early-stopping window reach a maximum of {_money(crisis['train_max_price'])}. Their 99th percentile is {_money(crisis['train_p99_price'])}. The asinh decode is clipped at sinh(10), about {_money(crisis['asinh_clip_dollars'])}. Calendar hours in 2021 missing from the matrix index: {crisis['calendar_hours_missing_from_matrix_index']}. Scored hours: {crisis['scored_2021_hours']}.",
        "",
        "| Slice | Model | n | MAE | RMSE | Bias | Pred max | Price max | Forecasts above $10,000 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, block in crisis["subsets"].items():
        for model in ("xgb_exog", "xgb_both", "persistence"):
            row = block[model]
            parts.append(
                f"| {label} | {model} | {row['n']} | {_money(row['mae'])} | {_money(row['rmse'])} | "
                f"{_money(row['bias'])} | {_money(row.get('pred_max'))} | {_money(row.get('y_max'))} | "
                f"{row.get('n_pred_above_10000', 'NA')} |"
            )
        if "raw_target_xgb_exog" in block:
            row = block["raw_target_xgb_exog"]
            parts.append(
                f"| {label} | xgb_exog raw target | {row['n']} | {_money(row['mae'])} | {_money(row['rmse'])} | "
                f"{_money(row['bias'])} | {_money(row.get('pred_max'))} | NA | NA |"
            )
    crisis_block = crisis["subsets"]["crisis_hours"]["xgb_exog"]
    clip_n = crisis_block["n_pred_above_10000"]
    clip_sentence = (
        "No saved forecast in the exogenous asinh model reached $10,000, so the asinh clip did not bind on this window."
        if clip_n == 0 else
        f"{clip_n} saved forecasts reached $10,000, so the asinh clip may have bound."
    )
    parts.append("")
    when = crisis.get("train_max_timestamp", "an earlier hour")
    parts.append(
        f"On the saved exogenous forecasts, the largest prediction in the crisis window is "
        f"{_money(crisis_block['pred_max'])} and the largest price is {_money(crisis_block['y_max'])}. "
        f"Bias is {_money(crisis_block['bias'])}. {clip_sentence} "
        f"Before the early-stopping window, the 99th percentile of prices is {_money(crisis['train_p99_price'])} "
        f"and the maximum is {_money(crisis['train_max_price'])}, at {when}. "
        "That maximum is the same order as the crisis peak, so the miss is not explained by the training sample never having contained a price near $9,000. "
        "The forecasts still stayed far below the storm. "
        "Persistence in this table is the last legal hourly price under the one-hour delay, not the price from hour t−1. "
        "On the crisis hours it has a much lower MAE than either tree. Outside that window the trees have lower MAE than persistence."
    )
    parts.append("")
    parts.append("Detail is in `runs/pit_verified/crisis_2021.json`.")
    return "\n".join(parts) + "\n"


def _regimes(regimes) -> str:
    if regimes is None:
        return "# Price regimes\n\nNot computed.\n"
    lines = [
        "# Price regimes in 2024",
        "",
        "Bins were fixed in `models/run_pit_verified.py` before the corrected predictions were read: negative, 0–50, 50–100, 100–200, 200–500, and above 500 dollars per megawatt-hour. They are subsets of the realized price. They are not model inputs. The gap is MAE(with Chronos) minus MAE(without) on the seed-averaged primary forecasts. A benefit is not claimed when the interval covers zero or the hour count is below 100. The bootstrap resamples the subset as one group because a price bin is not a contiguous block of hours.",
        "",
        "| Regime | Hours | MAE with Chronos | MAE without | Gap | 95% interval | Benefit claimed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in regimes.iterrows():
        if int(row["n"]) == 0:
            lines.append(f"| {row['regime']} | 0 | NA | NA | NA | NA | no |")
            continue
        lines.append(
            f"| {row['regime']} | {int(row['n'])} | {_money(row['mae_both'])} | {_money(row['mae_exog'])} | "
            f"{_money(row['mae_gap'])} | [{_money(row['ci_low'])}, {_money(row['ci_high'])}] | "
            f"{'yes' if bool(row['benefit_claim_allowed']) else 'no'} |"
        )
    lines.append("")
    lines.append(
        "The 0–50 bin is the only slice with at least 100 hours whose interval lies entirely below zero. "
        "The gap there is about eight cents per megawatt-hour. The bin is defined by the realized price, "
        "so the comparison describes those hours. It is not a regime the forecast could have selected in advance. "
        "The above-500 slice has fewer than 100 hours, and its interval is not treated as a benefit."
    )
    lines.append("")
    lines.append("Source: `runs/pit_verified/regimes.csv`.")
    return "\n".join(lines) + "\n"


def _uncertainty(primary, blocks, spikes) -> str:
    block_lines = [
        "| Block hours | Gap | 95% interval | Excludes zero | DM p |",
        "| --- | --- | --- | --- | --- |",
    ]
    if blocks is not None:
        for _, row in blocks.iterrows():
            block_lines.append(
                f"| {int(row['block_hours'])} | {_money(row['mae_gap'])} | "
                f"[{_money(row['ci_low'])}, {_money(row['ci_high'])}] | {row['interval_excludes_zero']} | "
                f"{_num(row['dm_p'], 3)} |"
            )
    spike_lines = [
        "| Model | Block | Precision | Recall | F1 | Undefined precision draws | Undefined recall draws | Undefined F1 draws |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if spikes is not None:
        for _, row in spikes.iterrows():
            spike_lines.append(
                f"| {row['model']} | {int(row['block_hours'])} | {_num(row['precision'], 3)} | "
                f"{_num(row['recall'], 3)} | {_num(row['f1'], 3)} | {int(row['n_undefined_precision'])} | "
                f"{int(row['n_undefined_recall'])} | {int(row['n_undefined_f1'])} |"
            )
    return f"""# Uncertainty

The primary gap uses 2,000 moving-block bootstrap draws and seed 0. The pre-specified block is 24 hours. The same draw count and seed are repeated at 48, 72, and 168 hours. The gap is MAE(with Chronos) minus MAE(without) for the seed-averaged one-hour-delay forecasts. The Diebold-Mariano lag matches the block length in this table. The 24-hour row is the one paired with the ablation.

{chr(10).join(block_lines)}

The primary 24-hour interval from the ablation file is [{_money(primary['bootstrap_ci_low'])}, {_money(primary['bootstrap_ci_high'])}].

Spike precision, recall, and F1 use the same blocks on the seed-averaged forecasts. A draw with no true spike leaves recall and F1 undefined. A draw with no predicted spike leaves precision and F1 undefined. Those draws are counted below and are left out of the quantile for that metric. They are not replaced with zero. Quantiles are omitted in the CSV when fewer than 100 draws are defined; the point estimates below are the full-sample scores.

{chr(10).join(spike_lines)}

An interval that covers zero does not establish a Chronos benefit. Changing the block length does not, by itself, repair an unverified publication time.
"""


def _baselines(baselines, exog) -> str:
    lines = [
        "# Baselines",
        "",
        "Persistence is the last legal hourly maximum under the one-hour delay, which is hour t−2. Seasonal naive forecasts use the price 24 and 168 hours earlier. Those hours have already ended under a delay of one to six hours. The new lasso uses the legal price lags and the Central calendar, standardized with 2019–2022 moments. Its penalty is the value with the lowest 2023 MAE. It is fit on 2019–2022 only.",
        "",
        "The historical three-lag lasso is the simplified model already in the repository. It is not the Lago, Marcjasz, De Schutter, and Weron LEAR specification, and it was not rebuilt on the delayed prices.",
        "",
        f"The latency-valid exogenous tree, for reference, has MAE {_money(exog['mae'])} on the seed-averaged forecast.",
        "",
        "| Model | Information set | n | MAE | RMSE | Median AE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if baselines is not None:
        for _, row in baselines.iterrows():
            lines.append(
                f"| {row['model']} | {row.get('information_set', 'NA')} | {row.get('n', 'NA')} | "
                f"{_money(row.get('mae'))} | {_money(row.get('rmse'))} | {_money(row.get('medae'))} |"
            )
    lines.append("")
    lines.append("Source: `runs/pit_verified/baselines.csv`.")
    return "\n".join(lines) + "\n"


def _external(probe) -> str:
    minutes = probe.get("estimated_full_cache_minutes", "not timed")
    hub = _load("hub_north.json") or {}
    if hub.get("mae_gap_a_minus_b") is None:
        hub_text = "The HB_NORTH refit is not in `runs/pit_verified/hub_north.json`."
    else:
        hub_text = (
            f"HB_NORTH, with its own one-hour-delay prices and its own Chronos cache, has a "
            f"seed-averaged gap of {_money(hub['mae_gap_a_minus_b'])}, interval "
            f"[{_money(hub['bootstrap_ci_low'])}, {_money(hub['bootstrap_ci_high'])}], "
            f"excludes zero: {hub['interval_excludes_zero']}. "
            f"The difference of the mean seed MAEs is {_money(hub.get('mean_of_seed_maes_gap'))}."
        )
    return f"""# External validity

No second market is in this study. HB_NORTH is another ERCOT hub. A previous transfer reused the HB_HUBAVG Chronos cache on HB_NORTH prices. That transfer is not an independent hub model, and it is not repeated here.

{hub_text}

The timed batch implied about {minutes} minutes for one full cache through 2024. The hub cache, when present, is `runs/pit_verified/chronos/hb_north_delay_1h.parquet`.

No 2025 or 2026 score is part of this record. Those years remain on disk from the earlier ablation and are not used here.
"""


def _target(hourly, series) -> str:
    if hourly:
        gap = (
            f"On the hourly-mean target, the seed-averaged gap is {_money(hourly['mae_gap_a_minus_b'])}, "
            f"interval [{_money(hourly['bootstrap_ci_low'])}, {_money(hourly['bootstrap_ci_high'])}], "
            f"excludes zero: {hourly['interval_excludes_zero']}. "
            f"The difference of the mean seed MAEs is {_money(hourly.get('mean_of_seed_maes_gap'))}."
        )
    else:
        gap = "The hourly-mean refit was not stored."
    return f"""# Target definition

The headline target is the maximum of the four 15-minute HB_HUBAVG settlement prices whose interval starts are t, t+15, t+30, and t+45 minutes, with t in UTC. The forecast is issued at t. The target is not observable at t. Under a one-hour posting delay it is not fully observable until t+2 hours, because the last interval ends at t+1 hour and the delay adds another hour.

Central time is a whole-hour offset from UTC, so a UTC hour is a Central clock hour. The feature calendar uses US/Central. The scored 2024 index is the existing matrix index, which runs through 2024-12-31 06:00 UTC. That is Central midnight at the end of December 31, not a full Central-year tail into January 1. No 2025 hour is added to repair that edge.

Incomplete hours are kept when a price exists. The 15-minute hourly maximum and the matrix target differ by more than one cent on {series.get('hours_abs_diff_gt_0_01', 'NA')} overlapping hours.

{gap}

The hourly mean is a secondary outcome on the same one-hour-delay features. Chronos statistics in that fit are still forecasts of the hourly maximum. A mean-target interval that excludes zero does not replace the hourly-maximum headline.
"""


def _verdict(primary, stamps, series) -> str:
    return f"""# Point-in-time verification verdict

**Partially verified, with unresolved concerns.**

This is not "fully verified." It is also not "valid only under stated publication-latency assumptions," because a latency shift does not answer the remaining problems below. The label is not based on the MAE.

What is in place:

- The issue time is the start of hour t, and the target is the hourly maximum inside that hour.
- Price and load features take an explicit extra delay. At one hour, no interval that ends at t enters the feature row. Tests cover that exclusion, rolling windows, the day-ahead gate, the storage gate, DST hour counts, training-only imputation, and the Chronos context end.
- Day-ahead prices stay behind the 14:00 US/Central gate. Storage stays behind the Friday 18:00 UTC buffer.
- The feature manifest is generated by `src/features/pit_latency.py`.

What is not established:

- {stamps.get('outcome', 'No publication timestamp was found.')} The price file is the weekly historical archive NP6-785-ER, report 13061, not the 15-minute NP6-905-CD feed.
- A one-hour delay is an assumption. So are two, three, six, and twenty-four hours. The source register does not verify any of them for 2019–2024.
- Even a correct delay uses the archived number. It does not recover the first posted price.
- Realized load comes from the daily NP6-345-CD archive. Lagging it does not turn it into a load forecast.
- Day-ahead files can contain prices corrected before 10:00 on the next business day. Those hours are not marked.
- The EIA holiday calendar for 2019–2024 was not applied row by row. The storage buffer is later than the standard Thursday release.
- Chronos is a new 20-path sample. Its context ends at the last legal hour, and the selected step forecasts hour t. The prices in that context are still the archived series. The 15-minute hourly maximum and the old matrix series differ by more than one cent on {series.get('hours_abs_diff_gt_0_01', 'NA')} hours. The new cache uses the 15-minute series.

The primary seed-averaged gap is {_money(primary['mae_gap_a_minus_b'])}, interval [{_money(primary['bootstrap_ci_low'])}, {_money(primary['bootstrap_ci_high'])}], excludes zero: {primary['interval_excludes_zero']}. That result does not upgrade this label. The pre-correction gap of about $-0.09 is an audit baseline, not a target.
"""


def _log(primary, stamps, series, probe, repro) -> str:
    return f"""# Research execution log

This log is the record for a later rewrite. Commands were run from `Ercot_Forecaster/ercot-grid-forecaster` with `/net/home/rfozdar/venvs/ercot/bin/python`. The study does not score 2025 or 2026.

## 1. Confirm the saved ablation

Task: reproduce the conservative 2024 gap without refitting. Evidence: `runs/pit_ablation/comparisons.json` and `ablation_metrics.csv`. Command: `python models/run_pit_verified.py` writes the summary before fitting. Outcome: seed-averaged gap {_money(repro.get('seed_averaged_forecast', {}).get('gap') if repro else None)} with an interval that covers zero. The mean of the seed MAEs is a different estimand. Artifact: `reports/pit_verified_reproduction.md`.

## 2. Source register

Task: retrieve ERCOT and EIA publication evidence. Pages retrieved on 2026-10-13 are listed in `reports/source_evidence_register.md`. Outcome: the price file is weekly archive NP6-785-ER. The live 15-minute product NP6-905-CD is not in the repository. Load is daily archive NP6-345-CD. Day-ahead posting is documented at no later than 13:30 Central, with a later correction window. EIA storage is Thursday 10:30 a.m. Eastern, with a code buffer of the following Friday 18:00 UTC. No historical first-publication timestamps were found.

## 3. Failed timestamp check

Task: look for a publication column in the price file. Outcome: {stamps.get('outcome', 'NA')}. Columns: {stamps.get('rt_columns')}. Artifact: `runs/pit_verified/timestamp_check.json`.

## 4. Availability code

Task: add an explicit delay and tests that fail if the interval ending at the issue time is used. Command: `pytest tests/test_pit_latency.py tests/test_pit_sets.py tests/test_feature_availability.py`. The manifest is written by code to `reports/feature_availability_manifest.csv`. Primary delay: 1 hour, chosen as the shortest pre-specified assumption, before the corrected MAE was known.

## 5. Series check

The 15-minute hourly maximum and the matrix target differ by more than one cent on {series.get('hours_abs_diff_gt_0_01', 'NA')} hours. Maximum absolute difference {_money(series.get('max_abs_diff'), 4)}. Artifact: `runs/pit_verified/series_check.json`.

## 6. Chronos caches

One batch of 256 was timed before the rebuild. Load seconds: {probe.get('load_seconds', 'NA')}. Batch seconds: {probe.get('one_batch_seconds', 'NA')}. Estimated full-cache minutes: {probe.get('estimated_full_cache_minutes', 'NA')}. That estimate was about half an hour, so the primary delay and the other whole-hour delays were rebuilt. The old cache in `data/features/chronos_hourly/` was not overwritten. Each new row forecasts hour t from a context that ends at the last legal hour. Artifacts: `runs/pit_verified/chronos_probe.json` and `runs/pit_verified/chronos/delay_*h.parquet`.

## 7. Refit

Command: `python models/run_pit_verified.py all`. Both trees, five seeds, delays of 1, 2, 3, and 6 hours, plus a 24-hour load lag and a dropped-load case. Primary seed-averaged gap: {_money(primary['mae_gap_a_minus_b'])}, interval [{_money(primary['bootstrap_ci_low'])}, {_money(primary['bootstrap_ci_high'])}]. Artifacts: `runs/pit_verified/metrics.csv`, `comparisons.json`, `master_metrics.csv`.

## 8. Remaining checks

February 2021 expanding window, regime bins, block lengths 24/48/72/168, spike intervals with the undefined-draw rule, latency-valid persistence and seasonal naive, and a 2023-tuned lasso on legal lags and calendar. Reports in this directory cite those files. The hourly-mean target is secondary. HB_NORTH, when scored, uses that hub's own delayed prices and its own Chronos cache. It is another ERCOT hub, not another market.

## 9. Verdict

Partially verified, with unresolved concerns. The MAE was not used to choose the label or the primary delay.
"""


def _handoff(primary, both, exog, crisis, baselines, hourly) -> str:
    crisis_mae = "NA"
    if crisis:
        crisis_mae = _money(crisis["subsets"]["crisis_hours"]["xgb_exog"]["mae"])
    mean_gap = _money(hourly["mae_gap_a_minus_b"]) if hourly else "NA"
    return f"""# Paper rewrite handoff

Write from this file and the reports next to it. Do not restore claims from the old PDF. Do not cite 2025 or 2026.

## Title

A point-in-time evaluation of Chronos features for one-hour-ahead ERCOT prices, under an explicit publication delay.

## Question

After removing features that are not known to be public, and after lagging real-time prices and realized load by a stated delay, do Chronos-T5-small statistics from that delayed context lower the absolute error of an XGBoost model that already sees the legal lags, day-ahead prices, storage, and calendar?

## What the study does

The target is the hourly maximum of four 15-minute HB_HUBAVG prices. The issue time is the start of the hour. The primary delay is one hour after interval end. It is an assumption, chosen as the shortest delay in a pre-specified list, not as the best MAE. Trees are matched: same rows, same seeds {{0, 1, 2, 3, 42}}, depth 6, learning rate 0.05, absolute error on asinh, early stopping on 2023, training on 2019–2022, score on the existing 2024 index. The gap is MAE(with Chronos) minus MAE(without).

## Numbers, and the files that hold them

- Pre-correction seed-averaged gap, audit baseline only: about $-0.09, interval covering zero. `reports/pit_verified_reproduction.md`.
- Corrected primary gap, MAE of the averaged forecast: {_money(primary['mae_gap_a_minus_b'])}, interval [{_money(primary['bootstrap_ci_low'])}, {_money(primary['bootstrap_ci_high'])}], excludes zero: {primary['interval_excludes_zero']}. With Chronos {_money(both['mae'])}, without {_money(exog['mae'])}. `runs/pit_verified/comparisons.json`.
- Difference of the mean seed-level MAEs: {_money(primary['gap_of_mean_seed_maes'])}. This is not the same estimand.
- Spike MAE with Chronos {_money(both['spike_mae'])}, without {_money(exog['spike_mae'])}. Recall {_num(both['spike_recall'], 3)} versus {_num(exog['spike_recall'], 3)}.
- February 2021 exogenous MAE on the crisis window: {crisis_mae}. `runs/pit_verified/crisis_2021.json`.
- Hourly-mean secondary gap: {mean_gap}. Its interval can exclude zero. It does not replace the hourly-maximum headline. `runs/pit_verified/hourly_mean.json`.
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
"""
