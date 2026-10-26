"""Write the manuscript and peer review from files under runs/pit_ablation.

Numbers are read from those files. Nothing in this module is a fitted result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value, digits=2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _money(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    value = float(value)
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):.2f}"


def _pct(value) -> str:
    return f"{100 * float(value):.2f}%"


def _load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return json.loads(path.read_text())


def _mean_row(metrics: pd.DataFrame, info: str, model: str) -> pd.Series | None:
    part = metrics[(metrics["role"] == "primary") & (metrics["split"] == "test")
                   & (metrics["info_set"] == info) & (metrics["model"] == model)]
    if part.empty:
        return None
    return part.mean(numeric_only=True)


def _find(comparisons: list[dict], name: str) -> dict | None:
    for row in comparisons:
        if row["name"] == name:
            return row
    return None


def _regime_table(out: Path) -> str:
    path = out / "regimes.csv"
    if not path.exists():
        return ""
    frame = pd.read_csv(path)
    keep = frame[(frame["info_set"] == "conservative") & frame["regime"].isin(
        ["all", "price_le_100", "price_gt_100", "price_gt_200", "negative_price"]
    )]
    if keep.empty:
        return ""
    lines = [
        "Conservative information set, seed-averaged forecast, 2024. The $200 threshold is the pre-specified spike rule.",
        "",
        "| Regime | Hours | MAE without Chronos | MAE with Chronos |",
        "| --- | ---: | ---: | ---: |",
    ]
    order = ["all", "price_le_100", "price_gt_100", "price_gt_200", "negative_price"]
    labels = {
        "all": "All hours",
        "price_le_100": "Price at or below $100",
        "price_gt_100": "Price above $100",
        "price_gt_200": "Price above $200",
        "negative_price": "Negative price",
    }
    for regime in order:
        part = keep[keep["regime"] == regime]
        if part.empty:
            continue
        exog = part[part["model"] == "xgb_exog_mean_seed"]
        both = part[part["model"] == "xgb_both_mean_seed"]
        if exog.empty or both.empty:
            continue
        lines.append(
            f"| {labels[regime]} | {int(exog['n'].iloc[0])} | {_money(exog['mae'].iloc[0])} | {_money(both['mae'].iloc[0])} |"
        )
    spike = keep[keep["regime"] == "price_gt_200"]
    if not spike.empty:
        exog = spike[spike["model"] == "xgb_exog_mean_seed"]
        both = spike[spike["model"] == "xgb_both_mean_seed"]
        if not exog.empty and not both.empty:
            gap = float(both["mae"].iloc[0] - exog["mae"].iloc[0])
            lines.append(
                "\nThe all-hours row is the MAE of the seed-averaged forecast. "
                "It is not the average of the five seed-level MAEs in the table above. "
                f"On hours above $200 the MAE gap, with Chronos minus without, is {_money(gap)}/MWh. "
                "That slice was not used to choose the model."
            )
    return "\n".join(lines)


def _year_2021_note(root: Path) -> str:
    path = root / "runs" / "pit_ablation" / "predictions" / "expanding_conservative__xgb_exog_2021__seed42.parquet"
    matrix_path = root / "data" / "features" / "feature_matrix_engineered_v2.parquet"
    if not path.exists() or not matrix_path.exists():
        return ""
    test = pd.read_parquet(path)
    test = test.loc[test["split"] == "test"]
    window = test.loc["2021-02-10":"2021-02-20"]
    outside = test.drop(index=window.index)
    prices = pd.read_parquet(matrix_path, columns=["HB_HUBAVG"])["HB_HUBAVG"]
    persist = prices.shift(freq="h").reindex(test.index)
    persist_mae = float(np.nanmean(np.abs(test["y_true"].to_numpy() - persist.to_numpy())))
    full = pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h", tz="UTC")
    present = prices.loc["2021-01-01":"2021-12-31"].index
    missing = full.difference(present)
    stamps = pd.Series(missing).reset_index(drop=True)
    cuts = stamps.index[stamps.diff().gt(pd.Timedelta(hours=1))].tolist()
    bounds = [0, *cuts, len(stamps)]
    stretches = []
    for start, end in zip(bounds, bounds[1:]):
        stretches.append(f"{stamps.iloc[start]} through {stamps.iloc[end - 1]}")
    inside_mae = float((window["y_true"] - window["y_pred"]).abs().mean())
    outside_mae = float((outside["y_true"] - outside["y_pred"]).abs().mean())
    return (
        f"The 2021 rows use the {len(test):,} hours present in the frozen matrix. "
        f"{len(missing):,} UTC hours in calendar 2021 are absent from that matrix, in "
        + " and ".join(stretches)
        + ". "
        f"Clock-hour persistence on the scored hours has MAE {_money(persist_mae)}/MWh. "
        f"The conservative tree without Chronos has MAE {_money(inside_mae)}/MWh on 2021-02-10 through 2021-02-20, "
        f"when the hourly maximum reaches {_money(float(window['y_true'].max()))}/MWh and the forecast stays at or below "
        f"{_money(float(window['y_pred'].max()))}/MWh, and MAE {_money(outside_mae)}/MWh on the other scored hours of 2021. "
        "Almost all of the year-level absolute error comes from that February window. "
        "It is not evidence that the tree fails in ordinary hours of 2021."
    )


def _paired_factor(frame: pd.DataFrame, column: str) -> float:
    keys = [c for c in ["weather_lag", "gdelt", "gas_train_p90", "plus_features", "window"] if c != column]
    left = frame[frame[column] == True].set_index(keys)["mae_2023"]  # noqa: E712
    right = frame[frame[column] == False].set_index(keys)["mae_2023"]  # noqa: E712
    return float((left - right).dropna().mean())


def _other_checks(root: Path, out: Path, metrics: pd.DataFrame) -> str:
    lines = []
    factorial = out / "factorial_2023.csv"
    if factorial.exists():
        frame = pd.read_csv(factorial)
        lines.append(
            "The 2023 factorial varies weather lag, whether GDELT is kept, the training-only gas threshold, "
            "the eight extra columns, and the training window, on the old feature matrix. "
            "Early stopping ends in 2022. 2024 is not used, and none of these contrasts is a share of the $11.56 result. "
            f"Across the matched pairs, clock-lagging weather changes 2023 MAE by {_money(_paired_factor(frame, 'weather_lag'))}/MWh, "
            f"keeping GDELT changes it by {_money(_paired_factor(frame, 'gdelt'))}/MWh, "
            f"the training-only gas threshold changes it by {_money(_paired_factor(frame, 'gas_train_p90'))}/MWh, "
            f"and the eight extra columns change it by {_money(_paired_factor(frame, 'plus_features'))}/MWh. "
            "A positive number means the named choice has higher MAE. "
            f"The 2019-start window averages {_money(float(frame.loc[frame['window']=='y2019', 'mae_2023'].mean()))}/MWh "
            f"and the full window averages {_money(float(frame.loc[frame['window']=='full', 'mae_2023'].mean()))}/MWh. "
            "The primary fits were not switched to whichever cell won on 2023."
        )
    hp = out / "hyperparameter_check_2023.csv"
    if hp.exists():
        frame = pd.read_csv(hp)
        frozen = frame[frame["frozen_primary"] == True]  # noqa: E712
        best = frame.loc[frame["mae_2023"].idxmin()]
        frozen_exog = frozen[frozen["model"] == "xgb_exog"]
        frozen_txt = _money(float(frozen_exog["mae_2023"].iloc[0])) if not frozen_exog.empty else "—"
        lines.append(
            "\n\nThe depth and learning-rate grid is scored on 2023 only, after early stopping on the last 90 days of 2022. "
            f"The lowest 2023 MAE in the grid is {_money(float(best['mae_2023']))}/MWh "
            f"at depth {int(best['depth'])}, learning rate {best['lr']}, model {best['model']}. "
            f"The frozen primary setting, depth 6 and learning rate 0.05, has 2023 MAE {frozen_txt}/MWh "
            "for exogenous features without Chronos. 2024 was not rescored with the 2023 winner."
        )
    hub = metrics[(metrics["role"] == "hub_transfer") & (metrics["split"] == "test")]
    if not hub.empty:
        lines.append(
            "\n\nHB_NORTH uses that hub's hourly maximum as the target, the same conservative columns, "
            "and the HB_HUBAVG Chronos cache without refitting it. "
            f"Across five seeds, XGBoost without Chronos has 2024 MAE {_money(float(hub.loc[hub['model']=='xgb_exog', 'mae'].mean()))}/MWh "
            f"and XGBoost with the Chronos statistics has {_money(float(hub.loc[hub['model']=='xgb_both', 'mae'].mean()))}/MWh. "
            + _hub_interval()
        )
    gfs = metrics[(metrics["role"] == "gfs_sensitivity") & (metrics["split"] == "test")]
    if not gfs.empty:
        def _g(info, model):
            part = gfs[(gfs["info_set"] == info) & (gfs["model"] == model)]
            return float(part["mae"].iloc[0]) if not part.empty else float("nan")
        lines.append(
            "\n\nGFS previous-run temperature at a one-day lead is scored from a training window that starts 2021-04-01, "
            "because that archive starts in March 2021. It is temperature, not a wind or solar generation forecast. "
            f"On 2024, seed 42, the temperature-augmented exogenous model has MAE {_money(_g('gfs_temp', 'xgb_exog'))}/MWh "
            f"and the same model with Chronos statistics has {_money(_g('gfs_temp', 'xgb_both'))}/MWh. "
            f"The conservative model trained on that same shorter window has {_money(_g('conservative_from_2021_04', 'xgb_exog'))}/MWh "
            f"without Chronos and {_money(_g('conservative_from_2021_04', 'xgb_both'))}/MWh with Chronos. "
            "Adding the temperature forecasts does not lower MAE relative to that matched window."
        )
    return "".join(lines)


def _hub_interval() -> str:
    try:
        from models.run_pit_ablation import compare_pair, mean_test_vectors
    except Exception:
        return "No interval was computed."
    try:
        index, y, both = mean_test_vectors("hub_HB_NORTH", "xgb_both")
        _, _, exog = mean_test_vectors("hub_HB_NORTH", "xgb_exog")
    except Exception:
        return "The saved hub forecasts were not both available."
    row = compare_pair(
        "hub_HB_NORTH__mean_seed__both_minus_exog", "hub_HB_NORTH", y, both, exog, index,
        "xgb_both_mean_seed", "xgb_exog_mean_seed",
    )
    cover = "covers zero" if not row["interval_excludes_zero"] else "does not cover zero"
    return (
        f"The seed-averaged MAE gap is {_money(row['mae_gap_a_minus_b'])}/MWh "
        f"with interval [{_money(row['bootstrap_ci_low'])}, {_money(row['bootstrap_ci_high'])}], which {cover}."
    )


def _later_note(later: dict, metrics: pd.DataFrame) -> str:
    years = later.get("years") or {}
    bits = []
    for year, info in years.items():
        extra = f" Fallback: {info['rt_fallback']}." if info.get("rt_fallback") else ""
        bits.append(f"{year}: real-time {info.get('rt')}; day-ahead {info.get('da')}.{extra}")
    year_text = " ".join(bits) if bits else "No per-year fetch record was written."
    primary = metrics[(metrics["role"] == "primary") & (metrics["split"] == "test")
                      & (metrics["info_set"] == "conservative") & (metrics["model"] == "xgb_exog")]
    rmse_2024 = f"{primary['rmse'].mean():.2f}" if not primary.empty else "—"
    cutoff = later.get("scored_through_exclusive_utc", "not recorded")
    return (
        f"Fetch record: {year_text} "
        f"Scores use Central calendar years and stop before {cutoff}. {later.get('cutoff_rule', '')} "
        f"Not scored: {later.get('not_scored', '')}. Load: {later.get('load', '')}. "
        f"The 2024 conservative model without Chronos has RMSE ${rmse_2024}/MWh on the earlier test year, "
        "so a lower MAE after 2024 is not, by itself, evidence that the model improved. "
        "This block does not score Chronos statistics."
    )


def write_manuscript(root: Path, out: Path, comparisons: list[dict], metrics: pd.DataFrame) -> None:
    cons = _find(comparisons, "conservative__mean_seed__both_minus_exog")
    assumed = _find(comparisons, "assumed_realtime__mean_seed__both_minus_exog")
    reported = _find(comparisons, "reported__mean_seed__both_minus_exog")
    if cons is None:
        raise FileNotFoundError("conservative comparison is missing")
    if cons["a_better_on_mae"]:
        title = "Do frozen Chronos forecast statistics improve point-in-time ERCOT real-time price forecasts?"
        chronos_sentence = (
            "On the conservative information set, adding the six Chronos statistics lowers mean absolute error "
            f"by {_money(-cons['mae_gap_a_minus_b'])}/MWh relative to the matched gradient-boosted model. "
            f"The 95% moving-block interval for the MAE gap is [{_money(cons['bootstrap_ci_low'])}, {_money(cons['bootstrap_ci_high'])}] "
            "and does not cover zero."
        )
    elif cons["bootstrap_ci_low"] > 0:
        title = "A point-in-time evaluation of foundation-model features for ERCOT real-time price forecasting"
        chronos_sentence = (
            "On the conservative information set, adding the six Chronos statistics raises mean absolute error "
            f"by {_money(cons['mae_gap_a_minus_b'])}/MWh relative to the matched gradient-boosted model. "
            f"The 95% interval is [{_money(cons['bootstrap_ci_low'])}, {_money(cons['bootstrap_ci_high'])}]."
        )
    else:
        title = "A point-in-time evaluation of foundation-model features for ERCOT real-time price forecasting"
        chronos_sentence = (
            "On the conservative information set, the MAE gap from adding the six Chronos statistics is "
            f"{_money(cons['mae_gap_a_minus_b'])}/MWh, with a 95% moving-block interval "
            f"[{_money(cons['bootstrap_ci_low'])}, {_money(cons['bootstrap_ci_high'])}] that covers zero. "
            "The statistics are not distinguishable from noise on this comparison."
        )

    def line(info, model, label):
        row = _mean_row(metrics, info, model)
        if row is None:
            return None
        return (
            f"| {label} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} | {_fmt(row['medae'])} | "
            f"{_pct(row['spike_recall'])} | {_pct(row['spike_precision'])} | {_fmt(row['spike_f1'], 3)} |"
        )

    primary_lines = []
    for info, model, label in (
        ("conservative", "xgb_exog", "Conservative, XGBoost without Chronos"),
        ("conservative", "xgb_chronos", "Conservative, XGBoost on Chronos statistics only"),
        ("conservative", "xgb_both", "Conservative, XGBoost with Chronos statistics"),
        ("assumed_realtime", "xgb_exog", "Assumed real-time, XGBoost without Chronos"),
        ("assumed_realtime", "xgb_both", "Assumed real-time, XGBoost with Chronos statistics"),
        ("reported", "xgb_exog", "Reported honest-tree features, XGBoost without Chronos"),
        ("reported", "xgb_both", "Reported honest-tree features, XGBoost with Chronos statistics"),
    ):
        text = line(info, model, label)
        if text:
            primary_lines.append(text)

    hist = metrics[(metrics["role"].isin(["historical", "replay"])) & (metrics["split"] == "test")]
    hist_lines = []
    for _, row in hist.iterrows():
        hist_lines.append(
            f"| {row['model']} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} | "
            f"{_pct(row['spike_recall'])} | {_pct(row['spike_precision'])} |"
        )

    expanding = _load(out / "expanding_window.csv")
    expand_lines = []
    if expanding is not None:
        for _, row in expanding.iterrows():
            expand_lines.append(
                f"| {int(row['year'])} | {row['info_set']} | {row['model']} | {row['role']} | "
                f"{int(row['n_test'])} | {int(row['n_spikes'])} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} |"
            )

    later = _load(out / "later_years.json") or {}
    later_metrics = _load(out / "later_years_metrics.csv")
    later_lines = []
    if later_metrics is not None:
        for _, row in later_metrics.iterrows():
            spikes = int(row["tp"] + row["fn"]) if pd.notna(row.get("tp")) else "—"
            later_lines.append(
                f"| {int(row['year'])} | {row['model']} | {int(row['n'])} | {spikes} | {_fmt(row['mae'])} | "
                f"{_fmt(row['rmse'])} | {_fmt(row['spike_f1'], 3)} |"
            )

    cons_both = _mean_row(metrics, "conservative", "xgb_both")
    cons_exog = _mean_row(metrics, "conservative", "xgb_exog")
    abstract_numbers = (
        f"Across five seeds, conservative XGBoost without Chronos has 2024 MAE ${_fmt(cons_exog['mae'])}/MWh "
        f"and the same model with the six Chronos statistics has MAE ${_fmt(cons_both['mae'])}/MWh."
        if cons_both is not None and cons_exog is not None else ""
    )

    parts = []
    parts.append(f"# {title}\n")
    parts.append("## Other titles that were not used\n")
    parts.append("- Do frozen Chronos forecast statistics improve point-in-time ERCOT real-time price forecasts?")
    parts.append("- When do zero-shot Chronos statistics help a gradient-boosted ERCOT price forecast?")
    parts.append("- A point-in-time evaluation of foundation-model features for ERCOT real-time price forecasting\n")
    parts.append("The title above is the one that matches the conservative-set interval. The other two are recorded so the choice is visible.\n")
    parts.append("## Abstract\n")
    parts.append(
        "This paper asks whether six forecast statistics from a frozen, zero-shot Chronos-T5-small model "
        "improve a one-hour-ahead forecast of the ERCOT real-time hub-average price. The target at hour t is "
        "HB_HUBAVG, the maximum of the four 15-minute settlement prices inside that hour. The forecast is issued "
        "at the start of hour t. A feature may be used only when its availability time is at or before that instant. "
        "Where a release time could not be verified, the feature is either lagged under an explicit assumption or removed. "
        "The conservative information set keeps price history, publication-gated day-ahead prices, a one-hour lag of realized load "
        "under an unverified posting assumption, weekly storage under a documented release offset, Central-time calendar variables, "
        "and the Chronos statistics. It removes archive weather, Henry Hub, and nuclear-outage status. "
        f"{abstract_numbers} {chronos_sentence} "
        "The 2024 evaluation year was inspected in earlier development and is not an untouched test set. "
        "An expanding-window backtest covers 2021–2024 under a rule fixed before those scores were read, "
        "but the column set itself was specified after that inspection. "
        "The study is one hub in one market. It does not establish a new forecasting architecture."
    )
    parts.append("\n## 1. Contributions\n")
    parts.append(
        "1. A column-level availability audit of the existing ERCOT hourly-maximum forecast, separating documented release rules, "
        "assumptions, and inputs that are not usable at the issue time.\n"
        "2. A matched comparison, on identical origins, of persistence, a Chronos zero-shot median, XGBoost without Chronos statistics, "
        "XGBoost on those statistics alone, and XGBoost on both, under three labeled information sets.\n"
        "3. Expanding-window scores for 2021–2024 and regime errors, with a block-bootstrap interval for the Chronos MAE gap.\n"
        "\nThese statements are limited to the runs in `runs/pit_ablation/`. They do not say that an earlier capstone leaderboard "
        "was reproduced, that news sentiment improves the forecast, or that the hybrid beats deep-learning models under a shared feature set."
    )
    parts.append("\n## 2. What the earlier manuscript claimed\n")
    parts.append(
        "The capstone PDF, `docs/static/pdfs/paper.pdf`, reports a Chronos–XGBoost hybrid at MAE $12.13/MWh, RMSE $93.98, "
        "spike recall 61.40%, and spike precision 50.43%, and a 29.3% MAE reduction from GDELT sentiment. "
        "The hybrid cells were not executed. On the frozen 2024 index of 8,767 hours and 114 spikes, 61.40% is 70/114, "
        "and precision 50.43% is then not a ratio of two integers. A faithful rerun has MAE $21.96. "
        "A later honest-tree protocol, with a different loss, target transform, window, and feature timing, has MAE $11.56 "
        "and a quantile stack has MAE $12.13 with RMSE $96.23. That stack is not the PDF row. "
        "The $11.56 fit always included the Chronos statistics, so it did not measure their incremental value. "
        "The audit is `reports/claim_audit.md`."
    )
    parts.append("\n## 3. Forecast definition\n")
    parts.append(
        "The market is ERCOT real-time settlement. The series is the hub-average settlement-point price. "
        "Hour t is the UTC hour beginning at t, which is the same four interval starts as the corresponding Central hour "
        "because Central time differs from UTC by a whole number of hours. The target is the maximum of those four prices. "
        "The v2 matrix does not require all four intervals to be present. Those hours were counted and retained so the index "
        "would stay the frozen 2024 index. The early-warning code drops incomplete hours; this study does not silently switch to that rule.\n\n"
        "The issue time is the start of hour t. The target hour has not occurred. It becomes observable when the last "
        "15-minute interval is posted, nominally at t plus one hour, plus a posting delay that this repository does not measure. "
        "This is a one-hour-ahead forecast. It is not a day-ahead bid submitted after the day-ahead market clears."
    )
    parts.append("\n## 4. Information sets\n")
    parts.append(
        "Three sets are scored. `reported` is the honest-tree column list, including same-hour realized load, trade-date gas, "
        "week-ending storage, report-date reactor status, UTC calendar features, and archive weather lagged one hour by the clock. "
        "It is a retrospective set with known gaps. Missing values are forward-filled inside each split and then set to zero, "
        "matching that protocol.\n\n"
        "`assumed_realtime` drops same-hour load. Weather is still archive reanalysis lagged one hour and is labeled an unverified "
        "station proxy. Gas uses the YAML weekly rule, which the file marks as assumed. Storage uses the documented Thursday release "
        "plus a buffer, implemented as the following Friday 18:00 UTC. Reactor status uses an assumed next-UTC-day rule. "
        "Day-ahead prices are masked unless their publication time, 14:00 Central on the previous local day, is at or before the issue time. "
        "Calendar features use US/Central time. Peak hours are weekday 16:00–22:00 local, which is what the PDF stated and not what the v2 matrix computed. "
        "Signed price features and the Chronos dollar statistics use `asinh`. Medians are fit on the training rows of that fit. "
        "A missingness indicator is kept for each family. Nothing is filled with zero.\n\n"
        "`conservative` starts from that set and removes archive weather, gas, and reactor status. Storage remains. "
        "Lagged realized load remains, under the assumption that the previous hour's load has been posted. That assumption is not a measured timestamp.\n\n"
        "The Chronos block is the existing gap-aware cache from `amazon/chronos-t5-small`: median, 10th and 90th percentiles, mean, "
        "the share of 20 paths above $200/MWh, and the gap between the 90th and 10th percentiles. The context check requires the 512-hour "
        "window to end at t−1 for every 2024 test hour. The cache was not recomputed. The check result is `runs/pit_ablation/chronos_context_check.json`."
    )
    parts.append("\n## 5. Models and uncertainty\n")
    parts.append(
        "The primary tree is XGBoost with histogram splits, absolute error, depth 6, learning rate 0.05, at most 2,000 trees, "
        "and early stopping after 50 rounds on 2023. The target inside the model is `asinh` of price, clipped to ±10 before `sinh`. "
        "Seeds are 0, 1, 2, 3, and 42. The same settings are used for exogenous features only, Chronos statistics only, and both. "
        "A depth and learning-rate grid is scored on 2023 after early stopping on the last 90 days of 2022. It is not used to replace the 2024 models.\n\n"
        "Persistence, seasonal persistence, the day-ahead price, the thin LEAR, the zero-shot Chronos median, the saved honest tree, "
        "and the saved quantile stack are copied from earlier artifacts. They are not refit on the new information sets. "
        "The thin LEAR is three price lags, the day-ahead price, and calendar dummies. It is not the LEAR benchmark of "
        "Lago, Marcjasz, De Schutter, and Weron, Applied Energy 293 (2021) 116983.\n\n"
        "The pre-registered contrast is MAE with the Chronos statistics minus MAE without them, on the same hours. "
        "Uncertainty is a moving-block bootstrap with 24-hour blocks and 2,000 resamples, inside contiguous stretches of the test index, "
        "and a Diebold–Mariano test on absolute error with HAC lag 24 and the Harvey–Leybourne–Newbold correction. "
        "The primary interval uses the seed-averaged forecast. Per-seed gaps are reported beside it. "
        "Spike precision and recall use the pre-specified threshold of $200/MWh. That threshold is not tuned on 2024."
    )
    parts.append("\n## 6. Results\n")
    parts.append("Means across the five primary seeds, 2024. Spike rates are means of the seed-level rates.\n")
    parts.append("| Model | MAE | RMSE | Median AE | Spike recall | Spike precision | Spike F1 |")
    parts.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    parts.extend(primary_lines)
    parts.append("\nUnchanged earlier forecasts on the same index:\n")
    parts.append("| Model | MAE | RMSE | Spike recall | Spike precision |")
    parts.append("| --- | ---: | ---: | ---: | ---: |")
    parts.extend(hist_lines)
    parts.append("\n" + chronos_sentence)
    seeds = cons.get("per_seed_mae_gap_both_minus_exog") or []
    if seeds:
        parts.append(
            " Per-seed gaps on that set, in seed order 0, 1, 2, 3, 42, are "
            + ", ".join(_money(v) for v in seeds)
            + "."
        )
    seed42 = _find(comparisons, "conservative__seed42__both_minus_exog")
    if seed42 is not None:
        parts.append(
            f" The seed-42 pair alone has gap {_money(seed42['mae_gap_a_minus_b'])} "
            f"[{_money(seed42['bootstrap_ci_low'])}, {_money(seed42['bootstrap_ci_high'])}]. "
            "That pair is not the pre-registered summary. The summary is the seed-averaged forecast."
        )
    if assumed is not None:
        assumed42 = _find(comparisons, "assumed_realtime__seed42__both_minus_exog")
        parts.append(
            f" On the assumed-real-time set the seed-averaged MAE gap is {_money(assumed['mae_gap_a_minus_b'])} "
            f"[{_money(assumed['bootstrap_ci_low'])}, {_money(assumed['bootstrap_ci_high'])}]."
        )
        if assumed["a_better_on_mae"]:
            parts.append(
                " That interval lies below zero. The set still includes archive weather, gas, and reactor status under rules that are not measured publication times."
            )
        if assumed42 is not None:
            parts.append(
                f" Its seed-42 interval is [{_money(assumed42['bootstrap_ci_low'])}, {_money(assumed42['bootstrap_ci_high'])}]."
            )
    if reported is not None:
        parts.append(
            f" On the reported honest-tree features the gap is {_money(reported['mae_gap_a_minus_b'])} "
            f"[{_money(reported['bootstrap_ci_low'])}, {_money(reported['bootstrap_ci_high'])}]. "
            "That set is not a clean publication-time comparison."
        )
    parts.append(
        "\n\nA negative gap means the model with Chronos statistics has lower MAE. "
        "An interval that covers zero means the study does not support a claim that Chronos helps. "
        "The pre-registered contrast is the conservative seed-averaged forecast. "
        "Large spike errors can dominate RMSE while leaving MAE almost unchanged.\n"
    )
    parts.append(_regime_table(out))
    if expand_lines:
        parts.append("\n### Expanding window\n")
        parts.append(
            "Each test year is scored by a model trained only on earlier hours, with early stopping on the last 90 days before that year. "
            "Seed 42 only. The column set was specified after 2024 had already been inspected, so these years are not an untouched confirmation "
            "of that specification. 2023 is the development year of the earlier protocol. 2024 is previously inspected.\n"
        )
        parts.append("| Year | Set | Model | Role | Hours | Spikes | MAE | RMSE |")
        parts.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |")
        parts.extend(expand_lines)
        parts.append("\n" + _year_2021_note(root))
    if later_lines:
        parts.append("\n### Hours after 2024\n")
        parts.append(
            "These hours were fetched after the 2024 comparison was frozen. The scored model is the conservative XGBoost fit without Chronos statistics, "
            "trained on 2019–2022. Chronos statistics were not scored, because the cache ends in 2024 and was not rerun. "
            "Load was not refetched; the load missingness indicator is on and the training median is used. "
            "Incomplete hours, those without four 15-minute prices, are dropped in this extension only.\n"
        )
        parts.append("| Year | Model | Hours | Spikes | MAE | RMSE | Spike F1 |")
        parts.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        parts.extend(later_lines)
        parts.append("\n" + _later_note(later, metrics))
    elif later.get("status"):
        parts.append(
            f"\n### Hours after 2024\n\nThe public-archive fetch status is `{later.get('status')}`"
            + (f": {later.get('error')}" if later.get("error") else "")
            + ". No 2025 or 2026 price score is claimed."
        )
    parts.append("\n### Other checks\n")
    parts.append(_other_checks(root, out, metrics))
    parts.append("\n## 7. Related work and what is not novel\n")
    parts.append(
        "Appending a frozen forecast to a gradient-boosted model is stacking. "
        "Lago, Marcjasz, De Schutter, and Weron, Applied Energy 293 (2021) 116983, already require ex-ante features, "
        "a LEAR baseline, multi-year evaluation, and dependence-aware comparison for day-ahead prices. "
        "Their target is a day-ahead auction price, not the real-time hourly maximum used here.\n\n"
        "The FutureBoosting preprint, arXiv:2603.06726, uses a frozen time-series foundation model to forecast variables that are not yet known "
        "and passes those forecasts to a gradient-boosted regressor for electricity prices. "
        "The present study instead feeds a univariate Chronos-T5-small forecast of the price itself into XGBoost, "
        "and its useful content is the publication-time audit and the matched ablation, not a new hybrid architecture.\n\n"
        "Chronos is Ansari and coauthors, arXiv:2403.07815. The capstone PDF's TimesFM reference, arXiv:2304.08424, is the TiDE paper, not TimesFM.\n\n"
        "Four framings were considered. Foundation-model feature augmentation is not a sufficient claim of novelty. "
        "Extreme-event skill is not the lead claim unless the regime table shows a gain on spikes that survives the same interval; "
        "the point metrics above are the ones that were pre-registered. "
        "Robustness across years is only as strong as the expanding-window table. "
        "The framing that matches the evidence is a point-in-time benchmark of when these Chronos statistics help or fail."
    )
    parts.append("\n## 8. Limitations\n")
    parts.append(
        "Real-time price posting is assumed to have happened by the top of the next hour. The delay was not measured. "
        "Lagged load uses the same kind of assumption. The gas and reactor rules that appear only in the assumed-real-time set are marked assumed in the YAML file. "
        "Archive weather is not a forecast vintage. The GFS sensitivity is temperature at a one-day lead from 2021 onward, not the operational ERCOT wind and solar forecasts.\n\n"
        "2024 was used in earlier development. Scoring a pre-specified comparison on it is not the same as confirming a model that was chosen on it, "
        "and it is also not a fresh test. The expanding window does not repair that, because the columns were chosen with knowledge of this history. "
        "A later period is a confirmation only to the extent the fetch section reports one.\n\n"
        "The bootstrap treats dependence inside 24-hour blocks. It is not a claim that 24 is the true dependence length. "
        "Five seeds describe fitting noise for this tree. They are not a posterior. "
        "The study is one hub-average series. The HB_NORTH transfer is the only geographic check, and it reuses the hub-average Chronos cache.\n\n"
        "Spike recall near one half, if that is what the table shows, still leaves a large share of $200 hours undetected, "
        "and absolute errors on the highest prices remain large. Predicting the hourly maximum is a different decision from predicting the hourly average "
        "or the day-ahead settlement. This paper does not claim the maximum is the right operational target. It keeps it because it is the target the earlier study defined."
    )
    parts.append("\n## 9. Reproducibility\n")
    parts.append(
        "From the repository root, with the local raw files and the Chronos cache present:\n\n"
        "```bash\n"
        "python models/run_pit_ablation.py all\n"
        "python -m pytest tests/test_pit_sets.py tests/test_feature_availability.py tests/test_honest_mae.py -q\n"
        "```\n\n"
        "The command writes `runs/pit_ablation/` and this manuscript's numerical sections. "
        "`artifacts/honest/` and `docs/static/pdfs/paper.pdf` are not modified. "
        "Data hashes, the git commit, package versions, and the GPU string are in `runs/pit_ablation/provenance.json` "
        "and `runs/pit_ablation/build_manifest.json`. Chronos weights are the public checkpoint `amazon/chronos-t5-small` and are not vendored. "
        "A 2025–2026 extension also needs network access to ERCOT's public archive and does not retrain Chronos."
    )
    parts.append("\n## Appendix A. Capstone PDF table, not a result of this protocol\n")
    parts.append(
        "These figures are what the PDF prints. They were not produced by one executed information set. "
        "They are not compared with the tables above.\n"
    )
    parts.append("| Model | MAE | RMSE | Spike recall | Spike precision |")
    parts.append("| --- | ---: | ---: | ---: | ---: |")
    for name, mae, rmse, rec, prec in (
        ("Chronos hybrid (PDF)", "12.13", "93.98", "61.40%", "50.43%"),
        ("CatBoost (PDF)", "13.75", "95.48", "44.35%", "56.94%"),
        ("NGBoost (PDF)", "14.59", "106.44", "25.44%", "59.18%"),
        ("Chronos (PDF)", "14.72", "115.59", "42.11%", "60.76%"),
        ("SARIMAX (PDF)", "18.45", "124.31", "45.61%", "38.81%"),
        ("LightGBM (PDF)", "19.76", "100.41", "34.21%", "45.35%"),
        ("LSTM (PDF)", "24.42", "148.72", "16.67%", "23.46%"),
        ("XGBoost (PDF)", "24.43", "112.46", "57.02%", "25.10%"),
        ("TimesFM (PDF)", "32.51", "136.94", "1.75%", "11.76%"),
        ("XGBoost, no sentiment (PDF)", "34.53", "122.57", "50.53%", "19.54%"),
        ("TFT (PDF)", "45.75", "134.60", "0.00%", "0.00%"),
        ("Prophet hybrid (PDF)", "48.36", "108.96", "71.05%", "14.16%"),
        ("Prophet (PDF)", "119.14", "168.75", "57.14%", "17.70%"),
    ):
        parts.append(f"| {name} | {mae} | {rmse} | {rec} | {prec} |")
    parts.append(
        "\nThe honest-tree MAE of $11.56 and the stack MAE of $12.13 with RMSE $96.23 belong to `reports/honest_mae.md`. "
        "They are traceable on the frozen index and are not rows of the PDF table."
    )
    text = "\n".join(parts) + "\n"
    (root / "reports" / "manuscript.md").write_text(text)
    _write_review(root, cons, later, expanding is not None, later_metrics is not None)
    print(f"  wrote {root / 'reports' / 'manuscript.md'}", flush=True)


def _write_review(root: Path, cons: dict, later: dict, expanding: bool, later_scored: bool) -> None:
    helps = bool(cons["a_better_on_mae"])
    covers = not cons["interval_excludes_zero"]
    gen = 5 if later_scored else (4 if expanding else 2)
    sound = 6
    rigor = 6 if expanding else 5
    if covers:
        significance = 4
        originality = 4
    elif helps:
        significance = 5
        originality = 4
    else:
        significance = 4
        originality = 4
    overall = "The revised draft is a possible workshop paper if the claims stay inside the tables. It is not ready for an applied energy journal."
    if not expanding:
        overall = "The ablation exists, but the generalization block did not finish. Do not submit."
    text = f"""# Peer review

This review scores the capstone PDF together with the $11.56 rewrite, and then scores only the work this protocol executed. Unrun extensions get no credit.

## A. Current paper, before this protocol

| Criterion | Score |
| --- | ---: |
| Originality | 2 |
| Technical soundness | 3 |
| Experimental rigor | 3 |
| Reproducibility | 2 for the PDF, 4 for the honest-tree artifacts |
| Clarity | 4 |
| Significance | 3 |
| Generalization | 2 |
| Publication readiness | reject |

The PDF's hybrid row was not executed, its spike precision is not an integer ratio on 114 events, and the sentiment result is not a stable matched ablation. The $11.56 tree is traceable and still uses inputs whose publication time is unverified. It never removes the Chronos columns, so it cannot show that those columns help. CatBoost's ordered boosting is described as if it blocked future prices. The TimesFM citation points at the TiDE preprint.

Strongest aspect: the target, the hub, and the frozen 2024 index are concrete, and the later protocol saved forecasts that can be recomputed.

Fatal weakness: a forecasting claim is being made with features that were not shown to be available at the issue time, and the central model has no matched ablation.

## B. Work executed in this protocol

| Criterion | Score |
| --- | ---: |
| Originality | {originality} |
| Technical soundness | {sound} |
| Experimental rigor | {rigor} |
| Reproducibility | 6 |
| Clarity | 6 |
| Significance | {significance} |
| Generalization | {gen} |
| Publication readiness | major revision before a workshop; not a journal submission |

Conservative-set MAE gap (with Chronos minus without), seed-averaged forecast: {_fmt(cons['mae_gap_a_minus_b'])} with interval [{_fmt(cons['bootstrap_ci_low'])}, {_fmt(cons['bootstrap_ci_high'])}]. Interval covers zero: {covers}. Chronos lower MAE with the interval entirely below zero: {helps}.

Strongest aspect: the comparison holds the origin, the target, the split, the loss, and the tree settings fixed, and it labels assumed release rules instead of treating them as measurements.

Potentially fatal if over-claimed: 2024 was already inspected; lagged load and the real-time posting delay are still assumptions; the result is one series. Treating a small MAE gap as a new hybrid method would recreate the original paper's problem.

Major concerns: no ERCOT wind or solar forecast vintage in the primary set; the gas rule remains assumed wherever gas is used; the block length of 24 hours is a choice; five seeds are a thin description of fitting noise.

Minor concerns: the v2 hourly maximum still keeps hours that do not have four intervals; the thin lasso is not the published LEAR benchmark; calendar peak hours follow the PDF's evening window rather than an ERCOT tariff definition.

Missing if one were submitting to a journal: a pre-registered later year scored with Chronos as well as without it, operational load and renewable forecasts joined on their release times, and at least one more market under the same rules.

Questions a reviewer would ask: Was any 2024 number used to decide that the conservative set was the one to emphasize? What happens if the real-time price is not available until fifteen minutes after the interval? Why is the hourly maximum the decision, rather than the hourly average or the day-ahead price?

Later-year fetch status: `{later.get("status", "not recorded")}`. Per year: `{later.get("years", {})}`. Expanding window present: {expanding}. Later years scored: {later_scored}.

Recommendation: {overall}
"""
    (root / "reports" / "peer_review.md").write_text(text)
