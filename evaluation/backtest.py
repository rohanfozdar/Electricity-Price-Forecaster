"""
Backtest the CALIBRATED v2 classifier against known ERCOT stress events.

Predictions come from the Platt-calibrated CalibratedClassifierCV wrapping
the v2 enhanced XGBoost classifier (models/artifacts/calibrated_classifier.pkl).
Calibration matters here because the operating threshold (0.3) is only
meaningful when probabilities are well-calibrated - the raw classifier
output was squeezed into [0.36, 0.66], which made any fixed threshold
either fire on everything or nothing.

Events span 2017-2024 across train, val, and test splits. The headline
metric is LEAD TIME: how many hours before the first observed spike
(>$200/MWh) did the calibrated probability first exceed 0.3?
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils.config import FEATURES_DIR
from models.utils import select_enhanced_features

# Stress events spanning 2016-2024, mapped to their split
STRESS_EVENTS = [
    ("Hurricane Harvey impact",   "2017-08-25", "2017-08-30", "train"),
    ("Jan 2018 cold snap",        "2018-01-15", "2018-01-18", "train"),
    ("Aug 2019 heatwave",         "2019-08-12", "2019-08-16", "train"),
    ("Winter Storm Uri",          "2021-02-13", "2021-02-19", "train"),
    ("June 2022 heat stress",     "2022-06-20", "2022-06-24", "train"),
    ("Aug 2023 heatwave",         "2023-08-20", "2023-08-30", "val"),
    ("Jan 2024 cold snap",        "2024-01-14", "2024-01-18", "test"),
    ("Aug 2024 heat",             "2024-08-15", "2024-08-25", "test"),
]

ALERT_THRESHOLD = 0.3
SPIKE_THRESHOLD = 200.0


def _predict_calibrated(df: pd.DataFrame) -> pd.Series:
    """Score the entire feature matrix with the calibrated classifier."""
    cal_path = Path("models/artifacts/calibrated_classifier.pkl")
    if not cal_path.exists():
        raise RuntimeError(f"Calibrated classifier not found at {cal_path}")
    cal = joblib.load(cal_path)
    # Use the exact feature ordering the calibrator was trained with
    if hasattr(cal, "feature_names_in_"):
        features = list(cal.feature_names_in_)
    else:
        features = select_enhanced_features(df)
    X = df[features].ffill().fillna(0)
    proba = cal.predict_proba(X)[:, 1]
    return pd.Series(proba, index=df.index, name="spike_probability_calibrated")


def _summarize_event(df: pd.DataFrame, name: str, start: str, end: str,
                     split: str) -> dict:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    pre_start = start_ts - pd.Timedelta(hours=48)

    pre = df[(df.index >= pre_start) & (df.index < start_ts)]
    event = df[(df.index >= start_ts) & (df.index < end_ts)]

    if event.empty:
        return {"event": name, "split": split, "skipped": "no data"}

    max_actual_price = float(event["HB_HUBAVG"].max())
    spikes = event[event["HB_HUBAVG"] > SPIKE_THRESHOLD]
    any_spike = len(spikes) > 0
    first_spike_idx = spikes.index.min() if any_spike else None

    max_pre_prob = float(pre["spike_probability_calibrated"].max()) if len(pre) else None
    max_event_prob = float(event["spike_probability_calibrated"].max())
    mean_event_prob = float(event["spike_probability_calibrated"].mean())

    # Lead time: first time prob crossed threshold strictly before first spike
    lead_time_hours = None
    first_alert_time = None
    if first_spike_idx is not None:
        lookback = df[(df.index >= pre_start) & (df.index < first_spike_idx)]
        above = lookback[lookback["spike_probability_calibrated"] > ALERT_THRESHOLD]
        if len(above) > 0:
            first_alert_time = above.index.min()
            lead_time_hours = (first_spike_idx - first_alert_time).total_seconds() / 3600

    # If no spike: did model correctly stay below threshold?
    quiet_correct = None
    if not any_spike:
        quiet_correct = bool(max_event_prob < ALERT_THRESHOLD)

    return {
        "event": name,
        "split": split,
        "start": start,
        "end": end,
        "max_actual_price": max_actual_price,
        "actual_spike_occurred": bool(any_spike),
        "first_spike_time": str(first_spike_idx) if first_spike_idx is not None else None,
        "max_pre_event_probability": max_pre_prob,
        "max_event_probability": max_event_prob,
        "mean_event_probability": mean_event_prob,
        "first_alert_time": str(first_alert_time) if first_alert_time is not None else None,
        "lead_time_hours": float(lead_time_hours) if lead_time_hours is not None else None,
        "quiet_event_correctly_silent": quiet_correct,
    }


def _print_event(rep: dict) -> None:
    if rep.get("skipped"):
        print(f"\n  [skip] {rep['event']}: {rep['skipped']}")
        return

    print(f"\n  {rep['event']:<30}  ({rep['split']} split, {rep['start']} -> {rep['end']})")
    print(f"    Max actual price:     ${rep['max_actual_price']:>8.2f}/MWh")
    print(f"    Spike (>${SPIKE_THRESHOLD:.0f}) occurred:   {rep['actual_spike_occurred']}")

    pre_p = rep.get("max_pre_event_probability")
    print(f"    Max prob 48h pre:     {pre_p:.3f}" if pre_p is not None else "    Max prob 48h pre:     n/a")
    print(f"    Max prob during:      {rep['max_event_probability']:.3f}")
    print(f"    Mean prob during:     {rep['mean_event_probability']:.3f}")

    lt = rep.get("lead_time_hours")
    if rep["actual_spike_occurred"]:
        if lt is not None:
            print(f"    Lead time @0.3:       {lt:>5.1f} hours  "
                  f"(first alert {rep['first_alert_time']})")
        else:
            print(f"    Lead time @0.3:       MISSED (probability never crossed {ALERT_THRESHOLD})")
    else:
        ok = rep.get("quiet_event_correctly_silent")
        verdict = "correctly silent" if ok else "false alarm"
        print(f"    No spike occurred -> {verdict} (max prob {rep['max_event_probability']:.3f})")


def _print_timeline(df: pd.DataFrame, name: str, start: str, end: str) -> None:
    """Print hour-by-hour timeline: 48h before through end of event."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    pre_start = start_ts - pd.Timedelta(hours=48)
    window = df[(df.index >= pre_start) & (df.index < end_ts)].copy()

    # Identify markers
    first_alert = window[window["spike_probability_calibrated"] > ALERT_THRESHOLD].index.min()
    first_spike = window[window["HB_HUBAVG"] > SPIKE_THRESHOLD].index.min()

    print(f"\n  --- HOURLY TIMELINE: {name} ---")
    print(f"  Window: {pre_start} -> {end_ts}")
    if pd.notna(first_alert):
        print(f"  First model alert (prob>{ALERT_THRESHOLD}): {first_alert}")
    else:
        print(f"  First model alert (prob>{ALERT_THRESHOLD}): NEVER")
    if pd.notna(first_spike):
        print(f"  First actual spike (>${SPIKE_THRESHOLD:.0f}):    {first_spike}")
    else:
        print(f"  First actual spike (>${SPIKE_THRESHOLD:.0f}):    NEVER (no spike during event)")

    if pd.notna(first_alert) and pd.notna(first_spike):
        gap = (first_spike - first_alert).total_seconds() / 3600
        if gap > 0:
            print(f"  ==> LEAD TIME: {gap:.1f} hours")
        else:
            print(f"  ==> LATE BY: {-gap:.1f} hours (alert came after spike)")

    # Compact print: every hour, but show only key marker rows + 6-hour samples
    # to keep output manageable
    print(f"\n  {'timestamp':<26}  {'price':>10}  {'prob':>6}  {'flag':<8}")
    print("  " + "-" * 60)
    last_printed = None
    for ts, row in window.iterrows():
        flag = ""
        is_marker = False
        if first_alert is not None and ts == first_alert:
            flag = "ALERT*"
            is_marker = True
        if first_spike is not None and ts == first_spike:
            flag = (flag + " SPIKE*").strip()
            is_marker = True
        # Show every 3rd hour, plus all marker rows, plus rows where
        # price > 100 or prob > 0.3
        important = (row["HB_HUBAVG"] > 100) or (row["spike_probability_calibrated"] > 0.3)
        sample = (last_printed is None) or ((ts - last_printed) >= pd.Timedelta(hours=3))
        if not (is_marker or important or sample):
            continue
        print(f"  {str(ts):<26}  ${row['HB_HUBAVG']:>8.2f}  "
              f"{row['spike_probability_calibrated']:>6.3f}  {flag:<8}")
        last_printed = ts


def run_backtest():
    print("\n" + "=" * 75)
    print("  BACKTEST: CALIBRATED v2 CLASSIFIER vs KNOWN STRESS EVENTS")
    print("=" * 75)

    matrix_path = FEATURES_DIR / "feature_matrix_engineered_v2.parquet"
    df = pd.read_parquet(matrix_path)
    print(f"  Loaded {len(df):,} rows from {matrix_path.name}")

    df["spike_probability_calibrated"] = _predict_calibrated(df)
    print(f"  Probability range: [{df['spike_probability_calibrated'].min():.3f}, "
          f"{df['spike_probability_calibrated'].max():.3f}]")

    reports = []
    for name, start, end, split in STRESS_EVENTS:
        rep = _summarize_event(df, name, start, end, split)
        _print_event(rep)
        reports.append(rep)

    # Summary table
    print("\n" + "=" * 75)
    print("  LEAD-TIME SUMMARY (in hours; '-' = not applicable, 'MISS' = no alert)")
    print("=" * 75)
    print(f"  {'Event':<28} {'Split':<6} {'Spike?':<7} {'Max prob':>9} "
          f"{'Lead (h)':>10}")
    print("  " + "-" * 73)
    for r in reports:
        if r.get("skipped"):
            continue
        spike_str = "YES" if r["actual_spike_occurred"] else "no"
        if r["actual_spike_occurred"]:
            lead = f"{r['lead_time_hours']:.1f}" if r["lead_time_hours"] is not None else "MISS"
        else:
            lead = "-"
        print(f"  {r['event']:<28} {r['split']:<6} {spike_str:<7} "
              f"{r['max_event_probability']:>9.3f} {lead:>10}")

    # Hourly timelines for the two key events
    print("\n" + "=" * 75)
    print("  HOURLY TIMELINES (48h pre + event)")
    print("=" * 75)
    for nm, st, en, _ in STRESS_EVENTS:
        if nm in ("Winter Storm Uri", "Jan 2024 cold snap"):
            _print_timeline(df, nm, st, en)

    out_path = Path("evaluation/backtest_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\n  Saved to {out_path}\n")
    return reports


if __name__ == "__main__":
    run_backtest()
