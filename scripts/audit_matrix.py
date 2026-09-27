"""Gate A evidence: timestamps, DST, row loss, same-hour shifts, and label integrity.

Read-only with respect to data. Loads the raw parquet files in data/raw/ and the
committed engineered matrix, then writes reports/audit_evidence.json.

    python scripts/audit_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.utils import (  # noqa: E402
    chronological_split,
    select_baseline_features,
    select_enhanced_features,
)

RAW = ROOT / "data" / "raw"
FEAT = ROOT / "data" / "features"
OUT = ROOT / "reports" / "audit_evidence.json"
THRESHOLD = 200.0


def _read(name: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW / f"{name}.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _gaps(idx: pd.DatetimeIndex, step: str) -> list[dict]:
    d = pd.Series(idx[1:] - idx[:-1], index=idx[1:])
    big = d[d > pd.Timedelta(step)]
    return [{"after": str(t - g), "resumes": str(t), "gap": str(g)} for t, g in big.items()]


def _eq_frac(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() == 0:
        return float("nan")
    return float((np.isclose(a[m].astype(float), b[m].astype(float), atol=1e-9)).mean())


def _lag_match(feature: pd.Series, source: pd.Series, lags=(0, 1, 2, 24)) -> dict:
    """Fraction of rows where feature(t) == source(t - lag), for each lag in hours."""
    return {f"lag_{k}h": _eq_frac(feature, source.shift(k, freq="h").reindex(feature.index))
            for k in lags}


def source_inventory() -> dict:
    inv = {}
    for p in sorted(RAW.glob("*.parquet")):
        df = _read(p.stem)
        idx = df.index
        step = pd.Series(idx[1:] - idx[:-1]).mode().iloc[0]
        inv[p.stem] = {
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
            "tz": str(idx.tz),
            "start": str(idx.min()),
            "end": str(idx.max()),
            "modal_step": str(step),
            "duplicates": int(idx.duplicated().sum()),
            "gaps_beyond_modal_step": _gaps(idx, str(step))[:10],
            "n_gaps_beyond_modal_step": len(_gaps(idx, str(step))),
            "index_hour_values": sorted(set(idx.hour.tolist()))[:5],
            "index_weekday_values": sorted(set(idx.dayofweek.tolist())),
        }
    return inv


def rt_evidence(rt: pd.DataFrame) -> dict:
    s = rt["HB_HUBAVG"]
    per_hour = s.resample("1h").count()
    per_hour = per_hour[(per_hour.index >= s.index.min().floor("h")) & (per_hour.index <= s.index.max())]
    central = s.index.tz_convert("US/Central")
    per_local_day = pd.Series(1, index=central).groupby(central.date).sum()
    dst_days = {str(d): int(n) for d, n in per_local_day.items() if n not in (96,)}
    hourly_max = s.resample("1h").max()
    hourly_mean = s.resample("1h").mean()
    return {
        "hours_with_4_intervals": int((per_hour == 4).sum()),
        "hours_with_other_counts": {str(k): int(v) for k, v in per_hour[per_hour != 4].value_counts().items()},
        "local_days_not_96_intervals": dict(list(dst_days.items())[:30]),
        "n_local_days_not_96": len(dst_days),
        "first_local_ts": str(central.min()),
        "last_local_ts": str(central.max()),
        "last_utc_ts": str(s.index.max()),
        "hours_missing_at_end_of_2024_utc": int((pd.Timestamp("2024-12-31 23:00", tz="UTC") - s.index.max().floor("h")) / pd.Timedelta("1h")),
        "spike_rate_hourly_max": float((hourly_max > THRESHOLD).mean()),
        "spike_rate_hourly_mean": float((hourly_mean > THRESHOLD).mean()),
        "spike_hours_hourly_max": int((hourly_max > THRESHOLD).sum()),
        "spike_hours_hourly_mean": int((hourly_mean > THRESHOLD).sum()),
        "negative_15min_share": float((s < 0).mean()),
    }


def diurnal_evidence(rt, load, weather) -> dict:
    """If a series is labelled UTC but actually Central local time, its diurnal
    profile shifts 5-6 hours earlier than the UTC-correct series."""
    price = rt["HB_HUBAVG"].resample("1h").mean()
    temp = weather[[c for c in weather.columns if c.endswith("_temperature_2m")]].mean(axis=1)
    L = load["load_actual_mw"]
    common = L.index.intersection(temp.index)
    L, T = L.loc[common], temp.loc[common]
    xcorr = {k: float(L.corr(T.shift(k))) for k in range(-8, 9)}
    # Summer-only (JJA) correlation peaks where load follows temperature most tightly
    summer = common[common.month.isin([6, 7, 8])]
    xcorr_summer = {k: float(L.loc[summer].corr(T.shift(k).loc[summer])) for k in range(-8, 9)}
    return {
        "load_mean_by_index_hour_argmin": int(L.groupby(L.index.hour).mean().idxmin()),
        "load_mean_by_index_hour_argmax": int(L.groupby(L.index.hour).mean().idxmax()),
        "temp_mean_by_index_hour_argmax": int(T.groupby(T.index.hour).mean().idxmax()),
        "rt_price_mean_by_index_hour_argmax": int(price.groupby(price.index.hour).mean().idxmax()),
        "load_temp_xcorr_summer_by_shift_h": xcorr_summer,
        "load_temp_xcorr_summer_best_shift_h": int(max(xcorr_summer, key=xcorr_summer.get)),
        "load_temp_xcorr_all_best_shift_h": int(max(xcorr, key=xcorr.get)),
        "load_rows_per_utc_day_values": sorted(set(L.groupby(L.index.date).size().tolist())),
        "load_coverage": {"start": str(load.index.min()), "end": str(load.index.max())},
    }


def matrix_evidence(mat: pd.DataFrame, raw_v2: pd.DataFrame | None, sources: dict) -> dict:
    ev: dict = {}
    idx = mat.index
    full = pd.date_range(idx.min(), idx.max(), freq="h")
    ev["rows"] = int(len(mat))
    ev["start"], ev["end"] = str(idx.min()), str(idx.max())
    ev["duplicates"] = int(idx.duplicated().sum())
    ev["missing_hours_inside_range"] = int(len(full.difference(idx)))
    ev["gaps"] = _gaps(idx, "1h")
    if raw_v2 is not None:
        ev["row_loss"] = {
            "raw_v2_rows": int(len(raw_v2)),
            "engineered_rows": int(len(mat)),
            "dropped_by_engineer_dropna": int(len(raw_v2) - len(mat)),
            "raw_v2_start": str(raw_v2.index.min()),
        }

    enh = select_enhanced_features(mat)
    base = select_baseline_features(mat)
    ev["n_enhanced_features"] = len(enh)
    ev["n_baseline_features"] = len(base)
    ev["enhanced_features"] = enh

    # NaN coverage per feature (training fills NaN with ffill then 0)
    nan_share = mat[enh].isna().mean().sort_values(ascending=False)
    ev["feature_nan_share_top"] = {k: float(v) for k, v in nan_share.head(25).items()}
    load_cols = [c for c in enh if c.startswith("load_")]
    ev["load_feature_nan_share_before_2019"] = {
        c: float(mat.loc[mat.index < "2019-01-01", c].isna().mean()) for c in load_cols}

    # Same-hour checks against raw sources
    da, weather, load = sources["ercot_da_prices"], sources["weather"], sources["ercot_load"]
    shift = {}
    for c in [c for c in enh if c.startswith("da_")]:
        shift[c] = _lag_match(mat[c], da[c])
    for c in [c for c in enh if c.endswith(("_temperature_2m", "_wind_speed_10m", "_precipitation",
                                             "_relative_humidity_2m", "_cloud_cover", "_wind_gusts_10m"))][:6]:
        shift[c] = _lag_match(mat[c], weather[c])
    for c in load_cols:
        shift[c] = _lag_match(mat[c], load["load_actual_mw"])
    temp_cols = [c for c in weather.columns if c.endswith("_temperature_2m")]
    shift["temp_max_across_zones"] = _lag_match(mat["temp_max_across_zones"], weather[temp_cols].max(axis=1))
    rt_h = sources["ercot_rt_prices"]["HB_HUBAVG"].resample("1h").max()
    shift["hubavg_lag_1h"] = _lag_match(mat["hubavg_lag_1h"], rt_h)
    ev["same_hour_match_fractions"] = shift
    ev["same_hour_features_in_enhanced_set"] = sorted(
        c for c, d in shift.items() if c in enh and d.get("lag_0h", 0) > 0.99)

    # Daily/weekly sources: how many hours see a value not yet published
    gd = sources["gdelt"]
    day = idx.floor("D")
    g_same_day = _eq_frac(mat["gdelt_tone"], gd["gdelt_tone"].reindex(day).set_axis(idx))
    storage = sources["eia_storage"]
    s_used = storage["storage_bcf"].reindex(idx, method="ffill")
    week_end = pd.Series(storage.index, index=storage.index).reindex(idx, method="ffill")
    release = week_end + pd.Timedelta(days=6, hours=14, minutes=30)   # Thursday 10:30 ET ~ 14:30/15:30 UTC
    s_early = (pd.Series(idx, index=idx) < release) & (mat["storage_bcf"] == s_used)
    gas = sources["eia_gas"]["henry_hub_price"]
    gas_d = gas.resample("1D").ffill()
    g_used_same_date = _eq_frac(mat["henry_hub_price"], gas_d.reindex(day).set_axis(idx))
    ev["daily_weekly_timing"] = {
        "gdelt_tone_equals_same_utc_day_value": g_same_day,
        "gdelt_first_valid": str(mat["gdelt_tone"].first_valid_index()),
        "henry_hub_equals_same_trade_date_value": g_used_same_date,
        "storage_rows_using_unreleased_report_share": float(s_early.mean()),
        "storage_index_weekdays": sorted(set(storage.index.dayofweek.tolist())),
        "nrc_equals_same_date_value": _eq_frac(
            mat["reactors_offline_pct_sum"],
            sources["nrc_reactors"]["reactors_offline_pct_sum"].reindex(day).set_axis(idx)),
    }

    # Full-sample statistics that cross split boundaries
    hh = mat["henry_hub_price"]
    train_mask = mat.index < "2023-01-01"
    ev["stress_gas_spike"] = {
        "threshold_full_sample_p90": float(hh.quantile(0.90)),
        "threshold_train_only_p90": float(hh[train_mask].quantile(0.90)),
        "reproduces_full_sample_flag": bool(((hh > hh.quantile(0.9)).astype(int) == mat["stress_gas_spike"]).all()),
    }

    # log1p(clip(lower=0)) information loss
    neg = {}
    for c in ["hubavg_lag_1h", "hubavg_lag_24h", "da_HB_HUBAVG", "hubavg_rollmean_24h"]:
        neg[c] = float((mat[c] < 0).mean())
    ev["negative_value_share_erased_by_log1p_clip"] = neg

    # Calendar features are computed on the UTC clock
    by_hour = mat.groupby("hour")["HB_HUBAVG"].median()
    ev["calendar"] = {
        "hour_feature_equals_utc_hour": bool((mat["hour"] == mat.index.hour).all()),
        "is_peak_hours_utc_window": "14-20 UTC = 08-15 CST / 09-15 CDT",
        "median_price_argmax_utc_hour": int(by_hour.idxmax()),
    }

    # Target and label integrity
    y = mat["HB_HUBAVG"]
    fut_max = y.rolling(24).max().shift(-24)
    tail = mat["future_price_max_24h"].isna()
    ev["labels"] = {
        "future_spike_24h_rows_with_nan_window": int(tail.sum()),
        "those_rows_labelled_zero": int((mat.loc[tail, "future_spike_24h"] == 0).sum()),
        "label_window_matches_rolling_definition": _eq_frac(mat["future_price_max_24h"], fut_max),
        "label_window": "max(y[t+1..t+24]); y[t] excluded",
    }
    tr, va, te = (mat[mat.index < "2023-01-01"],
                  mat[(mat.index >= "2023-01-01") & (mat.index < "2024-01-01")],
                  mat[mat.index >= "2024-01-01"])
    cross = {}
    for name, part, boundary in (("train", tr, "2023-01-01"), ("val", va, "2024-01-01")):
        b = pd.Timestamp(boundary, tz="UTC")
        crossing = part[part.index + pd.Timedelta(hours=24) >= b]
        cross[name] = {"rows_with_label_window_crossing_boundary": int(len(crossing)),
                       "positives_among_them": int(crossing["future_spike_24h"].sum())}
    ev["boundary_crossing_labels"] = cross
    ev["split_rows"] = {"train": len(tr), "val": len(va), "test": len(te),
                        "test_end": str(te.index.max()),
                        "test_spike_hours": int((te["HB_HUBAVG"] > THRESHOLD).sum()),
                        "val_spike_hours": int((va["HB_HUBAVG"] > THRESHOLD).sum()),
                        "train_spike_hours": int((tr["HB_HUBAVG"] > THRESHOLD).sum())}
    ev["price_summary"] = {"median": float(y.median()), "mean": float(y.mean()),
                           "max": float(y.max()), "share_gt_200": float((y > THRESHOLD).mean())}
    return ev


def chronos_cache_evidence() -> dict:
    p = FEAT / "chronos_pred.parquet"
    if not p.exists():
        return {"present": False}
    c = pd.read_parquet(p)
    return {"present": True, "rows": int(len(c)), "start": str(c.index.min()), "end": str(c.index.max())}


def main() -> None:
    sources = {p.stem: _read(p.stem) for p in RAW.glob("*.parquet")}
    mat = pd.read_parquet(FEAT / "feature_matrix_engineered_v2.parquet")
    raw_v2_path = FEAT / "feature_matrix_raw_v2.parquet"
    raw_v2 = pd.read_parquet(raw_v2_path) if raw_v2_path.exists() else None

    evidence = {
        "sources": source_inventory(),
        "rt_prices": rt_evidence(sources["ercot_rt_prices"]),
        "diurnal_alignment": diurnal_evidence(sources["ercot_rt_prices"], sources["ercot_load"], sources["weather"]),
        "matrix": matrix_evidence(mat, raw_v2, sources),
        "chronos_cache": chronos_cache_evidence(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2, default=str))
    m = evidence["matrix"]
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  matrix rows={m['rows']} {m['start']} -> {m['end']} missing_hours={m['missing_hours_inside_range']}")
    print(f"  enhanced features={m['n_enhanced_features']} same-hour in set: {m['same_hour_features_in_enhanced_set']}")
    print(f"  daily/weekly timing: {m['daily_weekly_timing']}")
    print(f"  diurnal: {json.dumps({k: v for k, v in evidence['diurnal_alignment'].items() if 'xcorr_summer_by' not in k})}")
    print(f"  labels: {m['labels']}  boundary: {m['boundary_crossing_labels']}")
    print(f"  rt: {json.dumps({k: v for k, v in evidence['rt_prices'].items() if k != 'local_days_not_96_intervals'})}")


if __name__ == "__main__":
    main()
