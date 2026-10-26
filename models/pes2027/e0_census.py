"""E0-CENSUS: hours, intervals, missing-hour causes, DST, and every Chronos endpoint.

Read-only with respect to the original runs. Rebuilds the delay-1h frame with the
original builder and the saved delay-1h Chronos cache.

    python -m models.pes2027.e0_census
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.pes2027.common import (
    CRISIS_END, CRISIS_START, RUNS, STUDY_END, TARGET, TEST_START, TRAIN_START, VAL_START, VERIFIED,
    git_revision, hourly_max, index_hash, interval_counts, read_json, utc_now, write_csv, write_json,
)
from models.run_pit_verified import build_spec, columns_for, finite_slice, study_sources
from models.run_pit_ablation import load_sources
from src.features.pit_latency import context_hash
from src.features.pit_sets import feature_subsets

OUT = RUNS / "E0-CENSUS"
CONTEXT = 512


def windows():
    es_start = pd.Timestamp("2021-01-01", tz="UTC") - pd.Timedelta(days=90)
    return {
        "train_2019_2022": (TRAIN_START, VAL_START),
        "val_2023": (VAL_START, TEST_START),
        "test_2024": (TEST_START, STUDY_END),
        "crisis_fit_train_pre_2020_10_03": (None, es_start),
        "crisis_fit_stop_last90d_2020": (es_start, pd.Timestamp("2021-01-01", tz="UTC")),
        "crisis_test_2021": (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC")),
        "year_2019": (pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC")),
        "year_2020": (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC")),
        "year_2021": (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC")),
        "year_2022": (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")),
    }


def explain_missing(cal: pd.DatetimeIndex, rows: pd.DatetimeIndex, counts: pd.Series,
                    raw_empty: pd.DatetimeIndex, matrix_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Classify calendar hours that are not scored rows."""
    empty = set(raw_empty)
    out = []
    for t in cal.difference(rows):
        n = int(counts.get(t, 0))
        recent = [t - pd.Timedelta(hours=k) for k in range(1, 169)]
        gap168 = [s for s in recent if s in empty]
        lag720 = (t - pd.Timedelta(hours=720)) in empty
        if t in matrix_index:
            reason = "in matrix index but target not finite"
        elif n == 0:
            reason = "no raw 15-minute price in this hour"
        elif gap168 or lag720:
            reason = "raw price present; dropped by inherited v2 rule (raw-empty hour within previous 168 h or at t-720 h)"
        else:
            reason = "raw price present; dropped from v2 index for an unidentified reason"
        out.append({
            "hour_utc": t, "hour_central": t.tz_convert("US/Central"), "raw_intervals": n, "reason": reason,
            "first_raw_empty_within_168h": min(gap168) if gap168 else pd.NaT,
            "raw_empty_at_t_minus_720h": lag720,
        })
    return pd.DataFrame(out)


def main() -> None:
    sources = study_sources(load_sources())
    rt = sources["rt"][TARGET]
    counts = interval_counts(rt)
    hourly = hourly_max(rt)
    grid = pd.date_range(rt.index.min().floor("h"), rt.index.max().floor("h"), freq="h", tz="UTC")
    raw_empty = grid.difference(counts.index)
    cache = pd.read_parquet(VERIFIED / "chronos" / "delay_1h.parquet")
    spec = {"name": "delay_1h", "price": 1, "load": 1, "drop_load": False, "role": "census"}
    frame, _prov, _h = build_spec(sources, spec, compute_hash=False, chronos_cache=cache)
    matrix_index = sources["matrix"].index
    columns = columns_for(frame, False)
    subsets = feature_subsets(columns)

    minutes = sorted(set(pd.DatetimeIndex(rt.dropna().index).minute))
    raw = {
        "raw_rt_first": rt.index.min(), "raw_rt_last": rt.index.max(),
        "raw_rt_rows": int(len(rt)), "raw_rt_nonnull": int(rt.notna().sum()),
        "duplicate_stamps": int(rt.index.duplicated().sum()), "minute_values": minutes,
        "raw_rt_tz": str(rt.index.tz),
        "grid_hours_first_to_last": int(len(grid)), "grid_hours_with_no_price": int(len(raw_empty)),
        "raw_empty_hours_2016_2024": [str(x) for x in raw_empty],
        "matrix_rows": int(len(matrix_index)), "matrix_first": matrix_index.min(), "matrix_last": matrix_index.max(),
        "n_features_xgb_exog": len(subsets["xgb_exog"]), "n_features_xgb_both": len(subsets["xgb_both"]),
        "features_xgb_exog": subsets["xgb_exog"], "features_chronos": subsets["xgb_chronos"],
    }
    # Runs of raw-empty hours.
    runs, start, prev = [], None, None
    for t in raw_empty:
        if prev is None or t - prev != pd.Timedelta(hours=1):
            if start is not None:
                runs.append((start, prev))
            start = t
        prev = t
    if start is not None:
        runs.append((start, prev))
    raw["raw_empty_runs"] = [{"start": str(a), "end": str(b), "hours": int((b - a) / pd.Timedelta(hours=1)) + 1} for a, b in runs]

    split_rows, missing_frames = [], []
    for name, (lo, hi) in windows().items():
        lo_eff = lo if lo is not None else matrix_index.min()
        hi_cal = min(hi, STUDY_END)
        cal = pd.date_range(lo_eff, hi_cal - pd.Timedelta(hours=1), freq="h", tz="UTC")
        in_matrix = matrix_index[(matrix_index >= lo_eff) & (matrix_index < hi_cal)]
        rows = finite_slice(frame, frame.index[(frame.index >= lo_eff) & (frame.index < hi_cal)])
        c = counts.reindex(rows).fillna(0).astype(int)
        miss = explain_missing(cal, rows, counts, raw_empty, matrix_index)
        if len(miss):
            miss.insert(0, "window", name)
            missing_frames.append(miss)
        split_rows.append({
            "window": name, "start_utc": lo_eff, "end_utc_exclusive": hi_cal,
            "calendar_hours": int(len(cal)), "matrix_index_hours": int(len(in_matrix)),
            "scored_rows_finite_target": int(len(rows)), "not_scored": int(len(cal) - len(rows)),
            "rows_4_intervals": int((c == 4).sum()), "rows_3_intervals": int((c == 3).sum()),
            "rows_2_intervals": int((c == 2).sum()), "rows_1_interval": int((c == 1).sum()),
            "missing_no_raw_price": int((miss["reason"] == "no raw 15-minute price in this hour").sum()) if len(miss) else 0,
            "missing_v2_rule": int(miss["reason"].str.startswith("raw price present; dropped by inherited").sum()) if len(miss) else 0,
            "missing_unidentified": int(miss["reason"].str.contains("unidentified").sum()) if len(miss) else 0,
            "index_hash": index_hash(rows),
        })
    splits = pd.DataFrame(split_rows)
    missing = pd.concat(missing_frames, ignore_index=True) if missing_frames else pd.DataFrame()

    # Feature missingness by window (before imputation).
    miss_rows = []
    for name, (lo, hi) in windows().items():
        lo_eff = lo if lo is not None else matrix_index.min()
        rows = finite_slice(frame, frame.index[(frame.index >= lo_eff) & (frame.index < min(hi, STUDY_END))])
        part = frame.loc[rows, subsets["xgb_both"]]
        for col in part.columns:
            miss_rows.append({"window": name, "feature": col, "n": int(len(part)),
                              "n_missing": int(part[col].isna().sum()),
                              "share_missing": float(part[col].isna().mean())})
    feature_missing = pd.DataFrame(miss_rows)

    # DST: local calendar hours on transition days.
    dst = {}
    for day in ("2024-03-10", "2024-11-03", "2023-03-12", "2023-11-05", "2021-03-14", "2021-11-07"):
        lo = pd.Timestamp(day, tz="US/Central").tz_convert("UTC")
        hi = (pd.Timestamp(day, tz="US/Central") + pd.Timedelta(days=1)).tz_convert("UTC")
        part = frame.loc[(frame.index >= lo) & (frame.index < hi), "hour_local"]
        dst[day] = {"rows": int(len(part)), "hour_local_sequence": [int(x) for x in part.to_numpy()],
                    "raw_15min_stamps_local_day": int(((rt.index >= lo) & (rt.index < hi)).sum())}

    # Chronos: every row's endpoint and context hash; NaN hours per context.
    grid_all = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", tz="UTC")
    values = hourly.reindex(grid_all).to_numpy(np.float32)
    pos_of = pd.Series(np.arange(len(grid_all)), index=grid_all)
    cache_rows, nan_rows = [], []
    for name, delay, series in (("delay_1h", 1, rt), ("delay_2h", 2, rt), ("delay_3h", 3, rt), ("delay_6h", 6, rt),
                                ("hb_north_delay_1h", 1, sources["rt"]["HB_NORTH"])):
        c = pd.read_parquet(VERIFIED / "chronos" / f"{name}.parquet")
        meta = read_json(VERIFIED / "chronos" / f"{name}.json")
        latest = pd.DatetimeIndex(c["latest_input_timestamp"])
        if latest.tz is None:
            latest = latest.tz_localize("UTC")
        expected = c.index - pd.Timedelta(hours=1 + delay)
        if series is not rt:
            h2 = hourly_max(series)
            g2 = pd.date_range(h2.index.min(), h2.index.max(), freq="h", tz="UTC")
            vals, pos = h2.reindex(g2).to_numpy(np.float32), pd.Series(np.arange(len(g2)), index=g2)
        else:
            vals, pos = values, pos_of
        end = pos.reindex(latest).to_numpy()
        ok_hash = 0
        nan_counts = np.empty(len(c), dtype=int)
        for i, e in enumerate(end):
            window = vals[int(e) - CONTEXT + 1:int(e) + 1]
            nan_counts[i] = int(np.isnan(window).sum())
            ok_hash += context_hash(window) == c["context_hash"].iat[i]
        cache_rows.append({
            "cache": name, "delay_hours": delay, "rows": int(len(c)), "first": c.index.min(), "last": c.index.max(),
            "meta_horizon": meta["horizon"], "meta_step_index": meta["step_index"],
            "all_latest_equal_t_minus_1_plus_d": bool((latest == expected).all()),
            "n_latest_mismatch": int((latest != expected).sum()),
            "context_hash_matches": int(ok_hash), "context_hash_mismatches": int(len(c) - ok_hash),
            "max_nan_hours_in_context": int(nan_counts.max()),
            "rows_with_any_nan_in_context": int((nan_counts > 0).sum()),
        })
        if name == "delay_1h":
            nan_series = pd.Series(nan_counts, index=c.index)
            for wname, (lo, hi) in windows().items():
                lo_eff = lo if lo is not None else matrix_index.min()
                rows = finite_slice(frame, frame.index[(frame.index >= lo_eff) & (frame.index < min(hi, STUDY_END))])
                x = nan_series.reindex(rows)
                nan_rows.append({
                    "window": wname, "rows": int(len(rows)), "rows_without_cache": int(x.isna().sum()),
                    "rows_with_any_nan_hour": int((x > 0).sum()), "mean_nan_hours": float(x.mean()),
                    "max_nan_hours": float(x.max()), "p99_nan_hours": float(x.quantile(0.99)),
                    "rows_with_ge_24_nan_hours": int((x >= 24).sum()),
                })
    caches = pd.DataFrame(cache_rows)
    nan_ctx = pd.DataFrame(nan_rows)

    # The aligned frame uses the cache row at the same timestamp (chronos_delay_hours=0 for rebuilt caches).
    aligned = frame.loc[frame.index >= TRAIN_START, "chronos_pred"]
    direct = cache["chronos_pred"].reindex(aligned.index)
    align = {"rows": int(len(aligned)),
             "equal_or_both_nan": int(((aligned == direct) | (aligned.isna() & direct.isna())).sum())}

    # Boundary label availability under the 1-hour delay assumption.
    boundary = {
        "last_training_label_hour": "2022-12-31 23:00 UTC; published by assumption at 2023-01-01 01:00 UTC",
        "last_early_stopping_label_hour": "2023-12-31 23:00 UTC; published by assumption at 2024-01-01 01:00 UTC",
        "test_origins_before_last_stop_label_public": ["2024-01-01 00:00 UTC"],
        "note": ("A model frozen after 2023 early stopping could not be finalized under the stated delay before "
                 "2024-01-01 01:00 UTC. The first test origin (00:00 UTC) is affected. Its features are legal; "
                 "the model fit is not. Effect is limited to one origin."),
        "crisis_protocol": "Early stopping ends 2020-12-31 23:00 UTC; the first 2021 origin has the same one-origin issue.",
    }

    crisis_hours = pd.date_range(CRISIS_START, CRISIS_END - pd.Timedelta(hours=1), freq="h", tz="UTC")
    crisis_cov = {"calendar_hours": int(len(crisis_hours)),
                  "in_scored_2021_rows": int(crisis_hours.isin(finite_slice(frame, frame.index)).sum())}

    write_json(OUT / "census.json", {
        "experiment": "E0-CENSUS", "created_utc": utc_now(), "code_revision": git_revision(),
        "raw": raw, "dst": dst, "chronos_alignment_in_frame": align, "boundary": boundary,
        "crisis_window_coverage": crisis_cov,
    })
    write_csv(OUT / "split_census.csv", splits)
    write_csv(OUT / "missing_hours.csv", missing)
    write_csv(OUT / "feature_missingness.csv", feature_missing)
    write_csv(OUT / "chronos_cache_rows.csv", caches)
    write_csv(OUT / "chronos_context_nan_by_window.csv", nan_ctx)
    print(splits.to_string())
    print(caches.to_string())
    print(nan_ctx.to_string())
    if len(missing):
        print(missing.groupby(["window", "reason"]).size().to_string())


if __name__ == "__main__":
    main()
