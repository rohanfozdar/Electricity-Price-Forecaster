"""Latency-filtered conservative features and the availability manifest.

The primary delay is the shortest pre-specified whole-hour assumption that
excludes the 15-minute interval ending at the issue time. It is not chosen
from forecast error. A delay shifts when an archived price may be used. It
does not recover the first number ERCOT posted.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.features.pit_sets import (
    CHRONOS_STATS,
    calendar_central,
    day_ahead_features,
    fundamental_frames,
    lagged_load,
    missing_indicators,
    previous_hour_intervals,
    price_history,
)

MANIFEST_COLUMNS = (
    "feature_name",
    "raw_source",
    "code_path",
    "observation_period",
    "timestamp_convention",
    "timezone",
    "assumed_publication_delay",
    "verified_publication_delay",
    "historical_evidence",
    "revision_or_backfill_risk",
    "available_at_issuance",
    "required_correction",
    "verification_status",
)

STATUS_VERIFIED = "verified historical availability"
STATUS_DOCUMENTED = "documented rule with uncertain historical applicability"
STATUS_ASSUMED = "assumed"
STATUS_UNVERIFIED = "unverified"
STATUS_LEAK = "known leak"

# Shortest listed assumption that excludes the interval ending at issue time.
# Not selected by comparing mean absolute errors.
PRIMARY_PRICE_DELAY_HOURS = 1
PRIMARY_LOAD_DELAY_HOURS = 1
LATENCY_HOURS = (1, 2, 3, 6)

PRICE_EVIDENCE = (
    "On-disk prices come from gridstatus Ercot.get_rtm_spp, report type 13061, "
    "EMIL NP6-785-ER, a weekly historical hub and load-zone archive. The live "
    "15-minute product is NP6-905-CD, report 12301. Neither the archive nor the "
    "EMIL page gives a first-publication timestamp for 2019-2024."
)
LOAD_EVIDENCE = (
    "pipelines/ercot_load.py reads report type 13101, EMIL NP6-345-CD, actual "
    "hourly load by weather zone, generated daily. A current public display "
    "shows same-day hours, but this file has no historical posting timestamp."
)
DAM_EVIDENCE = (
    "Nodal protocol language, a 2018 NPRR903 discussion, and an ERCOT day-ahead "
    "operations timeline put public posting at no later than 13:30 Central. "
    "The code waits until 14:00 Central. Later price corrections are not identified "
    "in the downloaded file."
)
STORAGE_EVIDENCE = (
    "EIA's current schedule releases the Weekly Natural Gas Storage Report at "
    "10:30 a.m. Eastern on Thursday, with holiday exceptions. The code waits "
    "until the following Friday 18:00 UTC. The 2019-2024 holiday calendar was "
    "not applied row by row."
)


def interval_end(start: pd.Timestamp) -> pd.Timestamp:
    return start + pd.Timedelta(minutes=15)


def interval_available(start: pd.Timestamp, issue: pd.Timestamp, delay: pd.Timedelta) -> bool:
    """True when the interval has ended and the stated delay has also elapsed."""
    return interval_end(start) + delay <= issue


def latest_complete_hour(issue: pd.Timestamp, delay_hours: int) -> pd.Timestamp:
    """Start of the latest clock hour whose last interval is legal at `issue`."""
    if delay_hours < 0:
        raise ValueError("delay_hours must be non-negative")
    return issue - pd.Timedelta(hours=1 + delay_hours)


def hourly_max_from_intervals(rt: pd.Series) -> pd.Series:
    rt = rt.sort_index().astype(float)
    hour = pd.DatetimeIndex(rt.index).floor("h")
    return rt.groupby(hour).max()


def hourly_mean_from_intervals(rt: pd.Series) -> pd.Series:
    rt = rt.sort_index().astype(float)
    hour = pd.DatetimeIndex(rt.index).floor("h")
    return rt.groupby(hour).mean()


def align_chronos(cache: pd.DataFrame, index: pd.DatetimeIndex, delay_hours: int) -> tuple[pd.DataFrame, pd.Series]:
    """Select the frozen one-step row whose context already ended at the legal hour.

    The cache row stored at hour s was built from prices through s-1. At issue t,
    with a whole-hour delay D, the last legal hour is t-(1+D), so the matching
    frozen forecast is the cache row at t-D. This is not a new random sample.
    """
    if delay_hours < 0:
        raise ValueError("delay_hours must be non-negative")
    src = pd.DatetimeIndex(index) - pd.Timedelta(hours=delay_hours)
    aligned = cache.reindex(src)
    aligned.index = pd.DatetimeIndex(index)
    latest = src - pd.Timedelta(hours=1)
    return aligned, pd.Series(latest, index=index, name="latest_input_timestamp")


def context_hash(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def context_hashes(hourly: pd.Series, latest: pd.Series, context: int = 512) -> pd.Series:
    """Hash of the context-length clock window ending at each latest input hour."""
    hourly = hourly.sort_index().astype(float)
    grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", tz=hourly.index.tz)
    values = hourly.reindex(grid).to_numpy(dtype=np.float64)
    pos = grid.get_indexer(pd.DatetimeIndex(latest))
    out = []
    for p in pos:
        if p < context - 1:
            out.append(pd.NA)
            continue
        out.append(context_hash(values[p - context + 1:p + 1]))
    return pd.Series(out, index=latest.index, name="context_hash")


def _row(name, source, code, observation, convention, timezone, assumed, verified,
         evidence, revision, available, correction, status) -> dict:
    return {
        "feature_name": name,
        "raw_source": source,
        "code_path": code,
        "observation_period": observation,
        "timestamp_convention": convention,
        "timezone": timezone,
        "assumed_publication_delay": assumed,
        "verified_publication_delay": verified,
        "historical_evidence": evidence,
        "revision_or_backfill_risk": revision,
        "available_at_issuance": available,
        "required_correction": correction,
        "verification_status": status,
    }


def availability_manifest(price_delay_hours: int = PRIMARY_PRICE_DELAY_HOURS,
                          load_delay_hours: int = PRIMARY_LOAD_DELAY_HOURS,
                          drop_load: bool = False) -> pd.DataFrame:
    """One row per conservative column and per Chronos context input.

    The status describes the corrected feature, not the withdrawn zero-delay claim.
    """
    price_assumed = (
        f"{price_delay_hours} hour(s) after the 15-minute interval ends; "
        "only a fully complete clock hour is used"
    )
    price_available = (
        f"yes under a {price_delay_hours}-hour delay applied to archived values; "
        "the archive may not be the first posted number"
    )
    load_assumed = f"{load_delay_hours} hour(s) after the load hour ends"
    rows = []
    for lag in (1, 12, 24, 48, 168, 720):
        rows.append(_row(
            f"hubavg_lag_{lag}h",
            "ERCOT RT SPP HB_HUBAVG, weekly historical archive NP6-785-ER",
            "src/features/pit_sets.py:price_history",
            f"clock hour t-({lag}+{price_delay_hours})h",
            "interval start; hour is the floor of the 15-minute start",
            "UTC",
            price_assumed,
            "none measured",
            PRICE_EVIDENCE,
            "high: weekly archive can be a revised settlement price",
            price_available,
            f"shift the hourly maximum by {lag}+{price_delay_hours} hours",
            STATUS_ASSUMED,
        ))
    for window in (24, 168):
        for stat in ("rollmean", "rollstd"):
            rows.append(_row(
                f"hubavg_{stat}_{window}h",
                "derived from HB_HUBAVG hourly maxima",
                "src/features/pit_sets.py:price_history",
                f"trailing {window} complete hours ending at t-(1+{price_delay_hours})h",
                "derived; right-closed rolling window then shifted",
                "UTC",
                price_assumed,
                "none measured",
                PRICE_EVIDENCE,
                "high, inherited from the archived prices inside the window",
                price_available,
                f"end the window at the last fully visible hour, {price_delay_hours} hour(s) later than hour t-1",
                STATUS_ASSUMED,
            ))
    for name, observation in (
        ("rt15_last", "last 15-minute interval of the last fully visible hour"),
        ("rt15_max", "maximum of the four intervals in that hour"),
        ("rt15_min", "minimum of the four intervals in that hour"),
        ("rt15_mean", "mean of the four intervals in that hour"),
    ):
        rows.append(_row(
            name,
            "derived from HB_HUBAVG 15-minute intervals",
            "src/features/pit_sets.py:previous_hour_intervals",
            observation,
            "15-minute interval start",
            "UTC",
            price_assumed,
            "none measured",
            PRICE_EVIDENCE,
            "high, inherited from the archived intervals",
            price_available,
            "do not use the interval that ends at the issue time when delay is positive",
            STATUS_ASSUMED,
        ))
    for hub in ("HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HUBAVG"):
        rows.append(_row(
            f"da_{hub}",
            "ERCOT day-ahead SPP",
            "src/features/pit_sets.py:day_ahead_features",
            "operating hour t",
            "hour start; available after the previous-day publication gate",
            "US/Central publication clock, UTC storage",
            "code gate 14:00 US/Central on the previous local day",
            "nominal deadline 13:30 US/Central; exact historical posting minute not re-measured",
            DAM_EVIDENCE,
            "medium: protocol text allows later correction; the file does not mark corrected hours",
            "yes when the 14:00 Central gate is at or before the issue time",
            "keep the publication gate; do not treat the archived number as proven first print",
            STATUS_DOCUMENTED,
        ))
    for name, observation in (
        ("da_shape_ramp", "current gated hub price minus the previous hour"),
        ("da_shape_day_max", "maximum gated hub price on the same Central operating day"),
        ("da_shape_day_min", "minimum gated hub price on the same Central operating day"),
    ):
        rows.append(_row(
            name,
            "derived from publication-gated day-ahead HB_HUBAVG",
            "src/features/pit_sets.py:day_ahead_features",
            observation,
            "derived from gated hourly day-ahead prices",
            "US/Central day",
            "same 14:00 US/Central gate as the day-ahead price",
            "same nominal 13:30 deadline",
            DAM_EVIDENCE,
            "medium, inherited from the day-ahead file",
            "yes for the already published operating day",
            "none beyond the day-ahead gate",
            STATUS_DOCUMENTED,
        ))
    rows.append(_row(
        "da_shape_spread_24h",
        "derived from lagged real-time HB_HUBAVG minus gated day-ahead HB_HUBAVG",
        "src/features/pit_sets.py:day_ahead_features",
        f"24-hour mean ending at t-(1+{price_delay_hours})h",
        "derived",
        "UTC",
        price_assumed + "; day-ahead side uses the 14:00 Central gate",
        "real-time side none measured; day-ahead side nominal 13:30 Central",
        PRICE_EVIDENCE + " " + DAM_EVIDENCE,
        "high on the real-time side",
        price_available,
        "shift the spread by the same price delay as the real-time lags",
        STATUS_ASSUMED,
    ))
    if not drop_load:
        rows.append(_row(
            "load_lag_1h",
            "ERCOT actual system load, report 13101 / NP6-345-CD",
            "src/features/pit_sets.py:lagged_load",
            f"clock hour t-(1+{load_delay_hours})h",
            "hour start",
            "UTC",
            load_assumed,
            "none measured",
            LOAD_EVIDENCE,
            "medium: daily archive may be revised; no revision flag is stored",
            f"yes only under the {load_delay_hours}-hour assumption",
            f"lag realized load by {1 + load_delay_hours} hours; do not use same-hour load",
            STATUS_ASSUMED,
        ))
        rows.append(_row(
            "load_rollmean_24h",
            "derived from lagged ERCOT actual load",
            "src/features/pit_sets.py:lagged_load",
            f"trailing 24 hours ending at t-(1+{load_delay_hours})h",
            "derived",
            "UTC",
            load_assumed,
            "none measured",
            LOAD_EVIDENCE,
            "medium, inherited from the load archive",
            f"yes only under the {load_delay_hours}-hour assumption",
            "end the load window at the last hour allowed by the load delay",
            STATUS_ASSUMED,
        ))
    for name, observation in (
        ("storage_bcf", "latest week whose release offset is at or before issue time"),
        ("storage_wow_change", "week-over-week change on that same released row"),
        ("storage_bcf_lag_1w", "the prior released storage row"),
    ):
        rows.append(_row(
            name,
            "EIA weekly natural gas storage",
            "src/features/pit_sets.py:fundamental_frames",
            observation,
            "week-ending Friday plus a fixed release offset",
            "UTC implementation of an Eastern release",
            "following Friday 18:00 UTC after the week-ending Friday",
            "standard Thursday 10:30 a.m. Eastern; holiday exceptions exist",
            STORAGE_EVIDENCE,
            "low for the timing buffer, because the offset is later than the standard release; holiday weeks in 2019-2024 were not checked one by one",
            "yes under the Friday 18:00 UTC buffer",
            "keep the buffer; do not join on the week-ending Friday",
            STATUS_DOCUMENTED,
        ))
    for name in (
        "hour_local", "day_of_week_local", "month_local", "is_weekend_local", "is_peak_local",
    ):
        rows.append(_row(
            name,
            "issue timestamp",
            "src/features/pit_sets.py:calendar_central",
            "the forecast issue hour",
            "US/Central civil time; peak is weekday 16:00-22:00",
            "US/Central",
            "none",
            "not applicable; the clock is known",
            "Calendar fields are functions of the issue timestamp.",
            "none",
            "yes",
            "none",
            STATUS_VERIFIED,
        ))
    for name in CHRONOS_STATS:
        rows.append(_row(
            name,
            "frozen amazon/chronos-t5-small sample, aligned to the legal price cutoff",
            "src/features/pit_latency.py:align_chronos",
            f"forecast of hour t; context ends at t-(1+{price_delay_hours})h",
            "new T5-small sample; 512 clock hours ending at the last legal hour; the selected step is hour t",
            "UTC",
            price_assumed,
            "none measured for the prices inside the context",
            PRICE_EVIDENCE + " The row is a new 20-path sample. Context prices are the latency-filtered hourly maxima.",
            "high: context prices are archived settlement values, and the sample is one 20-path draw",
            price_available,
            "rebuild the forecast of hour t from a context that ends at the last legal hour",
            STATUS_ASSUMED,
        ))
    if not drop_load:
        rows.append(_row(
            "miss_load",
            "missingness of load_lag_1h after the load delay",
            "src/features/pit_sets.py:missing_indicators",
            "issue time",
            "indicator",
            "UTC",
            load_assumed,
            "none measured",
            LOAD_EVIDENCE,
            "none beyond the parent column",
            "yes; it records that the delayed load is missing",
            "compute after the load delay, from training-safe imputation",
            STATUS_ASSUMED,
        ))
    rows.append(_row(
        "miss_storage",
        "missingness of storage_bcf after the weekly gate",
        "src/features/pit_sets.py:missing_indicators",
        "issue time",
        "indicator",
        "UTC",
        "same Friday 18:00 UTC buffer",
        "same Thursday 10:30 Eastern rule",
        STORAGE_EVIDENCE,
        "none beyond the parent column",
        "yes",
        "compute after the storage gate",
        STATUS_DOCUMENTED,
    ))
    rows.append(_row(
        "miss_da",
        "missingness of da_HB_HUBAVG after the publication gate",
        "src/features/pit_sets.py:missing_indicators",
        "issue time",
        "indicator",
        "UTC",
        "same 14:00 Central gate",
        "same nominal 13:30 deadline",
        DAM_EVIDENCE,
        "none beyond the parent column",
        "yes",
        "compute after the day-ahead gate",
        STATUS_DOCUMENTED,
    ))
    rows.append(_row(
        "miss_chronos",
        "missingness of the aligned Chronos row",
        "src/features/pit_sets.py:missing_indicators",
        "issue time",
        "indicator",
        "UTC",
        price_assumed,
        "none measured",
        PRICE_EVIDENCE,
        "none beyond the parent column",
        "yes",
        "compute after Chronos alignment",
        STATUS_ASSUMED,
    ))
    rows.append(_row(
        "chronos_context_hourly_maximum",
        "HB_HUBAVG hourly maximum inside the Chronos context",
        "models/chronos_hourly_features.py:context_bounds",
        f"512 clock hours ending at t-(1+{price_delay_hours})h",
        "hour start; target hour excluded",
        "UTC",
        price_assumed,
        "none measured",
        PRICE_EVIDENCE,
        "high: context uses the archived hourly maximum",
        price_available,
        "the context end must satisfy end + 1 hour + delay <= issue time",
        STATUS_ASSUMED,
    ))
    rows.append(_row(
        "chronos_context_latest_input_timestamp",
        "provenance column stored with the aligned cache",
        "src/features/pit_latency.py:align_chronos",
        f"t-(1+{price_delay_hours})h",
        "hour start of the last context price",
        "UTC",
        price_assumed,
        "none measured",
        "Computed from the issue time and the stated delay. It is not an ERCOT publication timestamp.",
        "none",
        "yes; it is a function of the issue time and the delay",
        "store it on every aligned cache row together with a hash of the context values",
        STATUS_ASSUMED,
    ))
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if list(frame.columns) != list(MANIFEST_COLUMNS):
        raise AssertionError("manifest column contract changed")
    return frame


def build_delayed_frame(index: pd.DatetimeIndex, rt_price: pd.Series, da: pd.DataFrame,
                        load: pd.Series, storage: pd.DataFrame, gas: pd.DataFrame,
                        nrc: pd.DataFrame, chronos: pd.DataFrame,
                        price_delay_hours: int, load_delay_hours: int,
                        drop_load: bool = False, target: str = "max",
                        compute_hash: bool = True,
                        chronos_delay_hours: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conservative matrix at one price delay and one load delay.

    The target is the hourly maximum by default. `target='mean'` changes only
    the outcome column. Features stay on the latency-filtered hourly maximum.
    """
    if index.max() >= pd.Timestamp("2025-01-01", tz="UTC"):
        raise AssertionError("study index enters 2025")
    price_delay = pd.Timedelta(hours=price_delay_hours)
    load_delay = pd.Timedelta(hours=load_delay_hours)
    hourly = hourly_max_from_intervals(rt_price)
    if target == "max":
        y = hourly.reindex(index)
    elif target == "mean":
        y = hourly_mean_from_intervals(rt_price).reindex(index)
    else:
        raise ValueError(target)
    parts = [
        price_history(hourly, index, delay=price_delay),
        previous_hour_intervals(rt_price, index, delay=price_delay),
        day_ahead_features(hourly.reindex(index), da.reindex(index), price_delay=price_delay),
    ]
    if not drop_load:
        parts.append(lagged_load(load, index, delay=load_delay))
    parts.append(fundamental_frames(gas, storage, nrc, index)["storage"])
    parts.append(calendar_central(index))
    if chronos_delay_hours is None:
        chronos_delay_hours = price_delay_hours
    aligned, latest = align_chronos(chronos, index, chronos_delay_hours)
    if chronos_delay_hours == 0 and "latest_input_timestamp" in chronos.columns:
        latest = chronos["latest_input_timestamp"].reindex(index)
    from src.features.pit_sets import chronos_block
    parts.append(chronos_block(aligned, index))
    frame = pd.concat(parts, axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    families = {"storage": "storage_bcf", "da": "da_HB_HUBAVG", "chronos": "chronos_pred"}
    if not drop_load:
        families = {"load": "load_lag_1h", **families}
    frame = pd.concat([frame, missing_indicators(frame, families)], axis=1)
    frame["HB_HUBAVG"] = y.to_numpy()
    provenance = pd.DataFrame(index=index)
    provenance["latest_input_timestamp"] = latest
    if compute_hash:
        provenance["context_hash"] = context_hashes(hourly, latest).reindex(index).to_numpy()
    else:
        provenance["context_hash"] = pd.NA
    if "load_actual_mw" in frame.columns:
        raise AssertionError("same-hour load entered the delayed matrix")
    return frame, provenance
