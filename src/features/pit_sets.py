"""Point-in-time features for a one-hour-ahead forecast of the hourly price maximum.

Issue time is the timestamp of the row. The row predicts the hour that starts then.
A source value may enter that row only when its availability time is at or before
the issue time. Rules come from configs/feature_availability.yaml.

`reported` is not built here. That set is the existing honest-tree matrix and is
labeled as a retrospective information set with known gaps. This module builds
`assumed_realtime` and the column filter for `conservative`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_yaml
from src.data.asof import available_at

UTC = "UTC"
CENTRAL = "US/Central"
RULES = load_yaml("configs/feature_availability.yaml")["sources"]

WEATHER_STRESS = (
    "stress_cold_snap",
    "stress_freeze",
    "stress_heat_wave",
    "stress_extreme_heat",
    "stress_low_wind",
)
# Thresholds match features/engineer_v2.py, not the prose in the PDF.
# cold < 20, freeze < 32, heat > 90, extreme heat > 105, low wind < 5.
CHRONOS_STATS = (
    "chronos_pred",
    "chronos_q10",
    "chronos_q90",
    "chronos_mean",
    "chronos_p_gt200",
    "chronos_spread",
)
ASINH_EXACT = set(CHRONOS_STATS) - {"chronos_p_gt200"}


def clock_shift(series: pd.Series, hours: int, index: pd.DatetimeIndex | None = None) -> pd.Series:
    """Value at t becomes the value that was observed at t-hours. Not a row shift.

    The result is indexed by `index` when that is given, so a value may land on
    an hour that was missing from the source series.
    """
    moved = series.copy()
    moved.index = pd.DatetimeIndex(series.index) + pd.Timedelta(hours=hours)
    return moved.reindex(series.index if index is None else index)


def _whole_delay_hours(delay: pd.Timedelta | None) -> int:
    """Extra whole hours after an interval or hour ends.

    Zero keeps the previous behavior: hour t-1 is treated as known at t.
    A positive delay is an assumption unless a source register verifies it.
    Partial hours are not used, so the delay has to be a whole number of hours.
    """
    if delay is None or delay == pd.Timedelta(0):
        return 0
    if not isinstance(delay, pd.Timedelta):
        delay = pd.Timedelta(delay)
    if delay < pd.Timedelta(0):
        raise ValueError("publication delay must be non-negative")
    if delay % pd.Timedelta(hours=1) != pd.Timedelta(0):
        raise ValueError("publication delay must be a whole number of hours")
    return int(delay / pd.Timedelta(hours=1))


def price_history(y: pd.Series, index: pd.DatetimeIndex,
                  delay: pd.Timedelta | None = None) -> pd.DataFrame:
    """Lags and trailing windows of the hourly price.

    Hour t is always excluded. `delay` adds whole hours after the hour ends.
    With a one-hour delay, the latest legal hourly price at issue t is hour t-2,
    because the last 15-minute interval of hour t-1 ends at t.
    """
    extra = _whole_delay_hours(delay)
    y = y.sort_index().astype(float)
    out = pd.DataFrame(index=index)
    for lag in (1, 12, 24, 48, 168, 720):
        out[f"hubavg_lag_{lag}h"] = clock_shift(y, lag + extra, index).to_numpy()
    for window in (24, 168):
        rolled_mean = y.rolling(f"{window}h", min_periods=window).mean()
        rolled_std = y.rolling(f"{window}h", min_periods=window).std()
        out[f"hubavg_rollmean_{window}h"] = clock_shift(rolled_mean, 1 + extra, index).to_numpy()
        out[f"hubavg_rollstd_{window}h"] = clock_shift(rolled_std, 1 + extra, index).to_numpy()
    return out


def previous_hour_intervals(rt: pd.Series, index: pd.DatetimeIndex,
                            delay: pd.Timedelta | None = None) -> pd.DataFrame:
    """Last, max, min, and mean of the four 15-minute prices in the last fully visible hour.

    With delay 0 that hour is [t-1h, t). A positive whole-hour delay moves the
    window earlier. The interval that ends at t is not in the window when delay > 0.
    """
    extra = _whole_delay_hours(delay)
    prev = index - pd.Timedelta(hours=1 + extra)
    offsets = [pd.Timedelta(minutes=m) for m in (0, 15, 30, 45)]
    arr = np.column_stack([
        rt.reindex(prev + off).to_numpy(dtype=float) for off in offsets
    ])
    with np.errstate(all="ignore"):
        rmax = np.nanmax(arr, axis=1)
        rmin = np.nanmin(arr, axis=1)
        rmean = np.nanmean(arr, axis=1)
    empty = np.isnan(arr).all(axis=1)
    rmax[empty] = np.nan
    rmin[empty] = np.nan
    rmean[empty] = np.nan
    last_stamp = prev + pd.Timedelta(minutes=45)
    return pd.DataFrame({
        "rt15_last": rt.reindex(last_stamp).to_numpy(dtype=float),
        "rt15_max": rmax,
        "rt15_min": rmin,
        "rt15_mean": rmean,
    }, index=index)


def gate_published(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Keep a row only when available_at(timestamp) <= timestamp.

    For a forecast issued at the row timestamp, that is the publication gate.
    """
    avail = available_at(pd.DatetimeIndex(frame.index), rule)
    visible = avail <= frame.index
    out = frame.copy()
    for col in out.columns:
        out[col] = out[col].where(visible).to_numpy()
    return out


def day_ahead_features(y: pd.Series, da: pd.DataFrame, hub_col: str = "da_HB_HUBAVG",
                       price_delay: pd.Timedelta | None = None) -> pd.DataFrame:
    """Publication-gated day-ahead prices, ramp, operating-day range, and lagged RT-DA spread.

    The day-ahead gate is unchanged. Only the real-time side of the spread is
    shifted by `price_delay`, because that side is a settlement price.
    """
    extra = _whole_delay_hours(price_delay)
    rule = RULES["ercot_da_prices"]
    gated = gate_published(da.sort_index(), rule)
    hub = gated[hub_col]
    day = hub.index.tz_convert(CENTRAL).floor("D")
    spread = (y.reindex(hub.index) - hub).rolling("24h", min_periods=12).mean()
    out = pd.DataFrame(index=da.index)
    for col in gated.columns:
        out[col] = gated[col]
    out["da_shape_ramp"] = (hub - clock_shift(hub, 1)).to_numpy()
    out["da_shape_day_max"] = hub.groupby(day).transform("max").to_numpy()
    out["da_shape_day_min"] = hub.groupby(day).transform("min").to_numpy()
    out["da_shape_spread_24h"] = clock_shift(spread, 1 + extra).to_numpy()
    return out


def lagged_load(load: pd.Series, index: pd.DatetimeIndex,
                delay: pd.Timedelta | None = None) -> pd.DataFrame:
    """Realized load lagged past the target hour. Same-hour load is not included.

    `delay` is an extra whole-hour lag after hour end. The posting time of the
    archive file was not measured, so any specific delay stays an assumption.
    """
    extra = _whole_delay_hours(delay)
    load = load.sort_index().astype(float)
    lag = clock_shift(load, 1 + extra, index)
    trail = clock_shift(load.rolling("24h", min_periods=12).mean(), 1 + extra, index)
    return pd.DataFrame({
        "load_lag_1h": lag.reindex(index).to_numpy(),
        "load_rollmean_24h": trail.reindex(index).to_numpy(),
    }, index=index)


def _zone_weather(weather: pd.DataFrame) -> pd.DataFrame:
    temp_cols = [c for c in weather.columns if c.endswith("temperature_2m")]
    wind_cols = [c for c in weather.columns if "wind_speed_10m" in c]
    out = weather.copy()
    out["temp_min_across_zones"] = out[temp_cols].min(axis=1)
    out["temp_max_across_zones"] = out[temp_cols].max(axis=1)
    out["temp_range_across_zones"] = out["temp_max_across_zones"] - out["temp_min_across_zones"]
    out["wind_mean_across_zones"] = out[wind_cols].mean(axis=1)
    out["wind_min_across_zones"] = out[wind_cols].min(axis=1)
    known_t = out["temp_min_across_zones"].notna()
    known_w = out["wind_min_across_zones"].notna()
    out["stress_cold_snap"] = (out["temp_min_across_zones"] < 20).astype(float).where(known_t)
    out["stress_freeze"] = (out["temp_min_across_zones"] < 32).astype(float).where(known_t)
    out["stress_heat_wave"] = (out["temp_max_across_zones"] > 90).astype(float).where(out["temp_max_across_zones"].notna())
    out["stress_extreme_heat"] = (out["temp_max_across_zones"] > 105).astype(float).where(out["temp_max_across_zones"].notna())
    out["stress_low_wind"] = (out["wind_min_across_zones"] < 5).astype(float).where(known_w)
    return out


def lagged_weather(weather: pd.DataFrame, index: pd.DatetimeIndex, hours: int = 1) -> pd.DataFrame:
    """Reanalysis weather shifted by the clock, then aggregated.

    The one-hour shift removes the same-hour observation. It does not make archive
    reanalysis a published forecast. Callers must keep that label.
    """
    shifted = pd.DataFrame(index=index)
    for col in weather.columns:
        shifted[col] = clock_shift(weather[col].astype(float), hours, index).to_numpy()
    return _zone_weather(shifted)


def asof_frame(values: pd.DataFrame, index: pd.DatetimeIndex, rule: dict) -> pd.DataFrame:
    """Latest source row with available_at <= issue time. Age columns are dropped."""
    frame = values.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    avail = available_at(pd.DatetimeIndex(frame.index), rule)
    order = np.argsort(avail.values, kind="stable")
    avail_sorted = avail.values[order]
    pos = np.searchsorted(avail_sorted, pd.DatetimeIndex(index).values, side="right") - 1
    valid = pos >= 0
    src = np.where(valid, order[np.clip(pos, 0, None)], 0)
    out = pd.DataFrame(index=index)
    for col in frame.columns:
        raw = frame[col].to_numpy(dtype=float)[src]
        out[col] = np.where(valid, raw, np.nan)
    return out


def fundamental_frames(gas: pd.DataFrame, storage: pd.DataFrame, nrc: pd.DataFrame,
                       index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Gas and NRC use assumed release rules. Storage uses the documented weekly offset."""
    g = gas.sort_index().astype(float)
    g = g[~g.index.duplicated(keep="last")]
    g["henry_hub_lag_1d"] = g["henry_hub_price"].shift(1)
    g["henry_hub_change_1d"] = g["henry_hub_price"] - g["henry_hub_price"].shift(1)
    g["henry_hub_change_7d"] = g["henry_hub_price"] - g["henry_hub_price"].shift(7)

    s = storage.sort_index().astype(float)
    s = s[~s.index.duplicated(keep="last")]
    s["storage_bcf_lag_1w"] = s["storage_bcf"].shift(1)

    n = nrc.sort_index().astype(float)
    n = n[~n.index.duplicated(keep="last")]
    n["reactors_offline_change_1d"] = n["reactors_offline_pct_sum"] - n["reactors_offline_pct_sum"].shift(1)
    return {
        "gas": asof_frame(g, index, RULES["eia_gas"]),
        "storage": asof_frame(s, index, RULES["eia_storage"]),
        "nrc": asof_frame(n, index, RULES["nrc_reactors"]),
    }


def calendar_central(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Clock features on US/Central. Peak is weekday 16:00 through 22:00, as the PDF stated.

    The v2 matrix instead marks UTC hours 14 through 20. This function does not.
    """
    local = pd.DatetimeIndex(index).tz_convert(CENTRAL)
    hour = local.hour
    dow = local.dayofweek
    return pd.DataFrame({
        "hour_local": hour.astype(float),
        "day_of_week_local": dow.astype(float),
        "month_local": local.month.astype(float),
        "is_weekend_local": (dow >= 5).astype(float),
        "is_peak_local": ((dow < 5) & (hour >= 16) & (hour <= 22)).astype(float),
    }, index=index)


def chronos_block(cache: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align a precomputed one-step cache. Does not recompute forecasts or shift them."""
    cols = [c for c in CHRONOS_STATS if c != "chronos_spread"]
    block = cache[cols].reindex(index)
    block = block.copy()
    block["chronos_spread"] = block["chronos_q90"] - block["chronos_q10"]
    return block


def gfs_target_hour(gfs: pd.DataFrame, index: pd.DatetimeIndex, lead_day: int = 1) -> pd.DataFrame:
    """Temperature forecast for hour t from the previous-day lead stored at t.

    Open-Meteo previous-day lead N is the forecast issued about 24N hours before
    the valid hour. Lead 1 was therefore issued before hour t. Coverage starts
    2021-03-23. These are not ERCOT wind or solar generation forecasts.
    """
    out = pd.DataFrame(index=index)
    for col in gfs.columns:
        zone, lead = col.split("|")
        if lead != f"d{lead_day}":
            continue
        out[f"gfs_{zone}_t_d{lead_day}"] = gfs[col].reindex(index).to_numpy(dtype=float)
    return out


def missing_indicators(frame: pd.DataFrame, families: dict[str, str]) -> pd.DataFrame:
    """One indicator per family, 1 when the named column is missing. Known at issue time."""
    out = pd.DataFrame(index=frame.index)
    for name, column in families.items():
        out[f"miss_{name}"] = frame[column].isna().astype(float)
    return out


def gas_spike_flag(gas_price: pd.Series, train_index: pd.DatetimeIndex) -> tuple[pd.Series, float]:
    """90th percentile on the training rows only. Later gas prices do not move it."""
    train = gas_price.reindex(train_index).dropna()
    if train.empty:
        raise ValueError("no training gas prices")
    p90 = float(train.quantile(0.90))
    flag = (gas_price > p90).astype(float).where(gas_price.notna())
    return flag, p90


def reactor_outage_flag(offline: pd.Series, threshold: float = 25.0) -> pd.Series:
    return (offline > threshold).astype(float).where(offline.notna())


def is_weather_column(name: str) -> bool:
    if name in WEATHER_STRESS:
        return True
    if name.startswith(("temp_", "wind_")):
        return True
    suffixes = (
        "temperature_2m", "relative_humidity_2m", "cloud_cover",
        "wind_speed_10m", "wind_gusts_10m", "precipitation",
    )
    return name.endswith(suffixes) or "temperature_2m" in name


def is_assumed_fundamental(name: str) -> bool:
    """Gas and NRC. Their release timestamps are assumptions in the YAML file."""
    return (
        name.startswith(("henry_hub", "reactor_", "reactors_"))
        or name in {"stress_gas_spike", "stress_reactor_outage", "miss_gas", "miss_nrc"}
    )


def conservative_columns(columns: list[str]) -> list[str]:
    """Drop reanalysis weather and fundamentals whose release time is only assumed.

    Storage stays: its weekly offset is the documented Thursday release plus a buffer.
    Price history, publication-gated day-ahead prices, lagged load, calendar, and
    Chronos statistics stay. Load remains under an assumed posting rule and is
    marked that way in the catalog.
    """
    kept = []
    for name in columns:
        if name == "load_actual_mw" or name.startswith("gfs_"):
            continue
        if is_weather_column(name) or name in {"miss_weather", "stress_score"}:
            continue
        if is_assumed_fundamental(name):
            continue
        kept.append(name)
    return kept


def feature_subsets(columns: list[str]) -> dict[str, list[str]]:
    """Exogenous columns, Chronos statistics, and both. Order follows `columns`."""
    chronos = [c for c in columns if c in CHRONOS_STATS]
    exog = [c for c in columns if c not in CHRONOS_STATS]
    if not chronos or not exog:
        raise ValueError("a subset comparison needs both exogenous columns and Chronos statistics")
    return {"xgb_exog": exog, "xgb_chronos": chronos, "xgb_both": list(columns)}


def signed_asinh_columns(columns: list[str]) -> list[str]:
    """Price-like columns, including Chronos dollar forecasts. Not the exceedance share."""
    out = []
    for name in columns:
        if name == "chronos_p_gt200" or name.startswith("miss_") or name.startswith("is_"):
            continue
        if name.startswith((
            "hubavg_", "rt15_", "da_", "chronos_",
        )):
            out.append(name)
    return out


def apply_asinh(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = np.arcsinh(out[col].astype(float))
    return out


def train_median_impute(train: pd.DataFrame, frames: list[pd.DataFrame],
                        columns: list[str]) -> tuple[list[pd.DataFrame], pd.Series]:
    """Medians from training rows only. No zero fill and no fill from later rows."""
    med = train[columns].median(numeric_only=True)
    filled = []
    for frame in frames:
        part = frame.copy()
        for col in columns:
            part[col] = part[col].fillna(med[col])
        filled.append(part)
    return filled, med


def assert_chronological(train: pd.DataFrame, eval_df: pd.DataFrame, test: pd.DataFrame) -> None:
    if not (train.index.max() < eval_df.index.min() <= eval_df.index.max() < test.index.min()):
        raise AssertionError(
            f"split overlap: train {train.index.max()}, eval {eval_df.index.min()}..{eval_df.index.max()}, "
            f"test {test.index.min()}")


def hourly_interval_counts(rt_index: pd.DatetimeIndex) -> pd.Series:
    """How many 15-minute stamps fall in each UTC hour."""
    hours = pd.DatetimeIndex(rt_index).floor("h")
    return hours.value_counts().sort_index()


def assemble_assumed_realtime(
    index: pd.DatetimeIndex,
    price: pd.Series,
    rt15: pd.Series,
    da: pd.DataFrame,
    load: pd.Series,
    weather: pd.DataFrame,
    gas: pd.DataFrame,
    storage: pd.DataFrame,
    nrc: pd.DataFrame,
    chronos: pd.DataFrame,
) -> pd.DataFrame:
    """Columns for assumed_realtime, before the split-specific gas flag and stress score.

    `load_actual_mw` is intentionally absent. Weather is lagged one hour.
    """
    funds = fundamental_frames(gas, storage, nrc, index)
    parts = [
        price_history(price, index),
        previous_hour_intervals(rt15, index),
        day_ahead_features(price.reindex(index), da.reindex(index)),
        lagged_load(load, index),
        lagged_weather(weather, index, hours=1),
        funds["gas"],
        funds["storage"],
        funds["nrc"],
        calendar_central(index),
        chronos_block(chronos, index),
    ]
    frame = pd.concat(parts, axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    indicators = missing_indicators(frame, {
        "load": "load_lag_1h",
        "gas": "henry_hub_price",
        "storage": "storage_bcf",
        "nrc": "reactors_offline_pct_sum",
        "weather": "temp_max_across_zones",
        "da": "da_HB_HUBAVG",
        "chronos": "chronos_pred",
    })
    frame = pd.concat([frame, indicators], axis=1)
    frame["HB_HUBAVG"] = price.reindex(index).to_numpy()
    if "load_actual_mw" in frame.columns:
        raise AssertionError("same-hour load entered the assumed-realtime matrix")
    return frame


def attach_split_flags(frame: pd.DataFrame, train_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, float]:
    """Gas-spike flag and stress score. The gas percentile sees only `train_index`."""
    out = frame.copy()
    flag, p90 = gas_spike_flag(out["henry_hub_price"], train_index)
    out["stress_gas_spike"] = flag
    out["stress_reactor_outage"] = reactor_outage_flag(out["reactors_offline_pct_sum"])
    parts = [c for c in list(WEATHER_STRESS) + ["stress_gas_spike", "stress_reactor_outage"] if c in out.columns]
    out["stress_score"] = out[parts].sum(axis=1, min_count=1)
    return out, p90


def catalog_rows(columns: list[str], information_set: str) -> pd.DataFrame:
    """One row per feature. `verified` is documented, assumed, or model_output."""
    rows = []
    for name in columns:
        rows.append({"information_set": information_set, "feature": name, **describe_feature(name)})
    return pd.DataFrame(rows)


def describe_feature(name: str) -> dict:
    if name in CHRONOS_STATS:
        return {
            "source": "amazon/chronos-t5-small gap-aware cache",
            "observation": "forecast of hour t from prices through t-1",
            "publication": "model output; context must end at t-1",
            "verified": "model_output",
            "leakage_risk": "low if context check passes",
            "notes": "Not refit. Statistics are the cached one-step forecast.",
        }
    if name.startswith("hubavg_") or name.startswith("rt15_"):
        return {
            "source": "ERCOT RT SPP HB_HUBAVG",
            "observation": "last fully visible hour under the delay passed to price_history",
            "publication": "interval end plus an explicit delay; zero delay is not established",
            "verified": "assumed",
            "leakage_risk": "medium",
            "notes": "The on-disk series is the weekly historical SPP archive, not a first-publication feed. A positive whole-hour delay drops every interval of hour t-1.",
        }
    if name.startswith("da_"):
        return {
            "source": "ERCOT day-ahead SPP",
            "observation": "operating hour, or a summary of the published operating day",
            "publication": "14:00 Central on the previous local day (30-minute buffer on the 13:30 posting)",
            "verified": "documented",
            "leakage_risk": "low",
            "notes": "Rows with available_at after the issue time are set to missing.",
        }
    if name.startswith("load_") or name == "miss_load":
        return {
            "source": "ERCOT hourly load",
            "observation": "hour t-1 or the trailing window ending at t-1",
            "publication": "assumed known when the hour ends",
            "verified": "assumed",
            "leakage_risk": "medium",
            "notes": "Realized load, not a published load forecast. Same-hour load is excluded.",
        }
    if name.startswith("gfs_"):
        return {
            "source": "Open-Meteo GFS previous runs",
            "observation": "valid hour t",
            "publication": "lead day 1, issued about 24 hours before the valid hour",
            "verified": "documented",
            "leakage_risk": "low",
            "notes": "Temperature only. Coverage starts 2021-03-23. Not wind or solar generation.",
        }
    if is_weather_column(name):
        return {
            "source": "Open-Meteo archive",
            "observation": "hour t-1 after the clock lag",
            "publication": "reanalysis; ERA5 is not available at a one-hour lag",
            "verified": "assumed",
            "leakage_risk": "high",
            "notes": "Treated as a station-observation proxy. Removed in the conservative set.",
        }
    if name.startswith("henry_hub") or name == "stress_gas_spike" or name == "miss_gas":
        return {
            "source": "EIA Henry Hub spot",
            "observation": "trade date of the last row whose assumed release is at or before t",
            "publication": "assumed Wednesday weekly release, five-day minimum lag",
            "verified": "assumed",
            "leakage_risk": "high",
            "notes": "YAML marks this rule assumed. Removed in the conservative set. Spike flag uses the training percentile only.",
        }
    if name.startswith("storage") or name == "miss_storage":
        return {
            "source": "EIA weekly storage",
            "observation": "week-ending Friday",
            "publication": "following Friday 18:00 UTC (Thursday 10:30 ET plus a holiday buffer)",
            "verified": "documented",
            "leakage_risk": "low",
            "notes": "Offset is the documented release plus a buffer, not the Friday stamp itself.",
        }
    if name.startswith(("reactor", "reactors")) or name in {"stress_reactor_outage", "miss_nrc"}:
        return {
            "source": "NRC daily reactor status",
            "observation": "report date",
            "publication": "assumed next UTC day",
            "verified": "assumed",
            "leakage_risk": "high",
            "notes": "Removed in the conservative set.",
        }
    if name.endswith("_local") or name in {"hour_local", "day_of_week_local", "month_local"}:
        return {
            "source": "clock",
            "observation": "issue timestamp",
            "publication": "known when the hour starts",
            "verified": "documented",
            "leakage_risk": "none",
            "notes": "US/Central. Peak is weekday 16:00-22:00 local.",
        }
    if name.startswith("miss_"):
        return {
            "source": "missingness of another column",
            "observation": "issue time",
            "publication": "known once that column's availability rule has been applied",
            "verified": "documented",
            "leakage_risk": "none",
            "notes": "Indicates missingness. The imputed value uses the training median.",
        }
    if name == "stress_score":
        return {
            "source": "sum of stress flags",
            "observation": "same rows as the component flags",
            "publication": "no better than the weakest flag",
            "verified": "assumed",
            "leakage_risk": "high",
            "notes": "Dropped from the conservative set together with weather, gas, and NRC.",
        }
    return {
        "source": "unclassified",
        "observation": "",
        "publication": "",
        "verified": "assumed",
        "leakage_risk": "high",
        "notes": "No availability rule was written for this column.",
    }


def reported_column_note(name: str) -> dict:
    """Availability of a column in the existing honest-tree list. Not a clean set."""
    base = describe_feature(name)
    if name == "load_actual_mw":
        base.update({
            "verified": "assumed",
            "leakage_risk": "high",
            "publication": "same hour as the target",
            "notes": "Realized load at hour t. Not lagged. Not usable at the start of hour t.",
        })
    elif name in {"hour", "day_of_week", "month", "is_weekend", "is_peak_hours"}:
        base.update({
            "source": "clock",
            "observation": "UTC timestamp",
            "publication": "known",
            "verified": "documented",
            "leakage_risk": "none",
            "notes": "UTC, not Central. is_peak_hours is UTC hours 14-20.",
        })
    elif name.startswith("henry_hub") or name == "stress_gas_spike":
        base.update({
            "notes": "Honest tree joins gas on the trade date. Only the spike threshold is fit before 2023.",
            "leakage_risk": "high",
        })
    elif name.startswith("storage"):
        base.update({
            "notes": "Honest tree joins storage on the week-ending Friday, before the documented release.",
            "leakage_risk": "high",
            "verified": "documented",
        })
    elif name.startswith(("reactor", "reactors")) or name == "stress_reactor_outage":
        base.update({
            "notes": "Honest tree joins NRC on the report date.",
            "leakage_risk": "high",
        })
    elif is_weather_column(name):
        base.update({
            "notes": "Honest tree clock-lags these columns by one hour. The source is still reanalysis.",
        })
    elif name.startswith("da_") and not name.startswith("da_shape"):
        base.update({
            "notes": "Honest tree uses the operating-hour price without the publication gate.",
            "leakage_risk": "medium",
        })
    return base
