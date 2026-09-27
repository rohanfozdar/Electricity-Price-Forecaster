"""Publication-time availability and as-of joins.

Every source row carries a UTC timestamp `ts`. `available_at(ts, rule)` returns
the first UTC instant at which that row could have been known. `asof_join`
then gives, for each issue time t, the latest row with available_at <= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CENTRAL = "US/Central"


def _td(s: str) -> pd.Timedelta:
    return pd.Timedelta(s)


def available_at(ts: pd.DatetimeIndex, rule: dict) -> pd.DatetimeIndex:
    """UTC availability time for each source timestamp under an availability rule."""
    ts = pd.DatetimeIndex(ts)
    if ts.tz is None:
        raise ValueError("source timestamps must be tz-aware UTC")
    kind = rule["rule"]
    if kind == "interval_end_plus":
        return ts + _td(rule["interval"]) + _td(rule.get("delay", "0min"))
    if kind == "daily_next_utc_day":
        return ts.floor("D") + pd.Timedelta(days=1) + _td(rule.get("delay", "0h"))
    if kind == "weekly_release":
        return ts + _td(rule["release_offset"])
    if kind == "eia_weekly_spot":
        base = ts.floor("D") + pd.Timedelta(days=int(rule["min_days"]))
        shift = (int(rule["release_weekday"]) - base.dayofweek) % 7
        hh, mm = map(int, rule["release_time_utc"].split(":"))
        return base + pd.to_timedelta(shift, unit="D") + pd.Timedelta(hours=hh, minutes=mm)
    if kind == "dam_publication":
        # Operating day D (local) is published at publish_local on D-1 (local).
        local = ts.tz_convert(rule.get("timezone", CENTRAL))
        day = local.normalize().tz_localize(None) - pd.Timedelta(days=1)
        hh, mm = map(int, rule["publish_local"].split(":"))
        pub_local = (day + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(
            rule.get("timezone", CENTRAL), ambiguous="NaT", nonexistent="shift_forward")
        return pub_local.tz_convert("UTC")
    raise ValueError(f"unknown availability rule {kind!r}")


def asof_join(values: pd.DataFrame | pd.Series, origins: pd.DatetimeIndex, rule: dict,
              suffix: str = "") -> pd.DataFrame:
    """For each origin, the latest source row whose availability time <= origin.

    Also returns `<name>_age_h`, the hours between the source timestamp and the origin.
    """
    frame = values.to_frame() if isinstance(values, pd.Series) else values.copy()
    frame = frame.sort_index()
    avail = available_at(frame.index, rule)
    order = np.argsort(avail.values, kind="stable")
    avail_sorted = avail.values[order]
    pos = np.searchsorted(avail_sorted, pd.DatetimeIndex(origins).values, side="right") - 1
    out = pd.DataFrame(index=pd.DatetimeIndex(origins))
    valid = pos >= 0
    src_rows = np.where(valid, order[np.clip(pos, 0, None)], 0)
    for col in frame.columns:
        v = frame[col].to_numpy(dtype=float)[src_rows]
        out[f"{col}{suffix}"] = np.where(valid, v, np.nan)
    src_ts = frame.index.values[src_rows]
    age = (pd.DatetimeIndex(origins).values - src_ts) / np.timedelta64(1, "h")
    out[f"{'_'.join(map(str, frame.columns[:1]))}{suffix}_age_h"] = np.where(valid, age, np.nan)
    return out


def dam_known_until(origins: pd.DatetimeIndex, publish_local: str = "14:00",
                    tz: str = CENTRAL) -> pd.DatetimeIndex:
    """End (exclusive, UTC) of the DA hours published by each origin.

    If local time >= publish_local on day L, operating day L+1 is published, so DA is
    known through the end of L+1; otherwise through the end of L.
    """
    local = pd.DatetimeIndex(origins).tz_convert(tz)
    hh, mm = map(int, publish_local.split(":"))
    minutes = local.hour * 60 + local.minute
    published_next = minutes >= hh * 60 + mm
    day = local.normalize().tz_localize(None)
    last_day = day + pd.to_timedelta(np.where(published_next, 1, 0), unit="D")
    end_local = (last_day + pd.Timedelta(days=1)).tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    return end_local.tz_convert("UTC")
