"""Raw-source loaders. Everything is returned on a UTC index; nothing is filled.

The files in data/raw are produced by pipelines/*.py (and, for GFS forecasts, by
`fetch_gfs_forecasts` below).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import ROOT

RAW = ROOT / "data" / "raw"
ZONES = {
    "Houston": (29.76, -95.37),
    "North (Dallas)": (32.78, -96.80),
    "South (San Antonio)": (29.42, -98.49),
    "West (Midland)": (31.99, -102.08),
}


def _read(name: str, raw_dir: Path = RAW) -> pd.DataFrame:
    df = pd.read_parquet(Path(raw_dir) / f"{name}.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _hourly_index(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end).floor("h"), freq="h", name="datetime")


def hourly_price(raw_dir: Path = RAW, column: str = "HB_HUBAVG") -> pd.Series:
    """Hourly max of 15-minute RT SPPs; NaN unless all four intervals are present."""
    rt = _read("ercot_rt_prices", raw_dir)[column]
    grp = rt.resample("1h")
    y = grp.max().where(grp.count() == 4)
    idx = _hourly_index(rt.index.min(), rt.index.max())
    return y.reindex(idx).rename("y")


def day_ahead(raw_dir: Path = RAW, column: str = "da_HB_HUBAVG") -> pd.Series:
    da = _read("ercot_da_prices", raw_dir)[column]
    return da.reindex(_hourly_index(da.index.min(), da.index.max())).rename("da")


def load(raw_dir: Path = RAW) -> pd.Series:
    s = _read("ercot_load", raw_dir)["load_actual_mw"]
    return s.reindex(_hourly_index(s.index.min(), s.index.max())).rename("load")


def weather(raw_dir: Path = RAW) -> pd.DataFrame:
    w = _read("weather", raw_dir)
    return w.reindex(_hourly_index(w.index.min(), w.index.max()))


def gdelt(raw_dir: Path = RAW) -> pd.DataFrame:
    return _read("gdelt", raw_dir)[["gdelt_tone", "gdelt_article_volume"]]


def henry_hub(raw_dir: Path = RAW) -> pd.Series:
    return _read("eia_gas", raw_dir)["henry_hub_price"].dropna()


def storage(raw_dir: Path = RAW) -> pd.DataFrame:
    s = _read("eia_storage", raw_dir)[["storage_bcf"]].dropna()
    s["storage_wow"] = s["storage_bcf"].diff()
    # deviation from the mean of the same ISO week over the prior five years (past data only)
    week = s.index.isocalendar().week.to_numpy()
    year = s.index.year.to_numpy()
    lvl = s["storage_bcf"].to_numpy(dtype=float)
    dev = np.full(len(s), np.nan)
    for i in range(len(s)):
        m = (week == week[i]) & (year < year[i]) & (year >= year[i] - 5)
        if m.sum() >= 3:
            dev[i] = lvl[i] - lvl[m].mean()
    s["storage_dev_5y"] = dev
    return s


def nrc(raw_dir: Path = RAW) -> pd.Series:
    return _read("nrc_reactors", raw_dir)["reactors_offline_pct_sum"].dropna()


GFS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def fetch_gfs_forecasts(start: str = "2021-03-23", end: str = "2024-12-31",
                        out: Path = RAW / "weather_forecast_gfs.parquet") -> pd.DataFrame:
    """GFS 2 m temperature (degF) at lead offsets of 1..7 days, one column per zone and lead."""
    frames = []
    hourly = ",".join(f"temperature_2m_previous_day{n}" for n in range(1, 8))
    for zone, (lat, lon) in ZONES.items():
        for y0 in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            s = max(pd.Timestamp(start), pd.Timestamp(f"{y0}-01-01")).strftime("%Y-%m-%d")
            e = min(pd.Timestamp(end), pd.Timestamp(f"{y0}-12-31")).strftime("%Y-%m-%d")
            params = {"latitude": lat, "longitude": lon, "hourly": hourly, "models": "gfs_seamless",
                      "start_date": s, "end_date": e, "timezone": "UTC", "temperature_unit": "fahrenheit"}
            for attempt in range(4):
                r = requests.get(GFS_URL, params=params, timeout=120)
                if r.status_code == 200:
                    break
                time.sleep(10 * (attempt + 1))
            r.raise_for_status()
            h = r.json()["hourly"]
            df = pd.DataFrame(h).rename(columns={"time": "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
            df = df.set_index("datetime")
            df.columns = [f"{zone}|d{c.rsplit('day', 1)[1]}" for c in df.columns]
            frames.append(df)
            time.sleep(1)
    by_zone: dict[str, list] = {}
    for f in frames:
        by_zone.setdefault(f.columns[0].split("|")[0], []).append(f)
    wide = pd.concat([pd.concat(v).sort_index() for v in by_zone.values()], axis=1)
    wide = wide[~wide.index.duplicated()].sort_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out)
    return wide


def gfs_forecasts(raw_dir: Path = RAW) -> pd.DataFrame | None:
    p = Path(raw_dir) / "weather_forecast_gfs.parquet"
    if not p.exists():
        return None
    return _read("weather_forecast_gfs", raw_dir)
