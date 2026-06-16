"""
Open-Meteo weather pipeline for ERCOT load zones.

Pulls hourly historical weather (temperature, wind, humidity, cloud cover,
precipitation) for the four ERCOT load zone centers. Open-Meteo is free and
does not require an API key.

Design notes:
- Hourly resolution matches the eventual feature matrix frequency.
- We pull one zone at a time to keep requests small and stay under any
  implicit rate limits.
- Temperature is converted to Fahrenheit since Texas grid thresholds are
  typically discussed in °F (e.g. "below 20°F", "above 105°F").
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from pipelines.base import DataPipeline
from utils.config import ERCOT_LOAD_ZONES, DATA_START, DATA_END, RAW_DIR
from utils.helpers import standardize_datetime, save_parquet

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
]


class WeatherPipeline(DataPipeline):
    def __init__(self) -> None:
        super().__init__(name="weather", output_dir=RAW_DIR)

    def _fetch_zone(self, zone: str, lat: float, lon: float,
                    start: str, end: str) -> pd.DataFrame:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(HOURLY_VARS),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
        }
        r = requests.get(OPEN_METEO_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()["hourly"]
        df = pd.DataFrame(data)
        df = df.rename(columns={"time": "datetime"})
        # Prefix every weather column with the zone so they don't collide
        # when we eventually concatenate horizontally.
        rename_map = {v: f"{zone}_{v}" for v in HOURLY_VARS}
        df = df.rename(columns=rename_map)
        return df

    def fetch(self, start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> pd.DataFrame:
        start = start_date or DATA_START
        end = end_date or DATA_END
        frames = []
        for zone, (lat, lon) in tqdm(ERCOT_LOAD_ZONES.items(),
                                     desc="Fetching weather by zone"):
            df = self._fetch_zone(zone, lat, lon, start, end)
            frames.append(df)
            time.sleep(1)  # polite spacing between API calls
        # Outer-join on datetime so all four zones align on the same index.
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on="datetime", how="outer")
        return merged

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = standardize_datetime(df, column="datetime", freq="hourly")
        df = df.sort_index()
        # Drop fully-empty rows; Open-Meteo occasionally returns a stub row.
        df = df.dropna(how="all")
        print(f"\n--- Weather Pipeline: Clean Summary ---")
        print(f"  Total rows:  {len(df):,}")
        print(f"  Date range:  {df.index.min()} → {df.index.max()}")
        print(f"  Columns:     {len(df.columns)}")
        print("---------------------------------------\n")
        return df


if __name__ == "__main__":
    pipe = WeatherPipeline()
    out = pipe.run()
    print(f"Saved to: {out}")
