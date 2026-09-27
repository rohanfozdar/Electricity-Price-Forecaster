"""Download archived GFS 2 m temperature forecasts (lead 1-7 days) for the four ERCOT zones.

    python scripts/fetch_gfs_forecasts.py

Writes data/raw/weather_forecast_gfs.parquet. Source: Open-Meteo Previous Runs API,
where `temperature_2m_previous_dayN` is the value predicted 24*N hours before valid time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.sources import fetch_gfs_forecasts  # noqa: E402

if __name__ == "__main__":
    df = fetch_gfs_forecasts()
    print(f"{len(df):,} hours x {df.shape[1]} columns, {df.index.min()} -> {df.index.max()}")
    print("non-null share by lead (Houston):")
    for n in range(1, 8):
        c = f"Houston|d{n}"
        print(f"  d{n}: {df[c].notna().mean():.3f}  first valid {df[c].first_valid_index()}")
