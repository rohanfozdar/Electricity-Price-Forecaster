"""
GDELT news sentiment pipeline.

Pulls daily counts and average tone of English-language news articles
about Texas energy/grid events from the GDELT 2.0 DOC API. GDELT
pre-computes a tone score for every article it indexes, so we get
sentiment without having to run our own NLP.

GDELT's DOC API is free, no key required, and responds in JSON.
Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

Design notes:
- We query with a text filter for Texas energy stress terms and restrict
  to English sources.
- GDELT's DOC API 'timeline' mode returns daily aggregated results,
  which is exactly what we want.
- We run one query per month to stay inside GDELT's soft response limits.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from pipelines.base import DataPipeline
from utils.config import DATA_START, DATA_END, RAW_DIR
from utils.helpers import standardize_datetime

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Query targets Texas grid / energy stress coverage. GDELT query syntax
# uses quoted phrases and boolean operators.
QUERY = (
    '("ERCOT" OR "Texas grid" OR "Texas power" OR "Texas electricity" '
    'OR "Texas blackout" OR "Texas outage") sourcelang:english'
)


class GDELTPipeline(DataPipeline):
    def __init__(self) -> None:
        super().__init__(name="gdelt", output_dir=RAW_DIR)

    def _fetch_month(self, start: datetime, end: datetime) -> pd.DataFrame:
        params = {
            "query": QUERY,
            "mode": "timelinetone",
            "format": "json",
            "startdatetime": start.strftime("%Y%m%d000000"),
            "enddatetime": end.strftime("%Y%m%d235959"),
        }
        r = requests.get(GDELT_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        timeline = data.get("timeline", [])
        if not timeline:
            return pd.DataFrame()
        # GDELT returns a list of {"data": [{"date": ..., "value": ...}]}
        records = timeline[0].get("data", [])
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df = df.rename(columns={"date": "datetime", "value": "gdelt_tone"})

        # Also fetch volume for the same window.
        # timelinevolraw returns two timeline series:
        #   [0] = normalized query volume (gdelt_norm, small per-unit values)
        #   [1] = raw total article count baseline (gdelt_article_volume, large integers)
        params_vol = dict(params)
        params_vol["mode"] = "timelinevolraw"
        rv = requests.get(GDELT_URL, params=params_vol, timeout=60)
        try:
            rv.raise_for_status()
            vol_data = rv.json().get("timeline", [])
            if vol_data:
                norm_records = vol_data[0].get("data", [])
                norm_df = pd.DataFrame(norm_records).rename(
                    columns={"date": "datetime", "value": "gdelt_norm"}
                )
                df = df.merge(norm_df, on="datetime", how="outer")
            if len(vol_data) > 1:
                vol_records = vol_data[1].get("data", [])
                vol_df = pd.DataFrame(vol_records).rename(
                    columns={"date": "datetime", "value": "gdelt_article_volume"}
                )
                df = df.merge(vol_df, on="datetime", how="outer")
            else:
                df["gdelt_article_volume"] = None
        except Exception:
            df["gdelt_norm"] = None
            df["gdelt_article_volume"] = None

        return df

    def fetch(self, start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> pd.DataFrame:
        start = pd.to_datetime(start_date or DATA_START)
        end = pd.to_datetime(end_date or DATA_END)
        months = pd.date_range(start, end, freq="MS")  # month start

        frames = []
        for i, month_start in enumerate(tqdm(months, desc="GDELT by month")):
            next_month = months[i + 1] if i + 1 < len(months) else end
            # Retry up to 3 times with exponential backoff for rate limits
            for attempt in range(3):
                try:
                    df = self._fetch_month(month_start, next_month)
                    if not df.empty:
                        frames.append(df)
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429 and attempt < 2:
                        wait = 10 * (attempt + 1)
                        print(f"  Rate limited, waiting {wait}s before retry...")
                        time.sleep(wait)
                    else:
                        print(f"  Month {month_start} failed: {e}")
                        break
                except Exception as e:
                    print(f"  Month {month_start} failed: {e}")
                    break
            time.sleep(5)

        if not frames:
            raise RuntimeError("GDELT returned no data")
        return pd.concat(frames, ignore_index=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # GDELT returns YYYYMMDDHHMMSS strings
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%dT%H%M%SZ", errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.set_index("datetime").sort_index()
        # Drop timezone info for consistency with our daily features
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        # Collapse to daily
        daily = df.resample("D").mean()
        print(f"\n--- GDELT Pipeline: Clean Summary ---")
        print(f"  Total rows:  {len(daily):,}")
        print(f"  Date range:  {daily.index.min()} → {daily.index.max()}")
        if "gdelt_tone" in daily.columns:
            print(f"  Mean tone:   {daily['gdelt_tone'].mean():.2f}")
            print(f"  Min tone:    {daily['gdelt_tone'].min():.2f} (most negative)")
        print("-------------------------------------\n")
        return daily


if __name__ == "__main__":
    pipe = GDELTPipeline()
    out = pipe.run()
    print(f"Saved to: {out}")
