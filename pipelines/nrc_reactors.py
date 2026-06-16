"""
NRC Power Reactor Status Report pipeline.

Tracks daily power output (as a % of nameplate capacity) for the four
Texas nuclear reactors:
  - Comanche Peak 1 & 2 (near Dallas-Fort Worth)
  - South Texas 1 & 2 (near Bay City)

Together these provide ~10% of ERCOT baseload capacity. When a reactor
drops below 100% (especially to 0% for unplanned outages), it's a direct
supply shock the pricing engine needs to know about.

Data source: NRC publishes annual pipe-delimited text files at
  https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/{YYYY}/{YYYY}PowerStatus.txt

Format: ReportDt|Unit|Power
Example row: 12/31/2023 12:00:00 AM|Comanche Peak 1|100

Design notes:
- One annual file per year contains all daily readings for that year.
- We fetch one file per year, filter to Texas reactors, and pivot to wide.
- If a day's reading is missing, we forward-fill from the previous day.
- The output is a wide dataframe with one column per reactor per day.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from pipelines.base import DataPipeline
from utils.config import DATA_START, DATA_END, RAW_DIR

NRC_BASE = "https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status"

TEXAS_REACTORS = [
    "Comanche Peak 1",
    "Comanche Peak 2",
    "South Texas 1",
    "South Texas 2",
]


class NRCReactorsPipeline(DataPipeline):
    def __init__(self) -> None:
        super().__init__(name="nrc_reactors", output_dir=RAW_DIR)

    def _fetch_year(self, year: int) -> pd.DataFrame:
        """Fetch and parse the annual PowerStatus.txt file for a given year."""
        url = f"{NRC_BASE}/{year}/{year}PowerStatus.txt"
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print(f"  [nrc] {year}: HTTP {r.status_code}, skipping")
                return pd.DataFrame()
            # Pipe-delimited: ReportDt|Unit|Power
            df = pd.read_csv(io.StringIO(r.text), sep="|", dtype=str)
            df.columns = df.columns.str.strip()
            # Filter to Texas reactors only
            df = df[df["Unit"].isin(TEXAS_REACTORS)].copy()
            if df.empty:
                print(f"  [nrc] {year}: no Texas reactor rows found")
                return pd.DataFrame()
            # Parse date — older years omit the time component
            df["date"] = pd.to_datetime(df["ReportDt"].str.split(" ").str[0],
                                        format="%m/%d/%Y", errors="coerce")
            df["Power"] = pd.to_numeric(df["Power"], errors="coerce")
            df = df.dropna(subset=["date", "Power"])
            return df[["date", "Unit", "Power"]]
        except Exception as e:
            print(f"  [nrc] {year}: error — {e}")
            return pd.DataFrame()

    def fetch(self, start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> pd.DataFrame:
        start = pd.to_datetime(start_date or DATA_START)
        end = pd.to_datetime(end_date or DATA_END)
        years = range(start.year, end.year + 1)

        frames = []
        for year in tqdm(years, desc="NRC reactor status by year"):
            df = self._fetch_year(year)
            if not df.empty:
                frames.append(df)
            time.sleep(1)

        if not frames:
            raise RuntimeError("NRC returned no data for any year")

        combined = pd.concat(frames, ignore_index=True)
        # Keep only dates in the requested range
        combined = combined[
            (combined["date"] >= start) & (combined["date"] <= end)
        ]
        return combined

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pivot: one row per date, one column per reactor
        pivoted = df.pivot_table(
            index="date",
            columns="Unit",
            values="Power",
            aggfunc="mean",
        )
        pivoted.columns.name = None
        # Rename columns to snake_case with reactor_ prefix
        col_map = {r: f"reactor_{r.replace(' ', '_').lower()}_pct" for r in TEXAS_REACTORS}
        pivoted = pivoted.rename(columns=col_map)

        # Set timezone-aware daily index
        pivoted.index = pd.DatetimeIndex(pivoted.index)
        if pivoted.index.tz is None:
            pivoted.index = pivoted.index.tz_localize("UTC")
        pivoted.index.name = "datetime"
        pivoted = pivoted.sort_index()

        # Forward-fill gaps (weekends/holidays where NRC skips entries)
        pivoted = pivoted.ffill()

        # Total offline proxy: (100 - power%) summed across all four reactors
        pct_cols = [c for c in pivoted.columns if c.startswith("reactor_") and c.endswith("_pct")]
        pivoted["reactors_offline_pct_sum"] = (100 - pivoted[pct_cols]).sum(axis=1)

        print(f"\n--- NRC Reactors Pipeline: Clean Summary ---")
        print(f"  Total rows:  {len(pivoted):,}")
        print(f"  Date range:  {pivoted.index.min()} → {pivoted.index.max()}")
        for reactor in TEXAS_REACTORS:
            col = f"reactor_{reactor.replace(' ', '_').lower()}_pct"
            if col in pivoted.columns:
                mean_pct = pivoted[col].mean()
                null_pct = pivoted[col].isnull().mean() * 100
                print(f"  {reactor:20s}  mean={mean_pct:.1f}%  null={null_pct:.1f}%")
        print("--------------------------------------------\n")
        return pivoted


if __name__ == "__main__":
    pipe = NRCReactorsPipeline()
    out = pipe.run()
    print(f"Saved to: {out}")
