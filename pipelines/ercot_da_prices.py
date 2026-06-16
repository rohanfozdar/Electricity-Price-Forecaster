"""ERCOT Day-Ahead Market (DAM) Settlement Point Price pipeline.

Uses gridstatus to pull historical DAM SPP data from ERCOT's public data
portal (no API key required). Data is fetched per-year, filtered to trading
hubs, then pivoted to one row per hour with a "da_" column prefix.
"""

from __future__ import annotations

import logging
import time

import gridstatus
import pandas as pd
from tqdm import tqdm

from pipelines.base import DataPipeline
from utils.config import RAW_DIR

logger = logging.getLogger(__name__)

TRADING_HUBS = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HUBAVG"]


class ErcotDayAheadPricesPipeline(DataPipeline):
    """Ingest ERCOT day-ahead hourly settlement point prices for trading hubs."""

    def __init__(self):
        super().__init__(name="ercot_da_prices", output_dir=RAW_DIR)
        self.ercot = gridstatus.Ercot()

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch DAM SPP data year-by-year for the given date range."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        years = range(start.year, end.year + 1)

        chunks = []
        for year in tqdm(years, desc="Fetching ERCOT DAM SPP by year"):
            try:
                logger.info("Fetching DAM SPP for %d", year)
                df = self.ercot.get_dam_spp(year, verbose=False)

                # Filter to trading hubs only
                df = df[df["Location"].isin(TRADING_HUBS)]

                # Trim to the requested date range
                df = df[
                    (df["Interval Start"] >= start.tz_localize("US/Central"))
                    & (df["Interval Start"] <= end.tz_localize("US/Central"))
                ]

                chunks.append(df)
                logger.info("Year %d: %d rows", year, len(df))
            except Exception:
                logger.exception("Failed to fetch year %d — skipping", year)
                continue

            # Small sleep between years to be polite to ERCOT servers
            if year != years[-1]:
                time.sleep(2)

        if not chunks:
            raise RuntimeError("No data fetched for any year in range")

        return pd.concat(chunks, ignore_index=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot hub prices into columns with da_ prefix."""
        # Convert Interval Start to UTC
        df["Interval Start"] = df["Interval Start"].dt.tz_convert("UTC")

        # Pivot: one row per hour, one column per hub
        pivoted = df.pivot_table(
            index="Interval Start",
            columns="Location",
            values="SPP",
            aggfunc="first",
        )
        pivoted.index.name = "datetime"
        pivoted.columns.name = None

        # Keep only the hubs we care about
        pivoted = pivoted[[c for c in TRADING_HUBS if c in pivoted.columns]]

        # Add da_ prefix to avoid collisions with real-time columns
        pivoted = pivoted.rename(columns={c: f"da_{c}" for c in pivoted.columns})

        # Drop rows where ALL prices are null
        pivoted = pivoted.dropna(how="all")

        # Sort chronologically
        pivoted = pivoted.sort_index()

        # Summary statistics
        avg = pivoted["da_HB_HUBAVG"]
        print("\n--- ERCOT DA Prices: Clean Summary ---")
        print(f"  Total rows:       {len(pivoted):,}")
        print(f"  Date range:       {pivoted.index.min()} → {pivoted.index.max()}")
        print(f"  da_HB_HUBAVG min: ${avg.min():.2f}/MWh")
        print(f"  da_HB_HUBAVG max: ${avg.max():.2f}/MWh")
        print(f"  da_HB_HUBAVG mean:${avg.mean():.2f}/MWh")
        print("--------------------------------------\n")

        return pivoted
