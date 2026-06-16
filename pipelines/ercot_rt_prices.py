"""ERCOT Real-Time 15-min Settlement Point Price pipeline.

Uses the gridstatus library to pull historical SPP data from ERCOT's
public data portal (no API key required). Data is fetched per-year,
filtered to trading hubs, then pivoted to one row per 15-min interval.
"""

from __future__ import annotations

import logging
import time

import gridstatus
import pandas as pd
from tqdm import tqdm

from pipelines.base import DataPipeline
from utils.config import RAW_DIR, SPIKE_THRESHOLD_MWH

logger = logging.getLogger(__name__)

TRADING_HUBS = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HUBAVG"]


class ErcotRealTimePricesPipeline(DataPipeline):
    """Ingest ERCOT real-time 15-min settlement point prices for trading hubs."""

    def __init__(self):
        super().__init__(name="ercot_rt_prices", output_dir=RAW_DIR)
        self.ercot = gridstatus.Ercot()

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch RTM SPP data year-by-year for the given date range.

        gridstatus.Ercot.get_rtm_spp() downloads a full year at a time from
        ERCOT's historical archive. We loop over each year in the range, filter
        to trading hubs, and concatenate.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        years = range(start.year, end.year + 1)

        chunks = []
        for year in tqdm(years, desc="Fetching ERCOT RTM SPP by year"):
            try:
                logger.info("Fetching RTM SPP for %d", year)
                df = self.ercot.get_rtm_spp(year, verbose=False)

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
        """Pivot hub prices into columns and add spike flag."""
        # Convert Interval Start to UTC
        df["Interval Start"] = df["Interval Start"].dt.tz_convert("UTC")

        # Pivot: one row per interval, one column per hub
        pivoted = df.pivot_table(
            index="Interval Start",
            columns="Location",
            values="SPP",
            aggfunc="first",
        )
        pivoted.index.name = "datetime"
        pivoted.columns.name = None

        # Keep only the hubs we care about (in case extras snuck in)
        pivoted = pivoted[[c for c in TRADING_HUBS if c in pivoted.columns]]

        # Drop rows where ALL prices are null
        pivoted = pivoted.dropna(how="all")

        # Sort chronologically
        pivoted = pivoted.sort_index()

        # Spike flag
        pivoted["price_spike_flag"] = pivoted["HB_HUBAVG"] > SPIKE_THRESHOLD_MWH

        # Summary statistics
        avg = pivoted["HB_HUBAVG"]
        spikes = pivoted["price_spike_flag"].sum()
        print("\n--- ERCOT RT Prices: Clean Summary ---")
        print(f"  Total rows:    {len(pivoted):,}")
        print(f"  Date range:    {pivoted.index.min()} → {pivoted.index.max()}")
        print(f"  HB_HUBAVG min: ${avg.min():.2f}/MWh")
        print(f"  HB_HUBAVG max: ${avg.max():.2f}/MWh")
        print(f"  HB_HUBAVG mean:${avg.mean():.2f}/MWh")
        print(f"  Spike events:  {spikes:,} (>{SPIKE_THRESHOLD_MWH} $/MWh)")
        print("--------------------------------------\n")

        return pivoted
