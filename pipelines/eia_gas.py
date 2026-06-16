"""
EIA Henry Hub natural gas spot price pipeline.

Daily Henry Hub natural gas spot prices read from a locally downloaded EIA
XLS file (data/raw/henry_hub_NG.xls). Gas sets the marginal electricity
price on ERCOT ~40-50% of the time, so gas spikes are a direct leading
indicator for electricity spikes.

EIA_API_KEY is no longer required. The XLS file can be refreshed manually
from: https://www.eia.gov/dnav/ng/hist/rngwhhdD.xls
(the .env.example entry is kept for Task 5 cleanup).

Design notes:
- Source: EIA series RNGWHHD (Henry Hub Natural Gas Spot Price, $/MMBtu)
- Raw values only; 5-day rolling volatility and 20-day percentile rank are
  computed in the feature engineering stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from pipelines.base import DataPipeline
from utils.config import RAW_DIR
from utils.helpers import standardize_datetime

XLS_PATH = RAW_DIR / "henry_hub_NG.xls"


def _find_header_row(path: Path, sheet: str) -> int:
    """Return the index of the first row whose col-0 is a short date-column label."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=15)
    for i, row in raw.iterrows():
        cell = str(row.iloc[0]).strip().lower()
        # Must be a short label (≤30 chars) that starts with a known header keyword
        if len(cell) <= 30 and any(cell.startswith(k) for k in ("date", "week", "period")):
            return int(i)
    return 0


class EIAGasPipeline(DataPipeline):
    def __init__(self) -> None:
        super().__init__(name="eia_gas", output_dir=RAW_DIR)

    def fetch(self, start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> pd.DataFrame:
        header_row = _find_header_row(XLS_PATH, "Data 1")
        df = pd.read_excel(XLS_PATH, sheet_name="Data 1", header=header_row)
        df = df.iloc[:, :2].copy()
        df.columns = ["datetime", "henry_hub_price"]
        df["henry_hub_price"] = pd.to_numeric(df["henry_hub_price"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = standardize_datetime(df, column="datetime", freq="daily")
        df = df.sort_index().dropna()
        print(f"\n--- EIA Gas Pipeline: Clean Summary ---")
        print(f"  Total rows:  {len(df):,}")
        print(f"  Date range:  {df.index.min()} → {df.index.max()}")
        print(f"  Min price:   ${df['henry_hub_price'].min():.2f}/MMBtu")
        print(f"  Max price:   ${df['henry_hub_price'].max():.2f}/MMBtu")
        print(f"  Mean price:  ${df['henry_hub_price'].mean():.2f}/MMBtu")
        print("---------------------------------------\n")
        return df


if __name__ == "__main__":
    pipe = EIAGasPipeline()
    out = pipe.run()
    print(f"Saved to: {out}")
