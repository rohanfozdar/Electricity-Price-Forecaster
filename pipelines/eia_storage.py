"""
EIA Weekly Natural Gas Storage Report pipeline.

Published every Thursday at 10:30 AM ET. Reads from a locally downloaded
EIA XLS file (data/raw/NG_storage.xls). Deviations from the 5-year average
are a classic leading indicator for gas prices, which in turn lead
electricity prices.

EIA_API_KEY is no longer required. The XLS file can be refreshed manually
from: https://ir.eia.gov/ngs/ngs.html  (Download -> XLS)
(the .env.example entry is kept for Task 5 cleanup).

Source: EIA series NW2_EPG0_SWO_R48_BCF (weekly, Lower 48 working gas, Bcf)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from pipelines.base import DataPipeline
from utils.config import RAW_DIR

XLS_PATH = RAW_DIR / "NG_storage.xls"


def _find_header_row(path: Path, sheet: str) -> int:
    """Return the index of the first row whose col-0 is a short date-column label."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=15)
    for i, row in raw.iterrows():
        cell = str(row.iloc[0]).strip().lower()
        # Must be a short label (≤30 chars) that starts with a known header keyword
        if len(cell) <= 30 and any(cell.startswith(k) for k in ("date", "week", "period")):
            return int(i)
    return 0


class EIAStoragePipeline(DataPipeline):
    def __init__(self) -> None:
        super().__init__(name="eia_storage", output_dir=RAW_DIR)

    def fetch(self, start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> pd.DataFrame:
        header_row = _find_header_row(XLS_PATH, "html_report_history")
        df = pd.read_excel(XLS_PATH, sheet_name="html_report_history",
                           header=header_row)
        date_col = df.columns[0]
        # Last column whose name contains "total" or "lower 48" is Total Lower 48
        total_col = next(
            c for c in reversed(list(df.columns))
            if "total" in str(c).lower() or "lower 48" in str(c).lower()
        )
        df = df[[date_col, total_col]].copy()
        df.columns = ["datetime", "storage_bcf"]
        df["storage_bcf"] = pd.to_numeric(df["storage_bcf"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index(pd.to_datetime(df["datetime"]))
        df = df.drop(columns=["datetime"], errors="ignore")
        df.index.name = "datetime"
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        df = df.sort_index().dropna()
        df["storage_wow_change"] = df["storage_bcf"].diff()
        print(f"\n--- EIA Storage Pipeline: Clean Summary ---")
        print(f"  Total rows:  {len(df):,}")
        print(f"  Date range:  {df.index.min()} → {df.index.max()}")
        print("-------------------------------------------\n")
        return df


if __name__ == "__main__":
    pipe = EIAStoragePipeline()
    out = pipe.run()
    print(f"Saved to: {out}")
