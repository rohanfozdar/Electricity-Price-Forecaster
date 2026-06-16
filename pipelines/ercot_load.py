"""
ERCOT system-wide load pipeline.

Pulls hourly actual load data from ERCOT's public data API. ERCOT
publishes "Actual System Load by Weather Zone" reports which give
hourly demand for each weather zone and system total.

No API key required — uses ERCOT's public MIS (Market Information System).
"""

from __future__ import annotations

import io
import time
import zipfile
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from pipelines.base import DataPipeline
from utils.config import DATA_START, DATA_END, RAW_DIR
from utils.helpers import standardize_datetime

# ERCOT public report: "Hourly Load Data Archives"
# Report ID NP6-345-CD contains native_load CSVs inside ZIP archives
ERCOT_DOCS_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
ERCOT_DOWNLOAD_URL = "https://www.ercot.com/misdownload/servlets/mirDownload"

# Report type IDs for load data
NATIVE_LOAD_RTID = 13101  # native system load archive


class ErcotLoadPipeline(DataPipeline):
    def __init__(self) -> None:
        super().__init__(name="ercot_load", output_dir=RAW_DIR)

    def _find_load_docs(self, year: int) -> list[dict]:
        """Query ERCOT MIS for load documents matching a year."""
        # Try the actual system load by weather zone report
        for rtid in [13101, 13100, 12311]:
            resp = requests.get(
                ERCOT_DOCS_URL,
                params={"reportTypeId": rtid},
                timeout=30,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            docs = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
            if not isinstance(docs, list):
                continue
            matches = [d for d in docs if str(year) in d.get("Document", {}).get("ConstructedName", "")]
            if matches:
                return matches
        return []

    def _download_and_read(self, doc: dict) -> pd.DataFrame:
        """Download a doc and try to read it as Excel or CSV."""
        doc_id = doc["Document"]["DocID"]
        url = f"{ERCOT_DOWNLOAD_URL}?doclookupId={doc_id}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        content = resp.content
        try:
            # Try ZIP first
            zf = zipfile.ZipFile(io.BytesIO(content))
            frames = []
            for name in zf.namelist():
                if name.endswith(".xlsx") or name.endswith(".xls"):
                    frames.append(pd.read_excel(io.BytesIO(zf.read(name)), sheet_name=None))
                elif name.endswith(".csv"):
                    frames.append({"csv": pd.read_csv(io.BytesIO(zf.read(name)))})
            if frames:
                all_dfs = []
                for item in frames:
                    if isinstance(item, dict):
                        all_dfs.extend(item.values())
                return pd.concat(all_dfs, ignore_index=True)
        except zipfile.BadZipFile:
            pass

        # Try direct Excel
        try:
            sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
            return pd.concat(sheets.values(), ignore_index=True)
        except Exception:
            pass

        # Try CSV
        return pd.read_csv(io.BytesIO(content))

    def fetch(self, start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> pd.DataFrame:
        start = pd.to_datetime(start_date or DATA_START)
        end = pd.to_datetime(end_date or DATA_END)
        years = list(range(start.year, end.year + 1))

        frames = []
        for year in tqdm(years, desc="ERCOT load by year"):
            try:
                docs = self._find_load_docs(year)
                if not docs:
                    print(f"  [load] No docs found for {year}, falling back to EIA")
                    continue
                df = self._download_and_read(docs[0])
                frames.append(df)
                print(f"  [load] year {year}: {len(df)} rows, cols={df.columns.tolist()[:5]}...")
            except Exception as e:
                print(f"  [load] year {year} failed: {e}")
            time.sleep(2)

        # If ERCOT archive didn't work, fall back to EIA hourly demand
        if not frames:
            print("  Falling back to EIA hourly demand API...")
            return self._fetch_from_eia(start, end)

        combined = pd.concat(frames, ignore_index=True)

        # Parse columns — ERCOT load archives have varied formats
        # Look for Hour Ending + date column, or Interval Start
        if "Hour Ending" in combined.columns:
            date_col = next(
                (c for c in combined.columns if "date" in c.lower() or "delivery" in c.lower()),
                combined.columns[0],
            )
            combined["datetime"] = (
                pd.to_datetime(combined[date_col])
                + pd.to_timedelta(combined["Hour Ending"].astype(float).astype(int) - 1, unit="h")
            )
        elif "Interval Start" in combined.columns:
            combined["datetime"] = pd.to_datetime(combined["Interval Start"])
        else:
            for col in ["Time", "Date", "Timestamp"]:
                if col in combined.columns:
                    combined["datetime"] = pd.to_datetime(combined[col])
                    break

        # Find system total column
        load_col = None
        for candidate in ["ERCOT", "System Total", "SystemTotal", "Total", "Load", "SYSTEM"]:
            matches = [c for c in combined.columns if candidate.lower() in c.lower()]
            if matches:
                load_col = matches[0]
                break
        if load_col is None:
            numeric = combined.select_dtypes(include="number").columns
            load_col = numeric[-1] if len(numeric) > 0 else combined.columns[-1]

        result = combined[["datetime", load_col]].copy()
        result = result.rename(columns={load_col: "load_actual_mw"})
        result["load_actual_mw"] = pd.to_numeric(result["load_actual_mw"], errors="coerce")
        result = result.dropna(subset=["datetime"])
        result = result[(result["datetime"] >= start) & (result["datetime"] <= end)]
        return result

    def _fetch_from_eia(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fallback: pull hourly demand from EIA API (free, no key for demand)."""
        import os
        api_key = os.getenv("EIA_API_KEY", "")
        url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "ERCO",
            "facets[type][]": "D",  # demand
            "start": start.strftime("%Y-%m-%dT00"),
            "end": end.strftime("%Y-%m-%dT23"),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }
        all_rows = []
        offset = 0
        while True:
            params["offset"] = offset
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("response", {}).get("data", [])
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < 5000:
                break
            offset += 5000
            time.sleep(1)

        if not all_rows:
            raise RuntimeError("EIA fallback returned no data")

        df = pd.DataFrame(all_rows)
        df["datetime"] = pd.to_datetime(df["period"])
        df = df.rename(columns={"value": "load_actual_mw"})
        df["load_actual_mw"] = pd.to_numeric(df["load_actual_mw"], errors="coerce")
        return df[["datetime", "load_actual_mw"]]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = standardize_datetime(df, column="datetime", freq="hourly")
        df = df.sort_index()
        print(f"\n--- ERCOT Load Pipeline: Clean Summary ---")
        print(f"  Total rows:  {len(df):,}")
        print(f"  Date range:  {df.index.min()} → {df.index.max()}")
        if "load_actual_mw" in df.columns:
            s = df["load_actual_mw"].dropna()
            if len(s):
                print(f"  Peak load:   {s.max():,.0f} MW")
                print(f"  Mean load:   {s.mean():,.0f} MW")
        print("------------------------------------------\n")
        return df


if __name__ == "__main__":
    pipe = ErcotLoadPipeline()
    out = pipe.run()
    print(f"Saved to: {out}")
