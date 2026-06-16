from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.helpers import save_parquet


class DataPipeline(ABC):
    """Abstract base class that every data pipeline must implement."""

    def __init__(self, name: str, output_dir: Path):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self, start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> pd.DataFrame:
        """Download raw data for the given date range."""

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the raw dataframe."""

    def save(self, df: pd.DataFrame) -> Path:
        """Save the cleaned dataframe to parquet."""
        path = self.output_dir / f"{self.name}.parquet"
        return save_parquet(df, path)

    def run(self, start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> Path:
        """Execute the full pipeline: fetch → clean → save."""
        raw = self.fetch(start_date, end_date)
        cleaned = self.clean(raw)
        return self.save(cleaned)
