from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def standardize_datetime_index(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    freq: str = "h",
    tz: str = "UTC",
) -> pd.DataFrame:
    """Standardize a dataframe to a consistent UTC datetime index.

    Args:
        df: Input dataframe.
        datetime_col: Column containing datetime values. Ignored if the index
            is already a DatetimeIndex.
        freq: Resampling frequency — "h" for hourly, "D" for daily.
        tz: Target timezone (default UTC).

    Returns:
        DataFrame with a tz-aware DatetimeIndex at the requested frequency.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df[datetime_col], format="mixed"))
        df = df.drop(columns=[datetime_col], errors="ignore")

    if df.index.tz is None:
        df = df.tz_localize(tz)
    else:
        df = df.tz_convert(tz)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq(freq)
    return df


_FREQ_MAP = {"hourly": "h", "daily": "D", "weekly": "W", "15min": "15min"}


def standardize_datetime(
    df: pd.DataFrame,
    column: str = "datetime",
    freq: str = "hourly",
    tz: str = "UTC",
) -> pd.DataFrame:
    """Convenience wrapper around standardize_datetime_index.

    Accepts human-readable freq strings ("hourly", "daily", "weekly")
    and a ``column`` parameter (mapped to ``datetime_col``).
    """
    pd_freq = _FREQ_MAP.get(freq, freq)
    return standardize_datetime_index(df, datetime_col=column, freq=pd_freq, tz=tz)


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Save a dataframe to parquet with logging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file with logging."""
    path = Path(path)
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def rate_limited_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    delay: float = 1.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> requests.Response:
    """GET request with rate-limiting, retry on 429/503.

    Args:
        url: Request URL.
        params: Query parameters.
        headers: Request headers.
        delay: Seconds to sleep before the request.
        max_retries: Number of retries on 429/503.
        backoff_factor: Multiplier applied to delay after each retry.

    Returns:
        requests.Response on success.

    Raises:
        requests.HTTPError: After exhausting retries.
    """
    time.sleep(delay)
    for attempt in range(1, max_retries + 1):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code in (429, 503):
            wait = delay * (backoff_factor ** attempt)
            logger.warning(
                "HTTP %d from %s — retrying in %.1fs (attempt %d/%d)",
                resp.status_code, url, wait, attempt, max_retries,
            )
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp  # unreachable, but satisfies type checkers
