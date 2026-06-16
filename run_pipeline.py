"""Run ERCOT price pipelines (real-time and/or day-ahead)."""

from __future__ import annotations

import argparse
import logging

from utils.config import DATA_START, DATA_END
from utils.helpers import load_parquet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_rt(start: str, end: str) -> str:
    from pipelines.ercot_rt_prices import ErcotRealTimePricesPipeline

    print(f"\n{'='*60}")
    print(f"Running ERCOT RT prices pipeline: {start} → {end}")
    print("=" * 60)
    pipeline = ErcotRealTimePricesPipeline()
    output_path = pipeline.run(start, end)
    df = load_parquet(output_path)
    print(f"Shape: {df.shape}")
    print("\nHead:")
    print(df.head())
    print("\nTail:")
    print(df.tail())
    return str(output_path)


def run_da(start: str, end: str) -> str:
    from pipelines.ercot_da_prices import ErcotDayAheadPricesPipeline

    print(f"\n{'='*60}")
    print(f"Running ERCOT DA prices pipeline: {start} → {end}")
    print("=" * 60)
    pipeline = ErcotDayAheadPricesPipeline()
    output_path = pipeline.run(start, end)
    df = load_parquet(output_path)
    print(f"Shape: {df.shape}")
    print("\nHead:")
    print(df.head())
    print("\nTail:")
    print(df.tail())
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Run ERCOT price pipelines")
    parser.add_argument(
        "--pipeline",
        choices=["rt", "da", "all"],
        default="all",
        help="Which pipeline to run (default: all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Pull the full {DATA_START} to {DATA_END} range (default: Jan 2023 only)",
    )
    args = parser.parse_args()

    if args.full:
        start, end = DATA_START, DATA_END
    else:
        start, end = "2023-01-01", "2023-01-31"

    results = {}

    if args.pipeline in ("rt", "all"):
        results["RT"] = run_rt(start, end)

    if args.pipeline in ("da", "all"):
        results["DA"] = run_da(start, end)

    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for name, path in results.items():
        print(f"  {name}: {path}")

    if not args.full:
        print(f"\nTest run complete. To pull the full {DATA_START} to {DATA_END} range:")
        if args.pipeline == "all":
            print("  python run_pipeline.py --full")
        else:
            print(f"  python run_pipeline.py --pipeline {args.pipeline} --full")
    print("=" * 60)


if __name__ == "__main__":
    main()
