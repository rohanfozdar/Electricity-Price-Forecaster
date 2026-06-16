"""Sanity check for project setup: config, helpers, pipeline ABC, and dependencies."""

import sys
import os
import logging

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Scorecard tracking ──────────────────────────────────────────────────────
results = {"Config": "PASS", "Helpers": "PASS", "Pipeline ABC": "PASS", "Dependencies": "PASS"}
dep_failures = []

# ═══════════════════════════════════════════════════════════════════════════
# 1. Config
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. CONFIG")
print("=" * 60)
try:
    from utils.config import (
        ERCOT_LOAD_ZONES,
        DATA_START,
        DATA_END,
        SPIKE_THRESHOLD_MWH,
        RAW_DIR,
        PROCESSED_DIR,
        FEATURES_DIR,
        MODELS_DIR,
        DATA_DIR,
    )

    print("\nLoad Zones:")
    for zone, (lat, lon) in ERCOT_LOAD_ZONES.items():
        print(f"  {zone}: lat={lat}, lon={lon}")

    print(f"\nDATA_START:          {DATA_START}")
    print(f"DATA_END:            {DATA_END}")
    print(f"SPIKE_THRESHOLD_MWH: ${SPIKE_THRESHOLD_MWH}/MWh")

    print("\nData Directories:")
    for label, path in [
        ("data/", DATA_DIR),
        ("data/raw/", RAW_DIR),
        ("data/processed/", PROCESSED_DIR),
        ("data/features/", FEATURES_DIR),
        ("models/", MODELS_DIR),
    ]:
        exists = "EXISTS" if path.exists() else "MISSING"
        print(f"  {label:<20} {path}  [{exists}]")

except Exception as e:
    results["Config"] = "FAIL"
    print(f"\n  ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Helpers
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. HELPERS")
print("=" * 60)
try:
    import pandas as pd
    import numpy as np
    from utils.helpers import standardize_datetime_index, save_parquet, load_parquet
    from utils.config import RAW_DIR

    # Build a messy dataframe
    messy_dates = [
        "2023-01-01",
        "01/02/2023",
        "2023-01-03 14:30:00",
        "2023-01-04",
        "01/05/2023",
        "2023-01-06 08:00:00",
        "2023-01-07",
        "01/08/2023",
        "2023-01-09 23:59:59",
        "2023-01-10",
    ]
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"datetime": messy_dates, "price": rng.uniform(20, 300, size=10)})

    print("\nOriginal (messy) dataframe:")
    print(df.to_string(index=False))

    # Standardize
    cleaned = standardize_datetime_index(df, datetime_col="datetime", freq="D")
    print("\nStandardized dataframe:")
    print(cleaned)

    # Round-trip parquet
    test_path = RAW_DIR / "test_output.parquet"
    save_parquet(cleaned, test_path)
    loaded = load_parquet(test_path)

    # Compare values — freq attribute is not preserved in parquet, so check data only
    pd.testing.assert_frame_equal(
        cleaned.reset_index(drop=True),
        loaded.reset_index(drop=True),
        check_names=False,
    )
    # Also verify the datetime index values survived
    assert (cleaned.index == loaded.index).all(), "Datetime index mismatch"
    print("\nParquet round-trip: MATCH ✓")

except Exception as e:
    results["Helpers"] = "FAIL"
    print(f"\n  ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Pipeline ABC
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. PIPELINE ABC")
print("=" * 60)
try:
    from pathlib import Path
    from pipelines.base import DataPipeline

    # Confirm ABC cannot be instantiated directly
    try:
        DataPipeline("bad", Path("/tmp"))
        results["Pipeline ABC"] = "FAIL"
        print("  ERROR: DataPipeline should not be instantiatable directly")
    except TypeError:
        print("  ABC enforcement working")

    # Create a concrete subclass
    class TestPipeline(DataPipeline):
        def fetch(self, start_date, end_date):
            return pd.DataFrame({"datetime": ["2023-01-01"], "value": [1.0]})

        def clean(self, df):
            return df

    tp = TestPipeline("sanity_test", RAW_DIR)
    out_path = tp.run("2023-01-01", "2023-01-31")
    print(f"  TestPipeline.run() produced: {out_path}")
    print(f"  File exists: {out_path.exists()}")

    # Clean up the pipeline test artifact
    out_path.unlink(missing_ok=True)

except Exception as e:
    results["Pipeline ABC"] = "FAIL"
    print(f"\n  ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Dependencies
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. DEPENDENCIES")
print("=" * 60)

packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "sklearn": "sklearn",
    "statsmodels": "statsmodels",
    "streamlit": "streamlit",
    "praw": "praw",
    "requests": "requests",
    "plotly": "plotly",
    "tqdm": "tqdm",
}

for display_name, import_name in packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {display_name:<15} {version}", flush=True)
    except ImportError as e:
        dep_failures.append(display_name)
        print(f"  {display_name:<15} IMPORT FAILED: {e}", flush=True)

if dep_failures:
    results["Dependencies"] = f"FAIL ({', '.join(dep_failures)})"

# ═══════════════════════════════════════════════════════════════════════════
# 5. Summary Scorecard
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY SCORECARD")
print("=" * 60)
all_pass = True
for section, status in results.items():
    indicator = "PASS" if status == "PASS" else "FAIL"
    if indicator == "FAIL":
        all_pass = False
    print(f"  {section:<20} {status}")

print("=" * 60)
if all_pass:
    print("All checks passed.")
else:
    print("Some checks failed — see details above.")
