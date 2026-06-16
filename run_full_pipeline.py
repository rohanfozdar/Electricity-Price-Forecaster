"""
Master orchestrator for the ERCOT grid stress forecaster.

Usage:
    python run_full_pipeline.py --stage all        # Run everything
    python run_full_pipeline.py --stage pipelines   # Only data pipelines
    python run_full_pipeline.py --stage features    # Only feature engineering
    python run_full_pipeline.py --stage models      # Only model training
    python run_full_pipeline.py --stage evaluate    # Only evaluation
    python run_full_pipeline.py --stage dashboard   # Launch Streamlit

Each stage depends on the previous one. If you run 'all', it executes
them in order. If a pipeline fails, it logs the error and continues
so you don't lose progress on the others.

Note: Steps 2 and 3 (ERCOT RT and DA prices) should already be done
from the earlier manual runs. This orchestrator skips them if the
parquet files already exist.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import RAW_DIR, FEATURES_DIR


def run_pipelines():
    print("\n" + "=" * 60)
    print("  STAGE 1: DATA PIPELINES")
    print("=" * 60 + "\n")

    # Check if RT and DA prices already exist (from Steps 2-3)
    if (RAW_DIR / "ercot_rt_prices.parquet").exists():
        print("[skip] ERCOT RT prices already pulled")
    else:
        print("[WARN] ERCOT RT prices missing - run: python run_pipeline.py --pipeline rt --full")

    if (RAW_DIR / "ercot_da_prices.parquet").exists():
        print("[skip] ERCOT DA prices already pulled")
    else:
        print("[WARN] ERCOT DA prices missing - run: python run_pipeline.py --pipeline da --full")

    # Run remaining pipelines
    pipeline_runs = [
        ("Weather (Open-Meteo)", "pipelines.weather", "WeatherPipeline"),
        ("ERCOT Load", "pipelines.ercot_load", "ErcotLoadPipeline"),
        ("GDELT News Sentiment", "pipelines.gdelt", "GDELTPipeline"),
        ("EIA Gas Prices", "pipelines.eia_gas", "EIAGasPipeline"),
        ("EIA Gas Storage", "pipelines.eia_storage", "EIAStoragePipeline"),
    ]

    # These are slow or flaky - run separately
    slow_pipelines = [
        ("NRC Reactors", "pipelines.nrc_reactors", "NRCReactorsPipeline"),
    ]

    all_pipelines = pipeline_runs + slow_pipelines

    results = {}
    for name, module_path, class_name in all_pipelines:
        print(f"\n--- Running: {name} ---")
        try:
            module = __import__(module_path, fromlist=[class_name])
            pipeline_class = getattr(module, class_name)
            pipeline = pipeline_class()
            output = pipeline.run()
            results[name] = f"SUCCESS -> {output}"
            print(f"  [OK] {name}")
        except Exception as e:
            results[name] = f"FAILED: {e}"
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:30s} {status}")
    print()


def run_features():
    print("\n" + "=" * 60)
    print("  STAGE 2: FEATURE ENGINEERING")
    print("=" * 60 + "\n")

    from features.build_matrix import build_feature_matrix
    from features.engineer import engineer_features
    from features.build_matrix_v2 import build_feature_matrix_v2
    from features.engineer_v2 import engineer_features_v2

    matrix = build_feature_matrix()
    engineer_features(matrix)

    matrix_v2 = build_feature_matrix_v2()
    engineer_features_v2(matrix_v2)


def run_models():
    print("\n" + "=" * 60)
    print("  STAGE 3: MODEL TRAINING")
    print("=" * 60 + "\n")

    from models.train_baseline import train_baseline
    from models.train_enhanced import train_enhanced
    from models.train_classifier import train_spike_classifier
    from models.train_all_v2 import main as train_all_v2
    from models.calibrate_classifier import calibrate

    print("\n--- Baseline Regressor ---")
    train_baseline()

    print("\n--- Enhanced Regressor ---")
    train_enhanced()

    print("\n--- Spike Classifiers ---")
    train_spike_classifier()

    print("\n--- All v2 Models ---")
    train_all_v2()

    print("\n--- Calibrated Classifier ---")
    calibrate()


def run_evaluate():
    print("\n" + "=" * 60)
    print("  STAGE 4: EVALUATION")
    print("=" * 60 + "\n")

    from evaluation.granger import run_granger_tests
    from evaluation.backtest import run_backtest
    from evaluation.benchmark_dam import benchmark_vs_dam

    print("\n--- Granger Causality Tests ---")
    run_granger_tests()

    print("\n--- Stress Event Backtest ---")
    run_backtest()

    print("\n--- Day-Ahead Market Benchmark ---")
    benchmark_vs_dam()


def run_dashboard():
    import subprocess
    subprocess.run(["streamlit", "run", "dashboard/app.py"])


def main():
    parser = argparse.ArgumentParser(
        description="ERCOT Grid Stress Forecaster - Master Orchestrator"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "pipelines", "features", "models", "evaluate", "dashboard"],
        default="all",
        help="Which stage to run (default: all)",
    )
    args = parser.parse_args()

    stages = {
        "pipelines": run_pipelines,
        "features": run_features,
        "models": run_models,
        "evaluate": run_evaluate,
        "dashboard": run_dashboard,
    }

    if args.stage == "all":
        for stage_name, stage_fn in stages.items():
            if stage_name == "dashboard":
                print("\nSkipping dashboard launch in 'all' mode.")
                print("Run separately: python run_full_pipeline.py --stage dashboard")
                continue
            stage_fn()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()
