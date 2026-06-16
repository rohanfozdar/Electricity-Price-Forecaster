"""
Granger causality test, v2.

Tests whether lagged sentiment features Granger-cause HB_HUBAVG at the
DAILY level. Uses the v2 engineered feature matrix (sentiment lags and
change features now computed at native daily resolution, so the test is
testing genuine day-ahead signal rather than ffill-corrupted noise).

Sentiment features tested:
    - gdelt_tone, gdelt_article_volume                     (raw daily)
    - gdelt_tone_change_1d, gdelt_tone_change_3d           (changes)
    - gdelt_volume_change_1d, gdelt_volume_zscore_30d      (changes)
Methodology:
    1. Resample hourly v2 matrix to daily mean for HB_HUBAVG and each
       sentiment feature (sentiment is naturally daily/weekly so the
       resample is just a no-op forward-fill, but it keeps everything on
       the same daily grid).
    2. For each feature, run statsmodels grangercausalitytests at lags
       1..7 (days). The null is "lagged feature does NOT predict price
       beyond what lagged price alone predicts."
    3. Report best lag, p-value, and 0.05 / 0.01 significance per feature.

Headline finding: if `gdelt_tone_change_1d` or `gdelt_volume_zscore_30d`
Granger-causes price at p<0.05, that's the core thesis - sentiment
CHANGES lead price beyond what historical price can explain.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from utils.config import FEATURES_DIR

warnings.filterwarnings("ignore")

# Sentiment features to test (curated from v2 column list)
SENTIMENT_FEATURES = [
    # Raw daily levels
    "gdelt_tone",
    "gdelt_article_volume",
    # Changes (the core thesis: changes lead price)
    "gdelt_tone_change_1d",
    "gdelt_tone_change_3d",
    "gdelt_volume_change_1d",
    "gdelt_volume_change_3d",
    "gdelt_volume_zscore_30d",
    "gdelt_tone_zscore_30d",
]

MAX_LAG_DAYS = 7


def run_granger_tests():
    print("\n" + "=" * 70)
    print("  GRANGER CAUSALITY TESTS (daily resolution, v2 matrix)")
    print("=" * 70)
    print("  Null: lagged sentiment does NOT improve prediction of HB_HUBAVG")
    print("  beyond what lagged HB_HUBAVG alone provides.")
    print("  Significance: p<0.05 (*), p<0.01 (**), p<0.001 (***)\n")

    matrix_path = FEATURES_DIR / "feature_matrix_engineered_v2.parquet"
    if not matrix_path.exists():
        raise RuntimeError(f"Missing {matrix_path} - run features/build_matrix_v2.py first")
    df = pd.read_parquet(matrix_path)

    # Resample to daily mean
    cols_present = [f for f in SENTIMENT_FEATURES if f in df.columns]
    missing = [f for f in SENTIMENT_FEATURES if f not in df.columns]
    if missing:
        print(f"  [warn] Missing from v2 matrix: {missing}\n")

    daily = df[["HB_HUBAVG"] + cols_present].resample("D").mean().dropna(how="all")

    results = {}
    for feature in cols_present:
        test_data = daily[["HB_HUBAVG", feature]].dropna()
        if len(test_data) < 100:
            print(f"  [skip] {feature}: insufficient data ({len(test_data)} rows)")
            continue

        # Skip features with zero variance after dropna (statsmodels chokes)
        if test_data[feature].std() < 1e-12:
            print(f"  [skip] {feature}: zero variance")
            continue

        try:
            res = grangercausalitytests(test_data, maxlag=MAX_LAG_DAYS, verbose=False)
            p_vals = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
            min_p = min(p_vals.values())
            best_lag = min(p_vals, key=p_vals.get)
            results[feature] = {
                "min_p_value": float(min_p),
                "best_lag_days": int(best_lag),
                "all_lags": {int(k): float(v) for k, v in p_vals.items()},
                "significant_at_0.05": bool(min_p < 0.05),
                "significant_at_0.01": bool(min_p < 0.01),
                "significant_at_0.001": bool(min_p < 0.001),
                "n_obs": int(len(test_data)),
            }
        except Exception as e:
            print(f"  [error] {feature}: {e}")
            continue

    # Print results table
    print(f"\n  {'Feature':<42} {'p-value':>10}  {'lag':>4}  {'sig':>6}  {'n':>5}")
    print("  " + "-" * 76)
    for feat, r in sorted(results.items(), key=lambda kv: kv[1]["min_p_value"]):
        if r["significant_at_0.001"]:
            star = "***"
        elif r["significant_at_0.01"]:
            star = "**"
        elif r["significant_at_0.05"]:
            star = "*"
        else:
            star = ""
        print(f"  {feat:<42} {r['min_p_value']:>10.4f}  {r['best_lag_days']:>4d}  "
              f"{star:>6}  {r['n_obs']:>5d}")

    # Highlight the headline finding
    print("\n" + "=" * 70)
    print("  HEADLINE FINDING")
    print("=" * 70)
    sig05 = {f: r for f, r in results.items() if r["significant_at_0.05"]}
    sig01 = {f: r for f, r in results.items() if r["significant_at_0.01"]}

    change_features_significant = [
        f for f in sig05
        if "change" in f or "zscore" in f or "wow" in f
    ]
    if change_features_significant:
        print(f"\n  CHANGE-BASED sentiment features that Granger-cause HB_HUBAVG (p<0.05):")
        for feat in change_features_significant:
            r = results[feat]
            print(f"    - {feat:<42} p={r['min_p_value']:.4f} at lag {r['best_lag_days']}d")
        print("\n  This is the core thesis: sentiment CHANGES (not levels) lead price")
        print("  movements beyond what historical price alone can explain.")
    else:
        print("\n  No change-based sentiment features achieved p<0.05.")
        print("  This weakens the lead-lag thesis.")

    print(f"\n  Total features significant at p<0.05: {len(sig05)} / {len(results)}")
    print(f"  Total features significant at p<0.01: {len(sig01)} / {len(results)}")

    out_path = Path("evaluation/granger_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}\n")
    return results


if __name__ == "__main__":
    run_granger_tests()
