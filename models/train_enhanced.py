"""
Sentiment-enhanced model: baseline features + Reddit + GDELT + Google Trends.

This is the model that tests the project thesis. If it meaningfully beats
the baseline on spike detection (even if average RMSE is only slightly
better), that is the main finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from utils.config import FEATURES_DIR
from models.utils import (
    TARGET_REG,
    chronological_split,
    regression_report,
    select_enhanced_features,
)

MODELS_DIR = Path("models/artifacts")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_enhanced():
    print("\n=== Training Sentiment-Enhanced Regression Model ===\n")
    df = pd.read_parquet(FEATURES_DIR / "feature_matrix_engineered_v2.parquet")
    df = df.dropna(subset=[TARGET_REG])

    train, val, test = chronological_split(df)
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    features = select_enhanced_features(df)
    print(f"  Using {len(features)} enhanced features")

    X_train = train[features].ffill().fillna(0)
    y_train = train[TARGET_REG]
    X_val = val[features].ffill().fillna(0)
    y_val = val[TARGET_REG]
    X_test = test[features].ffill().fillna(0)
    y_test = test[TARGET_REG]

    model = xgb.XGBRegressor(      # same model as baseline, but with sentiment features
        n_estimators=700,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        early_stopping_rounds=30,
        eval_metric="rmse",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10,
    )

    train_report = regression_report(y_train, model.predict(X_train), "enhanced-train")
    val_report = regression_report(y_val, model.predict(X_val), "enhanced-val")
    test_report = regression_report(y_test, model.predict(X_test), "enhanced-test")

    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)
    print("  Top 20 features by importance:")
    for feat, imp in importance.head(20).items():
        print(f"    {feat:40s} {imp:.4f}")

    # Check sentiment features specifically
    sentiment_features = [
        f for f in features
        if f.startswith(("reddit_", "gdelt_"))
    ]
    sent_importance = importance[sentiment_features].sum()
    print(f"\n  Total sentiment importance: {sent_importance:.4f}")
    print(f"  ({len(sentiment_features)} sentiment features out of {len(features)})")

    model_path = MODELS_DIR / "enhanced_regressor.json"
    model.save_model(str(model_path))
    importance.to_csv(MODELS_DIR / "enhanced_feature_importance.csv")
    with open(MODELS_DIR / "enhanced_report.json", "w") as f:
        json.dump(
            {"train": train_report, "val": val_report, "test": test_report},
            f, indent=2
        )
    print(f"\n  Saved to {model_path}\n")
    return model, test_report


if __name__ == "__main__":
    train_enhanced()
