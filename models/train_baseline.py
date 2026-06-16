"""
Baseline model: weather + historical price + load only.

This mirrors published electricity price forecasting models. No sentiment,
no Google Trends, no GDELT. If the sentiment-enhanced model cannot beat
this, we have no story.

Trains an XGBoost regressor to predict HB_HUBAVG one hour ahead.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from utils.config import FEATURES_DIR
from models.utils import (
    TARGET_REG,
    chronological_split,
    regression_report,
    select_baseline_features,
)

MODELS_DIR = Path("models/artifacts")   # defines where the trained model will be saved
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_baseline():
    print("\n=== Training Baseline Regression Model ===\n")
    df = pd.read_parquet(FEATURES_DIR / "feature_matrix_engineered_v2.parquet") 
    df = df.dropna(subset=[TARGET_REG])

    train, val, test = chronological_split(df)
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    features = select_baseline_features(df)  # excludes sentiment features
    print(f"  Using {len(features)} baseline features")

    X_train = train[features].ffill().fillna(0)
    y_train = train[TARGET_REG]       # TARGET_REG is set to "HB_HUBAVG" in models/utils.py
    X_val = val[features].ffill().fillna(0)
    y_val = val[TARGET_REG]
    X_test = test[features].ffill().fillna(0)
    y_test = test[TARGET_REG]

    model = xgb.XGBRegressor(
        n_estimators=500,  # stopped by early stopper if no improvement after 25 rounds
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8, # trained on random 80% of rows
        colsample_bytree=0.8,  # trained on random 80% of columns
        objective="reg:squarederror", # LOSS
        tree_method="hist",
        early_stopping_rounds=25,
        eval_metric="rmse",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10,
    )

    train_report = regression_report(y_train, model.predict(X_train), "baseline-train")
    val_report = regression_report(y_val, model.predict(X_val), "baseline-val")
    test_report = regression_report(y_test, model.predict(X_test), "baseline-test")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)
    print("  Top 15 features by importance:")
    for feat, imp in importance.head(15).items():
        print(f"    {feat:40s} {imp:.4f}")

    model_path = MODELS_DIR / "baseline_regressor.json"
    model.save_model(str(model_path))
    importance.to_csv(MODELS_DIR / "baseline_feature_importance.csv")
    with open(MODELS_DIR / "baseline_report.json", "w") as f:
        json.dump(
            {"train": train_report, "val": val_report, "test": test_report},
            f, indent=2
        )
    print(f"\n  Saved to {model_path}\n")
    return model, test_report


if __name__ == "__main__":
    train_baseline()
