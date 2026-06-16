"""
Spike classifier - predict whether a price spike (>$200/MWh) will occur
within the next 24 hours.

This is the most practically useful output of the whole project.
Evaluation prioritizes recall (catching spikes) over precision because
missing a real stress event costs far more than a false alarm.

Uses the enhanced feature set to directly test whether sentiment helps
on the classification task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from utils.config import FEATURES_DIR
from models.utils import (
    TARGET_CLASS,
    chronological_split,
    classification_report,
    select_baseline_features,
    select_enhanced_features,
)

MODELS_DIR = Path("models/artifacts")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _train_classifier(df, features, model_name: str):
    train, val, test = chronological_split(df)

    X_train = train[features].ffill().fillna(0)
    y_train = train[TARGET_CLASS]
    X_val = val[features].ffill().fillna(0)
    y_val = val[TARGET_CLASS]
    X_test = test[features].ffill().fillna(0)
    y_test = test[TARGET_CLASS]

    # Class imbalance handling: weight the positive (spike) class
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  Positive class weight: {pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=pos_weight,
        tree_method="hist",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    train_rep = classification_report(y_train, train_proba, f"{model_name}-train")
    val_rep = classification_report(y_val, val_proba, f"{model_name}-val")
    test_rep = classification_report(y_test, test_proba, f"{model_name}-test")

    # Try multiple thresholds to find the best recall/precision tradeoff
    print(f"\n  Threshold sweep on test set ({model_name}):")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        classification_report(y_test, test_proba, f"{model_name}@{thr}", threshold=thr)

    model_path = MODELS_DIR / f"{model_name}_classifier.json"
    model.save_model(str(model_path))
    with open(MODELS_DIR / f"{model_name}_classifier_report.json", "w") as f:
        json.dump({"train": train_rep, "val": val_rep, "test": test_rep}, f, indent=2)

    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)
    importance.to_csv(MODELS_DIR / f"{model_name}_classifier_importance.csv")
    return model, test_rep


def train_spike_classifier():
    print("\n=== Training Spike Classifiers ===\n")
    df = pd.read_parquet(FEATURES_DIR / "feature_matrix_engineered.parquet")
    df = df.dropna(subset=[TARGET_CLASS])

    print(f"\n--- Baseline classifier (no sentiment) ---")
    baseline_features = select_baseline_features(df)
    print(f"  {len(baseline_features)} features")
    _train_classifier(df, baseline_features, "baseline")

    print(f"\n--- Enhanced classifier (with sentiment) ---")
    enhanced_features = select_enhanced_features(df)
    print(f"  {len(enhanced_features)} features")
    _train_classifier(df, enhanced_features, "enhanced")


if __name__ == "__main__":
    train_spike_classifier()
