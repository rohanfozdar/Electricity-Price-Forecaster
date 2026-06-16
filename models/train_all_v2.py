"""
Retrain all models on the v2 feature matrix.

Produces:
    baseline_regressor_v2.json
    enhanced_regressor_v2.json
    baseline_classifier_v2.json
    enhanced_classifier_v2.json
    selective_classifier_v2.json   (baseline + top 5 sentiment features only)

Plus *_report.json and *_importance.csv for each.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from utils.config import FEATURES_DIR
from models.utils import (
    TARGET_REG,
    TARGET_CLASS,
    chronological_split,
    regression_report,
    classification_report,
    select_baseline_features,
    select_enhanced_features,
)

MODELS_DIR = Path("models/artifacts")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

V2_MATRIX = FEATURES_DIR / "feature_matrix_engineered_v2.parquet"


# --- Regression ----------------------------------------------------------

def _train_regressor(df, features, name):
    train, val, test = chronological_split(df)
    X_train = train[features].ffill().fillna(0); y_train = train[TARGET_REG]
    X_val   = val[features].ffill().fillna(0);   y_val   = val[TARGET_REG]
    X_test  = test[features].ffill().fillna(0);  y_test  = test[TARGET_REG]

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", tree_method="hist",
        early_stopping_rounds=25, eval_metric="rmse",
        n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    train_r = regression_report(y_train, model.predict(X_train), f"{name}-train")
    val_r   = regression_report(y_val,   model.predict(X_val),   f"{name}-val")
    test_r  = regression_report(y_test,  model.predict(X_test),  f"{name}-test")

    importance = pd.Series(model.feature_importances_, index=features)\
                   .sort_values(ascending=False)
    print(f"  Top 15 features for {name}:")
    for feat, imp in importance.head(15).items():
        print(f"    {feat:45s} {imp:.4f}")

    model.save_model(str(MODELS_DIR / f"{name}_regressor.json"))
    importance.to_csv(MODELS_DIR / f"{name}_feature_importance.csv")
    with open(MODELS_DIR / f"{name}_report.json", "w") as f:
        json.dump({"train": train_r, "val": val_r, "test": test_r}, f, indent=2)
    return model, test_r


# --- Classification ------------------------------------------------------

def _train_classifier(df, features, name):
    train, val, test = chronological_split(df)
    X_train = train[features].ffill().fillna(0); y_train = train[TARGET_CLASS]
    X_val   = val[features].ffill().fillna(0);   y_val   = val[TARGET_CLASS]
    X_test  = test[features].ffill().fillna(0);  y_test  = test[TARGET_CLASS]

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  Positive class weight: {pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=pos_weight, tree_method="hist",
        early_stopping_rounds=30, n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba   = model.predict_proba(X_val)[:, 1]
    test_proba  = model.predict_proba(X_test)[:, 1]

    train_r = classification_report(y_train, train_proba, f"{name}-train")
    val_r   = classification_report(y_val,   val_proba,   f"{name}-val")
    test_r  = classification_report(y_test,  test_proba,  f"{name}-test")

    print(f"\n  Threshold sweep on test set ({name}):")
    sweep = {}
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sweep[thr] = classification_report(y_test, test_proba, f"{name}@{thr}", threshold=thr)

    print(f"\n  Test-set probability distribution ({name}):")
    qs = np.quantile(test_proba, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"    min={qs[0]:.3f}  Q1={qs[1]:.3f}  median={qs[2]:.3f}  Q3={qs[3]:.3f}  max={qs[4]:.3f}")

    model.save_model(str(MODELS_DIR / f"{name}_classifier.json"))
    importance = pd.Series(model.feature_importances_, index=features)\
                   .sort_values(ascending=False)
    importance.to_csv(MODELS_DIR / f"{name}_classifier_importance.csv")
    with open(MODELS_DIR / f"{name}_classifier_report.json", "w") as f:
        json.dump({"train": train_r, "val": val_r, "test": test_r,
                   "threshold_sweep": {str(k): v for k, v in sweep.items()}},
                  f, indent=2)
    return model, test_r


# --- Selective sentiment features ---------------------------------------

def _select_top_sentiment_features(top_n: int = 5) -> list[str]:
    """Read existing v1 enhanced classifier importance, pick top N
    sentiment-family columns. Falls back to a sensible default list if
    the importance file is missing.
    """
    imp_path = MODELS_DIR / "enhanced_classifier_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0).iloc[:, 0].sort_values(ascending=False)
        sentiment_prefixes = ("gdelt_",)
        ranked = [feat for feat in imp.index if feat.startswith(sentiment_prefixes)]
        return ranked[:top_n]
    # Fallback
    return ["gdelt_article_volume_lag_24h", "gdelt_article_volume",
            "gdelt_tone", "gdelt_tone_lag_24h"]


def _select_selective_features(df: pd.DataFrame, top_sentiment: list[str]) -> list[str]:
    baseline = select_baseline_features(df)
    # Map old (v1) sentiment column names to v2 equivalents where needed.
    # In v2, "gdelt_X_lag_24h" no longer exists - replace with "gdelt_X_lag_1d".
    rename_map = {
        "gdelt_article_volume_lag_24h": "gdelt_volume_lag_1d",
        "gdelt_article_volume_lag_48h": "gdelt_volume_lag_2d",
        "gdelt_tone_lag_24h": "gdelt_tone_lag_1d",
        "gdelt_tone_lag_48h": "gdelt_tone_lag_2d",
    }
    mapped = []
    for f in top_sentiment:
        f2 = rename_map.get(f, f)
        if f2 in df.columns and f2 not in mapped:
            mapped.append(f2)
        elif f in df.columns and f not in mapped:
            mapped.append(f)
    print(f"  Selective sentiment features (top {len(mapped)}): {mapped}")
    return sorted(set(baseline + mapped))


# --- Main ---------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  RETRAIN ALL MODELS ON v2 FEATURE MATRIX")
    print("=" * 70)
    if not V2_MATRIX.exists():
        raise RuntimeError(f"Run features/build_matrix_v2.py first - missing {V2_MATRIX}")

    df = pd.read_parquet(V2_MATRIX)

    # Drop rows missing the targets
    df_reg = df.dropna(subset=[TARGET_REG])
    df_clf = df.dropna(subset=[TARGET_CLASS])

    baseline_feats = select_baseline_features(df)
    enhanced_feats = select_enhanced_features(df)
    print(f"\n  Baseline features: {len(baseline_feats)}")
    print(f"  Enhanced features: {len(enhanced_feats)}")
    new_in_enhanced = sorted(set(enhanced_feats) - set(baseline_feats))
    print(f"  Enhanced-only feats: {new_in_enhanced}")

    print("\n--- Baseline regressor (v2) ---")
    _train_regressor(df_reg, baseline_feats, "baseline_v2")

    print("\n--- Enhanced regressor (v2) ---")
    _train_regressor(df_reg, enhanced_feats, "enhanced_v2")

    print("\n--- Baseline classifier (v2) ---")
    _train_classifier(df_clf, baseline_feats, "baseline_v2")

    print("\n--- Enhanced classifier (v2) ---")
    _train_classifier(df_clf, enhanced_feats, "enhanced_v2")

    print("\n--- Selective classifier (baseline + top 5 sentiment) ---")
    top_sentiment = _select_top_sentiment_features(5)
    print(f"  Top sentiment from v1 importance: {top_sentiment}")
    sel_feats = _select_selective_features(df_clf, top_sentiment)
    print(f"  Total selective features: {len(sel_feats)}")
    _train_classifier(df_clf, sel_feats, "selective_v2")


if __name__ == "__main__":
    main()
