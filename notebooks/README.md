# Model Experiments

Index of the benchmark notebooks. The consolidated numbers in the root [README](../README.md) are a separate reported comparison. "Preserved output" means a saved notebook result, not that result's agreement with the consolidated table.

Inputs are the committed v2 matrix, `data/features/feature_matrix_engineered_v2.parquet`, unless a row says otherwise. The split is chronological: 2016–2022 train, 2023 validation, 2024 test.

## Structured Machine Learning

| Model | Notebook | Inputs | Regime | Preserved output |
| --- | --- | --- | --- | --- |
| XGBoost | [01_xgboost.ipynb](01_xgboost.ipynb) | Multimodal covariates | Supervised | Yes. With GDELT: MAE 22.21, RMSE 108.88, recall 54.39%, precision 30.24%, PR-AUC 0.422. This is the matched ablation, not the consolidated XGBoost row (MAE 24.43). |
| LightGBM | [02_lightgbm.ipynb](02_lightgbm.ipynb) | Multimodal covariates | Supervised | Yes. MAE 19.76, RMSE 100.41, recall 34.21%, precision 45.35%. Matches the consolidated row. |
| CatBoost | [03_catboost.ipynb](03_catboost.ipynb) | 63-feature subset of the matrix | Supervised | Yes, on that subset: MAE 20.90, RMSE 101.54, recall 44.74%, precision 48.11%. The consolidated row (MAE 13.75) is a different run. |
| NGBoost | [04_ngboost.ipynb](04_ngboost.ipynb) | Multimodal covariates | Supervised | No. The regressor cell was not executed. |

## Deep Sequence Models

| Model | Notebook | Inputs | Regime | Preserved output |
| --- | --- | --- | --- | --- |
| LSTM | [05_lstm.ipynb](05_lstm.ipynb) | Multivariate windows | Supervised | Yes. MAE 32.45, RMSE 134.37, recall 15.79%, precision 10.84%. Differs from the consolidated row (MAE 24.42). |
| Temporal Fusion Transformer | [06_tft.ipynb](06_tft.ipynb) | Multivariate sequence | Supervised | Yes. MAE 25.61, RMSE 132.76, recall 0%, precision 0%. Differs from the consolidated row (MAE 45.75). |

## Foundation Models

| Model | Notebook | Inputs | Regime | Preserved output |
| --- | --- | --- | --- | --- |
| Chronos | [07_chronos_and_hybrid.ipynb](07_chronos_and_hybrid.ipynb) | Price history | Zero-shot | Yes, for the zero-shot section: MAE 14.86, RMSE 116.42, recall 41.23%, precision 59.49%. Differs from the consolidated row (MAE 14.72). |
| TimesFM | [08_timesfm.ipynb](08_timesfm.ipynb) | Price history | Zero-shot | Yes. MAE 32.72, RMSE 138.07, recall 2.63%, precision 10.34%. Differs from the consolidated row (MAE 32.51). |

## Classical Models

| Model | Notebook | Inputs | Regime | Preserved output |
| --- | --- | --- | --- | --- |
| Prophet | [09_prophet_and_hybrid.ipynb](09_prophet_and_hybrid.ipynb) | Price history, seasonality, and a regressor subset | Series-specific | No. The notebook was not executed. |
| SARIMAX | [10_sarimax.ipynb](10_sarimax.ipynb) | Price history, with an exogenous variant | Series-specific | No. The notebook was not executed. |

## Hybrids

| Model | Notebook | Inputs | Regime | Preserved output |
| --- | --- | --- | --- | --- |
| Chronos–XGBoost | [07_chronos_and_hybrid.ipynb](07_chronos_and_hybrid.ipynb) | Chronos forecast plus multimodal covariates | Late fusion | No. The hybrid cells have no saved output. `models/train_chronos_hybrid.py` is the script form of the same design. The consolidated MAE 12.13 is not a saved notebook result. |
| Prophet–XGBoost | [09_prophet_and_hybrid.ipynb](09_prophet_and_hybrid.ipynb) | Prophet forecast plus multimodal covariates | Late fusion | No. The hybrid section was not executed. |
| TimesFM–XGBoost | `timesfm_xgboost.ipynb` (gitignored) | TimesFM forecast plus multimodal covariates | Late fusion | Yes, and not part of the consolidated table: MAE 23.52, RMSE 120.15, recall 57.89%, precision 30.28%. |

## Data construction and diagnostics

These notebooks are local working files and are gitignored.

| Notebook | Role |
| --- | --- |
| `data_inspection.ipynb` | Profiles the raw pulls |
| `build_raw_matrix.ipynb` | Walks through `features/build_matrix_v2.py` |
| `build_engineered_matrix.ipynb` | Walks through `features/engineer_v2.py` |
| `residual_diagnostics.ipynb` | Source of the log1p transforms in `models/utils.py` |
| `feature_diagnostics.ipynb` | Residual and transform checks on XGBoost |
| `rag_forecaster.ipynb` | Analog retrieval experiment, outside the twelve-model table |

`run_experiment_gpu.ipynb` trains the early-warning models on a GPU server. It is not one of the twelve benchmark configurations. The Chronos cache builder is `scripts/run_chronos_gpu.py`.
