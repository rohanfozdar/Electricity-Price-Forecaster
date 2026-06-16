# Electricity Price Forecaster

**A mathematics capstone project — forecasting real-time electricity prices on the ERCOT (Texas) grid.**

This project benchmarks **13 forecasting models** spanning gradient-boosted trees, deep learning, pretrained time-series foundation models, classical statistical methods, and hybrid approaches — all evaluated under identical conditions on the same data. The goal is two-fold: predict the **next-hour real-time price** (`HB_HUBAVG`, $/MWh) and flag **price spikes** (> $200/MWh), which are the rare, high-impact events that dominate grid economics.

---

## Why this problem is hard

ERCOT real-time prices have one of the most extreme distributions in any electricity market:

- Median price ≈ **$30/MWh**, but spikes exceed **$1,000** and can hit the **~$9,000/MWh** regulatory cap.
- Roughly **2.5%** of hours are spikes (> $200/MWh); about **1%** of hours are *negatively* priced (renewable oversupply).
- The target distribution has skewness > 10 and kurtosis > 100.

A squared-error objective is dominated by a handful of spike hours. This single fact drives every modeling decision in the project — the choice of loss function, the log-transformation of skewed inputs, and the evaluation metrics themselves.

---

## Results

All models were evaluated on a **held-out 2024 test set** (chronological split — no shuffling) using identical features and metric definitions. Spike recall/precision are measured at the **$200/MWh** threshold.

**Leaderboard (ordered by test MAE, lower is better):**

| Rank | Model | Family | MAE ($/MWh) | RMSE ($/MWh) | Spike Recall | Spike Precision |
|:----:|-------|--------|:-----------:|:------------:|:------------:|:---------------:|
| 1 | **Chronos Hybrid** | Hybrid | **12.13** | **93.98** | 61.40% | 50.43% |
| 2 | **CatBoost** | Gradient Boosted Trees | 13.75 | 95.48 | 44.35% | 56.94% |
| 3 | NGBoost | Gradient Boosted Trees | 14.59 | 106.44 | 25.44% | 59.18% |
| 4 | Chronos (Amazon) | Foundation (zero-shot) | 14.72 | 115.59 | 42.11% | **60.76%** |
| 5 | SARIMAX | Classical | 18.45 | 124.31 | 45.61% | 38.81% |
| 6 | LightGBM | Gradient Boosted Trees | 19.76 | 100.41 | 34.21% | 45.35% |
| 7 | LSTM | Deep Learning | 24.42 | 148.72 | 16.67% | 23.46% |
| 8 | XGBoost (sentiment-enhanced) | Gradient Boosted Trees | 24.43 | 112.46 | 57.02% | 25.10% |
| 9 | TimesFM (Google) | Foundation (zero-shot) | 32.51 | 136.94 | 1.75% | 11.76% |
| 10 | XGBoost (no sentiment) | Gradient Boosted Trees | 34.53 | 122.57 | 50.53% | 19.54% |
| 11 | TFT | Deep Learning | 45.75 | 134.60 | 0.00% | 0.00% |
| 12 | Prophet Hybrid | Hybrid | 48.36 | 108.96 | **71.05%** | 14.16% |
| 13 | Prophet (Meta) | Classical | 119.14 | 168.75 | 57.14% | 17.70% |

### Headline findings

- **The best overall model is the Chronos + XGBoost hybrid** — feeding Amazon's pretrained Chronos forecast in as a feature to XGBoost beats every standalone model on both MAE ($12.13) and RMSE ($93.98).
- **CatBoost is the strongest single model**, and gradient-boosted trees dominate the top of the table.
- **News sentiment improves the XGBoost baseline**: adding GDELT sentiment features cuts MAE from $34.53 to $24.43 and lifts spike precision from 19.5% to 25.1%.
- **There is a clear precision/recall tradeoff for spike detection.** Models like Chronos and NGBoost achieve high precision (you can trust their spike calls), while the Prophet hybrid achieves the highest recall (it catches the most spikes) at the cost of many false alarms.

---

## Models benchmarked

**Gradient Boosted Trees**
- **XGBoost** (with and without sentiment features — the project's ablation baseline)
- **LightGBM** — Microsoft's histogram-based GBT
- **CatBoost** — Yandex's ordered-boosting GBT
- **NGBoost** — probabilistic boosting (predicts a full price distribution)

**Deep Learning**
- **LSTM** — bidirectional recurrent network over a 168-hour look-back window
- **Temporal Fusion Transformer (TFT)** — attention-based multi-horizon forecaster

**Foundation Models (zero-shot)**
- **Chronos** (Amazon) — pretrained time-series foundation model
- **TimesFM** (Google) — pretrained decoder-only forecasting model

**Classical Time Series**
- **SARIMAX** — seasonal ARIMA with exogenous regressors (daily seasonality, m = 24)
- **Prophet** (Meta) — additive trend/seasonality decomposition with regressors

**Hybrid**
- **Chronos + XGBoost** and **Prophet + XGBoost** — the foundation/classical forecast is appended as an input feature to a gradient-boosted tree, combining a strong univariate prior with the full engineered feature set.

---

## Data sources

Six independent signal families, joined on an hourly timestamp from 2016–2024 (~78,000 observations):

| Source | Signals | API key |
|--------|---------|:-------:|
| **ERCOT** | Real-time + day-ahead hub prices, system load | No |
| **Open-Meteo** | Weather (temp, humidity, cloud, wind, precip) across 4 Texas cities | No |
| **EIA** | Henry Hub natural gas spot price, storage levels | local .xls |
| **GDELT** | News sentiment (tone, normalized tone, article volume) | No |
| **NRC** | Nuclear reactor outage status | No |

---

## Methodology

**Feature engineering** (`features/engineer_v2.py`): price lags (1, 12, 24, 48, 168, 720 h), rolling mean/std (24 h, 168 h), grid-stress flags (cold/freeze/heat/low-wind/gas/reactor + composite score), per-zone weather aggregates, and calendar features. The final matrix is **77,956 rows × 95 columns**, of which **87 are used as features**.

**Transformations** (`models/utils.py: apply_feature_transforms`): `log1p` applied to 34 heavily right-skewed columns (price lags, rolling stats, wind speed/gusts, day-ahead prices, gas/storage, precipitation) to stabilize variance and tame the heavy tail.

**Evaluation**: a strict chronological split prevents look-ahead leakage — **train 2016–2022, validate 2023, test 2024**. Every model is judged on the same splits with the same metric functions, making this a true apples-to-apples bake-off.

---

## Repository structure

```
pipelines/     → one ingestion script per data source
features/      → raw-matrix build + feature engineering
models/        → training scripts, shared utilities, metrics
evaluation/    → backtesting and benchmark comparisons
notebooks/     → one notebook per benchmarked model
dashboard/     → Streamlit app for interactive exploration
utils/         → shared helpers (config, date alignment, logging)
data/features/ → the engineered feature matrix (parquet)
tests/         → unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure API keys (only EIA is required, and only for regenerating gas data)
cp .env.example .env
```

## Usage

```bash
# Rebuild the full data → features → models → evaluation pipeline
python run_full_pipeline.py

# Or explore an individual model interactively
jupyter lab "notebooks/06_catboost (1).ipynb"

# Launch the dashboard
streamlit run dashboard/app.py
```

The canonical engineered feature matrix (`data/features/feature_matrix_engineered_v2.parquet`) is included so the model notebooks run out of the box; the raw intermediate matrices are regenerable via the pipeline.

---

## Notes & caveats

- Reported metrics are taken from the capstone presentation. The 2024 test window contains only ~29 spike hours, so spike recall/precision deltas of a few points are within sampling noise; MAE and RMSE (8,767 rows) are statistically robust.
- This repository contains the models presented in the capstone. Some exploratory work (a retrieval-augmented/analog forecaster and additional diagnostic notebooks) is intentionally omitted.

---

*Mathematics capstone project. Author: Rohan Fozdar.*
