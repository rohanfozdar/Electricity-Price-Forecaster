# A Multimodal Framework for Electricity Price Forecasting Using Sentiment-Augmented Time-Series Signals

Companion codebase for my capstone paper on **next-hour electricity price forecasting in ERCOT**, the Texas electricity market.

This project builds an hourly forecasting pipeline using **price history, system load, weather, natural gas data, nuclear outage status, and GDELT news sentiment** to predict the ERCOT hub-average real-time price:

```text
HB_HUBAVG ($/MWh)
```

The core question:

> Can multimodal signals — especially news sentiment — improve electricity price forecasting in a market dominated by rare but extreme price spikes?

---

## Why ERCOT?

ERCOT prices are highly nonlinear and heavy-tailed. Most hours are ordinary, but a small number of grid-stress events dominate the economics.

From 2016–2024:

* Median real-time price: about **$30/MWh**
* Spike threshold used here: **>$200/MWh**
* Spikes make up roughly **2.5%** of hourly observations
* Prices can surge toward the **$9,000/MWh** regulatory cap
* Negative prices occur during renewable oversupply conditions

This makes ERCOT forecasting less about predicting the average hour and more about handling the tail.

---

## What the Code Does

The repository implements the full forecasting workflow:

```text
raw data → hourly feature matrix → model training → benchmark evaluation → dashboard
```

Main folders:

```text
pipelines/       Data ingestion scripts
features/        Feature matrix construction and feature engineering
models/          Training utilities and metric functions
notebooks/       Model-specific experiments
evaluation/      Benchmark comparisons and backtests
dashboard/       Streamlit dashboard
data/features/   Engineered feature matrix
tests/           Unit tests
```

The final feature matrix contains:

```text
77,956 hourly observations × 87 features
```

All models use the same chronological split:

| Split      |     Years |
| ---------- | --------: |
| Train      | 2016–2022 |
| Validation |      2023 |
| Test       |      2024 |

No random shuffling is used.

---

## Models Benchmarked

The project compares twelve main forecasting architectures across five families, plus one no-sentiment XGBoost ablation.

| Family                      | Models                               |
| --------------------------- | ------------------------------------ |
| Gradient Boosted Trees      | XGBoost, LightGBM, CatBoost, NGBoost |
| Deep Learning               | LSTM, Temporal Fusion Transformer    |
| Zero-Shot Foundation Models | Chronos, TimesFM                     |
| Classical Time Series       | SARIMAX, Prophet                     |
| Hybrid Models               | Chronos + XGBoost, Prophet + XGBoost |
| Ablation                    | XGBoost without sentiment            |

The hybrid models append a foundation-model forecast as an additional feature before training XGBoost.

---

## Results

All metrics are reported on the held-out **2024 test set**.

Spike metrics use:

```text
HB_HUBAVG > $200/MWh
```

| Rank | Model                 | Family        |       MAE |      RMSE | Spike Recall | Spike Precision |
| :--: | --------------------- | ------------- | --------: | --------: | -----------: | --------------: |
|   1  | **Chronos Hybrid**    | Hybrid        | **12.13** | **93.98** |       61.40% |          50.43% |
|   2  | **CatBoost**          | GBT           |     13.75 |     95.48 |       44.35% |          56.94% |
|   3  | NGBoost               | GBT           |     14.59 |    106.44 |       25.44% |          59.18% |
|   4  | Chronos               | Foundation    |     14.72 |    115.59 |       42.11% |      **60.76%** |
|   5  | SARIMAX               | Classical     |     18.45 |    124.31 |       45.61% |          38.81% |
|   6  | LightGBM              | GBT           |     19.76 |    100.41 |       34.21% |          45.35% |
|   7  | LSTM                  | Deep Learning |     24.42 |    148.72 |       16.67% |          23.46% |
|   8  | XGBoost               | GBT           |     24.43 |    112.46 |       57.02% |          25.10% |
|   9  | TimesFM               | Foundation    |     32.51 |    136.94 |        1.75% |          11.76% |
|  10  | XGBoost, no sentiment | Ablation      |     34.53 |    122.57 |       50.53% |          19.54% |
|  11  | TFT                   | Deep Learning |     45.75 |    134.60 |        0.00% |           0.00% |
|  12  | Prophet Hybrid        | Hybrid        |     48.36 |    108.96 |   **71.05%** |          14.16% |
|  13  | Prophet               | Classical     |    119.14 |    168.75 |       57.14% |          17.70% |

---

## Key Takeaways

### 1. The best model is not purely supervised or purely zero-shot

The strongest result comes from the **Chronos + XGBoost hybrid**.

Chronos provides a univariate temporal forecast. XGBoost then combines that forecast with the full physical, market, weather, outage, and sentiment feature set.

```text
Best MAE:  $12.13/MWh
Best RMSE: $93.98/MWh
```

### 2. CatBoost is the best standalone supervised model

CatBoost is the strongest non-hybrid model and the best standalone gradient-boosted tree.

```text
CatBoost MAE:  $13.75/MWh
CatBoost RMSE: $95.48/MWh
```

Tree-based models handle ERCOT’s nonlinear feature interactions better than the deep learning models tested here.

### 3. News sentiment matters

The XGBoost ablation shows the value of GDELT sentiment features.

| Model                     |   MAE |   RMSE | Spike Recall | Spike Precision |
| ------------------------- | ----: | -----: | -----------: | --------------: |
| XGBoost without sentiment | 34.53 | 122.57 |       50.53% |          19.54% |
| XGBoost with sentiment    | 24.43 | 112.46 |       57.02% |          25.10% |

Adding sentiment reduces MAE by:

```text
$10.10/MWh, or 29.3%
```

### 4. Deep learning underperforms on this dataset

The LSTM performs reasonably on MAE but misses most spikes. The Temporal Fusion Transformer produces zero spike recall on the 2024 test set.

The issue is not model capacity alone. ERCOT spike behavior is rare, nonlinear, and event-driven.

### 5. Spike prediction is a tradeoff

Different models fail in different ways:

* **Chronos** is precise but misses some spikes.
* **Prophet Hybrid** catches many spikes but creates many false alarms.
* **Chronos Hybrid** gives the best overall balance.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

The engineered feature matrix is included so the notebooks can run without rebuilding every raw data source.

---

## Usage

Run the full pipeline:

```bash
python run_full_pipeline.py
```

Open a model notebook:

```bash
jupyter lab "notebooks/06_catboost (1).ipynb"
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

---

## Paper

This repository supports the paper:

**A Multimodal Framework for Electricity Price Forecasting Using Sentiment-Augmented Time-Series Signals**

The paper contains the full methodology, feature construction details, model descriptions, and discussion of results.


---



**Rohan Fozdar**
Knox College  
Mathematics Capstone Project

