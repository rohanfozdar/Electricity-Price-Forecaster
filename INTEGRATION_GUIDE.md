# ERCOT Grid Stress Forecaster — V1 Integration Guide

## What's in this zip

This zip contains the full first iteration of the project, covering
Steps 4–25 from the project plan. Your existing project already has
Steps 1–3 done (project setup, RT prices, DA prices).

### Files to copy into your existing `ercot-grid-forecaster/` directory:

```
pipelines/
  weather.py              # Step 4: Open-Meteo weather for 4 ERCOT zones
  eia_gas.py              # Step 6: Henry Hub natural gas spot prices
  eia_storage.py          # Step 7: Weekly gas storage reports
  nrc_reactors.py         # Step 8: Texas nuclear reactor status
  ercot_load.py           # Step 9: ERCOT load forecast & actual
  gdelt.py                # Step 10: GDELT news sentiment
  reddit_sentiment.py     # Step 12: Reddit sentiment via PRAW + VADER

features/
  build_matrix.py         # Step 13: Merge all sources into feature matrix
  engineer.py             # Steps 15-16: Lags, rolling stats, stress flags

models/
  utils.py                # Shared: train/test split, feature groups, metrics
  train_baseline.py       # Step 17: Weather + price only (XGBoost)
  train_enhanced.py       # Step 18: All features incl. sentiment (XGBoost)
  train_classifier.py     # Step 19: 24h spike classifier (baseline + enhanced)

evaluation/
  granger.py              # Step 20: Granger causality tests
  backtest.py             # Step 21: Backtest against known stress events
  benchmark_dam.py        # Step 22: Head-to-head vs day-ahead market

dashboard/
  app.py                  # Steps 24-25: Streamlit dashboard

run_full_pipeline.py      # Master orchestrator
```

## Setup before running

### 1. Install additional dependencies

In your project's virtual environment:

```bash
cd ercot-grid-forecaster
source .venv/bin/activate
pip install vaderSentiment
```

All other dependencies (xgboost, statsmodels, streamlit, praw,
plotly, etc.) should already be installed from Step 1.

### 2. Set up API keys

Copy `.env.example` to `.env` and fill in:

```
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here
REDDIT_USER_AGENT=ercot-forecaster/0.1
EIA_API_KEY=your_key_here
```

**Reddit:** Free at https://www.reddit.com/prefs/apps (create a "script" app)
**EIA:** Free at https://www.eia.gov/opendata/register.php

If you don't have these keys yet, the orchestrator will skip those
pipelines and continue with the rest. Weather, GDELT, and ERCOT load
don't need keys.

### 3. Copy the files

Copy every file from this zip into the corresponding directory in your
existing `ercot-grid-forecaster/` project. Don't overwrite your existing
`pipelines/base.py`, `utils/config.py`, or `utils/helpers.py` — the new
files import from those.

## How to run

### Option A: Run everything at once

```bash
python run_full_pipeline.py --stage all
```

This runs pipelines → features → models → evaluation in order.
Expect 30–60 minutes total (weather and NRC are the slowest pipelines).

### Option B: Run stage by stage

```bash
# 1. Pull all data (skip RT/DA since you already have them)
python run_full_pipeline.py --stage pipelines

# 2. Build and engineer features
python run_full_pipeline.py --stage features

# 3. Train all models
python run_full_pipeline.py --stage models

# 4. Run evaluation (Granger, backtest, DAM benchmark)
python run_full_pipeline.py --stage evaluate

# 5. Launch the dashboard
python run_full_pipeline.py --stage dashboard
```

### Option C: Run individual pipelines

Each pipeline file can be run standalone:

```bash
python -m pipelines.weather
python -m pipelines.eia_gas
python -m pipelines.gdelt
# etc.
```

## What to expect

### Pipelines that will definitely work without keys:
- Weather (Open-Meteo) — ~2 min
- ERCOT Load (gridstatus) — ~3 min
- GDELT (free API) — ~5 min

### Pipelines that need API keys:
- EIA Gas + Storage — need EIA_API_KEY
- Reddit — needs Reddit app credentials

### Pipelines that are slow:
- NRC Reactors — ~1 request per day × 1,461 days = ~8 min with rate limiting

### If a pipeline fails:
The orchestrator logs the error and continues. You can always rerun a
single pipeline later. The feature matrix builder (`build_matrix.py`)
is designed to work with whatever raw files exist — it skips any
missing sources and builds the matrix from what's available.

## After everything runs

Your `data/` directory will contain:
```
data/raw/
  ercot_rt_prices.parquet     (already exists from Step 2)
  ercot_da_prices.parquet     (already exists from Step 3)
  weather.parquet
  eia_gas.parquet
  eia_storage.parquet
  nrc_reactors.parquet
  ercot_load.parquet
  gdelt.parquet
  reddit.parquet

data/features/
  feature_matrix_raw.parquet
  feature_matrix_engineered.parquet
```

Your `models/artifacts/` directory will contain:
```
  baseline_regressor.json
  enhanced_regressor.json
  baseline_classifier.json
  enhanced_classifier.json
  *_feature_importance.csv
  *_report.json
```

Your `evaluation/` directory will contain:
```
  granger_results.json
  backtest_results.json
  dam_benchmark.json
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'pipelines'"**
Make sure you're running from the `ercot-grid-forecaster/` root directory.

**"EIA_API_KEY not set"**
Create a `.env` file with your key. See step 2 above.

**"vaderSentiment not installed"**
Run: `pip install vaderSentiment`

**NRC fetch is very slow**
It's fetching one text file per day for 4 years. This is normal.
You can reduce the date range by editing DATA_START in config.py.

**Feature matrix build skips sources**
This is by design. It builds with whatever data is available.
Get more pipelines working and rebuild: `python run_full_pipeline.py --stage features`
