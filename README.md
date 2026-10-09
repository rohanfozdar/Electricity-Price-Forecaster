# ERCOT Electricity Price Forecasting

This project forecasts the hourly ERCOT real-time hub-average price. The target for hour t is `HB_HUBAVG`, the maximum of the four 15-minute hub-average settlement prices inside that hour. The headline result is an honest tree with 2024 mean absolute error of $11.56/MWh. Three one-hour-ahead models are provided: that tree, a Lasso (LEAR), and a quantile stack of existing forecasts.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

The long write-up is [reports/honest_mae.md](reports/honest_mae.md). How the earlier paper rows did and did not reproduce is [reports/reproduction.md](reports/reproduction.md).

## Headline results, 2024

The test year is 8,767 hours, from 2024-01-01 00:00 UTC through 2024-12-31 06:00 UTC, index hash `b850034b0690e660`. It was not used to choose a training window, a penalty, or a stack weight. A spike is an hour with `HB_HUBAVG` above $200/MWh, the same rule as `SPIKE_PRICE` in `models/run_benchmark.py`. There are 114 such hours.

| Model | MAE | RMSE | Spike recall | Spike precision |
| --- | ---: | ---: | ---: | ---: |
| Honest tree | 11.56 | 99.96 | 50.00% (57 of 114) | 73.08% (57 of 78) |
| Quantile stack | 12.13 | 96.23 | 53.51% (61 of 114) | 64.21% (61 of 95) |
| LEAR | 13.39 | 97.92 | 34.21% (39 of 114) | 61.90% (39 of 63) |
| Chronos zero-shot, rescored, not refit | 14.54 | 113.62 | 41.23% | 61.04% |
| Persistence | 17.04 | 125.44 | 41.23% | 41.23% |
| Day-ahead price, rescored | 17.51 | 112.85 | 28.95% | 48.53% |

Units are $/MWh. These are the figures printed in [reports/honest_mae.md](reports/honest_mae.md). `python scripts/verify_artifacts.py` recomputes them from `artifacts/honest/predictions/`.

## What counts as the original

There are two originals. The honest tree should be compared with the second.

The paper hybrid, reproduced here as the faithful model, is Chronos-T5 small (`amazon/chronos-t5-small`), 512 hours of history, the median of 20 paths, plus the 87 engineered columns. XGBoost uses squared error, 500 trees, depth 6, early stopping of 25 rounds, and seed 42. On this machine the 2024 MAE is $21.96. A ten-seed sweep of that XGBoost stage ranges from $18.60 to $24.64. The paper's reported MAE of $12.13 sits outside that range. Detail is in [reports/reproduction.md](reports/reproduction.md).

The improved hybrid is the model this pass started from. It had already been chosen on 2023: gap-aware Chronos-T5, six Chronos statistics, an asinh target, absolute-error XGBoost, depth 6, learning rate 0.05, and the full pre-2023 history. Seed 42 of that setup, on the original features, scores $12.95. An earlier five-seed mean is $13.06 ± $0.17. That mean was not overwritten. The loss, the depth, the learning rate, and the Chronos checkpoint were not changed to get from $12.95 to $11.56.

## What changed in the data

The v2 matrix was not rebuilt. The existing log1p transforms on price lags, day-ahead prices, wind, precipitation, and gas stay in place. Four rewrites are applied on top of that frame.

Weather is lagged one hour by the clock, not by shifting rows. The lagged fields are temperature, wind, humidity, cloud cover, precipitation, and gusts at Houston, Dallas, San Antonio, and Midland, and the weather stress flags built from them. `stress_score` is then the sum of those lagged weather flags, the gas flag, and the reactor-outage flag.

The gas-spike flag uses $5.69/MMBtu, the 90th percentile of Henry Hub on hours before 2023. It does not use a percentile computed on 2016–2024.

Every column whose name starts with `gdelt_` is removed. The repository audit records that, in the v2 matrix, `gdelt_tone` at every hour equals that UTC day's daily mean, so the first hour of a day can see news from the end of the day (`reports/repository_audit.md`).

Eight columns are added in raw dollars, not log1p. From the previous hour's four 15-minute settlements: the last interval, and the max, min, and mean. The max matches the hourly price lagged one hour. From the day-ahead market: the hour-to-hour change in the hub-average day-ahead price, that operating day's max and min in Central time, and the average real-time minus day-ahead spread over the 24 hours ending at t−1. Same-hour day-ahead prices stay. They were published the previous afternoon.

Before the tree sees a row, missing values are forward-filled and whatever remains is set to 0. The target is asinh(price). Predictions are clipped to asinh ±10, about ±$11,000/MWh, and mapped back with sinh.

Chronos was not refit. The stored 512-hour context for hour t is the slice ending at t−1. Shifting those columns one extra hour raised 2023 MAE from $31.85 to $35.78, so the existing alignment is kept.

A one-hour weather lag by itself, with no new columns and no change of training window, scores $12.96 against $12.95 for the unshifted improved hybrid. Weather lag does not explain the drop to $11.56. This pass did not separately measure the training window, the eight new columns, the gas threshold, and the GDELT removal. `plus_features` in the report is the new columns on the original information set, scored on 2023 only, and it was not given a 2024 score.

## Architectures

### Honest tree

XGBoost, histogram splits, absolute error, up to 2,000 trees, depth 6, learning rate 0.05, minimum child weight 1, row and column subsampling 0.8, early stopping after 50 rounds without a gain in MAE, seed 42. The 87 inputs are the transformed v2 columns without GDELT, plus six Chronos statistics, plus the eight new columns. The Chronos statistics are the median path, the 10th and 90th percentiles, the mean, the share of paths above $200, and the gap between the 90th and 10th percentiles.

The training window was chosen by 2022 MAE only. Each candidate trained on earlier years of the same length and used the last 90 days of that slice for early stopping.

| Window scored on 2022 | 2022 MAE |
| --- | ---: |
| 2022 only, trained on 2021 | 25.70 |
| 2021–2022, trained on 2020–2021 | 23.75 |
| 2019–2022, trained on 2018–2021 | 22.98 |
| Full history through 2021 | 24.97 |

The winner is 2019–2022. The scored model is a refit on 34,850 hours from 1 January 2019 through 2022, with early stopping on 2023. It kept 494 trees. The 2024 score uses that saved booster. It is not a second fit. 2023 MAE is $29.47, against $31.85 for the unshifted hybrid.

The booster is `artifacts/honest/honest_tree_model.json` (XGBoost's native JSON, about 3.1 MB). Loading it and predicting the 2024 feature matrix matches `predictions/honest_tree.parquet` with a maximum absolute error of $0.00/MWh.

### LEAR

LEAR is a Lasso on asinh(price). The inputs are asinh of the hub price at t−1, t−24, and t−168, asinh of the same-hour day-ahead price, and hour, weekday, and month dummies in Central time. Those columns are standardized on the training rows. There is no Chronos, weather, load, or 15-minute path.

Sixteen penalties from 0.0001 to 10 were scored on 2022. The fit for that choice used only data before 2022. Alpha 0.01 won, at 2022 MAE $26.34. The scored model is a refit of that penalty on all hours before 2023.

The coefficient vector and the standardization mean and scale were not written to disk. Writing them now would require a refit, which was not done. The verifiable artifact is `artifacts/honest/predictions/lear.parquet`. The penalty grid is `artifacts/honest/lear_selection.json`.

### Quantile stack

The stack does not train a new price model. A median regression was fit on 8,760 hours of 2023 using five forecasts that already existed: the day-ahead price, persistence (the price at t−1), LEAR, Chronos zero-shot, and the honest tree.

| Rule, fit on 2023 only | 2023 MAE |
| --- | ---: |
| Median of the five forecasts | 29.85 |
| Weighted mean, weights at least 0 and summing to 1 | 29.05 |
| Median regression, weights free, with an intercept | 27.90 |

The regression won. With forecasts in $/MWh, the stored equation is

forecast = −4.40792754681336 − 0.04855996013392374 × day-ahead − 0.2006224730968082 × persistence + 0.516222432611746 × LEAR + 0.10954905833401914 × Chronos + 0.8669434315113034 × tree

The weights sum to 1.24353. They are not constrained to be positive, so the forecast can fall outside the range of the five inputs. That is why the constrained mean lost on 2023. Rounded to three decimals the same equation is −4.41 − 0.049 × day-ahead − 0.201 × persistence + 0.516 × LEAR + 0.110 × Chronos + 0.867 × tree. The stored coefficients, not the rounded ones, rebuild the saved stack forecast with a maximum absolute error of $0.00/MWh.

## Evaluation protocol

The split is chronological. Window length and the LEAR penalty were chosen on 2022. The stack weights and the tree's early stopping were chosen on 2023. The 2024 year was scored once per finalist.

Diebold-Mariano tests on absolute error, with a 24-hour HAC lag and the Harvey-Leybourne-Newbold correction, find that the tree, LEAR, and the stack each beat both the day-ahead price and persistence. The statistics are in [reports/honest_mae.md](reports/honest_mae.md). A 24-hour lag is used because consecutive hourly errors are serially correlated.

The saved forecasts are finite and are not a copy of the realized price. On hours at or below $100 the tree's mean forecast is $23.1, against a mean price of $24.4. On hours above $100 the mean forecast is $191, against a mean price of $345.

## Caveats

The tree ($11.56 MAE) and the stack ($12.13 MAE) were not ranked against each other on 2024. The tree has lower MAE and higher spike precision. The stack has lower RMSE and higher spike recall.

The stack's $12.13 is not the paper's $12.13. The paper row has RMSE $93.98 and spike recall 61.40%. The paper table was not replaced. It remains the consolidated comparison recorded in the capstone PDF at [docs/static/pdfs/paper.pdf](docs/static/pdfs/paper.pdf).

LSTM, TFT, Prophet, TimesFM, SARIMAX, and NGBoost were not rerun. No new loss, Chronos-2, N-BEATS, Optuna search, or conformal intervals were fit. Published wind, solar, and load forecasts were not added.

The honest tree is one seed, 42. The faithful paper hybrid moved from $18.60 to $24.64 across ten seeds, so seed variance is part of any comparison with that earlier setup. The improved hybrid's five-seed band is narrower, $13.06 ± $0.17, and the honest tree was not given the same five-seed treatment.

Compare models that share an information set. The honest tree is not allowed same-hour reanalysis weather, a full-sample gas percentile, or same-day GDELT. The earlier hybrid was.

## Repository layout

```text
notebooks/          01 through 10, one benchmark family each
models/honest_mae.py
reports/honest_mae.md
reports/reproduction.md
artifacts/honest/   booster, stack coefficients, prediction files, MANIFEST.json
scripts/verify_artifacts.py
runs/               gitignored; local training output, left untouched
```

`runs/` stays ignored. The files under `artifacts/honest/` are copies. `artifacts/honest/MANIFEST.json` lists each file, its size, its SHA-256, and which published number it reproduces.

## Using the saved models

The tree snippet below was run on this machine. The reloaded booster matched the saved 2024 forecasts exactly. It needs the local v2 matrix and the Chronos feature cache, which are not all in git.

```python
import json
import xgboost as xgb
from models.honest_mae import make_honest, predict_tree

frame, _feats, _p90, _val_index, test_index = make_honest()
meta = json.loads(open("artifacts/honest/honest_tree_fit.json").read())
model = xgb.XGBRegressor()
model.load_model("artifacts/honest/honest_tree_model.json")
pred = predict_tree(
    model, meta["cfg"], frame.loc[test_index],
    meta["features"], int(meta["best_iteration"]),
)
```

LEAR has no saved coefficients. Read the forecast that was stored:

```python
import pandas as pd
lear = pd.read_parquet("artifacts/honest/predictions/lear.parquet")
test = lear.loc[lear["split"] == "test"]
```

The stack snippet was run on the committed files. The rebuilt forecast matched `predictions/ensemble.parquet` exactly.

```python
import json
import numpy as np
import pandas as pd

spec = json.loads(open("artifacts/honest/stack_coefficients.json").read())
saved = pd.read_parquet("artifacts/honest/predictions/ensemble.parquet")
saved = saved.loc[saved["split"] == "test"]
columns = []
for name in spec["components"]:
    path = "artifacts/honest/" + spec["component_files"][name]
    part = pd.read_parquet(path)
    part = part.loc[part["split"] == "test", "y_pred"].reindex(saved.index)
    columns.append(part.to_numpy(dtype=float))
forecast = spec["intercept"] + np.column_stack(columns) @ np.asarray(spec["weights"])
```

## Reproducing the numbers

From the repository root, with the project environment:

```bash
python scripts/verify_artifacts.py
python scripts/verify_artifacts.py --replay-tree   # needs the local feature matrix and Chronos cache
```

`models/honest_mae.py` is the protocol that produced the forecasts. `python models/honest_mae.py check` rewrites [reports/honest_mae.md](reports/honest_mae.md) from predictions already stored under `runs/benchmark/` and does not refit a model that already has `predictions.parquet`. `python models/honest_mae.py all` is the full protocol. It was not rerun for this packaging pass. The earlier paper-row reproduction is described in [reports/reproduction.md](reports/reproduction.md) and is launched from `models/run_benchmark.py`.

## Data and model credits

Chronos-T5 small is the public checkpoint `amazon/chronos-t5-small`. This repository does not redistribute the weights. The feature matrix is built from ERCOT real-time and day-ahead prices, ERCOT or EIA load, Open-Meteo reanalysis, EIA Henry Hub and storage, NRC reactor status, GDELT, and calendar variables. GDELT columns are not inputs to the honest tree. Code is under the [MIT License](LICENSE). The data remain the property of their publishers.
