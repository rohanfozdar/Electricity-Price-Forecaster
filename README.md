# ERCOT Electricity Price Forecasting

This study forecasts the hourly ERCOT real-time hub-average price one hour ahead. The target for hour t is `HB_HUBAVG`, the maximum of the four 15-minute hub-average settlement prices inside that hour. On the 2024 test year the gradient-boosted model has mean absolute error $11.56/MWh. Two other forecasts are reported with it: a Lasso on price lags and the day-ahead price (LEAR), and a quantile stack of five forecasts.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

The full tables, Diebold-Mariano tests, and hour-level error checks are in [reports/honest_mae.md](reports/honest_mae.md).

## Results

The test year is 8,767 hours, from 2024-01-01 00:00 UTC through 2024-12-31 06:00 UTC. It was not used to choose a training window, a penalty, or a stack weight. A spike is an hour with `HB_HUBAVG` above $200/MWh. There are 114 such hours. Units below are $/MWh.

| Model | MAE | RMSE | Spike recall | Spike precision |
| --- | ---: | ---: | ---: | ---: |
| Gradient-boosted tree | 11.56 | 99.96 | 50.00% (57 of 114) | 73.08% (57 of 78) |
| Quantile stack | 12.13 | 96.23 | 53.51% (61 of 114) | 64.21% (61 of 95) |
| LEAR | 13.39 | 97.92 | 34.21% (39 of 114) | 61.90% (39 of 63) |
| Chronos zero-shot | 14.54 | 113.62 | 41.23% | 61.04% |
| Persistence | 17.04 | 125.44 | 41.23% | 41.23% |
| Day-ahead price | 17.51 | 112.85 | 28.95% | 48.53% |

The tree and the stack were not ranked against each other before 2024 was scored. The tree has the lower MAE and the higher spike precision. The stack has the lower RMSE and the higher spike recall. Both use one evaluation. The tree is seed 42.

`python scripts/verify_artifacts.py` recomputes this table from `artifacts/honest/predictions/`.

## Information set

The table above is the historical honest-tree protocol. It does not establish that every input was published at the start of hour t. Column-level availability is in [reports/claim_audit.md](reports/claim_audit.md). Same-hour realized load, trade-date gas, week-ending storage, report-date reactor status, and archive weather are still in that fit. The matched Chronos ablation is in [reports/pit_ablation.md](reports/pit_ablation.md).

Weather is lagged one hour by the clock. The lagged fields are temperature, wind, humidity, cloud cover, precipitation, and gusts at Houston, Dallas, San Antonio, and Midland, together with the weather stress flags built from them. `stress_score` is the sum of those lagged weather flags, the gas flag, and the reactor-outage flag.

The gas-spike flag is on when Henry Hub is above $5.69/MMBtu, the 90th percentile of gas prices on hours before 2023. News columns (`gdelt_`) are not used. A same-day news average would let the first hour of a day see articles from the end of that day.

Eight further columns are in raw dollars. From the previous hour's four 15-minute settlements: the last interval, and the max, min, and mean. The max matches the hourly price lagged one hour. From the day-ahead market: the hour-to-hour change in the hub-average day-ahead price, that operating day's max and min in Central time, and the average real-time minus day-ahead spread over the 24 hours ending at t−1. The day-ahead price for hour t itself is included. It is published the previous afternoon.

Price lags, day-ahead prices, wind, precipitation, and gas enter as log1p. Missing values are forward-filled, then any that remain are set to 0. The tree's target is asinh(price). Predictions are clipped to asinh ±10, about ±$11,000/MWh, and mapped back with sinh.

Chronos-T5 small reads the previous 512 hours and stops before hour t. The tree receives the median path, the 10th and 90th percentiles, the mean, the share of paths above $200, and the gap between the 90th and 10th percentiles. Moving that block one hour later raises 2023 MAE from $31.85 to $35.78, so the alignment above is the one used.

Lagging weather, and changing nothing else, gives 2024 MAE $12.96. The same model with weather left at the forecast hour gives $12.95. The selected model, which also shortens the training window, drops the news columns, fixes the gas threshold on pre-2023 prices, and adds the eight columns, has MAE $11.56. Those choices were scored together. Their separate contributions on 2024 were not measured.

## Models

### Gradient-boosted tree

XGBoost, histogram splits, absolute error, up to 2,000 trees, depth 6, learning rate 0.05, minimum child weight 1, row and column subsampling 0.8, early stopping after 50 rounds without a gain in MAE, seed 42. There are 87 inputs: the transformed covariates without news columns, the six Chronos statistics, and the eight new columns.

Training length was chosen by 2022 MAE only. Each candidate trained on earlier years of the same length and used the last 90 days of that slice for early stopping.

| Window scored on 2022 | 2022 MAE |
| --- | ---: |
| 2022 only, trained on 2021 | 25.70 |
| 2021–2022, trained on 2020–2021 | 23.75 |
| 2019–2022, trained on 2018–2021 | 22.98 |
| Full history through 2021 | 24.97 |

The selected window is 2019–2022. The scored fit uses 34,850 hours from 1 January 2019 through 2022, with early stopping on 2023, and keeps 494 trees. The 2024 forecast is that saved model. On 2023 its MAE is $29.47.

The booster is `artifacts/honest/honest_tree_model.json`. Loading it and predicting the 2024 feature matrix matches the saved forecast with a maximum absolute error of $0.00/MWh.

### LEAR

LEAR is a Lasso on asinh(price). The inputs are asinh of the hub price at t−1, t−24, and t−168, asinh of the same-hour day-ahead price, and hour, weekday, and month dummies in Central time, standardized on the training rows. It does not use Chronos, weather, load, or the 15-minute path.

Sixteen penalties from 0.0001 to 10 were scored on 2022, each fit only on data before 2022. Alpha 0.01 has the lowest 2022 MAE, $26.34. The scored model refits that penalty on all hours before 2023.

The coefficient vector was not stored. The forecast is `artifacts/honest/predictions/lear.parquet`. The penalty grid is `artifacts/honest/lear_selection.json`.

### Quantile stack

The stack does not fit a new price model. A median regression is fit on 8,760 hours of 2023 from five forecasts already in hand: the day-ahead price, persistence (the price at t−1), LEAR, Chronos zero-shot, and the tree.

| Rule, fit on 2023 only | 2023 MAE |
| --- | ---: |
| Median of the five forecasts | 29.85 |
| Weighted mean, weights at least 0 and summing to 1 | 29.05 |
| Median regression, weights free, with an intercept | 27.90 |

The regression is the one applied to 2024. With forecasts in $/MWh,

forecast = −4.40792754681336 − 0.04855996013392374 × day-ahead − 0.2006224730968082 × persistence + 0.516222432611746 × LEAR + 0.10954905833401914 × Chronos + 0.8669434315113034 × tree

The weights sum to 1.24353 and are not constrained to be positive, so the forecast can fall outside the range of the five inputs. The constrained mean lost on 2023. These stored coefficients rebuild the saved stack forecast with a maximum absolute error of $0.00/MWh.

## Evaluation

The split is chronological. Window length and the LEAR penalty are chosen on 2022. Early stopping for the tree, and the stack weights, are chosen on 2023. The year 2024 is scored once.

Diebold-Mariano tests on absolute error, with a 24-hour HAC lag and the Harvey-Leybourne-Newbold correction, find that the tree, LEAR, and the stack each have lower loss than the day-ahead price and than persistence. Hourly errors are serially correlated, so the variance estimate allows a day of dependence. The statistics are in [reports/honest_mae.md](reports/honest_mae.md).

The forecasts are finite and are not a copy of the realized price. On hours at or below $100 the tree's mean forecast is $23.1, against a mean price of $24.4. On hours above $100 the mean forecast is $191, against a mean price of $345. Those 294 hours still hold about 60% of the tree's absolute error. The largest misses are multi-thousand-dollar spikes.

## Repository

```text
notebooks/                 benchmark notebooks, numbered 01 through 10
models/honest_mae.py       training and scoring protocol
reports/honest_mae.md      full result tables
artifacts/honest/          booster, stack coefficients, forecasts, MANIFEST.json
scripts/verify_artifacts.py
```

Training output under `runs/` is not part of the repository. `artifacts/honest/MANIFEST.json` lists each saved file, its size, and its SHA-256.

## Saved forecasts

The tree loader below was run on the 2024 feature matrix. It needs that matrix and the Chronos feature cache, which are not all in git.

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

LEAR is read from the stored forecast:

```python
import pandas as pd
lear = pd.read_parquet("artifacts/honest/predictions/lear.parquet")
test = lear.loc[lear["split"] == "test"]
```

The stack equation, applied to the five stored forecasts, matches `predictions/ensemble.parquet`:

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

From the repository root:

```bash
python scripts/verify_artifacts.py
python scripts/verify_artifacts.py --replay-tree
```

`python models/honest_mae.py check` rewrites the result report from forecasts already stored under `runs/benchmark/`. `python models/honest_mae.py all` is the full protocol. A run that already has a forecast file is left as it is.

## Data and models

Chronos-T5 small is the public checkpoint `amazon/chronos-t5-small`. The weights are not included here. Prices come from ERCOT real-time and day-ahead settlements. Load comes from ERCOT or the EIA. Weather is Open-Meteo reanalysis. Gas and storage are from the EIA. Reactor status is from the NRC. Calendar variables are constructed from the timestamp. Code is under the [MIT License](LICENSE). The data remain the property of their publishers.
