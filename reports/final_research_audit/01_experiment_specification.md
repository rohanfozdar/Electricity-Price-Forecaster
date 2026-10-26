# Experiment specification

This is the configuration that was executed, not a target the code was forced to match. The run is `models/run_pit_verified.py` protocol `2026-10-13-pit-verified-v1`. Outputs remain in `runs/pit_verified/`. This audit does not refit those trees.

| Decision | Executed value | Code |
| --- | --- | --- |
| Target | Hourly maximum of HB_HUBAVG 15-minute settlement prices | `src/features/pit_latency.py:hourly_max_from_intervals` |
| Issue time | Start of hour t, UTC | Row index in `models/run_pit_verified.py` |
| Horizon | One hour ahead. The outcome is the maximum inside hour t. | Same |
| Price delay | 1 hour after interval end, so the latest legal completed hour at t is t−2 | `src/features/pit_sets.py:price_history` |
| Load delay | The same one-hour assumption, applied separately | `src/features/pit_sets.py:lagged_load` |
| Day-ahead gate | 14:00 US/Central on the previous local day | `src/data/asof.py` `dam_publication` |
| Storage gate | Friday 18:00 UTC after the week-ending Friday | `configs/feature_availability.yaml` `eia_storage` |
| Calendar | US/Central. Peak is weekday 16:00–22:00 | `src/features/pit_sets.py:calendar_central` |
| Training | 2019-01-01 through 2022 | `TRAIN_START`, `VAL_START` |
| Validation | 2023, early stopping only | `VAL_START`, `TEST_START` |
| Evaluation | Frozen 2024 index, 8767 hours, hash `b850034b0690e660` | Saved predictions |
| Chronos | amazon/chronos-t5-small, context 512, 20 paths, horizon 2, step index 1 for the one-hour delay | `models/chronos_delayed_cache.py` |
| Chronos features | pred, q10, q90, mean, p_gt200, spread | `src/features/pit_sets.py` `CHRONOS_STATS` |
| XGBoost | Depth 6, learning rate 0.05, min_child_weight 1, subsample 0.8, colsample 0.8, 2000 trees, early stopping 50, `reg:absoluteerror`, `eval_metric=mae` | `models/improve_hybrid.py:make_model` |
| Seeds | 0, 1, 2, 3, 42 | `SEEDS` |
| Target transform | asinh, decoded with sinh clipped at ±10 | `decode_target` |
| Primary estimand | MAE of the seed-averaged forecast. Gap is with Chronos minus without. | `run_comparisons` |
| Uncertainty | 2000 moving-block draws, seed 0, block 24 hours, plus Diebold–Mariano with lag 24 | `src/evaluation/statistical_tests.py` |
| Spike rule | Price above $200/MWh, fixed before this audit | `models/run_benchmark.py` `SPIKE_PRICE` |

Gas and NRC are not in the primary columns. `conservative_columns` drops them. The feature list is `runs/pit_ablation/matrices/conservative_columns.json` with values rebuilt under the delay.

Earlier hours, back to 2016, enter only as lags, rolling windows, and the 512-hour Chronos context. They are not evaluation hours.
