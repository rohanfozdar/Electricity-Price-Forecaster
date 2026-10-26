"""Out-of-fold first stage (E3) never uses labels from its own target year or later."""

import numpy as np
import pandas as pd
import pytest


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", "2019-12-31 23:00", freq="h", tz="UTC")
    hour = idx.tz_convert("US/Central").hour.to_numpy()
    base = 30 + 10 * np.sin(2 * np.pi * hour / 24)
    noise = np.empty(len(idx))
    noise[0] = 0
    e = rng.normal(0, 5, len(idx))
    for t in range(1, len(idx)):
        noise[t] = 0.8 * noise[t - 1] + e[t]
    price = pd.Series(base + noise, index=idx)
    da = pd.DataFrame({"da_HB_HUBAVG": base + rng.normal(0, 2, len(idx))}, index=idx)
    return price, da


ALPHAS = (1e-3, 1e-2)


def test_fold_model_is_unchanged_when_target_year_labels_change():
    from models.pes2027.features_extra import oof_lear

    price, da = _synthetic()
    pred0, folds0 = oof_lear(price, da, years=(2019,), alphas=ALPHAS)
    moved = price.copy()
    late = moved.index >= pd.Timestamp("2019-01-01 02:00", tz="UTC")
    moved[late] = moved[late] * 3 + 500
    pred1, folds1 = oof_lear(moved, da, years=(2019,), alphas=ALPHAS)
    assert folds0[0]["alpha"] == folds1[0]["alpha"]
    assert folds0[0]["inner_val_mae"] == folds1[0]["inner_val_mae"]
    first = pd.date_range("2019-01-01 00:00", periods=3, freq="h", tz="UTC")
    np.testing.assert_array_equal(pred0.loc[first].to_numpy(), pred1.loc[first].to_numpy())


def test_oof_value_ignores_prices_after_the_legal_hour():
    from models.pes2027.features_extra import oof_lear

    price, da = _synthetic(1)
    t = pd.Timestamp("2019-06-15 12:00", tz="UTC")
    pred0, _ = oof_lear(price, da, years=(2019,), alphas=ALPHAS)
    moved = price.copy()
    moved[moved.index >= t - pd.Timedelta(hours=1)] += 1000.0
    pred1, _ = oof_lear(moved, da, years=(2019,), alphas=ALPHAS)
    assert pred0.loc[t] == pred1.loc[t]
    assert pred0.loc[t + pd.Timedelta(hours=3)] != pred1.loc[t + pd.Timedelta(hours=3)]


def test_every_fold_trains_before_its_year():
    from models.pes2027.features_extra import oof_lear

    price, da = _synthetic(2)
    _, folds = oof_lear(price, da, years=(2018, 2019), alphas=ALPHAS)
    for f in folds:
        assert pd.Timestamp(f["train_end"]) < pd.Timestamp(f"{f['year']}-01-01", tz="UTC")
        assert f["n_inner_val"] > 8000
