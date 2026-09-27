"""Issue-time feature families.

Every function takes the origin grid and returns a DataFrame indexed by origin whose
values were available at the origin under configs/feature_availability.yaml. Hourly
series are on complete UTC indexes with NaN for missing hours; nothing is zero-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.data.asof import asof_join, dam_known_until
from src.features.targets import window_matrix

H1 = pd.Timedelta(hours=1)


def _at(series: pd.Series, when: pd.DatetimeIndex) -> np.ndarray:
    return series.reindex(when).to_numpy(dtype=float)


def _roll(series: pd.Series, window: int, how: str) -> pd.Series:
    r = series.rolling(window, min_periods=max(1, window // 2))
    return getattr(r, how)()


# --- price history -----------------------------------------------------------

def price_history(y: pd.Series, origins: pd.DatetimeIndex) -> pd.DataFrame:
    last = origins - H1                      # latest complete hour
    f = pd.DataFrame(index=origins)
    f["p_last"] = _at(y, last)
    f["p_lag24"] = _at(y, origins - 24 * H1)
    f["p_lag168"] = _at(y, origins - 168 * H1)
    for w in (6, 24, 72, 168, 720):
        f[f"p_max_{w}h"] = _at(_roll(y, w, "max"), last)
        f[f"p_mean_{w}h"] = _at(_roll(y, w, "mean"), last)
    for w in (24, 168):
        f[f"p_std_{w}h"] = _at(_roll(y, w, "std"), last)
    f["p_min_24h"] = _at(_roll(y, 24, "min"), last)
    for thr in (100, 200):
        exc = (y > thr).astype(float).where(y.notna())
        for w in (24, 168, 720):
            f[f"p_n_gt{thr}_{w}h"] = _at(exc.rolling(w, min_periods=1).sum(), last)
        ts = pd.Series(y.index.where(y > thr), index=y.index).ffill()
        since = pd.Series((origins - pd.DatetimeIndex(ts.reindex(last))) / H1, index=origins)
        f[f"p_hours_since_gt{thr}"] = since.clip(upper=8760).fillna(8760).to_numpy()
    return f


def calendar(origins: pd.DatetimeIndex, tz: str = "US/Central") -> pd.DataFrame:
    local = origins.tz_convert(tz)
    f = pd.DataFrame(index=origins)
    f["cal_hour"] = local.hour
    f["cal_dow"] = local.dayofweek
    f["cal_month"] = local.month
    f["cal_weekend"] = (local.dayofweek >= 5).astype(int)
    days = local.normalize().tz_localize(None)
    hol = USFederalHolidayCalendar().holidays(days.min(), days.max())
    f["cal_holiday"] = days.isin(hol).astype(int)
    doy = local.dayofyear.to_numpy()
    f["cal_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["cal_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f


# --- day-ahead ---------------------------------------------------------------

def _extend(series: pd.Series, hours: int = 400) -> pd.Series:
    idx = pd.date_range(series.index.min(), series.index.max() + hours * H1, freq="h")
    return series.reindex(idx)


def day_ahead_common(y: pd.Series, da: pd.Series, origins: pd.DatetimeIndex,
                     publish_local: str) -> pd.DataFrame:
    da = _extend(da)
    known_end = dam_known_until(origins, publish_local)
    last_day = window_matrix(da, pd.DatetimeIndex(known_end - 24 * H1).floor("h"), 24)
    f = pd.DataFrame(index=origins)
    f["da_lastday_max"] = np.nanmax(np.where(np.isnan(last_day), -np.inf, last_day), axis=1)
    f.loc[np.isnan(last_day).all(axis=1), "da_lastday_max"] = np.nan
    with np.errstate(all="ignore"):
        f["da_lastday_mean"] = np.nanmean(last_day, axis=1)
    spread = (y - da.reindex(y.index))
    f["da_rt_spread_24h"] = _at(_roll(spread, 24, "mean"), origins - H1)
    return f


def day_ahead_window(da: pd.Series, origins: pd.DatetimeIndex, horizon: int,
                     publish_local: str) -> pd.DataFrame:
    """DA prices published by t, over the part of the label window they cover."""
    known_end = dam_known_until(origins, publish_local)
    n_known = np.clip(((known_end - origins) / H1).to_numpy(), 0, horizon).astype(int)
    w = window_matrix(_extend(da), origins, horizon)
    mask = np.arange(horizon)[None, :] < n_known[:, None]
    w = np.where(mask, w, np.nan)
    f = pd.DataFrame(index=origins)
    has = ~np.isnan(w).all(axis=1)
    with np.errstate(all="ignore"):
        f["daw_max"] = np.where(has, np.nanmax(np.where(np.isnan(w), -np.inf, w), axis=1), np.nan)
        f["daw_mean"] = np.where(has, np.nanmean(w, axis=1), np.nan)
    f["daw_n_gt100"] = np.where(has, (w > 100).sum(axis=1), np.nan)
    f["daw_n_gt200"] = np.where(has, (w > 200).sum(axis=1), np.nan)
    f["daw_known_hours"] = (~np.isnan(w)).sum(axis=1)
    return f


# --- load and observed weather -------------------------------------------------

def load_features(load: pd.Series, origins: pd.DatetimeIndex) -> pd.DataFrame:
    last = origins - H1
    f = pd.DataFrame(index=origins)
    f["load_last"] = _at(load, last)
    f["load_max_24h"] = _at(_roll(load, 24, "max"), last)
    f["load_mean_24h"] = _at(_roll(load, 24, "mean"), last)
    f["load_chg_24h"] = f["load_last"] - _at(load, last - 24 * H1)
    f["load_chg_168h"] = f["load_last"] - _at(load, last - 168 * H1)
    return f


def _zone_cols(weather: pd.DataFrame, var: str) -> list[str]:
    return [c for c in weather.columns if c.endswith(f"_{var}")]


def weather_obs(weather: pd.DataFrame, origins: pd.DatetimeIndex, lag_hours: int = 1) -> pd.DataFrame:
    at = origins - lag_hours * H1
    temps = weather[_zone_cols(weather, "temperature_2m")]
    wind = weather[_zone_cols(weather, "wind_speed_10m")].mean(axis=1)
    tmin, tmax, tmean = temps.min(axis=1), temps.max(axis=1), temps.mean(axis=1)
    f = pd.DataFrame(index=origins)
    for c in temps.columns:
        f[f"wx_{c.split('_')[0].split(' ')[0].lower()}_temp"] = _at(temps[c], at)
    f["wx_tmin"] = _at(tmin, at)
    f["wx_tmax"] = _at(tmax, at)
    f["wx_tmin_24h"] = _at(_roll(tmin, 24, "min"), at)
    f["wx_tmax_24h"] = _at(_roll(tmax, 24, "max"), at)
    f["wx_wind_mean"] = _at(wind, at)
    f["wx_wind_min_24h"] = _at(_roll(wind, 24, "min"), at)
    f["wx_cdh_24h"] = _at((tmean - 75).clip(lower=0).rolling(24, min_periods=12).sum(), at)
    f["wx_hdh_24h"] = _at((50 - tmean).clip(lower=0).rolling(24, min_periods=12).sum(), at)
    return f


# --- fundamentals and news (as-of publication) --------------------------------

def fundamentals(gas: pd.Series, storage: pd.DataFrame, nrc: pd.Series,
                 origins: pd.DatetimeIndex, rules: dict) -> pd.DataFrame:
    g_now = asof_join(gas.rename("gas"), origins, rules["eia_gas"])
    g_wk = asof_join(gas.rename("gas"), origins - pd.Timedelta(days=7), rules["eia_gas"])
    s = asof_join(storage[["storage_bcf", "storage_wow", "storage_dev_5y"]], origins, rules["eia_storage"])
    n_now = asof_join(nrc.rename("nrc_off"), origins, rules["nrc_reactors"])
    n_prev = asof_join(nrc.rename("nrc_off"), origins - pd.Timedelta(days=1), rules["nrc_reactors"])
    f = pd.DataFrame(index=origins)
    f["gas_price"] = g_now["gas"].to_numpy()
    f["gas_chg_7d"] = g_now["gas"].to_numpy() - g_wk["gas"].to_numpy()
    f["gas_age_h"] = g_now["gas_age_h"].to_numpy()
    for c in ("storage_bcf", "storage_wow", "storage_dev_5y"):
        f[c] = s[c].to_numpy()
    f["nrc_offline"] = n_now["nrc_off"].to_numpy()
    f["nrc_chg_1d"] = n_now["nrc_off"].to_numpy() - n_prev["nrc_off"].to_numpy()
    return f


def news(gdelt: pd.DataFrame, origins: pd.DatetimeIndex, rule: dict) -> pd.DataFrame:
    d = gdelt.copy()
    d["tone_3d"] = d["gdelt_tone"].rolling(3, min_periods=2).mean()
    d["tone_7d"] = d["gdelt_tone"].rolling(7, min_periods=4).mean()
    for c, name in (("gdelt_tone", "tone_z30"), ("gdelt_article_volume", "vol_z30")):
        prior = d[c].shift(1).rolling(30, min_periods=10)
        d[name] = (d[c] - prior.mean()) / prior.std().replace(0, np.nan)
    cols = ["gdelt_tone", "gdelt_article_volume", "tone_3d", "tone_7d", "tone_z30", "vol_z30"]
    j = asof_join(d[cols], origins, rule)
    f = j[cols].rename(columns=lambda c: f"news_{c.replace('gdelt_', '')}")
    f["news_age_h"] = j.filter(like="_age_h").iloc[:, 0].to_numpy()
    return f


# --- window weather: GFS forecasts (operational) and realized (oracle) ---------

def lead_day_for_offset(k: np.ndarray, run_delay_h: int = 6) -> np.ndarray:
    """Smallest lead day N whose forecast for valid hour t+k was issued by t - run_delay."""
    return np.ceil((k + run_delay_h) / 24).astype(int).clip(min=1)


def forecast_temperature_cube(gfs: pd.DataFrame, origins: pd.DatetimeIndex, max_h: int,
                              run_delay_h: int = 6, max_lead: int = 7) -> np.ndarray:
    """(n_origins, max_h, n_zones) forecast temperatures usable at each origin."""
    zones = sorted({c.split("|")[0] for c in gfs.columns})
    idx = pd.date_range(gfs.index.min(), gfs.index.max() + (max_h + 1) * H1, freq="h")
    k = np.arange(max_h)
    leads = lead_day_for_offset(k, run_delay_h)
    if leads.max() > max_lead:
        raise ValueError(f"horizon {max_h} h needs lead day {leads.max()} > {max_lead}")
    cube = np.full((len(origins), max_h, len(zones)), np.nan)
    inside = (origins >= idx[0]) & (origins <= idx[-1] - max_h * H1)
    o_in = origins[inside]
    for zi, z in enumerate(zones):
        for n in np.unique(leads):
            series = gfs[f"{z}|d{n}"].reindex(idx)
            ks = k[leads == n]
            w = window_matrix(series, o_in, ks.max() + 1)
            cube[np.where(inside)[0][:, None], ks[None, :], zi] = w[:, ks]
    return cube


def realized_temperature_cube(weather: pd.DataFrame, origins: pd.DatetimeIndex, max_h: int) -> np.ndarray:
    cols = sorted(_zone_cols(weather, "temperature_2m"))
    return np.stack([window_matrix(weather[c], origins, max_h) for c in cols], axis=2)


def window_temperature_features(cube: np.ndarray, origins: pd.DatetimeIndex, horizon: int,
                                prefix: str, min_coverage: float = 0.9) -> pd.DataFrame:
    c = cube[:, :horizon, :]
    cov = (~np.isnan(c)).mean(axis=(1, 2))
    ok = cov >= min_coverage
    zmean = np.nanmean(c, axis=2) if ok.any() else np.full(c.shape[:2], np.nan)
    f = pd.DataFrame(index=origins)
    with np.errstate(all="ignore"):
        f[f"{prefix}_tmax"] = np.where(ok, np.nanmax(c, axis=(1, 2)), np.nan)
        f[f"{prefix}_tmin"] = np.where(ok, np.nanmin(c, axis=(1, 2)), np.nan)
        f[f"{prefix}_tmean"] = np.where(ok, np.nanmean(zmean, axis=1), np.nan)
        f[f"{prefix}_cdh"] = np.where(ok, np.nansum(np.clip(zmean - 75, 0, None), axis=1), np.nan)
        f[f"{prefix}_hdh"] = np.where(ok, np.nansum(np.clip(50 - zmean, 0, None), axis=1), np.nan)
    return f
