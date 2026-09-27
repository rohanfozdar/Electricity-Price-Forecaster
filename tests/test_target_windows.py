import numpy as np
import pandas as pd

from src.evaluation.events import alert_episodes, event_detection
from src.features.targets import extract_events, origin_grid, window_labels, window_max


def _flat(n=400):
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(20.0, index=idx)


def test_window_covers_t_through_t_plus_h_minus_1():
    y = _flat()
    t = y.index[100]
    origins = pd.DatetimeIndex([t])
    for offset, expected in ((-1, 0.0), (0, 1.0), (23, 1.0), (24, 0.0)):
        y2 = y.copy()
        y2.iloc[100 + offset] = 500.0
        assert window_labels(y2, origins, 24, 200.0).iloc[0] == expected, offset


def test_incomplete_window_is_missing_not_negative():
    y = _flat()
    y.iloc[110] = np.nan
    origins = pd.DatetimeIndex([y.index[100], y.index[-10]])
    z = window_labels(y, origins, 24, 200.0)
    assert z.isna().all()
    assert window_max(y, origins, 5).notna().tolist() == [True, True]


def test_origin_grid_alignment():
    y = _flat(1000)
    o = origin_grid(y, 6, "2024-01-01T03:00Z")
    assert (o.hour % 6 == 0).all() and o[0] >= pd.Timestamp("2024-01-01T03:00Z")


def test_event_merge_gap():
    y = _flat()
    for h in (10, 17, 30):          # 10 and 17 are 6 quiet hours apart; 30 is 12 apart from 17
        y.iloc[h] = 300.0
    ev = extract_events(y, 200.0, 6)
    assert len(ev) == 2 and ev.iloc[0]["n_hours"] == 2
    assert len(extract_events(y, 200.0, 12)) == 1


def test_event_detection_lead_time():
    y = _flat()
    y.iloc[204] = 300.0                                  # 12:00 UTC, on the 6-hourly grid offsets
    ev = extract_events(y, 200.0, 6)
    origins = pd.date_range(y.index[0], y.index[-1], freq="6h")
    s = y.index[204]
    alert = (origins == s - pd.Timedelta(hours=18)) | (origins == s - pd.Timedelta(hours=60))
    det = event_detection(ev, origins, alert, 48, (y.index[0], y.index[-1]))
    assert det["detected"].iloc[0] and det["lead_hours"].iloc[0] == 18.0
    det72 = event_detection(ev, origins, alert, 72, (y.index[0], y.index[-1]))
    assert det72["lead_hours"].iloc[0] == 60.0


def test_alert_episodes():
    origins = pd.date_range("2024-01-01", periods=10, freq="6h", tz="UTC")
    alert = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])
    z = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    ep = alert_episodes(origins, alert, z, 6)
    assert len(ep) == 3 and ep["true"].tolist() == [True, False, False]
