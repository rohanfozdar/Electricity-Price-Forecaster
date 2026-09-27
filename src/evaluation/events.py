"""Event-level early-warning metrics.

An event with onset s is detected at horizon H if some alerting origin t satisfies
s - H + 1 <= t <= s (the onset hour lies in the label window [t, t+H-1]).
Lead time is s - t for the earliest such alert. Only events whose opportunity window
[s-H+1, s] lies inside the evaluated period are counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

H1 = pd.Timedelta(hours=1)


def event_detection(events: pd.DataFrame, origins: pd.DatetimeIndex, alert: np.ndarray,
                    horizon: int, period: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    lo, hi = period
    alert_times = origins[np.asarray(alert, dtype=bool)]
    rows = []
    for ev in events.itertuples(index=False):
        s = ev.onset
        first = s - (horizon - 1) * H1
        if first < lo or s >= hi:
            continue
        cand = alert_times[(alert_times >= first) & (alert_times <= s)]
        lead = float((s - cand.min()) / H1) if len(cand) else np.nan
        rows.append({"onset": s, "peak": ev.peak, "n_hours": ev.n_hours,
                     "detected": bool(len(cand)), "lead_hours": lead})
    return pd.DataFrame(rows, columns=["onset", "peak", "n_hours", "detected", "lead_hours"])


def alert_episodes(origins: pd.DatetimeIndex, alert: np.ndarray, labels: np.ndarray,
                   every_hours: int) -> pd.DataFrame:
    a = np.asarray(alert, dtype=bool)
    if not a.any():
        return pd.DataFrame(columns=["start", "end", "n_origins", "true"])
    t = origins[a]
    z = np.asarray(labels)[a]
    step = np.asarray((t[1:] - t[:-1]) / H1)
    brk = np.where(step > every_hours)[0]
    starts, ends = np.r_[0, brk + 1], np.r_[brk, len(t) - 1]
    return pd.DataFrame([{"start": t[s], "end": t[e], "n_origins": int(e - s + 1),
                          "true": bool(z[s:e + 1].max() > 0)} for s, e in zip(starts, ends)])


def event_metrics(events: pd.DataFrame, origins: pd.DatetimeIndex, alert: np.ndarray, labels: np.ndarray,
                  horizon: int, every_hours: int, period: tuple[pd.Timestamp, pd.Timestamp]) -> dict:
    det = event_detection(events, origins, alert, horizon, period)
    eps = alert_episodes(origins, alert, labels, every_hours)
    days = len(origins) * every_hours / 24
    n_ev = len(det)
    leads = det.loc[det["detected"], "lead_hours"]
    n_eps = len(eps)
    false_eps = int((~eps["true"].astype(bool)).sum()) if n_eps else 0
    return {
        "n_events": n_ev,
        "detected_events": int(det["detected"].sum()) if n_ev else 0,
        "event_recall": float(det["detected"].mean()) if n_ev else float("nan"),
        "event_recall_lead_ge_12h": float((det["lead_hours"] >= 12).mean()) if n_ev else float("nan"),
        "median_lead_hours": float(leads.median()) if len(leads) else float("nan"),
        "missed_events": int((~det["detected"]).sum()) if n_ev else 0,
        "alert_episodes": n_eps,
        "event_precision": float(eps["true"].mean()) if n_eps else float("nan"),
        "false_alert_episodes_per_30d": float(false_eps / days * 30) if days else float("nan"),
        "leads": leads.tolist(),
    }
