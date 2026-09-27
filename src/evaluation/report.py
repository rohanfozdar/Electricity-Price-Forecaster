"""From stored predictions to metrics, confidence intervals, paired tests, and figures.

Only test-split rows are scored. Pooled results concatenate the test years, each
predicted by the model of its own fold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.evaluation.events import event_detection, event_metrics
from src.evaluation.statistical_tests import bootstrap_ci, diebold_mariano, holm, paired_bootstrap
from src.features.targets import extract_events

KEY = ["tag", "experiment", "horizon", "model"]


def _periods(cfg: dict) -> dict:
    return {f["name"]: tuple(pd.Timestamp(x, tz="UTC") for x in f["test"]) for f in cfg["folds"]}


def _sorted_test(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["split"] == "test"].sort_values(["fold", "origin"]).reset_index(drop=True)


def metrics_table(preds: pd.DataFrame, y: pd.Series, cfg: dict, gap_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    every = int(cfg["origins"]["every_hours"])
    periods = _periods(cfg)
    test = _sorted_test(preds)
    rows, lead_rows = [], []
    events_cache: dict = {}
    for key, g in test.groupby(KEY, sort=False):
        per_fold = []
        for fold, gf in g.groupby("fold", sort=False):
            thr = float(gf["threshold"].iloc[0])
            ev_key = (thr, gap_hours)
            if ev_key not in events_cache:
                events_cache[ev_key] = extract_events(y, thr, gap_hours)
            o = pd.DatetimeIndex(gf["origin"])
            em = event_metrics(events_cache[ev_key], o, gf["alert"].to_numpy(), gf["z"].to_numpy(),
                               int(key[2]), every, periods[fold])
            sm = M.summary(gf["z"].to_numpy(), gf["p"].to_numpy(), gf["alert"].to_numpy(), every)
            leads = em.pop("leads")
            rows.append({**dict(zip(KEY, key)), "fold": fold, "threshold": thr, **sm, **em})
            per_fold.append((em, len(gf), leads))
            lead_rows += [{**dict(zip(KEY, key)), "fold": fold, "lead_hours": x} for x in leads]
        if len(per_fold) > 1:
            sm = M.summary(g["z"].to_numpy(), g["p"].to_numpy(), g["alert"].to_numpy(), every)
            n_ev = sum(e["n_events"] for e, _, _ in per_fold)
            det = sum(e["detected_events"] for e, _, _ in per_fold)
            eps = sum(e["alert_episodes"] for e, _, _ in per_fold)
            true_eps = sum(e["event_precision"] * e["alert_episodes"] for e, _, _ in per_fold
                           if e["alert_episodes"])
            days = sum(n * every / 24 for _, n, _ in per_fold)
            false_eps = sum(e["false_alert_episodes_per_30d"] * n * every / 24 / 30 for e, n, _ in per_fold)
            all_leads = [x for _, _, ls in per_fold for x in ls]
            ge12 = sum(e["event_recall_lead_ge_12h"] * e["n_events"] for e, _, _ in per_fold if e["n_events"])
            rows.append({**dict(zip(KEY, key)), "fold": "pooled", "threshold": np.nan, **sm,
                         "n_events": n_ev, "detected_events": det,
                         "event_recall": det / n_ev if n_ev else np.nan,
                         "event_recall_lead_ge_12h": ge12 / n_ev if n_ev else np.nan,
                         "median_lead_hours": float(np.median(all_leads)) if all_leads else np.nan,
                         "missed_events": n_ev - det, "alert_episodes": eps,
                         "event_precision": true_eps / eps if eps else np.nan,
                         "false_alert_episodes_per_30d": false_eps / days * 30 if days else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(lead_rows)


def ap_intervals(preds: pd.DataFrame, cfg: dict, tags: list[str]) -> pd.DataFrame:
    st = cfg["statistics"]
    test = _sorted_test(preds[preds["tag"].isin(tags)])
    rows = []
    for key, g in test.groupby(KEY, sort=False):
        sizes = g.groupby("fold", sort=False).size().tolist()
        lo, hi = bootstrap_ci(M.average_precision, g["z"].to_numpy(), g["p"].to_numpy(), sizes,
                              st["block_length_origins"], st["bootstrap_resamples"], seed=0)
        rows.append({**dict(zip(KEY, key)), "ap_ci_low": lo, "ap_ci_high": hi})
    return pd.DataFrame(rows)


def _paired(test: pd.DataFrame, experiment: str, h: int, a: str, b: str, tag: str = "primary"):
    ga = test[(test.tag == tag) & (test.experiment == experiment) & (test.horizon == h) & (test.model == a)]
    gb = test[(test.tag == tag) & (test.experiment == experiment) & (test.horizon == h) & (test.model == b)]
    m = ga.merge(gb, on=["fold", "origin"], suffixes=("_a", "_b"))
    if not (m["z_a"] == m["z_b"]).all() or len(m) != len(ga) or len(m) != len(gb):
        raise AssertionError(f"{a} and {b} are not scored on identical rows at H={h}")
    return m.sort_values(["fold", "origin"]).reset_index(drop=True)


def hypothesis_tests(preds: pd.DataFrame, cfg: dict) -> dict:
    st = cfg["statistics"]
    test = _sorted_test(preds)
    block, B = st["block_length_origins"], st["bootstrap_resamples"]
    comparators = ["XGB_structured_news", "Chronos_only"]
    out = {"primary": {}, "secondary": []}
    horizons = sorted(test[test.tag == "primary"]["horizon"].unique())
    for h in horizons:
        for comp in comparators:
            m = _paired(test, "point_in_time", h, "Chronos_XGB_full", comp)
            sizes = m.groupby("fold", sort=False).size().tolist()
            z = m["z_a"].to_numpy()
            bs = paired_bootstrap(M.average_precision, z, m["p_a"].to_numpy(), m["p_b"].to_numpy(),
                                  sizes, block, B, seed=h)
            dm = diebold_mariano((m["p_a"] - z) ** 2, (m["p_b"] - z) ** 2, lag=max(1, h // cfg["origins"]["every_hours"]))
            rec = {"horizon": int(h), "comparator": comp, "n": int(len(m)),
                   "ap_diff": bs["diff"], "ap_ci_low": bs["ci_low"], "ap_ci_high": bs["ci_high"],
                   "ap_p": bs["p_value"], "brier_diff": dm["mean_diff"], "dm_stat": dm["stat"], "dm_p": dm["p_value"]}
            if h == 48:
                out["primary"][comp] = rec
            else:
                out["secondary"].append(rec)
    sec = pd.DataFrame(out["secondary"])
    sec["ap_p_holm"] = holm(sec["ap_p"].tolist())
    sec["dm_p_holm"] = holm(sec["dm_p"].tolist())
    out["secondary"] = sec.to_dict(orient="records")
    prim = out["primary"]
    out["primary_claim_supported"] = bool(all(r["ap_ci_low"] > 0 for r in prim.values()))
    out["primary_statement"] = st["primary_hypothesis"].strip()
    return out


def ablation_deltas(preds: pd.DataFrame, cfg: dict, tag: str = "primary") -> pd.DataFrame:
    st = cfg["statistics"]
    test = _sorted_test(preds)
    pairs = [("price", "XGB_price", "Chronos_XGB_price"),
             ("structured", "XGB_structured", "Chronos_XGB_structured"),
             ("structured_news", "XGB_structured_news", "Chronos_XGB_full")]
    rows = []
    for h in sorted(test[test.tag == tag]["horizon"].unique()):
        for fs, base, withc in pairs:
            m = _paired(test, "point_in_time", h, withc, base, tag)
            sizes = m.groupby("fold", sort=False).size().tolist()
            bs = paired_bootstrap(M.average_precision, m["z_a"].to_numpy(), m["p_a"].to_numpy(),
                                  m["p_b"].to_numpy(), sizes, st["block_length_origins"], 1000, seed=h)
            rows.append({"horizon": h, "feature_set": fs, "ap_without": M.average_precision(m["z_a"].to_numpy(), m["p_b"].to_numpy()),
                         "ap_with": M.average_precision(m["z_a"].to_numpy(), m["p_a"].to_numpy()),
                         "delta": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"]})
    return pd.DataFrame(rows)


def failure_analysis(preds: pd.DataFrame, y: pd.Series, cfg: dict, model: str = "Chronos_XGB_full",
                     h: int = 48, tag: str = "primary") -> pd.DataFrame:
    periods = _periods(cfg)
    test = _sorted_test(preds)
    rows = []
    for fold, (lo, hi) in periods.items():
        g = test[(test.tag == tag) & (test.experiment == "point_in_time") & (test.horizon == h) & (test.fold == fold)]
        if g.empty:
            continue
        thr = float(g["threshold"].iloc[0])
        ev = extract_events(y, thr, cfg["target"]["event_merge_gap_hours"])
        per_model = {}
        for mname, gm in g.groupby("model"):
            per_model[mname] = event_detection(ev, pd.DatetimeIndex(gm["origin"]), gm["alert"].to_numpy(), h, (lo, hi))
        base = per_model[model]
        for i, r in base.iterrows():
            rows.append({"fold": fold, "onset": r["onset"], "peak": r["peak"], "n_hours": r["n_hours"],
                         "month": r["onset"].month, "local_hour": r["onset"].tz_convert("US/Central").hour,
                         f"{model}_detected": r["detected"], f"{model}_lead": r["lead_hours"],
                         "caught_by_any_other": bool(any(per_model[m].loc[i, "detected"] for m in per_model if m != model))})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                         ("" if isinstance(v, float) and not np.isfinite(v) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)))
