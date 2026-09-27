"""Figures for the early-warning results. All inputs are tables produced by report.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.metrics import reliability  # noqa: E402

STYLE = {"prevalence": ("#9e9e9e", ":"), "persistence_max24h": ("#8d6e63", "--"),
         "day_ahead_max": ("#6d4c41", "-."), "XGB_structured_news": ("#1f77b4", "-"),
         "Chronos_only": ("#ff7f0e", "-"), "Chronos_XGB_full": ("#d62728", "-"),
         "Blend_full": ("#9467bd", "--"), "logistic_structured": ("#2ca02c", "--"),
         "seasonal_naive_168h": ("#bcbd22", ":")}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True, "font.size": 9})


def ap_by_horizon(tab: pd.DataFrame, ci: pd.DataFrame, out: Path, experiment="point_in_time", tag="primary"):
    t = tab[(tab.tag == tag) & (tab.experiment == experiment) & (tab.fold == "pooled")]
    t = t.merge(ci, on=["tag", "experiment", "horizon", "model"], how="left")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for model, (color, ls) in STYLE.items():
        g = t[t.model == model].sort_values("horizon")
        if g.empty:
            continue
        yerr = None
        if g["ap_ci_low"].notna().all():
            yerr = np.vstack([g["average_precision"] - g["ap_ci_low"], g["ap_ci_high"] - g["average_precision"]])
        ax.errorbar(g["horizon"], g["average_precision"], yerr=yerr, color=color, ls=ls, marker="o", ms=3,
                    capsize=2, lw=1.4, label=model)
    ax.set_xlabel("Horizon H (hours)")
    ax.set_ylabel("Average precision (pooled 2023+2024 test)")
    ax.set_xticks(sorted(t.horizon.unique()))
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_title("Spike warning skill by horizon ($200/MWh, point-in-time)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def ablation(ab: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    offsets = {"price": -6, "structured": 0, "structured_news": 6}
    colors = {"price": "#7f7f7f", "structured": "#1f77b4", "structured_news": "#d62728"}
    for fs, g in ab.groupby("feature_set"):
        g = g.sort_values("horizon")
        ax.errorbar(g["horizon"] + offsets[fs], g["delta"],
                    yerr=np.vstack([g["delta"] - g["ci_low"], g["ci_high"] - g["delta"]]),
                    fmt="o", capsize=2, color=colors[fs], label=f"{fs} + Chronos")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Horizon H (hours)")
    ax.set_ylabel("AP gain from Chronos features")
    ax.set_xticks(sorted(ab.horizon.unique()))
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Does adding Chronos features help? (95% block-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def reliability_plot(preds: pd.DataFrame, out: Path, h=48, models=("XGB_structured_news", "Chronos_only",
                                                                    "Chronos_XGB_full", "day_ahead_max")):
    t = preds[(preds.tag == "primary") & (preds.split == "test") & (preds.horizon == h)]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], color="black", lw=0.8)
    for m in models:
        g = t[t.model == m]
        if g.empty:
            continue
        r = pd.DataFrame(reliability(g["z"], g["p"], bins=10))
        ax.plot(r["p_mean"], r["y_rate"], marker="o", ms=3, color=STYLE.get(m, ("k", "-"))[0], label=m)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability, H = {h} h (test, pooled)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def lead_times(leads: pd.DataFrame, out: Path, models=("day_ahead_max", "XGB_structured_news", "Chronos_XGB_full")):
    t = leads[(leads.tag == "primary") & (leads.experiment == "point_in_time")]
    hs = sorted(t.horizon.unique())
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        data = [t[(t.model == m) & (t.horizon == h)]["lead_hours"].to_numpy() for h in hs]
        pos = np.arange(len(hs)) + (i - (len(models) - 1) / 2) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(STYLE.get(m, ("#555", "-"))[0])
            b.set_alpha(0.6)
        ax.plot([], [], color=STYLE.get(m, ("#555", "-"))[0], lw=6, alpha=0.6, label=m)
    ax.set_xticks(np.arange(len(hs)), [str(h) for h in hs])
    ax.set_xlabel("Horizon H (hours)")
    ax.set_ylabel("Lead time of first warning (hours)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Lead time for detected events (test, pooled)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def event_tradeoff(tab: pd.DataFrame, out: Path, models=("persistence_max24h", "day_ahead_max",
                                                         "XGB_structured_news", "Chronos_only", "Chronos_XGB_full")):
    t = tab[(tab.tag == "primary") & (tab.experiment == "point_in_time") & (tab.fold == "pooled")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for m in models:
        g = t[t.model == m].sort_values("horizon")
        c, ls = STYLE.get(m, ("#555", "-"))
        axes[0].plot(g["horizon"], g["event_recall"], color=c, ls=ls, marker="o", ms=3, label=m)
        axes[1].plot(g["horizon"], g["false_alert_episodes_per_30d"], color=c, ls=ls, marker="o", ms=3)
    axes[0].set_ylabel("Event recall")
    axes[1].set_ylabel("False alert episodes / 30 days")
    for a in axes:
        a.set_xlabel("Horizon H (hours)")
        a.set_xticks(sorted(t.horizon.unique()))
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("Event-level operating point (threshold chosen on validation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def case_study(preds: pd.DataFrame, y: pd.Series, start: str, end: str, out: Path, h=48,
               models=("XGB_structured_news", "Chronos_only", "Chronos_XGB_full")):
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    t = preds[(preds.tag == "primary") & (preds.split == "test") & (preds.horizon == h)
              & (preds.origin >= s) & (preds.origin <= e)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    yy = y[(y.index >= s) & (y.index <= e + pd.Timedelta(hours=h))]
    a1.plot(yy.index, yy.clip(upper=3000), color="black", lw=0.8)
    a1.axhline(200, color="red", lw=0.8, ls="--")
    a1.set_ylabel("RT hub price ($/MWh, clipped at 3000)")
    for m in models:
        g = t[t.model == m].sort_values("origin")
        a2.step(g["origin"], g["p"], where="post", color=STYLE.get(m, ("#555", "-"))[0], label=m)
    a2.set_ylabel(f"P(spike in next {h} h)")
    a2.legend(fontsize=7, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
