"""
Generate every figure used by the project page and the README.

All figures are derived from files that ship with the repository, so a fresh
clone can regenerate the whole set:

    data/features/feature_matrix_engineered_v2.parquet   the canonical matrix
    models/artifacts/*_feature_importance.csv            optional, see below

PAPER_LEADERBOARD is the consolidated capstone comparison, including the
unmatched no-sentiment row. It is not the matched GDELT ablation in
notebooks/01_xgboost.ipynb (MAE 24.64 without GDELT, 22.21 with GDELT).
Do not edit the table to match a later re-run.

Output goes to docs/static/images/, which the project page and README both read.

Usage:
    python figures/make_figures.py                  # all figures
    python figures/make_figures.py --only teaser    # one figure
    python figures/make_figures.py --list           # show figure names
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.utils import TARGET_REG  # noqa: E402

MATRIX = ROOT / "data/features/feature_matrix_engineered_v2.parquet"
ARTIFACTS = ROOT / "models/artifacts"
OUT_DIR = ROOT / "docs/static/images"

SPIKE_THRESHOLD = 200.0
TEST_START = "2024-01-01"

# Published capstone results on the 2024 test set. Do not edit.
PAPER_LEADERBOARD = [
    # name,                  family,          MAE,     RMSE,   recall, precision
    ("Chronos Hybrid",       "Hybrid",         12.13,   93.98,  0.6140, 0.5043),
    ("CatBoost",             "GBT",            13.75,   95.48,  0.4435, 0.5694),
    ("NGBoost",              "GBT",            14.59,  106.44,  0.2544, 0.5918),
    ("Chronos",              "Foundation",     14.72,  115.59,  0.4211, 0.6076),
    ("SARIMAX",              "Classical",      18.45,  124.31,  0.4561, 0.3881),
    ("LightGBM",             "GBT",            19.76,  100.41,  0.3421, 0.4535),
    ("LSTM",                 "Deep Learning",  24.42,  148.72,  0.1667, 0.2346),
    ("XGBoost",              "GBT",            24.43,  112.46,  0.5702, 0.2510),
    ("TimesFM",              "Foundation",     32.51,  136.94,  0.0175, 0.1176),
    ("XGBoost, no sentiment", "Ablation",      34.53,  122.57,  0.5053, 0.1954),
    ("TFT",                  "Deep Learning",  45.75,  134.60,  0.0000, 0.0000),
    ("Prophet Hybrid",       "Hybrid",         48.36,  108.96,  0.7105, 0.1416),
    ("Prophet",              "Classical",     119.14,  168.75,  0.5714, 0.1770),
]

HIGHLIGHT = "#c2410c"
PRIMARY = "#1f4e79"
MUTED = "#94a3b8"
ACCENT = "#0f766e"
INK = "#1e293b"

FAMILY_COLORS = {
    "Hybrid": HIGHLIGHT,
    "GBT": PRIMARY,
    "Foundation": ACCENT,
    "Classical": "#7c3aed",
    "Deep Learning": "#be123c",
    "Ablation": MUTED,
}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.axisbelow": True,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })


def load_matrix() -> pd.DataFrame:
    if not MATRIX.exists():
        raise SystemExit(
            "Missing {}\n"
            "This file ships with the repo. Regenerate it with:\n"
            "  python run_full_pipeline.py --stage pipelines\n"
            "  python run_full_pipeline.py --stage features".format(MATRIX)
        )
    mat = pd.read_parquet(MATRIX).sort_index()
    return mat.dropna(subset=[TARGET_REG])


def save(fig: plt.Figure, name: str, tight: bool = True, dpi: int | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=dpi or plt.rcParams["savefig.dpi"],
                bbox_inches="tight" if tight else None,
                pad_inches=0.1 if tight else 0)
    plt.close(fig)
    print("  wrote {}  ({:.0f} KB)".format(path.relative_to(ROOT), path.stat().st_size / 1024))


# --- Figures -----------------------------------------------------------------

def fig_teaser(mat: pd.DataFrame) -> None:
    """2024 test year: what the model is asked to predict, and the spikes."""
    test = mat.loc[mat.index >= TEST_START, TARGET_REG]
    spikes = test[test > SPIKE_THRESHOLD]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.2), gridspec_kw={"height_ratios": [2, 1], "hspace": 0.42},
    )

    ax1.plot(test.index, test.values, lw=0.55, color=PRIMARY, alpha=0.85)
    ax1.scatter(spikes.index, spikes.values, s=16, color=HIGHLIGHT, zorder=5,
                label="{} spike hours (> ${:.0f}/MWh)".format(len(spikes), SPIKE_THRESHOLD))
    ax1.axhline(SPIKE_THRESHOLD, color=HIGHLIGHT, ls="--", lw=1, alpha=0.7)
    ax1.set_yscale("symlog", linthresh=100)
    ax1.set_ylim(-60, 7000)
    ax1.set_ylabel("Real-time price ($/MWh)")
    ax1.set_title("ERCOT hub-average price, 2024 held-out test year")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.legend(loc="upper left")
    # Below the axis midline the series is quiet, so the caption stays legible there.
    # Two unescaped dollar signs in one label would be read as mathtext.
    ax1.text(
        0.012, 0.03,
        r"{:,} hours   median \${:.0f}   max \${:,.0f}".format(
            len(test), test.median(), test.max()),
        transform=ax1.transAxes, ha="left", va="bottom", fontsize=9.5, color="#475569",
    )

    # Zoom on the week containing the single highest-priced hour.
    peak = test.idxmax()
    window = test.loc[peak - pd.Timedelta(days=3):peak + pd.Timedelta(days=3)]
    ax2.fill_between(window.index, 0, window.values, color=PRIMARY, alpha=0.18)
    ax2.plot(window.index, window.values, lw=1.4, color=PRIMARY)
    ax2.axhline(SPIKE_THRESHOLD, color=HIGHLIGHT, ls="--", lw=1, alpha=0.7)
    ax2.annotate(
        "${:,.0f}/MWh".format(window.max()),
        xy=(peak, window.max()), xytext=(12, -6), textcoords="offset points",
        fontsize=10, fontweight="bold", color=HIGHLIGHT,
    )
    ax2.set_ylabel("$/MWh")
    ax2.set_title("The forecasting problem is the tail: one week around the 2024 peak",
                  fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    save(fig, "teaser.png")


def fig_leaderboard() -> None:
    """MAE for all 13 benchmarked configurations, best at the top."""
    rows = sorted(PAPER_LEADERBOARD, key=lambda r: r[2], reverse=True)
    names = [r[0] for r in rows]
    maes = [r[2] for r in rows]
    colors = [FAMILY_COLORS[r[1]] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(names, maes, color=colors, alpha=0.92, height=0.72)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_width() + 1.6, bar.get_y() + bar.get_height() / 2,
                "{:.2f}".format(mae), va="center", fontsize=9.5, color="#475569")

    best = bars[-1]
    best.set_edgecolor(INK)
    best.set_linewidth(1.3)
    ax.set_xlim(0, max(maes) * 1.13)
    ax.set_xlabel("Test MAE ($/MWh), 2024 - lower is better")
    ax.set_title("Consolidated capstone comparison: reported test MAE")
    ax.grid(axis="y", visible=False)

    # The top rows are the short bars, so the upper right is the only free space.
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, list(FAMILY_COLORS), loc="upper right", fontsize=9, ncol=2,
              title="Model family", title_fontsize=9)
    save(fig, "leaderboard_mae.png")


def fig_spike_tradeoff() -> None:
    """Spike precision against recall: every model fails differently."""
    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    # Hand-placed labels: several models sit close together in this space.
    offsets = {
        "Chronos": (-14, 10, "right"),
        "CatBoost": (10, -6, "left"),
        "NGBoost": (0, -16, "center"),
        "Chronos Hybrid": (12, 4, "left"),
        "SARIMAX": (0, -16, "center"),
        "LightGBM": (0, -16, "center"),
        "LSTM": (0, -16, "center"),
        "XGBoost": (12, 2, "left"),
        "TimesFM": (10, -4, "left"),
        "XGBoost, no sentiment": (-10, 6, "right"),
        "Prophet": (4, -16, "left"),
        "Prophet Hybrid": (0, -16, "center"),
    }

    for name, family, mae, _rmse, recall, precision in PAPER_LEADERBOARD:
        if recall == 0 and precision == 0:
            continue
        x, y = recall * 100, precision * 100
        ax.scatter(x, y, s=190 - mae, color=FAMILY_COLORS[family],
                   alpha=0.85, edgecolor="white", linewidth=1.2, zorder=4)
        dx, dy, ha = offsets.get(name, (0, -16, "center"))
        ax.annotate(name, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=8.5, color="#475569", zorder=5)

    for f1 in (0.3, 0.45, 0.6):
        r = np.linspace(max(f1 / 2 + 0.01, 0.05), 0.99, 200)
        p = (f1 * r) / (2 * r - f1)
        ok = (p > 0) & (p <= 1)
        ax.plot(r[ok] * 100, p[ok] * 100, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.text(101, (f1 * 0.99) / (2 * 0.99 - f1) * 100, "F1 {:.2f}".format(f1),
                fontsize=8, color=MUTED, va="center", ha="left")

    ax.set_xlim(-3, 100)
    ax.set_ylim(0, 72)
    ax.set_xlabel("Spike recall (%) - share of the 114 spike hours caught")
    ax.set_ylabel("Spike precision (%) - share of alarms that were real")
    ax.set_title("Spike detection is a tradeoff, and the hybrid balances it best")
    ax.text(0.015, 0.97,
            "Marker size is inversely proportional to MAE.\nTFT is omitted: it scored 0 on both.",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#64748b")
    save(fig, "spike_tradeoff.png")


def fig_price_distribution(mat: pd.DataFrame) -> None:
    """Why this is hard: most hours are ordinary, the tail sets the economics."""
    price = mat[TARGET_REG]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Restrict rather than clip: clipping piles the whole tail onto one bar.
    zoom = price[(price >= -50) & (price <= 400)]
    ax1.hist(zoom, bins=110, color=PRIMARY, alpha=0.85)
    ax1.axvline(price.median(), color=ACCENT, lw=1.6,
                label="median ${:.0f}".format(price.median()))
    ax1.axvline(SPIKE_THRESHOLD, color=HIGHLIGHT, lw=1.6, ls="--",
                label="spike threshold $200")
    ax1.set_xlabel("Price ($/MWh), restricted to [-50, 400]")
    ax1.set_ylabel("Hours")
    ax1.set_title("Most hours are unremarkable")
    ax1.legend(fontsize=9)

    bins = [-100, 0, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    counts, _ = np.histogram(price, bins=bins)
    labels = ["<0", "0-25", "25-50", "50-100", "100-200", "200-500",
              "0.5-1k", "1-2k", "2-5k", ">5k"]
    colors = [MUTED] + [PRIMARY] * 4 + [HIGHLIGHT] * 5
    ax2.bar(labels, np.maximum(counts, 0.6), color=colors, alpha=0.9)
    ax2.set_yscale("log")
    ax2.set_ylabel("Hours (log scale)")
    ax2.set_xlabel("Price band ($/MWh)")
    ax2.set_title("The tail is thin but decisive")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    n_spike = int((price > SPIKE_THRESHOLD).sum())
    ax2.text(0.98, 0.95,
             "{:,} of {:,} hours exceed \\$200\n"
             "({:.2f}% of the record)\n"
             "max observed \\${:,.0f}".format(
                 n_spike, len(price), 100 * n_spike / len(price), price.max()),
             transform=ax2.transAxes, ha="right", va="top", fontsize=9, color="#475569")

    fig.suptitle("ERCOT real-time hub-average price, 2016-2024",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "price_distribution.png")


def fig_sentiment_ablation() -> None:
    """Matched notebook ablation. Other historical pairs are labeled, not mixed in."""
    matched = {"without": 24.64, "with": 22.21}

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(2)
    bars = ax.bar(x, [matched["without"], matched["with"]],
                  width=0.55, color=[MUTED, PRIMARY])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
                "{:.2f}".format(bar.get_height()), ha="center", fontsize=10,
                color="#475569")

    ax.set_xticks(x)
    ax.set_xticklabels(["Without GDELT", "With GDELT"])
    ax.set_ylabel("Test MAE ($/MWh)")
    ax.set_ylim(0, 40)
    ax.set_title("Matched XGBoost ablation, notebooks/01_xgboost.ipynb")
    ax.text(0.02, 0.96,
            "MAE falls 9.9%.\n"
            "Other runs, not this ablation:\n"
            "capstone row 34.53 to 24.43;\n"
            "script re-run 24.58 to 19.17.",
            transform=ax.transAxes, va="top", fontsize=9, color="#475569")
    ax.grid(axis="x", visible=False)
    save(fig, "sentiment_ablation.png")


def _box(ax, xy, w, h, text, face, edge, fontsize=9.5, weight="normal", text_color=INK):
    ax.add_patch(FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=1.3,
    ))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color, linespacing=1.45)


def _arrow(ax, start, end, color=INK):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        color=color, linewidth=1.3, shrinkA=2, shrinkB=2,
    ))


def fig_architecture() -> None:
    """How the winning hybrid is wired together."""
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    _box(ax, (0.015, 0.60), 0.20, 0.33,
         "Eight data sources\n\nERCOT RT and DA prices\nsystem load, weather\nnatural gas, storage\nnuclear outages\nGDELT news sentiment",
         "#f1f5f9", "#cbd5e1", fontsize=8.8)

    _box(ax, (0.275, 0.66), 0.19, 0.21,
         "Feature matrix\n\n77,956 hours\n87 features",
         "#e0f2fe", PRIMARY, fontsize=9.5, weight="bold")

    _box(ax, (0.275, 0.13), 0.19, 0.21,
         "Raw price history\nHB_HUBAVG\n512-hour context",
         "#f1f5f9", "#cbd5e1", fontsize=9)

    _box(ax, (0.515, 0.13), 0.20, 0.21,
         "Chronos T5-small\nzero-shot, frozen\n1-hour-ahead forecast",
         "#ccfbf1", ACCENT, fontsize=9, weight="bold")

    _box(ax, (0.515, 0.66), 0.20, 0.21,
         "87 features\n+ chronos_pred\n= 88 inputs",
         "#e0f2fe", PRIMARY, fontsize=9.5)

    _box(ax, (0.785, 0.40), 0.20, 0.25,
         "XGBoost\n500 trees, depth 6\nlr 0.05\n\nhourly price\n+ spike flag",
         "#ffedd5", HIGHLIGHT, fontsize=9.3, weight="bold")

    _arrow(ax, (0.215, 0.765), (0.275, 0.765))
    _arrow(ax, (0.215, 0.70), (0.275, 0.30))
    _arrow(ax, (0.465, 0.235), (0.515, 0.235), ACCENT)
    _arrow(ax, (0.615, 0.34), (0.615, 0.66), ACCENT)
    _arrow(ax, (0.465, 0.765), (0.515, 0.765), PRIMARY)
    _arrow(ax, (0.715, 0.73), (0.855, 0.65), PRIMARY)

    ax.text(0.63, 0.50, "appended as\none column", fontsize=8.5, color=ACCENT,
            ha="left", style="italic")
    ax.text(0.5, 0.99, "Chronos + XGBoost hybrid", ha="center", va="top",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(0.5, 0.035,
            "Chronos is never fine-tuned. Its forecast enters as a single feature, and the "
            "gradient-boosted trees learn when to trust it.",
            ha="center", fontsize=9, color="#64748b", style="italic")
    save(fig, "architecture.png")


def fig_feature_importance() -> None:
    """Top features of the 87-feature model, if artifacts have been built."""
    csv = ARTIFACTS / "enhanced_v2_feature_importance.csv"
    if not csv.exists():
        print("  skipping feature_importance.png - run "
              "`python run_full_pipeline.py --stage models` first")
        return

    imp = pd.read_csv(csv, index_col=0).iloc[:, 0].sort_values(ascending=False).head(20)
    imp = imp.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    colors = [HIGHLIGHT if f.startswith("gdelt_") else PRIMARY for f in imp.index]
    ax.barh(imp.index, imp.values, color=colors, alpha=0.9, height=0.74)
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Top 20 features, 87-feature model")
    ax.grid(axis="y", visible=False)
    ax.text(0.97, 0.05,
            "Sentiment features in orange", transform=ax.transAxes,
            ha="right", fontsize=9.5, color=HIGHLIGHT)
    save(fig, "feature_importance.png")


def fig_social_preview(mat: pd.DataFrame) -> None:
    """1200x630 card for link previews on Scholar, LinkedIn, and Twitter."""
    test = mat.loc[mat.index >= TEST_START, TARGET_REG]

    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor("#0f172a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.grid(False)

    spark = fig.add_axes([0.06, 0.08, 0.88, 0.30])
    spark.plot(test.index, np.clip(test.values, None, 900), lw=0.7, color="#38bdf8", alpha=0.9)
    spark.axhline(SPIKE_THRESHOLD, color="#fb923c", lw=0.9, ls="--", alpha=0.8)
    spark.set_facecolor("#0f172a")
    spark.axis("off")

    ax.text(0.06, 0.86, "A Multimodal Framework for Electricity Price Forecasting",
            fontsize=25, fontweight="bold", color="white", va="top")
    ax.text(0.06, 0.73, "Using Sentiment-Augmented Time-Series Signals",
            fontsize=25, fontweight="bold", color="#38bdf8", va="top")
    ax.text(0.06, 0.60, "Rohan Fozdar   |   Knox College   |   Mathematics Capstone",
            fontsize=13.5, color="#cbd5e1", va="top")
    ax.text(0.06, 0.48,
            "Retrospective ERCOT price estimation, 2016-2024\n"
            "Consolidated benchmark reports Chronos + XGBoost at MAE $12.13/MWh",
            fontsize=13, color="#e2e8f0", va="top", linespacing=1.6)
    # Exactly 1200x630, the size declared in the page's Open Graph tags.
    save(fig, "social_preview.png", tight=False, dpi=100)


def fig_favicon() -> None:
    """Replace the template's favicon, which is otherwise the template author's."""
    from PIL import Image

    fig = plt.figure(figsize=(1.28, 1.28), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)
    series = [1.6, 1.9, 1.7, 2.1, 1.8, 8.6, 2.4, 1.9, 2.2, 1.7]
    ax.plot(range(len(series)), series, color="#38bdf8", lw=2.6, solid_joinstyle="round")
    ax.scatter([5], [8.6], s=40, color="#fb923c", zorder=5)

    tmp = OUT_DIR / "_favicon_tmp.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(tmp, facecolor="#0f172a", bbox_inches=None, pad_inches=0, dpi=200)
    plt.close(fig)

    img = Image.open(tmp).convert("RGBA")
    ico = OUT_DIR / "favicon.ico"
    img.save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    tmp.unlink()
    print("  wrote {}  ({:.0f} KB)".format(ico.relative_to(ROOT), ico.stat().st_size / 1024))


FIGURES = {
    "teaser": ("matrix", fig_teaser),
    "leaderboard": ("none", fig_leaderboard),
    "spike_tradeoff": ("none", fig_spike_tradeoff),
    "price_distribution": ("matrix", fig_price_distribution),
    "sentiment_ablation": ("none", fig_sentiment_ablation),
    "architecture": ("none", fig_architecture),
    "feature_importance": ("none", fig_feature_importance),
    "social_preview": ("matrix", fig_social_preview),
    "favicon": ("none", fig_favicon),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", choices=sorted(FIGURES),
                    help="Generate only this figure (repeatable)")
    ap.add_argument("--list", action="store_true", help="List figure names and exit")
    args = ap.parse_args()

    if args.list:
        for name in sorted(FIGURES):
            print(name)
        return

    wanted = args.only or list(FIGURES)
    style()

    mat = None
    if any(FIGURES[n][0] == "matrix" for n in wanted):
        mat = load_matrix()
        print("  matrix: {:,} rows x {} cols  ({} to {})".format(
            mat.shape[0], mat.shape[1], mat.index.min().date(), mat.index.max().date()))

    print("\nWriting figures to {}/".format(OUT_DIR.relative_to(ROOT)))
    for name in wanted:
        needs, fn = FIGURES[name]
        fn(mat) if needs == "matrix" else fn()
    print("\nDone.")


if __name__ == "__main__":
    main()
