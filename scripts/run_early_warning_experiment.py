"""Run the frozen early-warning experiment end to end (CPU only).

Needs the Chronos caches in data/features/chronos/, produced on a GPU by
scripts/run_chronos_caches.sh (see GPU_CHRONOS.md). This script never runs Chronos.

    python scripts/run_early_warning_experiment.py                 # everything
    python scripts/run_early_warning_experiment.py --export-chronos-input
    python scripts/run_early_warning_experiment.py --no-sensitivity --horizons 48   # quick check

Outputs go to runs/<hash>/, where the hash covers the frozen config, the availability
rules, the raw data files, the Chronos caches and the git commit. Summary tables and
figures are also copied to reports/results/.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import file_hash, load_yaml, stable_hash  # noqa: E402
from src.data import sources  # noqa: E402
from src.evaluation import figures, report  # noqa: E402
from src.experiment import Dataset, RunSpec, run  # noqa: E402
from src.models.chronos_features import ChronosCache  # noqa: E402

CONFIG = "configs/experiment_early_warning.yaml"
CHRONOS_DIR = ROOT / "data" / "features" / "chronos"
CHRONOS_INPUT = ROOT / "data" / "features" / "chronos_input_y.parquet"

BASELINES = ["prevalence", "persistence_max24h", "seasonal_naive_168h", "day_ahead_max", "logistic_structured"]
XGB = ["XGB_price", "XGB_structured", "XGB_structured_news"]
CHRONOS = ["Chronos_only", "Chronos_XGB_price", "Chronos_XGB_structured", "Chronos_XGB_full", "Blend_full"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def export_chronos_input() -> Path:
    y = sources.hourly_price()
    CHRONOS_INPUT.parent.mkdir(parents=True, exist_ok=True)
    y.to_frame("y").to_parquet(CHRONOS_INPUT)
    log(f"wrote {CHRONOS_INPUT.relative_to(ROOT)} ({len(y):,} hours, {y.index.min()} -> {y.index.max()})")
    return CHRONOS_INPUT


def find_cache(variant: str, context: int, required: bool = False) -> Path | None:
    """Chronos caches are produced on a GPU machine (GPU_CHRONOS.md); this runner never generates them."""
    path = CHRONOS_DIR / f"{variant}_ctx{context}.npz"
    if path.exists():
        return path
    if required:
        sys.exit(f"missing {path}: generate it on a GPU with "
                 f"scripts/run_chronos_caches.sh (see GPU_CHRONOS.md), then rerun")
    log(f"  {path.name} not found; skipping that sensitivity row")
    return None


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-chronos-input", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--horizons", type=int, nargs="*")
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs")
    args = ap.parse_args()

    if args.export_chronos_input:
        export_chronos_input()
        return

    cfg = load_yaml(CONFIG)
    horizons = args.horizons or cfg["target"]["horizons_hours"]
    fs = cfg["feature_sets"]
    if not CHRONOS_INPUT.exists():
        export_chronos_input()

    ch_cfg = cfg["chronos"]
    variant_of = {"amazon/chronos-bolt-small": "bolt-small", "amazon/chronos-bolt-base": "bolt-base",
                  "amazon/chronos-t5-small": "t5-small"}
    primary_cache = find_cache(variant_of[ch_cfg["primary"]["checkpoint"]], ch_cfg["primary"]["context_hours"],
                               required=True)
    sens_caches = []
    if not args.no_sensitivity:
        for s in ch_cfg["sensitivity"]:
            p = find_cache(variant_of[s["checkpoint"]], s["context_hours"])
            if p:
                sens_caches.append(p)
        t5 = find_cache("t5-small", ch_cfg["gpu_handoff"]["context_hours"])
        if t5:
            sens_caches.append(t5)

    raw_files = sorted(p for p in sources.RAW.glob("*.parquet"))
    manifest = {
        "config": cfg, "availability": load_yaml(cfg["data"]["availability"]),
        "raw_hashes": {p.name: file_hash(p) for p in raw_files},
        "chronos_caches": {p.name: file_hash(p) for p in [primary_cache, *sens_caches]},
        "git_commit": git_commit(), "horizons": horizons, "sensitivity": not args.no_sensitivity,
    }
    run_id = stable_hash({k: v for k, v in manifest.items() if k != "git_commit"})
    out = args.out_root / run_id
    (out / "figures").mkdir(parents=True, exist_ok=True)
    import numpy, sklearn, xgboost  # noqa: F401,E401
    manifest.update({"run_id": run_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": numpy.__version__, "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__})
    log(f"run {run_id} -> {out}")

    t0 = time.time()
    ds = Dataset(cfg)
    log(f"dataset: {len(ds.origins):,} origins, built in {time.time() - t0:.1f}s")
    cache = ChronosCache(primary_cache)
    exps = cfg["experiments"]
    specs = [RunSpec("point_in_time", exps["point_in_time"]["folds"], fs, BASELINES + XGB + CHRONOS,
                     horizons=horizons, tag="primary")]
    if not args.no_sensitivity:
        specs += [
            RunSpec("oracle_weather", exps["oracle_weather"]["folds"], fs,
                    ["XGB_structured_news_oracle", "Chronos_XGB_full_oracle"], horizons=horizons, tag="oracle"),
        ]
        if ds.fc_cube is not None:
            specs.append(RunSpec("forecast_weather", exps["forecast_weather"]["folds"], fs,
                                 ["persistence_max24h", "day_ahead_max", "XGB_structured_news",
                                  "XGB_structured_news_wxfc", "Chronos_only", "Chronos_XGB_full",
                                  "Chronos_XGB_full_wxfc"],
                                 horizons=horizons, period_start=exps["forecast_weather"]["period_start"],
                                 tag="forecast_weather"))
        for thr in cfg["target"]["sensitivity_thresholds"]:
            specs.append(RunSpec("point_in_time", exps["point_in_time"]["folds"], fs,
                                 ["prevalence", "persistence_max24h", "day_ahead_max", "XGB_structured_news",
                                  "Chronos_only", "Chronos_XGB_full"],
                                 threshold=thr, horizons=horizons, tag=f"thr_{thr}"))
        for seed in cfg["statistics"]["robustness_seeds"]:
            specs.append(RunSpec("point_in_time", exps["point_in_time"]["folds"], fs,
                                 ["XGB_structured_news", "Chronos_XGB_full"], seed=seed, horizons=horizons,
                                 tag=f"seed_{seed}"))

    all_preds, all_info = [], []
    for spec in specs:
        log(f"spec {spec.tag} ({spec.experiment}, {len(spec.models)} models)")
        p, i = run(ds, spec, cache, log=log)
        all_preds.append(p)
        all_info += i
    if not args.no_sensitivity:
        for path in sens_caches:
            c = ChronosCache(path)
            tag = f"chronos_{path.stem}"
            log(f"spec {tag}")
            p, i = run(ds, RunSpec("point_in_time", exps["point_in_time"]["folds"], fs,
                                   ["Chronos_only", "Chronos_XGB_full"], horizons=horizons, tag=tag), c, log=log)
            all_preds.append(p)
            all_info += i
        log("spec weather_lag_120h")
        ds_lag = Dataset(cfg, weather_lag_hours=120)
        p, i = run(ds_lag, RunSpec("point_in_time", exps["point_in_time"]["folds"], fs,
                                   ["XGB_structured_news", "Chronos_XGB_full"], horizons=horizons,
                                   tag="weather_lag_120h"), cache, log=log)
        all_preds.append(p)
        all_info += i

    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_parquet(out / "predictions.parquet")
    report.write_json(all_info, out / "models.json")
    log(f"training done in {(time.time() - t0) / 60:.1f} min; {len(preds):,} prediction rows")

    # --- evaluation -----------------------------------------------------------
    gap = cfg["target"]["event_merge_gap_hours"]
    tab, leads = report.metrics_table(preds, ds.y, cfg, gap)
    tab.to_csv(out / "metrics.csv", index=False)
    leads.to_csv(out / "lead_times.csv", index=False)
    for g in cfg["target"]["sensitivity_event_merge_gap_hours"]:
        tg, _ = report.metrics_table(preds[preds.tag == "primary"], ds.y, cfg, g)
        tg.to_csv(out / f"metrics_gap{g}h.csv", index=False)
    ci = report.ap_intervals(preds, cfg, ["primary"])
    ci.to_csv(out / "ap_intervals.csv", index=False)
    tests = None
    if set(horizons) >= {48} and len(horizons) > 1:
        tests = report.hypothesis_tests(preds, cfg)
        report.write_json(tests, out / "hypothesis_tests.json")
        ab = report.ablation_deltas(preds, cfg)
        ab.to_csv(out / "ablation.csv", index=False)
        figures.ablation(ab, out / "figures" / "ablation_chronos_gain.png")
    fa = report.failure_analysis(preds, ds.y, cfg, h=48 if 48 in horizons else horizons[0])
    fa.to_csv(out / "failure_analysis.csv", index=False)

    figures.ap_by_horizon(tab, ci, out / "figures" / "ap_by_horizon.png")
    figures.reliability_plot(preds, out / "figures" / "reliability_48h.png", h=48 if 48 in horizons else horizons[0])
    figures.lead_times(leads, out / "figures" / "lead_times.png")
    figures.event_tradeoff(tab, out / "figures" / "event_tradeoff.png")

    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["minutes"] = round((time.time() - t0) / 60, 1)
    report.write_json(manifest, out / "manifest.json")

    dest = ROOT / "reports" / "results" if (not args.no_sensitivity and args.horizons is None) else out / "summary"
    dest.mkdir(parents=True, exist_ok=True)
    for f in ["metrics.csv", "ap_intervals.csv", "hypothesis_tests.json", "ablation.csv", "failure_analysis.csv",
              "lead_times.csv", "manifest.json", "models.json", *[f"metrics_gap{g}h.csv" for g in
                                                                  cfg["target"]["sensitivity_event_merge_gap_hours"]]]:
        if (out / f).exists():
            shutil.copy(out / f, dest / f)
    shutil.copytree(out / "figures", dest / "figures", dirs_exist_ok=True)
    (dest / "RUN_ID").write_text(run_id + "\n")
    if tests:
        log("primary: " + json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                          for kk, vv in v.items()} for k, v in tests["primary"].items()}, default=str))
        log(f"primary claim supported: {tests['primary_claim_supported']}")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
