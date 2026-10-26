# Reproduction of the saved conservative ablation

No model was refit for this summary. No 2025 or 2026 score is included. The files read were `runs/pit_ablation/ablation_metrics.csv`, `runs/pit_ablation/comparisons.json`, `runs/pit_ablation/saved_booster_replay.json`, and `runs/pit_ablation/build_manifest.json`.

The saved honest-tree booster replays at MAE $11.5595 with a maximum absolute difference of 0.0. The frozen 2024 index hash is `b850034b0690e660`.

Two different estimands were stored in the earlier run:

- MAE of the seed-averaged forecast: $11.3330 with Chronos and $11.4184 without. The gap is $-0.0854. The 24-hour block bootstrap interval is [-0.2700, 0.1125], bootstrap p 0.368, Diebold-Mariano p 0.383. The interval covers zero.
- Mean of the five seed-level MAEs: $11.4609 with Chronos and $11.5155 without. The difference of those means is $-0.0546. That number is not the gap above.

Per-seed gaps, Chronos minus exogenous, in seed order 0, 1, 2, 3, 42: 0.0579, -0.0085, -0.1714, 0.1756, -0.3267. The seed-42 pair alone has gap $-0.3267 and an interval that excludes zero. That pair was not the pre-registered summary.

2024 had already been inspected when that ablation was run. This file confirms the saved arithmetic. It does not re-establish that every feature was public at the issue time. The zero-delay label on real-time prices is withdrawn in `reports/source_evidence_register.md`.
