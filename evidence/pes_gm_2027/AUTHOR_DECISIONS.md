# Author decisions

Decisions the evidence agent made because the plan required a choice and the author was not asked. Each is dated. None was chosen by looking at 2024 evaluation performance after the protocol freeze.

| ID | Decision | Date (UTC) | Why |
| --- | --- | --- | --- |
| AD-01 | Venue is IEEE PES GM 2027, not GT&D LA 2027 | 2026-10-25 | Author selected GM explicitly. |
| AD-02 | Rolling windows use pre-2025 data only | 2026-10-25 | Author selected this when asked. |
| AD-03 | Audit target is manuscript copy B | 2026-10-25 | Copy B has the latest edits. Copy C is a truncated B. |
| AD-04 | Git identity for commits is `pes2027-evidence-agent` / `noreply@localhost` | 2026-10-25 | Local-only; no push. |
| AD-05 | GPU 0 only | 2026-10-25 | GPU 1 is in use by another job. |
| AD-06 | E1 representations are selected on 2023 five-seed MAE only | 2026-10-26 | Frozen in protocol v1 before any new 2024 score. |
| AD-07 | E6 does not add a load daily-posting gate | 2026-10-26 | NP6-345-CD posting minute is not documented. Frozen in protocol v1. |
| AD-08 | E8 official windows: onset 2021-02-10 06:00Z–2021-02-15 06:00Z; emergency 2021-02-15 06:00Z–2021-02-19 15:00Z; recovery 2021-02-19 15:00Z–2021-02-21 06:00Z | 2026-10-26 | Taken from FERC/NERC, ERCOT ops messages, PUC 51812, and ERCOT legislative slides. Committed as `protocol/e8_official_subwindows.yaml` (f53057f) before sub-window metrics. Hour-aligned to include any hour that overlaps the official instant. |
| AD-09 | Crisis comparator names are `xgb_delayed_history_baseline`, `xgb_delayed_history_plus_chronos`, `persistence_latest_legal_t_minus_2` | 2026-10-26 | Corrects the manuscript “exogenous-only” label (CL-01). |
| AD-10 | E6 and E0 reproduce from saved predictions; they do not refit | 2026-10-26 | Avoids writing the original run directories and avoids needless GPU time. |
| AD-11 | E1/E3/E7/E8 drop non-finite target rows with `finite_slice` | 2026-10-26 | Matches `run_pit_verified`. The inherited matrix has at least one 2019–2022 hour with a non-finite HB_HUBAVG. |
| AD-12 | Primary committee recommendation is PSOPE / Energy Forecasting and Analytics | 2026-10-26 | Official WG scope includes price forecasting. Not a promise of assignment. |
| AD-13 | $200/MWh remains an inherited constant | 2026-10-26 | Scientific rationale is [MISSING]. Not re-tuned. |
| AD-14 | No equivalence test | 2026-10-26 | No tolerance was justified before 2024 was scored. Frozen in protocol v1. |
| AD-15 | E1/E3 baseline columns come from `columns_for`, not `frame.columns` | 2026-10-26 | `feature_subsets(frame.columns)` treated the target as an exogenous feature. The leaked 2023 selection (R2, val MAE 9.65) was discarded uncommitted. The committed selection is R1 (val MAE 38.44) on the original 37 columns plus dense lags. |

Pending author items (not decided here):

- Scientific rationale for the $200 threshold.
- Whether the writing agent should mention the GitHub URL in the camera-ready paper.
- Whether to submit; this package does not submit.
