# Peer review

This review scores the capstone PDF together with the $11.56 rewrite, and then scores only the work this protocol executed. Unrun extensions get no credit.

## A. Current paper, before this protocol

| Criterion | Score |
| --- | ---: |
| Originality | 2 |
| Technical soundness | 3 |
| Experimental rigor | 3 |
| Reproducibility | 2 for the PDF, 4 for the honest-tree artifacts |
| Clarity | 4 |
| Significance | 3 |
| Generalization | 2 |
| Publication readiness | reject |

The PDF's hybrid row was not executed, its spike precision is not an integer ratio on 114 events, and the sentiment result is not a stable matched ablation. The $11.56 tree is traceable and still uses inputs whose publication time is unverified. It never removes the Chronos columns, so it cannot show that those columns help. CatBoost's ordered boosting is described as if it blocked future prices. The TimesFM citation points at the TiDE preprint.

Strongest aspect: the target, the hub, and the frozen 2024 index are concrete, and the later protocol saved forecasts that can be recomputed.

Fatal weakness: a forecasting claim is being made with features that were not shown to be available at the issue time, and the central model has no matched ablation.

## B. Work executed in this protocol

| Criterion | Score |
| --- | ---: |
| Originality | 4 |
| Technical soundness | 6 |
| Experimental rigor | 6 |
| Reproducibility | 6 |
| Clarity | 6 |
| Significance | 4 |
| Generalization | 5 |
| Publication readiness | major revision before a workshop; not a journal submission |

Conservative-set MAE gap (with Chronos minus without), seed-averaged forecast: -0.09 with interval [-0.27, 0.11]. Interval covers zero: True. Chronos lower MAE with the interval entirely below zero: False.

Strongest aspect: the comparison holds the origin, the target, the split, the loss, and the tree settings fixed, and it labels assumed release rules instead of treating them as measurements.

Potentially fatal if over-claimed: 2024 was already inspected; lagged load and the real-time posting delay are still assumptions; the result is one series. Treating a small MAE gap as a new hybrid method would recreate the original paper's problem.

Major concerns: no ERCOT wind or solar forecast vintage in the primary set; the gas rule remains assumed wherever gas is used; the block length of 24 hours is a choice; five seeds are a thin description of fitting noise.

Minor concerns: the v2 hourly maximum still keeps hours that do not have four intervals; the thin lasso is not the published LEAR benchmark; calendar peak hours follow the PDF's evening window rather than an ERCOT tariff definition.

Missing if one were submitting to a journal: a pre-registered later year scored with Chronos as well as without it, operational load and renewable forecasts joined on their release times, and at least one more market under the same rules.

Questions a reviewer would ask: Was any 2024 number used to decide that the conservative set was the one to emphasize? What happens if the real-time price is not available until fifteen minutes after the interval? Why is the hourly maximum the decision, rather than the hourly average or the day-ahead price?

Later-year fetch status: `fetched`. Per year: `{'2025': {'rt': 'fetched', 'da': 'fetched'}, '2026': {'rt_fallback': 'empty trailing month sheets were dropped before parsing', 'rt': 'fetched', 'da': 'fetched'}}`. Expanding window present: True. Later years scored: True.

Recommendation: The revised draft is a possible workshop paper if the claims stay inside the tables. It is not ready for an applied energy journal.
