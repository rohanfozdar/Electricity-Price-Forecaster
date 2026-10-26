# Claim–evidence matrix

Status: reproduced = recomputed from saved original forecasts. newly_measured = new run after protocol freeze. author-reported = manuscript or repo claim not re-derived here. proposed = interpretation. unavailable = [MISSING].

Monetary units are $/MWh. The primary 2024 index has 8,767 hours, hash `b850034b0690e660`.

## Original 2024 experiment (keep distinct)

| Claim | Number | Status | Artifact |
| --- | --- | --- | --- |
| Seed-averaged baseline MAE | 15.068204 | reproduced | `runs/pes2027/E0-REPRO/paired_comparisons.csv` scenario delay_1h; `tables/T_e0_paired.csv` |
| Seed-averaged augmented MAE | 14.918317 | reproduced | same |
| Gap (both − exog) | −0.149888 | reproduced | same; matches `runs/pit_verified/comparisons.json` |
| 24 h bootstrap 95% interval | [−0.317435, 0.063750] | reproduced | same |
| Bootstrap p (interval inversion) | 0.131 | reproduced | same |
| DM p, original h=25 | 0.124383 | reproduced | same |
| DM p, primary h=1 | 0.123353 | newly_measured | independent stats; not in the manuscript |
| Evaluation hours | 8767 | reproduced | index hash `b850034b0690e660` |
| Spike hours y>200 | 114 | reproduced | E0-REPRO levels |
| Persistence MAE | 25.839600 | reproduced | `tables/T_e0_baselines.csv` |
| Seasonal naive 24 / 168 | 29.58 / 32.35 (rounded) | reproduced | same; RMSE exists (manuscript dashes are omissions, CL) |
| Complete-four “refit” equals saved subset | max abs 0.0 | reproduced | Concern C |
| Crisis “exogenous-only” is xgb_exog | 37 features including price lags | reproduced | CL-01 |
| Crisis trees are seed 42 only | 3836.73 / 3604.11 | reproduced | CL-02; `tables/T_e8_original_seed42.csv` |
| Primary trees refit identically | max abs 0.0 | newly_measured | `runs/pes2027/E0-REPRO/rerun/rerun_fits.csv` |
| Manuscript vs full precision | 105 rounding; 4 omitted; 0 substantive | newly_measured | `tables/T_manuscript_precision_check.csv` |

Do not read non-rejection as equivalence. No equivalence test was predeclared.

## New experiments (after protocol freeze b353520)

| Experiment | Estimand | Result | Status | Artifact |
| --- | --- | --- | --- | --- |
| E1 selection | 2023 five-seed MAE | R1 38.442; R2 38.968; R3 38.801. Selected R1. | newly_measured | `protocol/e1_selection.json` committed d027226 **before** 2024 |
| E1 2024 B512 | seed-avg MAE | 15.105367 | newly_measured | `runs/pes2027/E1-H512/score_2024.json` |
| E1 2024 B512+Chronos | seed-avg MAE | 14.885307 | newly_measured | same |
| E1 matched gap | B512+Chronos − B512 | −0.220060; CI [−0.489, 0.092]; p_boot 0.142 | newly_measured | same. Interval covers 0 |
| E1 original-both vs B512 | gap | −0.190437; CI covers 0; p_boot 0.119 | newly_measured | same |
| E2 standalone Chronos | chronos_pred MAE | 17.794443 | newly_measured | `runs/pes2027/E2-CHR/standalone.json` |
| E2 vs persistence | gap | −8.045; CI excludes 0 | newly_measured | same |
| E2 vs xgb_exog | gap | +2.726; CI excludes 0 | newly_measured | Chronos alone is worse than the local tree |
| E3 XGB (re-fit baseline) | seed-avg MAE | 15.068204 (matches original) | newly_measured | `runs/pes2027/E3-LEARAUG/summary.json` |
| E3 XGB+LEAR | seed-avg MAE | 15.099304 | newly_measured | same. Label: not strictly matched |
| E3 XGB+LEAR+Chronos | seed-avg MAE | 14.903786 | newly_measured | same |
| E3 LEAR − XGB | gap | +0.031099; CI [−0.113, 0.162]; p_boot 0.627 | newly_measured | same |
| E3 +Chronos − LEAR | gap | −0.195517; CI [−0.448, 0.138]; p_boot 0.208 | newly_measured | same. Holm family of 2 does not reject |
| E3 2024 LEAR α | selected penalty | 1e-6, at the grid edge | newly_measured | `runs/pes2027/E3-LEARAUG/folds.json` |
| E4 cache seed 0 | vs fixed xgb_exog | gap −0.149888; identical to original | newly_measured | `tables/T_e4_summary.json` |
| E4 cache seed 1 | vs fixed xgb_exog | gap −0.139605; CI excludes 0; p_boot 0.046 | newly_measured | same. **Holm family of 4:** 4×0.046=0.184. Not significant after Holm |
| E4 cache seed 2 / 3 | vs fixed xgb_exog | −0.045 / −0.129; both CIs cover 0 | newly_measured | same |
| E4 cache MAE SD | across 4 seeds | 0.0478 | newly_measured | same |
| E5 cache wall | seconds / seed | 284.6–287.2 | newly_measured | `tables/T_e5_cost.json` |
| E5 Chronos batch 1 / 256 | ms / origin | 17.86 / 3.07 | newly_measured | same; RTX A6000 |
| E5 XGB train / predict | s / fit; ms / origin | ~1.8 s; 0.003 ms | newly_measured | same |
| E6 delays 1/2/3/6 | Holm p | 0.372 / 0.372 / 0.418 / 0.324 | reproduced | `tables/T_e6_sensitivities.csv` |
| E6 hourly mean | secondary | gap −0.176; Holm p 0.0015 | reproduced | same. Does not replace the max-target headline |
| E6 HB_NORTH | secondary | gap −0.153; Holm p 0.062 | reproduced | same |
| E6 load daily-posting gate | — | not added | protocol | posting minute [MISSING] |
| E7 E2022 | matched gap | −4.756; CI excludes 0; Holm 0.0015 | newly_measured | `tables/T_e7_summary.json`. Val year is 2021 (crisis). Retrospective |
| E7 E2023 | matched gap | +0.193; CI covers 0; Holm 0.297 | newly_measured | same |
| E7 E2024 | matched gap | −0.149888; identical to original | newly_measured | same |
| E8 official windows | hours | onset 120; emergency 105; recovery 39; sum 264 | newly_measured | frozen in `protocol/e8_official_subwindows.yaml` (f53057f) before metrics |
| E8 seed-42 crisis MAE | baseline / +Chronos / persistence | 3836.73 / 3604.11 / 540.13 | reproduced | `tables/T_e8_original_seed42.csv` |
| E8 five-seed crisis MAE | same models | 3819.60 / 3606.62 / 540.13 | newly_measured | `tables/T_e8_five_seed.csv` |
| E8 emergency (official) five-seed | baseline / +Chronos / persistence | 8310.79 / 7919.06 / 368.34 | newly_measured | same. Persistence still dominates. No causal claim |

## What the writing agent must not say

- That 2024 is an untouched holdout.
- That delays are verified first-publication times.
- That Chronos never saw ERCOT prices (ERCOT Load is in Benchmark II; prices are not listed).
- That a pre-2021 tree split dates the foundation-model weights.
- That non-rejection is equivalence.
- That E4 seed 1 is the selected result (it is one member of a Holm family).
- That E7 E2022 is confirmatory (crisis sits in its validation year).
- That the crisis diagnostic shows resilience or a mechanism.
- Trading, dispatch, or bill savings.

## Unresolved

- Scientific rationale for $200: [MISSING].
- NP6-345-CD posting minute: [MISSING].
- Hornek et al. 2025 IEEE venue name: [MISSING].
- E3 2024 LEAR penalty is at the 1e-6 grid edge.
