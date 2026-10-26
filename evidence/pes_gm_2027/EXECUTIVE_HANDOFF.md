# Executive handoff for the paper-writing agent

Date: 2026-10-26. Branch: `pes-gm-2027-evidence`. This package does not write the paper and does not submit it.

## Venue (author-selected)

IEEE PES General Meeting 2027, San Francisco, 25–29 July 2027. Deadline **10 November 2026**. Maximum **5 pages including references, figures, and tables**. Submission through **MIRA**. Official CFP: https://pes-gm.org/2027-san-francisco/2027-call-for-papers/ (accessed 2026-10-26). Details: `VENUE_AND_NOVELTY.md`.

A dated note: the handoff prompt below still names GT&D Latin America 2027 as the default. The author selected GM on 2026-10-25. Use GM.

## Research question (unchanged)

Do six frozen Chronos–T5-small forecast summaries improve forecasts of the forthcoming ERCOT hourly maximum price beyond a strong local model, under the same assumed information cutoff?

## What was already true (original 2024 experiment)

Reproduced from saved per-origin forecasts, independent statistics, 410 checks, 0 failures (`runs/pes2027/E0-REPRO/`). A full refit of the 10 primary trees matches the saved forecasts at max abs 0.0.

Headline: seed-averaged MAE 15.068204 without Chronos and 14.918317 with Chronos. Gap −0.149888. 24-hour moving-block 95% interval [−0.317435, 0.063750]. Bootstrap p (interval inversion) 0.131. Original DM p (h=25) 0.124383. The interval covers zero. This is not an equivalence result.

2024 is a retrospective year. It was inspected from June 2026 onward. The 1-hour delay was chosen after delay-0 2024 results. Label: archived-data evaluation under assumed delays.

## What the new experiments add

Protocol v1 was committed (`b353520`) before any new score. E8 official windows were committed (`f53057f`) before sub-window metrics. E1 selection was committed (`d027226`) before 2024 scoring.

- **E1.** Representation selected on 2023 only: **R1** (dense lags). 2024 B512 MAE 15.105; B512+Chronos 14.885; gap −0.220, CI covers 0 (p 0.142). Same qualitative conclusion as the original 37-feature baseline.
- **E2.** Standalone Chronos median MAE 17.794 on the same 8,767 hours. Better than persistence (25.840), worse than the local tree (15.068).
- **E3.** Not strictly matched. XGB 15.068; XGB+LEAR 15.099 (gap +0.031, CI covers 0); XGB+LEAR+Chronos 14.904 (gap −0.196 vs LEAR, CI covers 0). The 2024 LEAR penalty is at the 1e-6 grid edge.
- **E4.** Regenerated torch seed 0 reproduces the headline gap. Seeds 1–3 give gaps −0.140, −0.045, −0.129. Seed 1’s unadjusted interval excludes zero; Holm on the four-seed family does not (adjusted p 0.184). Do not promote seed 1.
- **E5.** Cache build ~285 s/seed on one RTX A6000. Recurring Chronos cost ~3.1 ms/origin at batch 256, ~18 ms at batch 1. Tree train ~2 s.
- **E6.** Secondary reproductions. After Holm, delay and block families remain non-significant. Hourly-mean is nominally significant (Holm p 0.0015) and remains secondary. No load daily-posting gate (posting minute [MISSING]).
- **E7.** Retrospective windows. E2024 matches the original. E2023 gap +0.193, CI covers 0. E2022 gap −4.76, CI excludes 0; its validation year is 2021. Not an untouched confirmation.
- **E8.** Names corrected (delayed-history baseline, not exogenous-only). Official onset/emergency/recovery hours 120/105/39. Five-seed crisis MAE 3819.60 / 3606.62 / 540.13. Persistence still dominates the emergency slice. Diagnostic only.

## Corrections the clean paper must absorb

See `CORRECTIONS_LOG.md`. The two that change wording: “exogenous-only” is the 37-feature delayed-history tree; crisis numbers are seed 42 unless the five-seed table is used. The complete-target “refit” did not change training rows.

## Guardrails for the writing agent

Five pages including references. Prefer one primary table, one crisis or design figure, and a short bibliography from `VERIFIED_REFERENCES.bib` only. Recommended committee: PSOPE / Energy Forecasting and Analytics. Disclose AI use in Acknowledgments if the paper uses generated text.

Original run directories were not written. After-work checksum: 525/525 unchanged.

## Verbatim writing-agent prompt (supplied 2026-10-25)

The following block is the prompt the author asked to pass through unchanged, except for the GM-selection note above.

---

Write the full IEEE PES conference paper using the attached original manuscript and Opus’s completed evidence package.

TARGET

Primary venue: IEEE PES GT&D Latin America 2027.
Use IEEE PES General Meeting 2027 instead only if I explicitly select it.
Verify the current official requirements.
Aim for the strongest defensible submission; do not promise acceptance.

START WITH AN EVIDENCE CHECK

Read:
- EXECUTIVE_HANDOFF.md
- RESULTS_MANIFEST.csv
- CLAIM_EVIDENCE_MATRIX.md
- PROTOCOL_AND_METHODS.md
- CORRECTIONS_LOG.md
- AUTHOR_DECISIONS.md
- VERIFIED_REFERENCES.bib
- The relevant tables, figures, configurations, and source manuscript.

Check that each headline number has a traceable artifact, a clear estimand, and consistent units and sample counts. Do not rely solely on Opus’s narrative summary. Recompute key table arithmetic and inspect saved evidence where practical.

If a missing fact affects the central scientific conclusion, identify it before writing a clean submission version. Continue independent work on supported sections.

WRITING OBJECTIVE

Write a focused empirical paper answering:

Do frozen Chronos forecast features improve forthcoming-hour ERCOT maximum-price forecasts beyond a strong local model under explicit publication-delay assumptions?

Adjust the final emphasis to the verified evidence. Do not force a favorable result or change the research question to match whichever comparison is significant.

PRESERVE SCIENTIFIC ACCURACY

- Keep the original 2024 experiment distinct from new experiments.
- Apply only documented corrections, with an explanation outside the clean manuscript.
- Distinguish retrospective robustness from genuinely untouched confirmation.
- Do not describe assumed availability as verified real-time availability.
- Do not claim pretraining independence unless established.
- Do not interpret non-significance as equivalence.
- Do not claim economic savings, trading performance, grid resilience, or causal mechanisms without direct evidence.
- Do not turn exploratory findings into confirmatory claims.
- Use [MISSING: ...] and [CITATION NEEDED] in the review copy when needed.
- Never hide a central unresolved limitation to make the paper appear ready.

WRITE THE COMPLETE PAPER

Include:
- A precise, restrained title.
- Abstract and keywords.
- A clear power-market problem and target definition.
- Focused related work and a supported contribution statement.
- Data and publication-availability assumptions.
- Sufficient methods for a reviewer to assess the experiment.
- Main results, uncertainty, and selected robustness evidence.
- A crisis diagnostic only if its protocol is verified and it supports the central argument.
- Concise limitations.
- A conclusion limited to the evidence.
- Verified references and an appropriate reproducibility statement.

Use plain, precise academic English. Follow ASD-STE100 principles where they improve clarity without distorting technical meaning. Keep most sentences under 25 words. Use consistent terms and retain good original prose.

Remove repetitive defenses of the null result, generic transitions, inflated novelty claims, and statements about how responsible the authors are. Let the evidence establish credibility.

FORMAT AND VALIDATE

Use the official conference template and verified page limit. Include figures and references within that limit where required. Do not meet the limit by shrinking fonts or margins.

Select the strongest evidence and move nonessential detail to the repository while retaining essential methods in the paper.

Compile the LaTeX. Inspect every rendered page. Check equations, table values, captions, references, units, legibility, overflow, and consistency between the abstract, results, and conclusion.

DELIVER

1. Complete LaTeX source with bibliography and figure assets.
2. A compiled PDF, if the manuscript is scientifically complete.
3. A separate review memo with section-by-section rationale, documented corrections, remaining limitations, and reviewer risks.
4. A separate annotated review copy if unresolved questions remain.

A clean submission copy must contain no unresolved placeholders. If central evidence is missing, label the paper a review draft and state exactly what prevents submission.

Do not submit the paper or contact the venue. Finish with a brief readiness assessment based on verified evidence.
