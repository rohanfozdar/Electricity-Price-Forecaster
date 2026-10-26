# Venue and novelty

Access date for venue pages: 2026-10-26. Every bibliographic record that enters `VERIFIED_REFERENCES.bib` was checked against Crossref, a publisher page, an official PDF, or a product catalog on that date. Records that could not be confirmed stay out of the `.bib` or are marked `[MISSING]`.

## Selected venue

IEEE Power & Energy Society General Meeting 2027, San Francisco, 25–29 July 2027. The author selected this venue on 2026-10-25. GT&D Latin America 2027 (deadline 1 November 2026, 5 pages) was the venue that was not selected.

Official CFP: https://pes-gm.org/2027-san-francisco/2027-call-for-papers/ (accessed 2026-10-26).

| Item | Official statement | Source |
| --- | --- | --- |
| Dates / place | 25–29 July 2027, San Francisco, CA, USA | CFP page |
| Theme | Grid Modernization: Scaling the Grid for Emerging Demand | CFP page |
| Paper length | Maximum five (5) pages | CFP page; PES Authors Kit Part 3 |
| References in the limit | Yes. “Conference papers are limited to five pages including any references, attributions, figures, tables, etc.” | https://ieee-pes.org/publications/authors-kit/preparation-and-submission-of-conference-technical-papers/ (Part 3, last revised October 2018; accessed 2026-10-26) |
| Template | PES conference-paper template (Word or LaTeX). LaTeX is on the IEEE Publications site; PES says to use the PES kit links for Word, not the generic IEEE Word templates. | https://ieee-pes.org/publications/authors-kit/templates-and-sample-of-pes-technical-papers/ |
| Submission system | MIRA. Authors Kit: create a MIRA account; CFP: submission link on PES and GM sites from 6 October 2026. | https://ieee-pes.org/publications/how-to-get-published/how-to-submit-a-conference-paper/ ; CFP |
| Deadline | 10 November 2026 | CFP page; https://ieee-pes.org/events/call-for-participation/ |
| Accept/reject notice | Mid March 2027 | CFP page |
| Volume | “Up to approximately 800” conference papers accepted | CFP page |
| Presentation | Required for Xplore. Most papers: Monday poster. Top 60–80: Monday Best Conference Paper (8 min, ≤6 slides). Up to 240 more: Paper Forum (4 min, ≤4 slides). | CFP page |
| Part 1 / Part 2 | Not accepted | CFP page |
| AI disclosure | IEEE: AI-generated content must be disclosed in Acknowledgments (system, sections, level of use). Editing/grammar disclosure is recommended, not required. Reviewers must not use public AI on manuscripts. PES Publications Board restated this on 8 December 2025. | PES Transactions author page; https://ieee-pes.org/wp-content/uploads/2025/12/PES-Publications-Policies-12-11-25.pdf |
| Originality | Original work, not in review elsewhere. Conference papers later submitted to PES journals need at least 60% new content. | Authors Kit; PES conference-paper page |

MIRA was scheduled to open on 6 October 2026. This package does not confirm that a submission record already exists.

## Committee recommendation

Primary committee: **Power System Operation, Planning and Economics (PSOPE)**.

Closer fit: Power System Economics Subcommittee, Working Group on Energy Forecasting and Analytics (scope includes price forecasting, machine learning, forecast evaluation). Source: https://site.ieee.org/pes-psope/subcommittees/power-system-economics-subcommittee/working-group-on-energy-forecasting-and-analytics/ (accessed 2026-10-26).

Secondary: Power System Analysis, Computing and Economics topics that treat data availability, or the Analytics / AI-adjacent PSOPE sessions if the submission site lists them. Super-session topics on the 2027 CFP (large loads, situational awareness, storage) are not a close match to a one-hour price-forecast ablation.

This is a recommendation for the writing agent. It is not a claim that PSOPE will accept the paper.

## Scope fit

The study is an archived-data evaluation of incremental forecast accuracy under assumed publication delays. That is inside PES GM conference-paper scope (electric power and energy; forecasting is a standing PSOPE topic). The 2027 theme does not have to be the paper’s theme. Super-session alignment is optional.

Five pages including references is tight. The writing agent should keep one primary table, one design or crisis figure, and a short reference list. Extra experiments belong in the repository.

## Novelty comparison (verified neighbors only)

What this study does, if the evidence holds: it asks whether six frozen Chronos–T5-small forecast summaries improve a matched local tree for the forthcoming ERCOT hourly *maximum* hub-average real-time price, under an explicit delay assumption, with the original 2024 experiment kept distinct from later controls.

| Neighbor | What it studies | How this study differs | Status |
| --- | --- | --- | --- |
| Lago et al., *Applied Energy* 293:116983 (2021), Crossref 10.1016/j.apenergy.2021.116983 | Open-access day-ahead EPF benchmark; LEAR; best-practice rules | Different product (RT hourly max, not DA 24-hour vector), different market, different question (incremental FM features, not a new DA algorithm). The local LEAR-style lasso is not claimed to be their LEAR. | Verified |
| Ansari et al., Chronos, TMLR Oct 2024 (arXiv:2403.07815v3 PDF header; OpenReview gerNCVqqtR) | Pretrained T5 token-forecasting; Benchmark II includes ERCOT *Load* as a zero-shot series | This study uses frozen Chronos summaries as *features* for ERCOT *prices*. ERCOT prices are not on the published corpus list. 2021 is a stress diagnostic, not a claim that the weights predate 2021. | Verified from the arXiv v3 PDF and CHRONOS_PROVENANCE.md. TMLR has no Crossref DOI. |
| Hornek et al., arXiv:2506.08113 (2025 IEEE copyright line on the preprint) | Zero-shot TSFMs vs statistical/ML models on 2024 European *day-ahead* auction prices; no TSFM beat biseasonal MSTL | Different market, horizon, and estimand. They compare standalone TSFMs to retrained locals. This study holds the local tree fixed and adds six Chronos summaries. Published IEEE venue name is [MISSING]. | Preprint verified; venue [MISSING] |
| Lerch et al., *Statistical Science* 32(1):106–127 (2017), Crossref 10.1214/16-STS588 | Forecaster’s dilemma: scoring only on extreme outcomes is not a proper evaluation | Used as a caution for the $200 spike subset and the 2021 crisis slice. Those slices remain descriptive. | Verified |
| Croushore and Stark, *Journal of Econometrics* 105(1):111–130 (2001), DOI 10.1016/S0304-4076(01)00072-0 | Real-time data vintages change forecast evaluation | Motivates the study label “archived-data evaluation under assumed delays.” This study does not reconstruct ERCOT first-publication vintages. | Publisher page verified; Crossref DOI as printed |
| Diebold and Mariano (1995); Harvey, Leybourne and Newbold (1997) | Predictive-accuracy tests | Used for the paired loss differential. Primary HLN horizon is h=1; the original code used h=25. | Crossref verified |

What is *not* claimed as novel: Chronos itself; XGBoost; LEAR; a new electricity-price architecture; trading value; a first-publication ERCOT archive; proof that Chronos never saw ERCOT prices.

A defensible contribution sentence, if the writing agent keeps it, is: a matched, delay-constrained ablation of frozen Chronos summaries for an ERCOT real-time hourly-maximum target, with an independent audit of the original 2024 experiment and a bounded set of predeclared controls. That sentence is a description of the design. It is not a guarantee of acceptance.
