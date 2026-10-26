# Preserved inputs

Created 2026-10-26 (UTC) by the evidence agent. These files are the audit inputs. Do not edit them; checksums are in `inputs.sha256`.

## Manuscript

The author pasted the LaTeX manuscript into the chat on 2026-10-25 22:09 (UTC-5). The paste contained three copies nested inside each other.

| File | Content | How produced |
|---|---|---|
| `manuscript_paste_raw.txt` | The full paste, byte for byte (122,304 characters). | Extracted programmatically from the chat record. |
| `manuscript_copy_A_complete.tex` | First complete copy (396 lines). | Paste offsets 125–47,944; verified as an exact substring of the raw paste. |
| `manuscript_copy_B_complete.tex` | Second complete copy (397 lines). **Audit target.** | Paste offsets 47,953–95,914; verified as an exact substring of the raw paste. |
| `manuscript_copy_C_incomplete_reconstructed.tex` | Outer, incomplete copy. | Raw paste with copies A and B removed. The fragments `frozen 2019--2024`, ` experim`, and `ent only.` join to form copy C's second line. Rejoining A, B, and C reproduces the raw paste exactly. |

### Structure of the paste

`C[0:125]` + `A` + `"\n experim"` + `B` + `"\nent only. …rest of C"`. The newline before each fragment is the trailing newline of copy A or copy B.

### Differences (`diff_A_vs_B.diff`)

Copy B differs from copy A in six places:

1. B loads `\usepackage{xurl}`.
2. Abstract: A says "Our design trains"; B says "Our retrospective design trains".
3. Abstract: A says "the mean absolute error  is" (double space, missing qualifier); B says "the mean absolute error of five-seed averaged forecasts is".
4. Table I column specification: A uses `@{}p{2.65cm}X p{2.7cm}@{}`; B uses ragged-right `p{2.45cm}`/`X`/`p{2.45cm}` columns.
5. Spike paragraph: A writes `$y_t>\$200/\MWh\` (the `\$200` is inside math mode); B writes `$y_t>200$~\USD/\MWh\`.
6. Data statement: A uses `\texttt{...\_...}` paths; B uses `\path{...}`.

No number, claim, or citation differs between A and B.

### Copy C (`diff_B_vs_C.diff`)

Copy C equals copy B's lines 1–96 followed by copy B's lines 290–397. It has B's formatting choices (xurl, ragged-right table, `\path`). It is missing B's lines 97–289. The gap runs from the "Realized ERCOT load" row of Table I to the paragraph before "\paragraph{Scope of generalization.}". Copy C contains no text that is absent from B.

### Identification

Copy B is the most complete and most recently edited version. It contains every edit present in A and C, so it is the audit target. Every manuscript claim audited in this package is quoted from copy B.

## Writing-agent handoff prompt

`writing_agent_handoff_prompt.md` holds the prompt that the author supplied on 2026-10-25 in a structured answer field (question "handoff"). The stored chat record does not keep the raw text of structured answers. The wording was preserved from the agent's working record. Line breaks and bullet layout were restored from the section structure (TARGET, START WITH AN EVIDENCE CHECK, …, DELIVER). If exact whitespace matters, replace the file with the author's original.

### Author venue selection (dated note, not part of the prompt)

On 2026-10-25 the author explicitly selected **IEEE PES General Meeting 2027**, full scope (structured answer `deadline = gm_full`). The prompt's condition "Use IEEE PES General Meeting 2027 instead only if I explicitly select it" is therefore met. On the same date the author also chose to build rolling retrospective windows from **pre-2025 data only** (`later_years = pre2025`).
