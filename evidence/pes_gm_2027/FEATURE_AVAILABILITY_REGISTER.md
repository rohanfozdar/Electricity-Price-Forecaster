# Feature-availability register

Access date for live pages: 2026-10-26. The study is an **archived-data evaluation under assumed delays**. A delay applied to a final archive is not proof of first-publication validity. Historical first-print vintages for real-time prices and load were not recovered.

Machine-readable rows: `FEATURE_AVAILABILITY_REGISTER.csv` (one row per feature family). Source captures and extracts: `wp2_raw/`.

## Products checked

| Product | EMIL / report | Generation frequency (current catalog) | In this study |
| --- | --- | --- | --- |
| Historical RTM Load Zone and Hub Prices | NP6-785-ER / 13061 | Chron-Weekly; first run 2011-09-30; rule NP6.3.2(2) | Yes. `pipelines/ercot_rt_prices.py` via `gridstatus.Ercot.get_rtm_spp`. |
| Settlement Point Prices at Resource Nodes, Hubs and Load Zones | NP6-905-CD / 12301 | Chron-15 Minutes; display duration 7 | No. This is the live 15-minute feed. It is not stored. |
| Historical DAM Load Zone and Hub Prices | NP4-180-ER / 13060 | Chron-Weekly; first run 2011-10-01; rule NP4.5.3(2)(b) | The on-disk DA file is a gridstatus DAM extract, not this weekly workbook. The product exists and is the historical DAM analog of NP6-785-ER. |
| DAM Settlement Point Prices | NP4-190-CD / 12331 | Event-Per DAM Run; display duration 31 | Not stored. Current catalog cites the same 4.5.3(2)(b) rule. |
| Actual System Load by Weather Zone | NP6-345-CD / 13101 | Chron-Daily; display duration 31; first run 2013-06-13 | Intended load source in `pipelines/ercot_load.py`. The on-disk parquet does not match Native_Load (see Load). |
| Hourly Load Data Archives (Native_Load) | gridinfo/load/load_hist (not an EMIL id) | Annual workbooks; previous month typically posted around the 9th | Retrieved for 2020 and 2024 as the timestamp-convention check. Not the on-disk series. |

Current EMIL pages are operational catalog entries as of the access date. They do not give 2019–2024 posting timestamps.

The 2011 settlements notice that introduced NP6-785-ER and NP4-180-ER says the weekly files run on Sundays, replace the current-year file each week, and that the initial implementation did not include price corrections. A 2012 follow-up notice later updated historical files for corrections. That is a posting cadence, not a first-print clock.

## Day-ahead 13:30 deadline

Current Nodal Protocols Section 4 (file `04-080126_Nodal.docx`, listed as Aug 1, 2026 on the current-protocols page), §4.5.2(2):

> As soon as practicable, but no later than 1330, ERCOT shall post on the ERCOT website the hourly: … (b) DASPPs for each Settlement Point for each hour of the Operating Day.

§4.5.2(6):

> All DAM LMPs, MCPCs, and Settlement Point Prices are final at 1000 of the second Business Day after the Operating Day.

NPRR903 (posted 2018-10-02, Board approved 2020-08-11, effective 2020-09-01) does not create the 13:30 deadline. Its description treats “the 1330 DAM publishing deadline” as already existing, requires an Advisory if that deadline is delayed, and memorializes 1900 as the latest time DAM can publish.

The code waits until 14:00 US/Central on D−1, a 30-minute buffer after 13:30. The gate is later than the nominal deadline on a timely day. If DAM is delayed past 14:00, the code still unlocks at 14:00 and would use the archived price.

A 2011 Board item still used “10:00 of the next Business Day.” That is not the current protocol text. The current text is the second Business Day.

## Real-time price corrections and the 2020-12-31 gap

Current Section 6 (`06-082826_Nodal.docx`) §6.3(7):

> All Real-Time LMPs, Real-Time Settlement Point Prices, … are final at 1600 of the second Business Day after the Operating Day.

Board review after that time remains possible with notice within 30 days. The on-disk price file has no correction or vintage column.

The 2020 annual NP6-785-ER file (MIS DocID 751356439, published 2021-01-01 08:41:25 Central, retrieved 2026-10-26) contains **96 intervals on 2020-12-31**, hours 1–24. The on-disk file ends at 2020-12-31 06:00 UTC and resumes at 2021-01-01 06:00 UTC. That is the same `Interval Start <= DATA_END.tz_localize("US/Central")` cut as 2024-12-31. The prices exist in ERCOT’s archive.

On the 35,041 overlapping 2020 intervals, disk and the 2026-retrieved archive differ on **eight** timestamps: the two ambiguous fall-back hours on 2020-11-01 (06:00–07:45 UTC). The values are swapped across those two hours (max abs $0.83). That is a DST-order disagreement, not a Board price correction. No other overlapping interval differed by more than $1e-6.

## Load convention

`pipelines/ercot_load.py` maps integer Hour Ending k to `date + (k-1) hours` and then `standardize_datetime` localizes naive stamps as UTC. It also accepts an Interval Start column and can fall back to EIA hourly demand.

The official Native_Load workbooks (2020, 2024) use Hour Ending as a Central clock with HE=24 and a DST-labeled repeated hour (`11/03/2024 02:00 DST`). That is the market-local hour-ending convention.

The on-disk series is **not** that workbook under that convention:

| Alignment | 2024 MAE (MW) | 2024 corr | 2020 MAE (MW) | 2020 corr |
| --- | --- | --- | --- | --- |
| Native HE-as-end, hour-start Central | 1523 | 0.984 | 1477 | 0.980 |
| Native HE stamp treated as Central hour-start | 275 | 0.9997 | 101 | 0.99994 |
| Native localized as UTC | 6600–7800 | 0.51–0.68 | 6600–7500 | 0.51–0.62 |

On-disk load has 24 rows on every UTC day and a single NaN at 2021-11-07 06:00 UTC, the fall-back hour. That profile is consistent with a UTC hour-beginning series (EIA-like), not with Native_Load’s 23/25 local DST days.

Which fetch path wrote `ercot_load.parquet` is [MISSING]. The one-hour publication delay on load remains an assumption. The 24-hour lag stays a sensitivity only.

## EIA WNGSR 2019–2024 exceptions

Standard release: Thursday 10:30 a.m. Eastern, stocks as of the previous Friday (live schedule page; EIA Today in Energy, 22 June 2023).

The live page on 2026-10-26 lists 2025–2026 exceptions only. Holiday weeks in 2019–2024 were read from Wayback snapshots of the same URL. Every listed exception is either Wednesday 12:00 p.m. Eastern or Friday 10:30 a.m. Eastern. Both are before the code’s Friday 18:00 UTC buffer. Table: `wp2_raw/eia_wngsr_exceptions_2019_2024.csv`.

The implementation is still a later buffer, not a join on the week-ending Friday or on the actual release clock.

## What is not claimed

- No source measures a real-time price or load posting delay of 0, 1, 2, 3, or 6 hours during 2019–2024.
- No source reconstructs the first posted settlement price from the weekly archive.
- A delay applied to a final archive does not make that archive a first print.
- Current EMIL pages are 2026 catalog entries.
