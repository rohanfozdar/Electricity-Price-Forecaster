# Source evidence register

Access date for every page below: 2026-10-13. Paraphrases are from the retrieved text. This file does not treat a current webpage as a measurement of what was posted during 2019–2024. No first-publication timestamp for real-time prices or load was found in the study files.

The corrected model therefore does not claim a verified zero posting delay. Its primary price and load delay is one hour after interval end, which is an assumption. That assumption is the shortest delay in the pre-specified set {1, 2, 3, 6} hours. It was not chosen by comparing forecast errors.

## Real-time settlement prices

| Field | Record |
| --- | --- |
| Title | Historical RTM Load Zone and Hub Prices |
| URL | https://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-ER |
| What the page says | EMIL id NP6-785-ER, report type 13061. Historical real-time settlement-point prices for hubs and load zones. Generation frequency is weekly. First run date 2011-09-30. The cited rule is NP6.3.2(2). The EMIL record was last updated 2026-10-06. The page does not state a number of minutes after interval end. |
| Applies throughout 2019–2024? | The product existed before 2019. The page retrieved in 2026 is a current catalog entry, not a saved copy of the 2019–2024 posting rule. |
| How it enters this study | `pipelines/ercot_rt_prices.py` calls `gridstatus.Ercot.get_rtm_spp`. In the installed gridstatus package that function downloads report type 13061. `data/raw/ercot_rt_prices.parquet` is that weekly archive. |

| Field | Record |
| --- | --- |
| Title | Settlement Point Prices at Resource Nodes, Hubs and Load Zones |
| URL | https://www.ercot.com/mp/data-products/data-product-details?id=NP6-905-CD |
| What the page says | EMIL id NP6-905-CD, report type 12301. Settlement-point prices produced from SCED LMPs every 15 minutes. Generation frequency is every 15 minutes. First run date 2010-11-30. Display duration 7. The cited rule is again NP6.3.2(2). The EMIL record was last updated 2026-02-25. The page does not state how many minutes after the interval the file appears. |
| Applies throughout 2019–2024? | The product existed before 2019. This is a current catalog entry. It does not give historical posting timestamps. |
| How it enters this study | It does not. The repository does not store this feed or its publication times. |

Failed check: the price parquet has no column for a publication time, a posting time, or a revision flag. The hourly timestamp is the interval start, not the time the number became public. A zero delay in `configs/feature_availability.yaml` was labeled documented. That label is withdrawn. The delay value in the YAML file is left at zero only so older code paths stay explicit; the corrected features pass a separate positive delay.

A positive delay still uses the archived number. It changes when that number is allowed into the model. It does not reconstruct the first posted price, and it does not undo later settlement revisions.

## Load

| Field | Record |
| --- | --- |
| Title | Actual System Load by Weather Zone |
| URL | https://www.ercot.com/mp/data-products/data-product-details?id=NP6-345-CD |
| What the page says | EMIL id NP6-345-CD, report type 13101. Actual hourly load by weather zone and ERCOT total. Generation frequency is daily. First run date 2013-06-13. Display duration 31. The cited rule is NP6.3.2(4). The EMIL record was last updated 2026-02-24. The page does not say that hour t−1 is public at the start of hour t. |
| Applies throughout 2019–2024? | The product existed before 2019. The retrieved page is a current catalog entry. |
| How it enters this study | `pipelines/ercot_load.py` requests report type 13101 and stores the series as `load_actual_mw`. |

The corrected model lags this realized load. A one-hour lag is an assumption, not a verified live-display delay. A 24-hour lag is a second assumption, closer to a next-day archive, and is reported only as a sensitivity. Same-hour realized load is not a feature.

## Day-ahead prices

| Field | Record |
| --- | --- |
| Title | DAM execution language in a PUC interchange PDF, and NPRR903 |
| URLs | https://interchange.puc.texas.gov/Documents/31540_100_495900.PDF and https://www.ercot.com/mktrules/issues/NPRR903 |
| What was retrieved | The PUC PDF says ERCOT shall post day-ahead settlement-point prices on the MIS public area as soon as practicable, but no later than 13:30 in the day-ahead. The same PDF says those prices are final at 10:00 of the next business day after the operating day, unless the Board finds that a software or data error significantly affected them. The PDF is a filing copy of protocol language. This session did not establish which protocol revision it reproduces. NPRR903, posted 2018-10-02 and approved 2020-08-11 with an effective date of 2020-09-01, discusses delays of the existing 13:30 publishing deadline. It does not create that deadline. |
| Applies throughout 2019–2024? | The 13:30 deadline is treated as an existing deadline in a 2018 filing, so it predates the study window. The correction clause in the retrieved PDF was not checked against every protocol edition from 2019 through 2024. |
| How the code uses it | `src/data/asof.py` keeps day-ahead hours only after 14:00 US/Central on the previous local day, a 30-minute buffer after 13:30. |

Status: documented rule with uncertain historical applicability. The gate is later than the nominal deadline. The downloaded file can still contain a price that was corrected before 10:00 on the next business day. Those corrections are not marked.

## EIA weekly storage

| Field | Record |
| --- | --- |
| Title | Weekly Natural Gas Storage Report schedule, and an EIA explainer |
| URLs | https://ir.eia.gov/ngs/schedule.html and https://www.eia.gov/todayinenergy/detail.php?id=56900 |
| What was retrieved | The schedule page says the standard release is 10:30 a.m. Eastern on Thursday, with holiday exceptions. The exception table on the page retrieved 2026-10-13 lists dates in 2025 and 2026 only. The June 22, 2023 explainer says the report is released Thursday at 10:30 a.m. Eastern except certain holiday weeks, and that it reports working gas as of the previous Friday. |
| Applies throughout 2019–2024? | Thursday 10:30 a.m. Eastern is the stated standard, and the 2023 explainer shows that this was already the public description during the study window. The holiday calendar for 2019–2024 was not retrieved row by row. |
| How the code uses it | `configs/feature_availability.yaml` waits until the Friday 18:00 UTC after the week-ending Friday. That instant is after Thursday 10:30 a.m. Eastern in both standard time and daylight time, and it is also after the Friday-morning and Wednesday-noon exception times listed on the current schedule. |

Status: documented rule with uncertain historical applicability for the exact holiday weeks. The implementation is a later buffer, not a join on the week-ending Friday.

## What is not claimed

- No source above measures a real-time price posting delay of zero, one, two, three, or six hours during 2019–2024.
- No source reconstructs the first posted settlement price from the weekly archive.
- The current EMIL pages are operational catalog entries as of the access date.
