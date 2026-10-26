# WP2 primary-source access log

Access date for every live page below: 2026-10-26. Wayback snapshots keep the capture timestamp in the URL. Retrieved files are in `/tmp/pes2027_wp2` (not committed) and selected extracts are in this directory.

| Source | URL | What was retrieved |
| --- | --- | --- |
| EMIL NP6-785-ER | https://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-ER | Historical RTM Load Zone and Hub Prices; report 13061; Chron-Weekly; first run 2011-09-30; rule NP6.3.2(2); EMIL last updated 2026-10-06 |
| EMIL NP6-905-CD | https://www.ercot.com/mp/data-products/data-product-details?id=NP6-905-CD | Settlement Point Prices at Resource Nodes, Hubs and Load Zones; report 12301; Chron-15 Minutes; first run 2010-11-30; display duration 7; rule NP6.3.2(2) |
| EMIL NP6-345-CD | https://www.ercot.com/mp/data-products/data-product-details?id=NP6-345-CD | Actual System Load by Weather Zone; report 13101; Chron-Daily; first run 2013-06-13; display duration 31; rule NP6.3.2(4) |
| EMIL NP4-180-ER | https://www.ercot.com/mp/data-products/data-product-details?id=np4-180-er | Historical DAM Load Zone and Hub Prices; report 13060; Chron-Weekly; first run 2011-10-01; rule NP4.5.3(2)(b) |
| EMIL NP4-190-CD | https://www.ercot.com/mp/data-products/data-product-details?id=NP4-190-CD | DAM Settlement Point Prices; report 12331; Event-Per DAM Run; display duration 31; rule NP4.5.3(2)(b) |
| NPRR903 | https://www.ercot.com/mktrules/issues/NPRR903 | Posted 2018-10-02; Board approved 2020-08-11; effective 2020-09-01; treats the 1330 DAM publishing deadline as already existing |
| Current Nodal Protocols index | https://www.ercot.com/mktrules/nprotocols/current/ | Section 4 listed Aug 1, 2026; Section 6 listed Aug 28, 2026 |
| Nodal Protocols Section 4 | https://www.ercot.com/files/docs/2024/06/28/04-080126_Nodal.docx | Extracted `protocol_section4_paragraphs.txt`; 4.5.2 posts DASPPs no later than 1330; prices final at 1000 of the second Business Day |
| Nodal Protocols Section 6 | https://www.ercot.com/files/docs/2024/06/28/06-082826_Nodal.docx | Extracted `protocol_section6_paragraphs.txt`; RT Settlement Point Prices final at 1600 of the second Business Day |
| 2011 settlements notice | https://lists.ercot.com/cgi-bin/wa?A3=1109&B=--_000_B117FDA9B7BC68479362C1197F77D879062A1BCPW0005ercotcom_&E=quoted-printable&L=NOTICE_SETTLEMENTS&P=119827&T=text%2Fhtml%3B+charset%3Dus-ascii&XSS=3&header=1 | Weekly Sunday run of NP6-785-ER / NP4-180-ER; initial implementation did not include price corrections |
| Hourly Load Data Archives | https://www.ercot.com/gridinfo/load/load_hist | Native_Load annual zips; previous month typically posted around the 9th |
| Native_Load 2020 | https://www.ercot.com/files/docs/2021/01/12/Native_Load_2020.zip | Hour Ending Central, HE=24, one DST-labeled row |
| Native_Load 2024 | https://www.ercot.com/files/docs/2024/02/06/Native_Load_2024.zip | Same convention; 8,784 rows |
| NP6-785-ER 2020 annual file | MIS report 13061, DocID 751356439, published 2021-01-01T08:41:25-05:00 | Complete 2020-01-01 through 2020-12-31 (96 intervals on Dec 31) |
| EIA WNGSR schedule (live) | https://ir.eia.gov/ngs/schedule.html | Standard Thursday 10:30 a.m. Eastern; live table lists 2025–2026 exceptions only |
| EIA WNGSR Wayback 2019–2024 | see `eia_wngsr_exceptions_2019_2024.csv` | Contemporaneous exception tables |
| EIA Today in Energy 2023-06-22 | https://www.eia.gov/todayinenergy/detail.php?id=56900 | Thursday 10:30 a.m. eastern except certain holiday weeks; stocks as of previous Friday |
| Chronos paper (arXiv html v1) | https://arxiv.org/html/2403.07815v1 | ERCOT Load is in Benchmark II (zero-shot only); 8 regions, hourly, 2004–2021 |
| Hugging Face amazon/chronos-t5-small | https://huggingface.co/amazon/chronos-t5-small | T5-small, 46M; citation TMLR 2024 |
