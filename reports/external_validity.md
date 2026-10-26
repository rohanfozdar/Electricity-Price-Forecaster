# External validity

No second market is in this study. HB_NORTH is another ERCOT hub. A previous transfer reused the HB_HUBAVG Chronos cache on HB_NORTH prices. That transfer is not an independent hub model, and it is not repeated here.

HB_NORTH, with its own one-hour-delay prices and its own Chronos cache, has a seed-averaged gap of $-0.15, interval [$-0.29, $-0.02], excludes zero: True. The difference of the mean seed MAEs is $-0.16.

The timed batch implied about 26.01 minutes for one full cache through 2024. The hub cache, when present, is `runs/pit_verified/chronos/hb_north_delay_1h.parquet`.

No 2025 or 2026 score is part of this record. Those years remain on disk from the earlier ablation and are not used here.
