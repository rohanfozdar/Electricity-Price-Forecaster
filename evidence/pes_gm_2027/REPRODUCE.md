# Reproduce

Interpreter: `/net/home/rfozdar/venvs/ercot/bin/python`. Repository root: `Ercot_Forecaster/ercot-grid-forecaster`. Branch: `pes-gm-2027-evidence`. GPU: device 0 only (`CUDA_VISIBLE_DEVICES=0`). Do not write to `runs/pit_verified/`, `runs/pit_ablation/`, `runs/final_audit/`, or `reports/final_research_audit/`. Do not run `python models/run_pit_verified.py all`.

Protocol file `evidence/pes_gm_2027/protocol/protocol_v1.yaml` must be tracked, unmodified, and committed. Runners call `require_frozen()`.

## Tests

```bash
/net/home/rfozdar/venvs/ercot/bin/python -m pytest -q \
  tests/test_pes2027_stats.py tests/test_pes2027_availability.py \
  tests/test_pes2027_chronos.py tests/test_pes2027_target_splits.py \
  tests/test_pes2027_oof.py tests/test_pit_latency.py tests/test_pit_sets.py
```

## Audit of the original experiment (no new fits except E0-RERUN)

```bash
/net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e0_census
/net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e0_reproduce
/net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e0_precision
/net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e0_rerun
```

## New experiments, in order

1. Freeze protocol v1 (already committed as `b353520`).
2. Freeze E8 official windows (already committed as `f53057f`).
3. Chronos caches, seeds 0–3, delay 1 h, GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 /net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.chronos_cache --seed 0 --delay 1
# repeat --seed 1, 2, 3. Existing files are refused.
```

4. E1 selection on 2023 only, then **commit** `runs/pes2027/E1-H512/selection.json` before scoring 2024:

```bash
ERCOT_XGB_DEVICE=cpu CUDA_VISIBLE_DEVICES= /net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e1_e3 e1-select
# commit selection.json
CUDA_VISIBLE_DEVICES=0 /net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e1_e3 e1-score
```

5. Standalone Chronos and OOF LEAR:

```bash
/net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e1_e3 e2
CUDA_VISIBLE_DEVICES=0 /net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e1_e3 e3
```

6. Cache-seed trees, cost, saved-artifact sensitivities, rolling windows:

```bash
CUDA_VISIBLE_DEVICES=0 /net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e4_e7 e4
/net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e4_e7 e5
/net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e4_e7 e6
CUDA_VISIBLE_DEVICES=0 /net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e4_e7 e7
```

7. Crisis diagnostic. Official windows must already be committed.

```bash
/net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e8_crisis original
CUDA_VISIBLE_DEVICES=0 /net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e8_crisis refit
/net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e8_crisis figure
/net/home/rfozdar/venvs/ercot/bin/python -m models.pes2027.e8_crisis target-doc
```

## Outputs

All new scores are under `runs/pes2027/<ID>/`. Evidence prose is under `evidence/pes_gm_2027/`. Original run checksums: `evidence/pes_gm_2027/checksums/original_runs_before.sha256`.

## Environment

See `evidence/pes_gm_2027/env/`. The `ercot` venv is not modified.
