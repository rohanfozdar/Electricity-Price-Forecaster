#!/usr/bin/env bash
# Generate every Chronos cache the early-warning experiment uses (GPU).
#   bash scripts/run_chronos_caches.sh            # all five caches
#   bash scripts/run_chronos_caches.sh --smoke    # 64 origins each, writes to /tmp
# Existing caches are skipped, so the script can be rerun after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-64}"
INPUT="data/features/chronos_input_y.parquet"
OUTDIR="data/features/chronos"
EXTRA=()
if [[ "${1:-}" == "--smoke" ]]; then
  OUTDIR="/tmp/chronos_smoke"
  EXTRA=(--limit 64)
fi
mkdir -p "$OUTDIR"

# variant context: primary first, then the frozen sensitivities, then the T5 robustness run
for job in "bolt-small 512" "bolt-small 256" "bolt-small 1024" "bolt-base 512" "t5-small 512"; do
  read -r variant ctx <<<"$job"
  out="$OUTDIR/${variant}_ctx${ctx}.npz"
  if [[ -f "$out" ]]; then
    echo "skip $out (exists)"
    continue
  fi
  echo "=== $variant ctx=$ctx -> $out"
  python scripts/run_chronos_cache.py --variant "$variant" --context "$ctx" \
    --input "$INPUT" --out "$out" --device "$DEVICE" --batch-size "$BATCH" ${EXTRA[@]+"${EXTRA[@]}"}
done
ls -la "$OUTDIR"
