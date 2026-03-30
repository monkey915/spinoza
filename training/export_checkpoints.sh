#!/usr/bin/env bash
# Export replays from all checkpoints to visualize training progress.
# Creates one JSON per checkpoint → web/replays_progress/ckpt_NNNN.json
set -eo pipefail
cd "$(dirname "$0")"

CKPT_DIR="models/ppo_aggressive_stage1_checkpoints"
OUT_DIR="../web/replays_progress"
DIFFICULTY=1
N_REPLAYS=20

mkdir -p "$OUT_DIR"

echo "=== Exporting checkpoint replays ==="
echo "  Source: $CKPT_DIR"
echo "  Output: $OUT_DIR"
echo ""

# Also export the final model
FINAL="models/ppo_aggressive_stage1.zip"

for ckpt in "$CKPT_DIR"/ckpt_*_steps.zip; do
    [ -f "$ckpt" ] || continue
    name=$(basename "$ckpt" .zip)
    steps=$(echo "$name" | grep -oP '\d+')
    out="$OUT_DIR/${name}.json"

    echo -n "  $name (${steps} steps) ... "
    python3 export_replays.py "$ckpt" -o "$out" -n "$N_REPLAYS" -d "$DIFFICULTY" 2>&1 | tail -1
done

# Final model
if [ -f "$FINAL" ]; then
    echo -n "  final model ... "
    python3 export_replays.py "$FINAL" -o "$OUT_DIR/final.json" -n "$N_REPLAYS" -d "$DIFFICULTY" 2>&1 | tail -1
fi

echo ""
echo "=== Done! Replay files in $OUT_DIR ==="
ls -lh "$OUT_DIR"/*.json 2>/dev/null
