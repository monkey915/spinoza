#!/bin/bash
# Monitor training progress: check every 20 minutes, report stats at each 20M step milestone
LOG="logs/v6_stage4b_20260401_105621.log"
LAST_MILESTONE=0

while true; do
    if [ ! -f "$LOG" ]; then
        echo "Log not found: $LOG"
        sleep 60
        continue
    fi

    # Get latest step count
    LATEST=$(grep "step=" "$LOG" | tail -1 | grep -oP 'step=\s*\K[0-9]+')
    if [ -z "$LATEST" ]; then
        sleep 60
        continue
    fi

    # Calculate current milestone (every 20M)
    MILESTONE=$(( (LATEST / 20000000) * 20000000 ))

    if [ "$MILESTONE" -gt "$LAST_MILESTONE" ] && [ "$MILESTONE" -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "  MILESTONE: ${MILESTONE}M steps reached"
        echo "  $(date)"
        echo "=========================================="
        # Show latest stats
        grep "step=" "$LOG" | tail -3
        echo ""

        # Find latest checkpoint
        CKPT_DIR="models/ppo_stage4_checkpoints"
        if [ -d "$CKPT_DIR" ]; then
            LATEST_CKPT=$(ls -t "$CKPT_DIR"/rl_model_*_steps.zip 2>/dev/null | head -1)
            if [ -n "$LATEST_CKPT" ]; then
                echo "  Latest checkpoint: $LATEST_CKPT"
            fi
        fi

        LAST_MILESTONE=$MILESTONE
    fi

    # Check if training finished
    if grep -q "Training complete" "$LOG" 2>/dev/null; then
        echo ""
        echo "=========================================="
        echo "  TRAINING COMPLETE! $(date)"
        echo "=========================================="
        tail -10 "$LOG"
        break
    fi

    sleep 300  # check every 5 minutes
done
