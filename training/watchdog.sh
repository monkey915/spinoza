#!/usr/bin/env bash
# Training Watchdog: monitors progress every 30 min, kills on degeneration.
# Usage: bash watchdog.sh <logfile> <training_pid>
# Stops training if:
#   - success drops >10% from peak
#   - miss rate rises >10% from minimum
#   - stagnant for 3 hours (6 consecutive checks with <1% change)
TRAINING_DIR="$(cd "$(dirname "$0")" && pwd)"
WATCHDOG_LOG="$TRAINING_DIR/logs/watchdog.log"

LOG="${1:?Usage: watchdog.sh <logfile> <pid>}"
TRAIN_PID="${2:?Usage: watchdog.sh <logfile> <pid>}"
CHECK_INTERVAL=1800  # 30 minutes

PEAK_SUCCESS=0
MIN_MISS=100
STAGNANT_COUNT=0
LAST_SUCCESS=0
CHECK_NUM=0

wlog() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"; }

wlog "=== Watchdog started (PID $$) ==="
wlog "  Training PID: $TRAIN_PID"
wlog "  Log: $LOG"
wlog "  Check interval: ${CHECK_INTERVAL}s (30 min)"

while true; do
    sleep $CHECK_INTERVAL

    # Check if training still running
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        if grep -q "Training complete" "$LOG" 2>/dev/null; then
            wlog "TRAINING COMPLETED NORMALLY"
            tail -10 "$LOG" | while read -r line; do wlog "  $line"; done
        else
            wlog "TRAINING CRASHED or killed externally"
            tail -5 "$LOG" | while read -r line; do wlog "  $line"; done
        fi
        break
    fi

    CHECK_NUM=$((CHECK_NUM + 1))

    # Parse latest stats
    LATEST_LINE=$(grep "step=" "$LOG" | tail -1)
    [ -z "$LATEST_LINE" ] && continue

    STEP=$(echo "$LATEST_LINE" | grep -oP 'step=\s*\K[0-9]+')
    SUCCESS=$(echo "$LATEST_LINE" | grep -oP 'success=\s*\K[0-9.]+')
    MISS=$(echo "$LATEST_LINE" | grep -oP 'miss=\s*\K[0-9.]+')
    STEP_M=$(echo "scale=0; $STEP / 1000000" | bc)

    # Update peaks
    if (( $(echo "$SUCCESS > $PEAK_SUCCESS" | bc -l) )); then
        PEAK_SUCCESS="$SUCCESS"
    fi
    if (( $(echo "$MISS < $MIN_MISS" | bc -l) )); then
        MIN_MISS="$MISS"
    fi

    # Check stagnation (±1% from last check)
    ABS_DIFF=$(echo "scale=1; a=$SUCCESS - $LAST_SUCCESS; if (a < 0) -a else a" | bc -l)
    if (( $(echo "$ABS_DIFF < 1.0" | bc -l) )); then
        STAGNANT_COUNT=$((STAGNANT_COUNT + 1))
    else
        STAGNANT_COUNT=0
    fi
    LAST_SUCCESS="$SUCCESS"

    # Check failure conditions
    DROP_FROM_PEAK=$(echo "scale=1; $PEAK_SUCCESS - $SUCCESS" | bc -l)
    MISS_RISE=$(echo "scale=1; $MISS - $MIN_MISS" | bc -l)
    STATUS="OK"
    KILL_REASON=""

    if (( $(echo "$DROP_FROM_PEAK > 10.0" | bc -l) )); then
        KILL_REASON="SUCCESS DROPPED >10% from peak (${PEAK_SUCCESS}% -> ${SUCCESS}%)"
    elif (( $(echo "$MISS_RISE > 10.0" | bc -l) )); then
        KILL_REASON="MISS RATE ROSE >10% from min (${MIN_MISS}% -> ${MISS}%)"
    elif [ "$STAGNANT_COUNT" -ge 6 ]; then
        KILL_REASON="STAGNANT for 3h (6 checks, success ~${SUCCESS}%)"
    fi

    if [ -n "$KILL_REASON" ]; then
        STATUS="KILLED"
    fi

    wlog "Check #$CHECK_NUM | ${STEP_M}M steps | success=${SUCCESS}% (peak=${PEAK_SUCCESS}%) | miss=${MISS}% (min=${MIN_MISS}%) | stagnant=${STAGNANT_COUNT}/6 | $STATUS"

    if [ -n "$KILL_REASON" ]; then
        wlog ">>> STOPPING: $KILL_REASON"
        kill "$TRAIN_PID" 2>/dev/null || true
        sleep 5
        wlog "Training killed at ${STEP_M}M steps"
        break
    fi
done

wlog "=== Watchdog ended ==="
