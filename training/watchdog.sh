#!/usr/bin/env bash
# Autonomer Training-Watchdog: überwacht Fortschritt, greift bei Stagnation/Kollaps ein.
# KEIN set -e: Script darf nie lautlos sterben, alle Fehler werden geloggt.
TRAINING_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$TRAINING_DIR/logs"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
RESTART_SCRIPT="$TRAINING_DIR/run_all_stages.sh"

CHECK_INTERVAL=120
STAGNATION_STEPS_COLD=15000000
STAGNATION_WINDOW=5000000
COLLAPSE_THRESHOLD_HIGH=30.0
COLLAPSE_THRESHOLD_LOW=2.0
COLLAPSE_WINDOW=2000000

wlog() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"; }

get_latest_log() {
    ls -t "$LOG_DIR"/training_[0-9]*.log 2>/dev/null | head -1
}

get_training_pids() {
    # Gibt alle PIDs von laufenden Training-Prozessen zurück
    ps aux | grep -E "run_all_stages|train\.py" | grep -v grep | awk '{print $2}' | tr '\n' ' '
}

parse_progress() {
    local log="$1"
    tail -50 "$log" 2>/dev/null | grep "^  step=" | tail -1 | \
        sed 's/.*step=\s*\([0-9]*\).*success=\s*\([0-9.]*\)%.*/\1 \2/'
}

kill_training() {
    local reason="$1"
    wlog "EINGRIFF: $reason"
    local pids
    pids=$(get_training_pids)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            kill "$pid" 2>/dev/null || true
        done
        sleep 3
        pids=$(get_training_pids)
        for pid in $pids; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
    wlog "Training gestoppt."
}

restart_training() {
    wlog "Starte Training neu..."
    nohup bash "$RESTART_SCRIPT" >> "$WATCHDOG_LOG" 2>&1 &
    local new_pid=$!
    wlog "Training gestartet (PID $new_pid)"
}

wlog "=== Watchdog gestartet (PID $$) ==="

history_step=0
history_success=0
restarts=0
MAX_RESTARTS=3

while true; do
    sleep "$CHECK_INTERVAL"

    log=$(get_latest_log)
    [ -z "$log" ] && { wlog "Kein Log gefunden."; continue; }

    progress=$(parse_progress "$log")
    [ -z "$progress" ] && { wlog "Kein Fortschritt im Log."; continue; }

    cur_step=$(echo "$progress" | awk '{print $1}')
    cur_success=$(echo "$progress" | awk '{print $2}')
    pids=$(get_training_pids)

    wlog "step=$cur_step success=${cur_success}% pids=[${pids:-keine}]"

    # Training nicht mehr aktiv
    if [ -z "$pids" ]; then
        wlog "Training nicht aktiv. Warte..."
        sleep 300
        continue
    fi

    [ "$restarts" -ge "$MAX_RESTARTS" ] && { wlog "Max Restarts erreicht, Watchdog stoppt."; exit 1; }

    # Kaltstart-Stagnation: nach 3M Steps < 1% success
    if [ "$cur_step" -ge "$STAGNATION_STEPS_COLD" ] && \
       [ "$(echo "$cur_success < 1.0" | bc -l)" = "1" ]; then
        kill_training "Kaltstart-Stagnation: ${cur_success}% nach ${cur_step} Steps"
        restarts=$((restarts + 1))
        sleep 5; restart_training
        history_step=0; history_success=0; continue
    fi

    # Kollaps: von >20% auf <5% in 1M Steps
    if [ "$(echo "$history_success > $COLLAPSE_THRESHOLD_HIGH" | bc -l)" = "1" ] && \
       [ "$(echo "$cur_success < $COLLAPSE_THRESHOLD_LOW" | bc -l)" = "1" ] && \
       [ "$((cur_step - history_step))" -le "$COLLAPSE_WINDOW" ]; then
        kill_training "KOLLAPS: ${history_success}% -> ${cur_success}%"
        restarts=$((restarts + 1))
        sleep 5; restart_training
        history_step=0; history_success=0; continue
    fi

    # Stagnation: ab >10% kein Wachstum >1% in 5M Steps
    if [ "$(echo "$history_success > 10.0" | bc -l)" = "1" ] && \
       [ "$((cur_step - history_step))" -ge "$STAGNATION_WINDOW" ] && \
       [ "$(echo "$cur_success - $history_success < 1.0" | bc -l)" = "1" ]; then
        kill_training "Stagnation: ${history_success}% -> ${cur_success}% in $((cur_step-history_step)) Steps"
        restarts=$((restarts + 1))
        sleep 5; restart_training
        history_step=0; history_success=0; continue
    fi

    # History alle 1M Steps aktualisieren
    if [ "$((cur_step - history_step))" -ge 1000000 ]; then
        history_step=$cur_step
        history_success=$cur_success
    fi
done
