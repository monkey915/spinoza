#!/usr/bin/env bash
# auto_switch.sh – Überwacht MLP-Training, prüft Topspin-Qualität,
# und wechselt automatisch zu LSTM falls omega.x zu niedrig ist.
# KEIN set -e: Script darf nie lautlos sterben, alle Fehler werden geloggt.
#
# Ablauf:
#   1. Warte bis MLP-Training (run_all_stages.sh) fertig ist
#   2. Prüfe omega.x aus exportierten Replays
#   3. Falls omega.x < SPIN_THRESHOLD → starte LSTM-Training
#   4. Falls omega.x >= SPIN_THRESHOLD → alles gut, berichte Erfolg
#
# (kein set -e — Script muss robust laufen und darf nicht lautlos sterben)
cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
SWITCH_LOG="$LOG_DIR/auto_switch.log"

# Schwelle: mittleres omega.x muss > 30 rad/s sein (Topspin gelernt)
SPIN_THRESHOLD=30.0
REPLAY_FILE="../web/replays.json"
CHECK_INTERVAL=60

wlog() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SWITCH_LOG"; }

wlog "=== Auto-Switch gestartet (PID $$) ==="
wlog "    Topspin-Schwelle: omega.x > ${SPIN_THRESHOLD} rad/s"
wlog "    Prüft alle ${CHECK_INTERVAL}s ob MLP-Training fertig ist"

# ── Phase 1: Warte auf MLP-Training ─────────────────────────────────────────
wlog "Warte auf MLP-Training (run_all_stages.sh) ..."

wait_for_mlp_done() {
    while true; do
        # Training läuft noch?
        pids=$(ps aux | grep -E "run_all_stages\.sh|train\.py" | grep -v grep | grep -v lstm | awk '{print $2}' | tr '\n' ' ')
        if [ -z "$pids" ]; then
            wlog "MLP-Training beendet (keine Prozesse mehr aktiv)."
            return 0
        fi

        # "FERTIG" im neuesten Log?
        latest_log=$(ls -t "$LOG_DIR"/training_[0-9]*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ] && grep -q "^FERTIG$" "$latest_log" 2>/dev/null; then
            wlog "MLP-Training abgeschlossen (FERTIG in $latest_log)."
            return 0
        fi

        # Fortschritt loggen
        if [ -n "$latest_log" ]; then
            progress=$(tail -3 "$latest_log" | grep "step=" | tail -1 | \
                sed 's/.*step=\s*\([0-9]*\).*success=\s*\([0-9.]*\)%.*/step=\1 success=\2%/')
            [ -n "$progress" ] && wlog "  MLP läuft: $progress  pids=[$pids]"
        fi

        sleep "$CHECK_INTERVAL"
    done
}

wait_for_mlp_done

# ── Phase 2: Prüfe Topspin-Qualität ─────────────────────────────────────────
wlog ""
wlog "=== Prüfe Topspin-Qualität ==="

check_spin() {
    # Berechne mittleres omega.x aus replays.json
    if [ ! -f "$REPLAY_FILE" ]; then
        wlog "  WARNUNG: $REPLAY_FILE nicht gefunden!"
        echo "0"
        return
    fi

    python3 - <<'PYEOF'
import json, sys, os

replay_file = "../web/replays.json"
if not os.path.exists(replay_file):
    sys.stdout.write("0\n")
    sys.exit(0)

with open(replay_file) as f:
    data = json.load(f)

replays = data.get("replays", [])
omegas = [r["hit_omega"][0] for r in replays if "hit_omega" in r]

if not omegas:
    # Altes Format ohne hit_omega → als kein Topspin werten
    sys.stdout.write("0\n")
    sys.exit(0)

mean_omega = sum(omegas) / len(omegas)
topspin_count = sum(1 for x in omegas if x > 10)
backspin_count = sum(1 for x in omegas if x < -10)
flat_count = len(omegas) - topspin_count - backspin_count

log_path = os.environ.get("SWITCH_LOG", "/dev/stderr")
with open(log_path, "a") as log:
    log.write(f"[auto_switch] omega.x: n={len(omegas)}, mean={mean_omega:.1f} rad/s\n")
    log.write(f"[auto_switch]   Topspin(>10): {topspin_count}, Flat: {flat_count}, Backspin(<-10): {backspin_count}\n")

sys.stdout.write(f"{mean_omega:.2f}\n")
PYEOF
}

mean_spin=$(SWITCH_LOG="$SWITCH_LOG" check_spin)
wlog "  Mittleres omega.x = ${mean_spin} rad/s (Schwelle: ${SPIN_THRESHOLD})"

# ── Phase 3: Entscheidung ────────────────────────────────────────────────────
result=$(python3 -c "print('ok' if float('$mean_spin') >= $SPIN_THRESHOLD else 'lstm')" 2>/dev/null || echo "lstm")

if [ "$result" = "ok" ]; then
    wlog ""
    wlog "✅ TOPSPIN GELERNT! omega.x=${mean_spin} rad/s >= ${SPIN_THRESHOLD} rad/s"
    wlog "   MLP-Modell ist ausreichend. Kein LSTM nötig."
    wlog "   Replays verfügbar in: $REPLAY_FILE"
else
    wlog ""
    wlog "⚠️  KEIN TOPSPIN: omega.x=${mean_spin} rad/s < ${SPIN_THRESHOLD} rad/s"
    wlog "   Wechsle zu RecurrentPPO (LSTM) ..."
    wlog ""

    # Stoppe ggf. noch laufende Watchdog-Instanzen
    watchdog_pids=$(ps aux | grep watchdog | grep -v grep | awk '{print $2}' | tr '\n' ' ')
    if [ -n "$watchdog_pids" ]; then
        wlog "  Stoppe alten Watchdog: $watchdog_pids"
        for pid in $watchdog_pids; do
            kill "$pid" 2>/dev/null || true
        done
    fi

    wlog "=== Starte LSTM-Training ==="
    bash run_lstm_stages.sh 2>&1 | tee -a "$SWITCH_LOG"
fi

wlog ""
wlog "=== Auto-Switch beendet ==="
