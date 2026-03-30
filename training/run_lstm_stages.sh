#!/usr/bin/env bash
# Spinoza – LSTM Curriculum-Training (Stage 1 → 2 → 3) mit RecurrentPPO
# Wird automatisch gestartet wenn MLP kein Topspin gelernt hat.
set -eo pipefail
cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p models "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/lstm_training_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== LSTM Training gestartet ==="
log ""

# ── Stage 1: Kaltstart ───────────────────────────────────────────────────────
# n_envs=32 (LSTM braucht mehr Speicher pro Env), n_steps=128 (kürzere Rollouts)
# batch_size=4096 = 32*128 = 4096 (muss n_envs*n_steps teilen)
log "=== LSTM Stage 1 | 35M Steps | lr=3e-4→1e-5 | ent=0.02 ==="
python3 train_lstm.py \
    --n-envs        32 \
    --difficulty    1 \
    --total-timesteps 35000000 \
    --lr            3e-4 \
    --lr-final      1e-5 \
    --ent-coef      0.02 \
    --lstm-hidden   256 \
    --lstm-layers   1 \
    --net-arch      256 256 \
    --n-steps       128 \
    --batch-size    4096 \
    --n-epochs      10 \
    --log-interval  100000 \
    --output        models/lstm_stage1 \
    2>&1 | tee -a "$LOG_FILE"

log "=== Replay-Export LSTM Stage 1 ==="
python3 export_replays.py models/lstm_stage1 \
    -o ../web/replays_lstm_stage1.json -n 50 -d 1 \
    2>&1 | tee -a "$LOG_FILE"

log "LSTM Stage 1 abgeschlossen."
log ""

# ── Stage 2 ───────────────────────────────────────────────────────────────────
log "=== LSTM Stage 2 | 25M Steps | Difficulty 2 | Fine-Tune ==="
python3 train_lstm.py \
    --n-envs        32 \
    --difficulty    2 \
    --total-timesteps 25000000 \
    --lr            1e-5 \
    --ent-coef      0.02 \
    --lstm-hidden   256 \
    --lstm-layers   1 \
    --net-arch      256 256 \
    --n-steps       128 \
    --batch-size    4096 \
    --n-epochs      5 \
    --target-kl     0.05 \
    --log-interval  100000 \
    --load          models/lstm_stage1 \
    --output        models/lstm_stage2 \
    2>&1 | tee -a "$LOG_FILE"

log "=== Replay-Export LSTM Stage 2 ==="
python3 export_replays.py models/lstm_stage2 \
    -o ../web/replays_lstm_stage2.json -n 50 -d 2 \
    2>&1 | tee -a "$LOG_FILE"

log "LSTM Stage 2 abgeschlossen."
log ""

# ── Stage 3 ───────────────────────────────────────────────────────────────────
log "=== LSTM Stage 3 | 40M Steps | Difficulty 3 | Fine-Tune ==="
python3 train_lstm.py \
    --n-envs        32 \
    --difficulty    3 \
    --total-timesteps 40000000 \
    --lr            5e-6 \
    --ent-coef      0.02 \
    --lstm-hidden   256 \
    --lstm-layers   1 \
    --net-arch      256 256 \
    --n-steps       128 \
    --batch-size    4096 \
    --n-epochs      5 \
    --target-kl     0.01 \
    --log-interval  100000 \
    --load          models/lstm_stage2 \
    --output        models/lstm_stage3 \
    2>&1 | tee -a "$LOG_FILE"

log "=== Replay-Export LSTM Stage 3 (Hauptdatei) ==="
python3 export_replays.py models/lstm_stage3 \
    -o ../web/replays.json -n 100 -d 3 \
    2>&1 | tee -a "$LOG_FILE"

log ""
log "=== LSTM Training komplett! ==="
log "    Modelle: models/lstm_stage1.zip, lstm_stage2.zip, lstm_stage3.zip"
log "    Log: $LOG_FILE"

echo ""
echo "FERTIG"
