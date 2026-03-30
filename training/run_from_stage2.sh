#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"

LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_stage2restart_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== NEUSTART Stage 2 mit lr=5e-5 (Policy Collapse bei lr=1e-4 beobachtet) ==="

python3 train.py \
    --n-envs        64 \
    --difficulty    2 \
    --total-timesteps 20000000 \
    --lr            5e-5 \
    --ent-coef      0.02 \
    --net-arch      256 256 \
    --n-steps       256 \
    --batch-size    8192 \
    --n-epochs      10 \
    --log-interval  100000 \
    --load          models/ppo_stage1 \
    --output        models/ppo_stage2 \
    2>&1 | tee -a "$LOG_FILE"

log "Stage 2 abgeschlossen."

log "=== Stage 3: Voller Spin | 30M Steps | lr=2e-5 ==="
python3 train.py \
    --n-envs        64 \
    --difficulty    3 \
    --total-timesteps 30000000 \
    --lr            2e-5 \
    --ent-coef      0.02 \
    --net-arch      256 256 \
    --n-steps       256 \
    --batch-size    8192 \
    --n-epochs      10 \
    --log-interval  100000 \
    --load          models/ppo_stage2 \
    --output        models/ppo_stage3 \
    2>&1 | tee -a "$LOG_FILE"

log "Stage 3 abgeschlossen."

log "=== Finale Evaluation (500 Episoden) ==="
python3 evaluate.py models/ppo_stage3 --difficulty 3 --n-episodes 500 \
    2>&1 | tee -a "$LOG_FILE"

log "=== Fertig! ==="
