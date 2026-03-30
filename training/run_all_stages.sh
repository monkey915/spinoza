#!/usr/bin/env bash
# Spinoza – vollständiges Curriculum-Training (Stage 1 → 2 → 3)
# Alle Entscheidungen sind in entscheidungen.md dokumentiert.
set -eo pipefail
cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p models "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ── Build Rust / PyO3 Modul ─────────────────────────────────────────────────
log "=== Baue Rust-Modul (maturin build --release) ==="
cd ..
maturin build --release --features python 2>&1 | tee -a "training/$LOG_FILE"
WHEEL=$(ls target/wheels/spinoza-*.whl | head -1)
pip install "$WHEEL" --force-reinstall --break-system-packages 2>&1 | tee -a "training/$LOG_FILE"
cd training

python3 -c "import spinoza; print('spinoza OK')" | tee -a "$LOG_FILE"
log ""

# ── Stage 1: Kein Spin ───────────────────────────────────────────────────────
log "=== Stage 1: Kein Spin | 20M Steps | lr=3e-4→1e-5 (decay) | ent=0.01 ==="
# Kein target_kl: bei einer schwachen Policy ist die KL-Divergenz pro Update hoch →
# target_kl würde nach Epoch 1 abbrechen und das Lernen drastisch verlangsamen.
# 20M Steps (statt 10M) weil die neue Reward-Funktion (Qualitätsboni) das Lernen etwas verlangsamt.
python3 train.py \
    --n-envs        64 \
    --difficulty    1 \
    --total-timesteps 20000000 \
    --lr            3e-4 \
    --lr-final      1e-5 \
    --ent-coef      0.02 \
    --net-arch      512 512 \
    --n-steps       256 \
    --batch-size    8192 \
    --n-epochs      10 \
    --log-interval  100000 \
    --output        models/ppo_stage1 \
    2>&1 | tee -a "$LOG_FILE"

log "=== Replay-Export Stage 1 (50 Replays, Difficulty 1) ==="
python3 export_replays.py models/ppo_stage1 \
    -o ../web/replays_stage1.json -n 50 -d 1 \
    2>&1 | tee -a "$LOG_FILE"

log "Stage 1 abgeschlossen."
log ""

# ── Stage 2: Leichter bis mittlerer Spin ────────────────────────────────────
log "=== Stage 2: Leichter Spin | 25M Steps | lr=1e-5 | ent=0.02 (Fine-Tune von Stage 1) ==="
# target_kl=0.05 (permissiv): Stage 1 endet mit ~40-50% → Modell noch schwach genug
# dass KL-Divergenz hoch ist. 0.05 lässt Lernen zu, verhindert aber extreme Updates.
# Collapse-Risiko bei <50% Erfolgsrate gering.
python3 train.py \
    --n-envs        64 \
    --difficulty    2 \
    --total-timesteps 25000000 \
    --lr            1e-5 \
    --ent-coef      0.02 \
    --net-arch      512 512 \
    --n-steps       256 \
    --batch-size    8192 \
    --n-epochs      5 \
    --target-kl     0.05 \
    --log-interval  100000 \
    --load          models/ppo_stage1 \
    --output        models/ppo_stage2 \
    2>&1 | tee -a "$LOG_FILE"

log "=== Replay-Export Stage 2 (50 Replays, Difficulty 2) ==="
python3 export_replays.py models/ppo_stage2 \
    -o ../web/replays_stage2.json -n 50 -d 2 \
    2>&1 | tee -a "$LOG_FILE"

log "Stage 2 abgeschlossen."
log ""

# ── Stage 3: Voller Spin ─────────────────────────────────────────────────────
log "=== Stage 3: Voller Spin | 40M Steps | lr=5e-6 | ent=0.02 (Fine-Tune von Stage 2) ==="
python3 train.py \
    --n-envs        64 \
    --difficulty    3 \
    --total-timesteps 40000000 \
    --lr            5e-6 \
    --ent-coef      0.02 \
    --net-arch      512 512 \
    --n-steps       256 \
    --batch-size    8192 \
    --n-epochs      5 \
    --target-kl     0.01 \
    --log-interval  100000 \
    --load          models/ppo_stage2 \
    --output        models/ppo_stage3 \
    2>&1 | tee -a "$LOG_FILE"

log "=== Replay-Export Stage 3 (100 Replays, alle Difficulties) ==="
# Hauptdatei für Web UI: Stage-3-Modell auf voller Schwierigkeit
python3 export_replays.py models/ppo_stage3 \
    -o ../web/replays.json -n 100 -d 3 \
    2>&1 | tee -a "$LOG_FILE"
# Zusätzlich: Stage 3 Modell auch auf Difficulty 1+2 zum Vergleich
python3 export_replays.py models/ppo_stage3 \
    -o ../web/replays_stage3_d1.json -n 30 -d 1 \
    2>&1 | tee -a "$LOG_FILE"
python3 export_replays.py models/ppo_stage3 \
    -o ../web/replays_stage3_d2.json -n 30 -d 2 \
    2>&1 | tee -a "$LOG_FILE"

log "Stage 3 abgeschlossen."
log ""

# ── Finale Evaluation ────────────────────────────────────────────────────────
log "=== Finale Evaluation Stage 3 (200 Episoden pro Difficulty) ==="
python3 evaluate.py models/ppo_stage3 --stages 1 2 3 --episodes 200 \
    2>&1 | tee -a "$LOG_FILE"

log ""
log "=== Alle Stages abgeschlossen! Modelle in training/models/ ==="
log "    ppo_stage1.zip, ppo_stage2.zip, ppo_stage3.zip"
log "    Vollständiges Log: training/$LOG_FILE"
