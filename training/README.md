# Training Guide

## Setup

```bash
# Create virtualenv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install torch stable-baselines3 gymnasium maturin numpy

# Build the Rust PyO3 module
cd /path/to/spinoza
maturin develop --release

# Verify
python3 -c "from spinoza import SimEnv; print('OK')"
```

## Training Commands

### Stage 1: No Spin (baseline)
```bash
cd training
python3 train.py --n-envs 64 --difficulty 1 --total-timesteps 10000000 \
  --ent-coef 0.01 --net-arch 256 256 --lr 3e-4 --n-steps 256 \
  --batch-size 8192 --log-interval 100000 --output models/ppo_stage1
```

### Stage 2: Topspin/Backspin (curriculum from Stage 1)
```bash
python3 train.py --n-envs 64 --difficulty 2 --total-timesteps 20000000 \
  --ent-coef 0.02 --net-arch 256 256 --lr 1e-4 --n-steps 256 \
  --batch-size 8192 --log-interval 500000 \
  --load models/ppo_stage1 --output models/ppo_stage2
```

### Stage 3: Full Spin (curriculum from Stage 2)
```bash
python3 train.py --n-envs 64 --difficulty 3 --total-timesteps 30000000 \
  --ent-coef 0.02 --net-arch 256 256 --lr 5e-5 --n-steps 256 \
  --batch-size 8192 --log-interval 500000 \
  --load models/ppo_stage2 --output models/ppo_stage3
```

## Evaluation
```bash
python3 evaluate.py models/ppo_stage1 --difficulty 1 --n-episodes 500
```

## Generate Replays for Web Visualization
```bash
python3 export_replays.py models/ppo_stage1 -o ../web/replays.json -n 50 -d 1
```

## Hyperparameter Notes

### What Works
- **MLP [256, 256]**: 150k params, ~250μs inference. Sufficient for this task.
- **ent_coef 0.01**: Good balance of exploration vs. exploitation for Stage 1
- **ent_coef 0.02**: Slightly more exploration for spin stages (new trajectories)
- **batch_size 8192 with 64 envs**: Good throughput (~9-10k steps/sec on 64-core)
- **lr 3e-4 for fresh training, 1e-4 for fine-tuning**: Standard PPO rates

### What Doesn't Work
- **ent_coef 0.05**: Too much noise — agent can't learn fine paddle control
- **No near-miss reward gradient**: Agent gets +0.3 for any contact but has zero signal about WHERE to aim → converges to 0% success despite 70% contact rate
- **reset() returning zeros**: Agent is blind, converges to single fixed action

### Training History (pre-physics-fix, for reference)

These results were with OLD physics (no net collision, no ball radius, unrealistic serves).
After the physics fix, all models need retraining from scratch.

| Model | Steps | Difficulty | Success | Notes |
|-------|-------|-----------|---------|-------|
| v1 (blind) | 10M | 1 | 46% (fake) | Agent blind: reset() returned zeros |
| v2 (fixed obs) | 10M | 1 | 0% | Saw ball but no aim gradient |
| v3 (+ reward) | 10M | 1 | 77% | Near-miss gradient breakthrough |
| v4 (30M) | 40M total | 1 | 97% | Blocking strategy (all params at limits) |
| stage2_v1 | 20M | 2 | ~85% | Spin, curriculum from v4 |

### Known Issues with Pre-Fix Training
1. **Action saturation**: Agent found blocking exploit — paddle_z, tilt_x, swing_speed, swing_elevation all at action space limits. All returns look identical.
2. **Return diversity**: Need to check if post-physics-fix training produces more varied actions.

## Architecture Details

### Environment (env.py)
- `TableTennisEnv`: Gymnasium wrapper around Rust `SimEnv`
- Single-step episodic: reset() → observe ball → step(action) → reward → done
- `make_env(seed, difficulty)`: Factory for SubprocVecEnv

### Rust SimEnv (pymodule.rs)
- `reset()`: Generate serve → simulate 2 bounces → return 30-float observation
- `step(action)`: Apply 7-param paddle → evaluate return → reward + auto-reset
- `replay(action)`: Full trajectory data for visualization (uses last serve from reset)

### Reward Function (rally.rs: evaluate_return_result)
The reward has smooth gradients at every level:
1. **Miss** (paddle doesn't reach ball): -0.1 – distance×0.2
2. **Contact** (paddle touches ball): +0.3 base
3. **Direction bonus** (ball heads toward opponent): +0.1
4. **Hit net** (ball hits net): +0.3 + clearance_ratio×0.15
5. **Crossed net** but missed table: +0.3 + 0.2 + near_miss_bonus (up to 0.4)
6. **Success** (landed on opponent's half): +0.3 + 0.2 + 0.5 + placement_bonus

The near-miss bonus (`(1.0 - dist/2.0) × 0.4`) computes where the ball would land on the ground plane and how far that is from the table. This creates a smooth gradient from "hit ball randomly" to "hit ball onto table".
