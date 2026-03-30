# spinoza – RL-Trained Table Tennis Return Agent

## Overview

**spinoza** is a hybrid Rust+Python project that trains an RL agent to return table tennis serves with an **aggressive playing style** (topspin, flat over net, fast returns). The Rust core provides high-speed physics simulation (aerodynamics, bounce, paddle contact, net collision). Python (Stable-Baselines3 PPO) handles training via PyO3 bindings. A Three.js web viewer visualizes replays.

## Language Policy

All code comments, documentation files, CLI output, and UI strings **must be in English**. Do not use German or any other language in the codebase.

## Architecture

```
Rust (speed)                          Python (ML ecosystem)
┌──────────────────────┐              ┌──────────────────────┐
│ physics/             │              │ training/            │
│   forces, bounce,    │   PyO3      │   env.py (Gymnasium) │
│   paddle, integrator │◄────────────│   train.py (PPO)     │
│ simulation.rs        │   maturin   │   evaluate.py        │
│ rally.rs             │              │   export_replays.py  │
│ serve.rs             │              └──────────────────────┘
│ pymodule.rs (SimEnv) │
└──────────────────────┘
         │
         ▼
┌──────────────────────┐
│ web/ (Three.js)      │
│   AI Replay viewer   │
│   Physics sim viewer │
└──────────────────────┘
```

## Coordinate System

- **Origin**: Server-side left corner of the table surface
- **+X**: To the right (table width: 0 → 1.525 m)
- **+Y**: Toward the opponent (table length: 0 → 2.74 m)
- **+Z**: Upward; table surface at z = 0.76 m, ball rests at z = 0.78 m (surface + ball radius)
- **Net**: at y = 1.37 m (table center), top at z = 0.9125 m

### Spin Convention
- **Return direction**: -Y (toward opponent/server)
- **Topspin**: omega.x > 0 (right-hand rule: thumb +X, top of ball moves in -Y = forward)
- **Backspin**: omega.x < 0
- Ball acquires ~77 rad/s backspin from table bounce friction (physically correct)

### Three.js Coordinate Mapping
- `s2t(sx, sy, sz) = Vector3(sx, sz, sy)` — swaps Y↔Z (Z-up → Y-up)
- **IMPORTANT**: Angular velocity (omega) is a pseudo-vector. The Y/Z swap (det=-1) requires negating all components: `omega_threejs = s2t(-ωx, -ωy, -ωz)`
- Positions use `s2t()` directly (no negation needed)

## Project Structure

```
Cargo.toml              # Rust 2024 edition; cdylib+rlib, optional pyo3 feature
pyproject.toml          # Maturin build config for Python bindings
src/
  main.rs               # CLI entry point (clap args, output formatting)
  lib.rs                # Library crate, exports all modules, conditional PyO3
  pymodule.rs           # PyO3 bindings: SimEnv class (reset/step/replay)
  simulation.rs         # Core sim loop: RK4 stepping, bounce detection, net collision
  rally.rs              # RL episode logic: prepare_serve, evaluate_return, reward shaping
  serve.rs              # Randomized serve generator with 3 difficulty stages
  table.rs              # ITTF table geometry + helpers (contact_z, net_y, net_top_z)
  physics/
    mod.rs              # Module declarations
    state.rs            # Vec3 + BallState (pos, vel, omega)
    constants.rs        # Ball mass/radius, air density, drag/lift coefficients
    forces.rs           # Gravity + drag + Magnus effect; spin decay
    integrator.rs       # RK4 integrator for (pos, vel, omega)
    bounce.rs           # Table bounce: restitution, friction (slip vs. grip), spin
    paddle.rs           # Paddle contact: 7-param action, tilt, swing physics
training/
  env.py                # Gymnasium wrapper (TableTennisEnv)
  train.py              # PPO training script with SubprocVecEnv, progress logging
  train_lstm.py         # LSTM variant training script
  evaluate.py           # Detailed evaluation with per-stage breakdown
  export_replays.py     # Generate replay JSON from trained model for web viewer
  run_all_stages.sh     # Full curriculum pipeline: Stage 1→2→3
web/
  index.html            # Three.js visualization + AI Replay section
  js/main.js            # Scene, animation, replay viewer with autoplay/filter
  js/physics.js         # JS port of physics engine for interactive mode
  PHYSICS.md            # Physics documentation
```

## RL Environment Design

### Single-Step Episodes
Each episode is one serve → one paddle action → reward. No sequential decisions.
This is why MLP works fine (LSTM doesn't help — it operates across steps, not within an observation).

### Gymnasium Flow
```
reset() → simulate serve (2 bounces: server half → receiver half)
        → return 30 ball positions at ~60Hz after 2nd bounce
        → agent sees ball trajectory (implicitly encodes speed, spin, direction)

step(action) → apply 7-param paddle action at agent's chosen Y position
             → simulate return flight with net collision
             → return reward based on outcome
```

### Observation Space (90 floats)
30 frames × 3 coords (x, y, z) of ball positions sampled at ~60Hz after the serve's second bounce. Right-aligned with zero padding if fewer than 30 frames available.

### Action Space (7 continuous floats)
| Parameter | Range | Description |
|-----------|-------|-------------|
| paddle_x | 0.0 – 1.525 | Lateral position |
| paddle_y | 1.8 – 3.5 | Where along table to intercept |
| paddle_z | 0.85 – 1.40 | Paddle height (min raised to avoid table clipping) |
| tilt_x | -0.8 – 0.8 | Forward/backward paddle tilt (<0 = open, >0 = closed) |
| tilt_y | -0.8 – 0.8 | Left/right paddle tilt |
| swing_speed | 1.0 – 12.0 | Swing speed (m/s) |
| swing_elevation | -0.3 – 0.7 | Swing angle (>0 = upward, for topspin) |

### Reward Shaping — Aggressive Profile v6

The reward is designed so that **topspin success is the only profitable strategy**:

```
Base rewards (on success):
  Contact:       +0.30  (paddle touches ball)
  Net cleared:   +0.20
  On table:      +0.50
  → Total base:   1.00

Quality bonuses (only on success):
  Apex timing:   0.00–1.00  hitting ball at peak = maximum control
  Net-skim:     -0.50–1.00  sweet spot 2–5cm over net, lob = penalty
  Return speed: -0.20–0.30  fast returns rewarded, blocks penalized
  TOPSPIN:       0.00–3.00  ← MAIN INCENTIVE
  Backspin:      0.00–-1.00 ← punishment, but not so harsh agent avoids table
  Placement:     0.00–0.20  edge/baseline bonus

Failure rewards:
  Near miss:     0.30–0.90  gradient based on distance to table
  Hit net:       0.30–0.45  based on clearance ratio
  Paddle miss:  -0.10–-0.30 distance-based

Gradient: Topspin(5.50) >>> Neutral(2.50) >> Block(0.50) > Near-miss(0.30)
```

**Key design insight**: The expected value of attempting topspin (20% success × 5.0 reward = 1.0) must exceed the EV of blocking (100% success × 0.5 reward = 0.5). This forces the agent to learn topspin even though it's risky.

### Serve Generation (serve.rs)
Realistic model: ball launched from z=1.0–1.1m with downward angle, bounces on server's half, clears net, bounces on receiver's half.

3 difficulty stages (curriculum learning):
- **Stage 1**: Speed 5–7 m/s, no spin
- **Stage 2**: Speed 5–10 m/s, topspin/backspin up to 100 rad/s
- **Stage 3**: Speed 4–14 m/s, heavy spin (200+ rad/s), sidespin

## Physics Model

### Flight (forces.rs, integrator.rs)
- **Gravity**: -9.81 m/s² in Z
- **Drag**: `F_D = -½·C_D·ρ·A·|v|·v` (C_D = 0.40)
- **Magnus**: `F_M = C_L·ρ·A·r·(ω × v)` (C_L = 0.60)
- **Integration**: RK4 with dt = 0.5 ms

### Bounce (bounce.rs)
- Normal restitution: e_n = 0.93
- Tangential: slip vs. grip (μ = 0.25)
- Ball contacts surface at z = surface_z + ball_radius (0.78 m)

### Net Collision (simulation.rs)
- Detected when ball crosses y = net_y with z < net_top_z
- Returns SimOutcome::HitNet

### Paddle (paddle.rs)
- 7-param action with tilt and swing
- Restitution e_n = 0.85, friction μ = 0.45
- Hit detection uses `paddle_radius + BALL_RADIUS` (not point-ball)
- PADDLE_RADIUS = 0.15m (physics detection zone, NOT visual size)
- Paddle_z clamped ≥ table contact_z + 0.09m (avoid table clipping)
- **Biomechanical dampening**: `effective_speed = speed × cos(elevation)^0.6` prevents unrealistic upward swing speeds

## Build & Train

### Quick Start
```bash
# Build PyO3 module
maturin build --release --features python
pip install target/wheels/spinoza-*.whl --force-reinstall

# Train Stage 1 (no spin, 20M steps)
cd training
python3 train.py --n-envs 64 --difficulty 1 --total-timesteps 20000000 \
  --lr 3e-4 --lr-final 1e-5 --ent-coef 0.02 --net-arch 512 512 \
  --n-steps 256 --batch-size 8192 --n-epochs 10 \
  --log-interval 100000 --output models/ppo_stage1

# Continue with Stage 2 (spin)
python3 train.py --n-envs 64 --difficulty 2 --total-timesteps 25000000 \
  --lr 1e-5 --ent-coef 0.02 --net-arch 512 512 \
  --n-steps 256 --batch-size 8192 --n-epochs 5 --target-kl 0.05 \
  --load models/ppo_stage1 --output models/ppo_stage2

# Full curriculum (automated)
bash run_all_stages.sh
```

### Export Replays for Web Viewer
```bash
python3 export_replays.py models/ppo_stage1 -o ../web/replays.json -n 50 -d 1
```
**Note**: `SimEnv.replay(action)` generates a NEW random serve — it does NOT replay the last `step()`. Use the replay's own `reward` and `outcome` fields, not the values from `env.step()`.

### Web Viewer
```bash
cd web && python3 -m http.server 8080
# Open http://localhost:8080
```

## Training Hyperparameters
| Parameter | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| Steps | 20M | 25M | 40M |
| LR | 3e-4 → 1e-5 | 1e-5 | 5e-6 |
| Entropy | 0.02 | 0.02 | 0.02 |
| Network | MLP 512×512 | MLP 512×512 | MLP 512×512 |
| Batch | 8192 | 8192 | 8192 |
| n_steps | 256 | 256 | 256 |
| n_epochs | 10 | 5 | 5 |
| target_kl | — | 0.05 | 0.01 |
| n_envs | 64 | 64 | 64 |
| Speed | ~3k steps/sec | ~3k steps/sec | ~3k steps/sec |

## Lessons Learned

1. **Blind agent bug**: If reset() returns zeros, the agent never sees the ball and converges to a single fixed action (std=0.000). Always return real observations.
2. **Near-miss reward gradient is essential**: Binary success/fail rewards cause policy collapse. Smooth distance-based gradients for near-misses were the breakthrough.
3. **Entropy coefficient matters**: 0.05 = too much noise for fine motor control. 0.01–0.02 works well.
4. **Blocking exploit**: Agent finds local optimum: max closed face + downward swing = safe lob with backspin. Gets ~1.0 reward without topspin. Fix: make topspin the dominant reward component and penalize backspin.
5. **Expected Value trap**: If block EV (100% × low_reward) > topspin EV (20% × high_reward), agent will always block. Topspin bonus must be high enough that even at low success rates, topspin EV exceeds blocking EV.
6. **Backspin penalty balance**: Too harsh (-3.0) → agent prefers missing the table entirely over risking backspin. Moderate penalty (-1.0) + high topspin bonus (+3.0) works better.
7. **Action fixation diagnostic**: After training, check action statistics. If std < 0.01 on 3+ parameters, agent is stuck in a degenerate strategy. The key params to watch: tilt_x, swing_speed, swing_elevation.
8. **Pseudo-vector transform bug**: When swapping coordinate axes (Z-up → Y-up), angular velocity must be negated (det(M) = -1). Positions transform normally.
9. **Replay export mismatch**: `SimEnv.replay(action)` uses a different random serve than `env.step()`. Never mix rewards/outcomes between them.
10. **Serve realism**: Serves must bounce on server's half first, clear the net, then bounce on receiver's half.

## CLI Usage

```bash
cargo run -- [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-v, --speed` | Launch speed (m/s) | 8.0 |
| `-e, --elevation` | Elevation angle (degrees, negative=downward) | 10.0 |
| `-a, --azimuth` | Azimuth from +Y axis (degrees) | 0.0 |
| `--topspin/--backspin/--sidespin` | Spin (rad/s) | 0.0 |
| `--trajectory` | Print full trajectory CSV | false |
