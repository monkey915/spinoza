# spinoza – RL-Trained Table Tennis Return Agent

## Overview

**spinoza** is a hybrid Rust+Python project that trains an RL agent to return table tennis serves. The Rust core provides high-speed physics simulation (aerodynamics, bounce, paddle contact, net collision). Python (Stable-Baselines3 PPO) handles training via PyO3 bindings.

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
  evaluate.py           # Detailed evaluation with per-stage breakdown
  export_replays.py     # Generate replay JSON from trained model for web viewer
web/
  index.html            # Three.js visualization + AI Replay section
  js/main.js            # Scene, animation, replay viewer with autoplay/filter
  js/physics.js         # JS port of physics engine for interactive mode
  PHYSICS.md            # Physics documentation
```

## RL Environment Design

### Two-Step Gymnasium Flow
```
reset() → simulate serve (2 bounces: server half → receiver half)
        → return 10 ball positions at ~60Hz after 2nd bounce
        → agent sees ball trajectory (implicitly encodes speed, spin, direction)

step(action) → apply 7-param paddle action at agent's chosen Y position
             → simulate return flight with net collision
             → return reward based on outcome
```

### Observation Space (30 floats)
10 frames × 3 coords (x, y, z) of ball positions sampled at ~60Hz after the serve's second bounce. Right-aligned with zero padding if fewer than 10 frames available.

### Action Space (7 continuous floats)
| Parameter | Range | Description |
|-----------|-------|-------------|
| paddle_x | 0.0 – 1.525 | Lateral position |
| paddle_y | 1.8 – 3.5 | Where along table to intercept |
| paddle_z | 0.78 – 1.40 | Paddle height (clamped ≥ contact_z) |
| tilt_x | -0.5 – 0.5 | Forward/backward paddle tilt |
| tilt_z | -0.5 – 0.5 | Left/right paddle tilt |
| swing_speed | 1.0 – 12.0 | Swing speed (m/s) |
| swing_elevation | -0.3 – 0.8 | Swing angle |

### Reward Shaping
```
Contact:       +0.3  (paddle touches ball)
Net cleared:   +0.2  (additive)
On table:      +0.5  (additive)
Placement:     0.0–0.2 bonus for edge placement
Near-miss:     0.0–0.4 gradient based on distance to table (CRITICAL for learning)
Direction:     +0.1  if ball heads toward opponent without crossing net
Hit net:       0.3 + 0.0–0.15 based on clearance ratio
Miss penalty:  -0.1 – distance×0.2
```

The near-miss gradient was the breakthrough that took success from 0% → 77%. Without it, the agent gets +0.3 for any contact but has no signal about WHERE to aim.

### Serve Generation (serve.rs)
Realistic model: ball launched from z=1.0–1.1m with downward angle (-5° to -35°), bounces on server's half (y < 1.37), clears net, bounces on receiver's half.

3 difficulty stages:
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
- Paddle_z clamped ≥ table contact_z

## Training Guide

See `training/README.md` for detailed commands and hyperparameter history.

### Quick Start
```bash
# Build PyO3 module
source .venv/bin/activate
maturin develop --release

# Train Stage 1 (no spin)
cd training
python3 train.py --n-envs 64 --difficulty 1 --total-timesteps 10000000 \
  --ent-coef 0.01 --net-arch 256 256 --lr 3e-4 --batch-size 8192 \
  --output models/ppo_stage1

# Continue with Stage 2 (spin)
python3 train.py --n-envs 64 --difficulty 2 --total-timesteps 20000000 \
  --ent-coef 0.02 --net-arch 256 256 --lr 1e-4 --batch-size 8192 \
  --load models/ppo_stage1 --output models/ppo_stage2
```

## Lessons Learned

1. **Blind agent bug**: If reset() returns zeros, the agent never sees the ball and converges to a single fixed action (std=0.000). Always return real observations.
2. **Near-miss reward gradient is essential**: Binary success/fail rewards cause policy collapse. Smooth distance-based gradients for near-misses were the breakthrough.
3. **Entropy coefficient matters**: 0.05 = too much noise for fine motor control. 0.01–0.02 works well.
4. **Action space saturation**: The agent may find a "blocking" exploit (all params at limits). Check action statistics after training with `evaluate.py`.
5. **Serve realism**: Serves must bounce on server's half first, clear the net, then bounce on receiver's half. Flat trajectories from y=0 phase through the net.

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
