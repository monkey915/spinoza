# spinoza – Table Tennis Ball Flight & Bounce Simulator

## Overview

**spinoza** is a physics-based CLI tool written in Rust that simulates the trajectory and bounce of a table tennis ball on an ITTF-standard table. It models aerodynamic drag, Magnus effect (spin-induced forces), gravity, and realistic bounce mechanics (friction, restitution, spin transfer).

## Language Policy

All code comments, documentation files, CLI output, and UI strings **must be in English**. Do not use German or any other language in the codebase.

## Coordinate System

- **Origin**: Server-side left corner of the table surface
- **+X**: To the right (table width: 0 → 1.525 m)
- **+Y**: Toward the opponent (table length: 0 → 2.74 m)
- **+Z**: Upward; table surface is at z = 0.76 m
- Default launch position: `(0.7625, 0.0, 0.90)` — centre-width, server end, 14 cm above surface

## Project Structure

```
Cargo.toml              # Rust 2024 edition, depends on clap v4
src/
  main.rs               # CLI entry point (clap-based arg parsing, output formatting)
  simulation.rs          # Main simulation loop — steps the ball forward with RK4, detects bounce
  table.rs               # ITTF table geometry (dimensions, surface intersection tests)
  physics/
    mod.rs               # Module declarations
    state.rs             # Vec3 (3D vector math) and BallState (pos, vel, omega)
    constants.rs         # Physical constants: ball mass/radius, air density, drag/lift coefficients, gravity
    forces.rs            # Aerodynamic forces: gravity + drag + Magnus effect; spin decay
    integrator.rs        # RK4 (Runge-Kutta 4th order) integrator for position, velocity, and spin
    bounce.rs            # Bounce model: normal restitution, tangential friction (slip vs. grip), spin update
web/
  index.html            # Three.js-based 3D visualization frontend
  PHYSICS.md            # Physics documentation (differential equations, RK4, bounce model)
  js/
    main.js             # Three.js scene, UI controls, animation loop
    physics.js          # JS port of the Rust physics engine
```

## Physics Model

### Flight (forces.rs, integrator.rs)

- **Gravity**: constant −9.81 m/s² in Z
- **Drag**: `F_D = -½ · C_D · ρ · A · |v| · v` (C_D = 0.40)
- **Magnus effect**: `F_M = C_L · ρ · A · r · (ω × v)` (C_L = 0.60) — creates curved trajectories from spin
- **Spin decay**: Stokes-like torque, very slow during flight (k_spin ≈ 5e-7 N·m·s)
- **Integration**: RK4 with dt = 0.5 ms, applied to position, velocity, and angular velocity simultaneously

### Bounce (bounce.rs)

- Based on Gardin / Haake & Goodwill model
- **Normal restitution**: e_n = 0.93 (coefficient of restitution)
- **Tangential**: Computes contact-point velocity (includes spin contribution), then determines slip vs. grip
  - If friction impulse ≤ μ · normal impulse → **sticking** (rolling contact)
  - Otherwise → **sliding** (kinetic friction, μ = 0.25)
- Spin is updated from the tangential impulse at the contact point

### Ball Constants

- Mass: 2.7 g, Radius: 20 mm (ITTF standard 40 mm plastic ball)
- Moment of inertia: hollow sphere `I = (2/3)·m·r²`

## CLI Usage

```bash
cargo run -- [OPTIONS]
```

### Key Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-v, --speed` | Launch speed (m/s) | 8.0 |
| `-e, --elevation` | Elevation angle above horizontal (degrees) | 10.0 |
| `-a, --azimuth` | Azimuth from +Y axis (degrees, 0 = straight) | 0.0 |
| `--topspin` | Topspin angular velocity (rad/s) | 0.0 |
| `--backspin` | Backspin angular velocity (rad/s) | 0.0 |
| `--sidespin` | Sidespin angular velocity (rad/s) | 0.0 |
| `--x0, --y0, --z0` | Launch position (m) | centre of width, y=0, z=0.90 |
| `--trajectory` | Print full trajectory as CSV | false |

### Example

```bash
# Topspin serve, 9 m/s, 5° elevation, 150 rad/s topspin
cargo run -- -v 9 -e 5 --topspin 150
```

## Simulation Logic (simulation.rs)

1. Steps the ball forward using RK4 at 0.5 ms intervals
2. At each step, checks if the ball is descending toward the table surface
3. When a surface crossing is detected within the next timestep, sub-steps exactly to the surface
4. Verifies the impact point is within table bounds (x, y)
5. Applies the bounce model and returns `SimResult` with landing point, pre/post bounce state, and full trajectory
6. Errors: `MissedTable`, `HitFloor`, `Timeout` (5 s limit)

## Development Notes

- Rust 2024 edition
- No tests yet
- `web/` directory exists but is empty — presumably planned for a visualization frontend
- The simulation currently handles only the **first bounce** — it stops after one table contact
