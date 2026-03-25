# 🏓 spinoza

A physics-based table tennis ball flight & bounce simulator, written in Rust. Spinoza models aerodynamic drag, the Magnus effect (spin-induced forces), gravity, and realistic bounce mechanics on an ITTF-standard table.

Includes a **3D web visualization** built with Three.js for interactive exploration of trajectories.

## Quick Start

### CLI

```bash
cargo run -- -v 9 -e 5 --topspin 150
```

```
=== Table Tennis Ball Simulation ===

Launch position: x=0.7625 m  y=0.0000 m  z=0.9000 m
Launch velocity: vx=0.000 m/s  vy=8.966 m/s  vz=0.784 m/s
Spin (ω):        ωx=-150.0 rad/s  ωy=0.0 rad/s  ωz=0.0 rad/s

--- Impact after 0.1685 s ---
Impact point:    x=0.7625 m  y=1.4498 m  (z=0.7600 m)
...
```

### Web Visualization

Open `web/index.html` in a browser — no build step required. Adjust launch parameters, spin, and camera angle interactively.

## CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-v, --speed` | Launch speed (m/s) | 8.0 |
| `-e, --elevation` | Elevation angle above horizontal (°) | 10.0 |
| `-a, --azimuth` | Azimuth from +Y axis (°, 0 = straight) | 0.0 |
| `--topspin` | Topspin angular velocity (rad/s) | 0.0 |
| `--backspin` | Backspin angular velocity (rad/s) | 0.0 |
| `--sidespin` | Sidespin angular velocity (rad/s) | 0.0 |
| `--x0, --y0, --z0` | Launch position (m) | centre, y=0, z=0.90 |
| `--trajectory` | Print full trajectory as CSV | false |

## Physics Model

### Coordinate System

- **Origin**: Server-side left corner of the table surface
- **+X**: To the right (table width: 0 → 1.525 m)
- **+Y**: Toward the opponent (table length: 0 → 2.74 m)
- **+Z**: Upward; table surface is at z = 0.76 m

### Flight

The ball state is integrated using **RK4** (4th-order Runge-Kutta) at 0.5 ms timesteps. Three forces act on the ball during flight:

- **Gravity**: −9.81 m/s² in Z
- **Drag**: `F_D = −½ · C_D · ρ · A · |v| · v` (C_D = 0.40)
- **Magnus effect**: `F_M = C_L · ρ · A · r · (ω × v)` (C_L = 0.60) — topspin makes the ball dip, backspin makes it float, sidespin curves it sideways

Spin decays slowly via Stokes-like air friction (k_spin ≈ 5×10⁻⁷ N·m·s).

### Bounce

Based on the Gardin / Haake & Goodwill model:

- **Normal restitution**: e_n = 0.93
- **Tangential**: Computes contact-point velocity (including spin), then determines slip vs. grip
  - Friction impulse ≤ μ · normal impulse → **sticking** (rolling contact)
  - Otherwise → **sliding** (kinetic friction, μ = 0.25)
- Spin is updated from the tangential impulse

### Ball Constants (ITTF standard)

- Mass: 2.7 g
- Radius: 20 mm (40 mm diameter)
- Moment of inertia: hollow sphere `I = (2/3)·m·r²`

For a detailed derivation of the differential equations and numerical methods, see [`web/PHYSICS.md`](web/PHYSICS.md).

## Project Structure

```
src/
  main.rs                CLI entry point (clap argument parsing, output)
  simulation.rs          Simulation loop (RK4 stepping, bounce detection)
  table.rs               ITTF table geometry
  physics/
    state.rs             Vec3 and BallState types
    constants.rs         Physical constants
    forces.rs            Gravity + drag + Magnus; spin decay
    integrator.rs        RK4 integrator
    bounce.rs            Bounce model (restitution, friction, spin transfer)
web/
  index.html             Three.js 3D visualization
  PHYSICS.md             Physics documentation
  js/
    main.js              Scene, controls, animation
    physics.js           JS port of the physics engine
```

## Building

Requires Rust 2024 edition (1.85+):

```bash
cargo build --release
```

## License

MIT
