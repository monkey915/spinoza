# spinoza – Differential Equations and Numerical Methods

## Coordinate System

The simulation uses a right-handed coordinate system aligned with the table:

    X  → table width       (0 to 1.525 m)
    Y  → table length      (0 to 2.74 m, serve direction = +Y)
    Z  → height            (table surface = 0.76 m, up = +Z)

    Net at y = 1.37 m.  Agent (receiver) stands at large Y (1.8 – 3.5 m).

**Spin convention — topspin direction depends on travel direction:**

    Incoming ball (travels +Y):  topspin = ωₓ < 0  (ω × v points downward)
    Return ball   (travels −Y):  topspin = ωₓ > 0  (ω × v points downward)

The Magnus force always produces downward curvature for topspin —
the sign of ωₓ just has to match the sign of vel_Y.


## The State Vector

The ball is described by 9 quantities:

    s(t) = ( x, y, z,  vₓ, vᵧ, vᵤ,  ωₓ, ωᵧ, ωᵤ )
             ─────────  ─────────────  ─────────────
             Position      Velocity        Spin

This yields a system of 9 coupled first-order ordinary differential equations.


## The 9 Differential Equations

### Group 1: Kinematics (trivial)

    dx/dt = vₓ
    dy/dt = vᵧ
    dz/dt = vᵤ

Position changes with velocity — straightforward.


### Group 2: Dynamics (where the physics lives)

    dvₓ/dt = aₓ(v, ω)
    dvᵧ/dt = aᵧ(v, ω)
    dvᵤ/dt = aᵤ(v, ω)

The acceleration a is composed of three forces:

#### a) Gravity

    a_grav = (0, 0, −9.81)  m/s²

Simply downward, constant.

#### b) Aerodynamic Drag

    F_D = −½ · C_D · ρ · A · |v| · v

    → a_drag = F_D / m = −½ · C_D · ρ · A · |v| · v / m

Drag is proportional to the square of the velocity and always acts
opposite to the direction of flight. The factor |v|·v (not v²) ensures
the correct direction.

Values: C_D = 0.40, ρ = 1.2 kg/m³, A = π·r² = π·0.02² m², m = 2.7g

#### c) Magnus Effect (Spin → Curved Trajectory)

    F_M = C_L(S) · ρ · A · r · (ω × v)

    → a_magnus = F_M / m

The cross product ω × v is the key: the Magnus force is perpendicular
to BOTH the flight direction and the spin axis.

Examples (incoming ball, travels +Y):
  - Topspin (ωₓ < 0): force DOWNWARD → ball dips, bounces flat
  - Backspin (ωₓ > 0): force UPWARD  → ball floats, bounces steep
  - Sidespin (ωᵤ ≠ 0): lateral force → ball curves left/right

**S-dependent lift coefficient (non-linear):**

The lift coefficient C_L is not constant — it saturates at high spin,
matching real table-tennis measurements (Nakashima 2010, Cross 2014):

    S = r · |ω| / |v|               (dimensionless spin parameter)

    C_L(S) = 0.60 · (1 − e^{−4.5·S})

    S = 0.10 → C_L ≈ 0.21   (gentle loop)
    S = 0.25 → C_L ≈ 0.43   (medium topspin)
    S = 0.50 → C_L ≈ 0.54   (hard topspin)
    S ≥ 1.00 → C_L ≈ 0.60   (saturation)

This saturation is important: doubling spin above S≈0.5 gives
only a small additional curve — the effect plateaus.

Values: C_L_max = 0.60, k = 4.5, r = 0.02 m


### Combined: Total Acceleration

    a(v, ω) = a_grav + a_drag(v) + a_magnus(v, ω)

The system is nonlinear — no analytical solution exists, hence
numerical integration. The nonlinearity has two sources:
  1. Drag: |v|·v is quadratic in v (nonlinear even without spin)
  2. Magnus: ω × v couples two unknowns multiplicatively


### Group 3: Spin Decay

    dωₓ/dt = −(k_spin / I) · ωₓ
    dωᵧ/dt = −(k_spin / I) · ωᵧ
    dωᵤ/dt = −(k_spin / I) · ωᵤ

Spin is slowly damped by air friction (Stokes torque).
This is a simple exponential decay.

Values: k_spin = 5·10⁻⁷ N·m·s, I = ⅔·m·r² (hollow sphere)

Spin barely changes during flight — the decay time constant is ~2.4 s,
while a typical flight lasts only ~0.2–0.4 s.


## Numerical Method: 4th-Order Runge-Kutta (RK4)

### Why not just Euler?

Euler method: s_{n+1} = s_n + dt · f(s_n)

That would be 1st order — the error per step is O(dt²).
With dt = 0.5 ms and 600 steps, this accumulates significantly.

### RK4: 4 Slopes per Step

Idea: Instead of only evaluating the slope at the beginning of the
interval, RK4 computes four slopes and takes a weighted average:

    k₁ = f(tₙ, sₙ)                          ← slope at the start
    k₂ = f(tₙ + dt/2, sₙ + dt/2 · k₁)      ← slope at the midpoint (using k₁)
    k₃ = f(tₙ + dt/2, sₙ + dt/2 · k₂)      ← slope at the midpoint (using k₂)
    k₄ = f(tₙ + dt,   sₙ + dt · k₃)        ← slope at the end

    sₙ₊₁ = sₙ + (dt/6) · (k₁ + 2·k₂ + 2·k₃ + k₄)

Each kᵢ is itself a 9-component vector (pos, vel, omega).

### Error Order

Local error:  O(dt⁵) per step
Global error: O(dt⁴) over the entire simulation

With dt = 0.5 ms = 5·10⁻⁴ s:
  dt⁴ = 6.25·10⁻¹⁴

This is absurdly precise for this application. Even dt = 5 ms would
be sufficient, but 0.5 ms allows precise impact detection.


### Implementation (integrator.rs / physics.js)

    fn rk4_step(state, dt):
        // k1: slope at the current point
        k1_pos   = state.vel
        k1_vel   = acceleration(state)        ← the 3 forces
        k1_omega = angular_deceleration(state) ← spin damping

        // k2: half step using k1
        s2 = BallState(pos + dt/2·k1_pos, vel + dt/2·k1_vel, ω + dt/2·k1_omega)
        k2_pos   = s2.vel
        k2_vel   = acceleration(s2)
        k2_omega = angular_deceleration(s2)

        // k3: half step using k2
        s3 = BallState(pos + dt/2·k2_pos, vel + dt/2·k2_vel, ω + dt/2·k2_omega)
        ...analogous...

        // k4: full step using k3
        s4 = BallState(pos + dt·k3_pos, vel + dt·k3_vel, ω + dt·k3_omega)
        ...analogous...

        // Weighted sum
        return BallState(
            pos   + dt/6 · (k1_pos   + 2·k2_pos   + 2·k3_pos   + k4_pos),
            vel   + dt/6 · (k1_vel   + 2·k2_vel   + 2·k3_vel   + k4_vel),
            omega + dt/6 · (k1_omega + 2·k2_omega + 2·k3_omega + k4_omega),
        )

Each timestep evaluates all forces 4 times.
For 600 steps: 2400 force evaluations — trivial for the CPU.


## What is NOT Solved by ODEs: The Bounce

The bounce is not a continuous problem but an instantaneous impulse
problem (bounce.rs / physics.js):

### Detection

When v_z < 0 (ball descending) and the trajectory crosses the table
plane at z = 0.76 m, the exact impact time is determined via linear
interpolation, and the RK4 integrator sub-steps to that point.

### Impulse Model (Gardin / Haake & Goodwill)

1. Normal component:  v'_z = −e_n · v_z     (e_n = 0.93)
   → Ball rebounds with 93% of the normal velocity.

2. Tangential component: contact-point velocity
   v_contact = v_tangential + ω × r_contact

   Then case distinction:
   a) |friction impulse| ≤ μ·|normal impulse| → STICKING (rolling contact)
      The ball grips the table surface.
   b) Otherwise → SLIDING (kinetic friction, μ = 0.25)
      The ball slides across the surface.

3. Spin update from the tangential impulse:
   Δω = (r_contact × J_tangential) / I

   Topspin is AMPLIFIED by the bounce (~150 → ~213 rad/s),
   because friction drives the ball further into rolling.


## What is NOT Solved by ODEs: The Paddle Hit

Like the bounce, the paddle contact is an instantaneous impulse event,
not a continuous ODE. It is triggered once when the ball reaches the
paddle's Y-plane.

### Paddle Orientation: tilt_x and tilt_z

The paddle face default normal points in −Y (toward the server).
Two angles tilt this normal:

    tilt_x  rotates around X-axis  →  nz = sin(tilt_x)   (face leans forward/backward)
    tilt_z  rotates around Z-axis  →  nx = sin(tilt_z)   (face leans sideways)
    ny = −√(1 − nx² − nz²)         (unit vector, always points toward −Y side)

    tilt_x < 0  →  open face (tilts upward)   → generates topspin on return
    tilt_x > 0  →  closed face (tilts down)   → generates backspin on return
    tilt_x = 0  →  face perpendicular to table → flat hit

The swing direction is set by `swing_elevation`:

    v_swing = speed · (−cos(elev), sin(elev), 0)   (in Y-Z plane)

    elevation = 0      → horizontal swing
    elevation > 0      → upward swing (brushes ball upward → topspin)

**Topspin is generated by the combination of an open face (tilt_x < 0)
and an upward swing (elevation > 0): the paddle brushes the top-back
of the ball, creating ωₓ > 0 on the return.**

### Paddle Impulse Model (paddle.rs)

Identical structure to the table bounce, with different constants:

    e_paddle = 0.85    (restitution, slightly lower than table's 0.93)
    μ_paddle = 0.45    (friction, higher than table's 0.25 — rubber grip)

Steps:

1. **Relative velocity at contact:**

       v_rel = v_ball − v_swing

2. **Decompose into normal and tangential:**

       v_n = (v_rel · n̂) · n̂         (normal component)
       v_t = v_rel − v_n             (tangential component)

3. **Normal restitution:**

       v_n_out = −e · v_n

4. **Contact-point velocity (ball spin + tangential sliding):**

       r_contact = −r · n̂            (vector from ball center to contact point)
       v_contact = v_t + ω × r_contact

5. **Normal impulse magnitude:**

       J_n = m · (1 + e) · |v_rel · n̂|

6. **Tangential impulse (sticking vs. sliding):**

       J_stick   = m · |v_contact| / (1 + m·r² / I)    (would fully stop slip)
       J_t_mag   = min(J_stick, μ · J_n)               (Coulomb limit)
       J_t       = −J_t_mag · v_contact / |v_contact|  (opposes slip)

7. **Spin update:**

       Δω = (r_contact × J_t) / I

8. **Reconstruct velocity:**

       v_out = v_n_out + v_t_out + v_swing

The new spin Δω depends on both tilt_x (which way the normal points)
and swing_elevation (which way J_t points). Typical topspin return:

    tilt_x = −0.3, elevation = +0.5, speed = 6 m/s  →  ωₓ ≈ +120 rad/s
