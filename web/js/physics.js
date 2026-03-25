// physics.js – Table Tennis Ball Physics Engine
// Faithful port of the spinoza Rust simulation

// ===== Vec3 =====

export class Vec3 {
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  static zero() {
    return new Vec3(0, 0, 0);
  }

  add(v) {
    return new Vec3(this.x + v.x, this.y + v.y, this.z + v.z);
  }
  sub(v) {
    return new Vec3(this.x - v.x, this.y - v.y, this.z - v.z);
  }
  scale(s) {
    return new Vec3(this.x * s, this.y * s, this.z * s);
  }
  neg() {
    return new Vec3(-this.x, -this.y, -this.z);
  }

  dot(v) {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }

  cross(v) {
    return new Vec3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x,
    );
  }

  norm() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }

  normalized() {
    const n = this.norm();
    return n < 1e-12 ? Vec3.zero() : this.scale(1.0 / n);
  }

  clone() {
    return new Vec3(this.x, this.y, this.z);
  }
}

// ===== Physical Constants (matching Rust constants.rs) =====

export const BALL_MASS = 0.0027; // kg
export const BALL_RADIUS = 0.020; // m
export const BALL_AREA = Math.PI * BALL_RADIUS * BALL_RADIUS; // m²
export const AIR_DENSITY = 1.2; // kg/m³
export const CD = 0.4; // drag coefficient
export const CL = 0.6; // Magnus lift coefficient
export const BALL_INERTIA =
  (2.0 / 3.0) * BALL_MASS * BALL_RADIUS * BALL_RADIUS;
export const G = 9.81; // m/s²

const K_SPIN = 5e-7; // spin decay constant (N·m·s)
const E_N = 0.93; // normal coefficient of restitution
const MU = 0.25; // Coulomb friction coefficient

// ===== BallState =====

export class BallState {
  constructor(pos, vel, omega) {
    this.pos = pos;
    this.vel = vel;
    this.omega = omega;
  }

  clone() {
    return new BallState(this.pos.clone(), this.vel.clone(), this.omega.clone());
  }
}

// ===== Forces (matching Rust forces.rs) =====

export function acceleration(state) {
  const vel = state.vel;
  const speed = vel.norm();
  const gravity = new Vec3(0, 0, -G);

  if (speed < 1e-9) return gravity;

  // Drag: F_D = -½ · C_D · ρ · A · |v| · v
  const dragFactor = (-0.5 * CD * AIR_DENSITY * BALL_AREA * speed) / BALL_MASS;
  const drag = vel.scale(dragFactor);

  // Magnus: F_M = C_L · ρ · A · r · (ω × v)
  const magnusFactor =
    (CL * AIR_DENSITY * BALL_AREA * BALL_RADIUS) / BALL_MASS;
  const magnus = state.omega.cross(vel).scale(magnusFactor);

  return gravity.add(drag).add(magnus);
}

export function angularDeceleration(state) {
  return state.omega.scale(-K_SPIN / BALL_INERTIA);
}

// ===== RK4 Integrator (matching Rust integrator.rs) =====

export function rk4Step(state, dt) {
  const k1_pos = state.vel;
  const k1_vel = acceleration(state);
  const k1_omega = angularDeceleration(state);

  const s2 = new BallState(
    state.pos.add(k1_pos.scale(dt / 2)),
    state.vel.add(k1_vel.scale(dt / 2)),
    state.omega.add(k1_omega.scale(dt / 2)),
  );
  const k2_pos = s2.vel;
  const k2_vel = acceleration(s2);
  const k2_omega = angularDeceleration(s2);

  const s3 = new BallState(
    state.pos.add(k2_pos.scale(dt / 2)),
    state.vel.add(k2_vel.scale(dt / 2)),
    state.omega.add(k2_omega.scale(dt / 2)),
  );
  const k3_pos = s3.vel;
  const k3_vel = acceleration(s3);
  const k3_omega = angularDeceleration(s3);

  const s4 = new BallState(
    state.pos.add(k3_pos.scale(dt)),
    state.vel.add(k3_vel.scale(dt)),
    state.omega.add(k3_omega.scale(dt)),
  );
  const k4_pos = s4.vel;
  const k4_vel = acceleration(s4);
  const k4_omega = angularDeceleration(s4);

  const f = dt / 6.0;
  return new BallState(
    state.pos.add(
      k1_pos
        .add(k2_pos.scale(2))
        .add(k3_pos.scale(2))
        .add(k4_pos)
        .scale(f),
    ),
    state.vel.add(
      k1_vel
        .add(k2_vel.scale(2))
        .add(k3_vel.scale(2))
        .add(k4_vel)
        .scale(f),
    ),
    state.omega.add(
      k1_omega
        .add(k2_omega.scale(2))
        .add(k3_omega.scale(2))
        .add(k4_omega)
        .scale(f),
    ),
  );
}

// ===== Bounce Model (matching Rust bounce.rs) =====

export function applyBounce(state) {
  const v = state.vel;
  const omega = state.omega;
  const r = BALL_RADIUS;
  const m = BALL_MASS;
  const I = BALL_INERTIA;

  const v_nz = v.z;
  const v_tx = v.x;
  const v_ty = v.y;

  // Contact point velocity (bottom of ball)
  const vc_x = v_tx + -r * omega.y;
  const vc_y = v_ty + r * omega.x;

  // Normal impulse magnitude
  const j_n = m * (1 + E_N) * Math.abs(v_nz);

  // Tangential impulse for sticking (rolling contact)
  const denom = 1 + (m * r * r) / I; // = 5/2 for hollow sphere
  const j_stick_x = (-m * vc_x) / denom;
  const j_stick_y = (-m * vc_y) / denom;
  const j_stick_mag = Math.sqrt(
    j_stick_x * j_stick_x + j_stick_y * j_stick_y,
  );

  const maxFriction = MU * j_n;

  let j_tx, j_ty;
  if (j_stick_mag <= maxFriction) {
    // Sticking – enough friction to stop slip
    j_tx = j_stick_x;
    j_ty = j_stick_y;
  } else {
    // Sliding – kinetic friction caps the impulse
    const scale = maxFriction / j_stick_mag;
    j_tx = j_stick_x * scale;
    j_ty = j_stick_y * scale;
  }

  const new_vx = v_tx + j_tx / m;
  const new_vy = v_ty + j_ty / m;
  const new_vz = -E_N * v_nz;

  // Spin update: Δω = r_contact × J_t / I
  const d_omega_x = (r * j_ty) / I;
  const d_omega_y = (-r * j_tx) / I;

  return new BallState(
    state.pos.clone(),
    new Vec3(new_vx, new_vy, new_vz),
    new Vec3(omega.x + d_omega_x, omega.y + d_omega_y, omega.z),
  );
}

// ===== Table (matching Rust table.rs) =====

export class Table {
  constructor() {
    this.length = 2.74;
    this.width = 1.525;
    this.height = 0.76;
    this.netHeight = 0.1525;
  }

  surfaceZ() {
    return this.height;
  }

  coversXY(x, y) {
    return x >= 0 && x <= this.width && y >= 0 && y <= this.length;
  }

  timeToSurface(pos, vel) {
    if (vel.z >= 0) return null;
    const t = (this.surfaceZ() - pos.z) / vel.z;
    if (t < 0) return null;
    const hitX = pos.x + vel.x * t;
    const hitY = pos.y + vel.y * t;
    return this.coversXY(hitX, hitY) ? t : null;
  }
}

// ===== Simulation =====

const DT = 0.0005; // 0.5 ms
const T_MAX = 5.0;

export function simulate(params) {
  const table = new Table();

  const elevRad = (params.elevation * Math.PI) / 180;
  const azimRad = (params.azimuth * Math.PI) / 180;

  const vx = params.speed * Math.cos(elevRad) * Math.sin(azimRad);
  const vy = params.speed * Math.cos(elevRad) * Math.cos(azimRad);
  const vz = params.speed * Math.sin(elevRad);

  const omegaX = -params.topspin + params.backspin;
  const omegaY = 0;
  const omegaZ = -params.sidespin;

  const initial = new BallState(
    new Vec3(params.x0, params.y0, params.z0),
    new Vec3(vx, vy, vz),
    new Vec3(omegaX, omegaY, omegaZ),
  );

  let state = initial.clone();
  let t = 0;
  const trajectory = [{ t: 0, state: initial.clone() }];
  const bounces = [];

  while (t < T_MAX) {
    if (state.pos.z < 0) break;

    // Check for table surface crossing
    if (state.vel.z < 0 && state.pos.z > table.surfaceZ()) {
      const tHit = table.timeToSurface(state.pos, state.vel);
      if (tHit !== null && tHit <= DT) {
        if (tHit > 1e-9) {
          state = rk4Step(state, tHit);
          t += tHit;
          trajectory.push({ t, state: state.clone() });
        }

        if (table.coversXY(state.pos.x, state.pos.y)) {
          const preBounce = state.clone();
          state = applyBounce(state);
          state.pos = new Vec3(state.pos.x, state.pos.y, table.surfaceZ());
          if (state.vel.z < 0)
            state.vel = new Vec3(state.vel.x, state.vel.y, 0.001);

          bounces.push({
            landing: preBounce.pos.clone(),
            preBounce,
            postBounce: state.clone(),
            time: t,
          });

          trajectory.push({ t, state: state.clone() });

          if (bounces.length >= 3) break;
        }
        continue;
      }
    }

    // Normal RK4 step
    state = rk4Step(state, DT);
    t += DT;
    trajectory.push({ t, state: state.clone() });

    // Stop if ball is far from table
    if (
      state.pos.y > table.length + 3 ||
      state.pos.y < -3 ||
      state.pos.x > table.width + 3 ||
      state.pos.x < -3
    )
      break;
  }

  // Post-process: check if ball crossed net below net height
  const netY = table.length / 2;
  const netTopZ = table.surfaceZ() + table.netHeight;
  let hitNet = false;

  for (let i = 1; i < trajectory.length; i++) {
    const prev = trajectory[i - 1].state.pos;
    const curr = trajectory[i].state.pos;
    if (
      (prev.y < netY && curr.y >= netY) ||
      (prev.y > netY && curr.y <= netY)
    ) {
      const frac = (netY - prev.y) / (curr.y - prev.y);
      const zAtNet = prev.z + frac * (curr.z - prev.z);
      if (zAtNet - BALL_RADIUS < netTopZ) {
        hitNet = true;
      }
      break;
    }
  }

  return { trajectory, bounces, hitNet, table };
}
