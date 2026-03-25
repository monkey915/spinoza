use super::constants::*;
use super::state::{BallState, Vec3};
use crate::table::Table;

/// Table-surface normal points upward: n̂ = (0, 0, 1)
#[allow(dead_code)]
const NORMAL: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 1.0 };

/// Coefficient of restitution (normal direction) for plastic ball on ITTF-approved table
const E_N: f64 = 0.93;

/// Coulomb friction coefficient between ball and table surface
const MU: f64 = 0.25;

/// Apply one bounce off the table surface.
///
/// Model (based on Gardin / Haake & Goodwill):
///   1. Decompose velocity into normal (z) and tangential (x,y) parts.
///   2. Normal: v_nz' = -e_n · v_nz
///   3. Tangential: depends on whether contact point slips or grips.
///      - Slip contact point velocity:  v_c = v_t + r × (n × ω)
///        For flat table (n = ẑ):  v_c = (v_x - r·ω_y, v_y + r·ω_x)
///      - If |friction impulse| < μ·|normal impulse| → sliding, apply kinetic friction
///      - Otherwise → sticking (rolling contact)
///   4. Update spin from impulse.
pub fn apply_bounce(state: &BallState, _table: &Table) -> BallState {
    let v = state.vel;
    let omega = state.omega;
    let r = BALL_RADIUS;
    let m = BALL_MASS;
    let i = BALL_INERTIA;

    // Normal component (z)
    let v_nz = v.z;
    // Tangential velocity at ball centre
    let v_tx = v.x;
    let v_ty = v.y;

    // Velocity of contact point (bottom of ball) due to spin
    // v_contact = v_centre + ω × (-r·ẑ)
    // ω × (-r·ẑ) = (-r) * (ω × ẑ) = (-r) * (ω_y·x̂ - ω_x·ŷ)  → (+r·ω_y, -r·ω_x)
    // Wait: ω × ẑ = (ω_y·1 - ω_z·0, ω_z·0 - ω_x·1, ω_x·0 - ω_y·0) = (ω_y, -ω_x, 0)
    // Contact point at -r·ẑ relative to centre:
    // v_contact = v + ω × r_contact   where r_contact = (0,0,-r)
    // ω × (0,0,-r) = (ω_y·(-r) - ω_z·0, ω_z·0 - ω_x·(-r), ...) = (-r·ω_y, r·ω_x, 0)
    let vc_x = v_tx + (-r * omega.y);
    let vc_y = v_ty + (r * omega.x);

    // Normal impulse magnitude (pointing up, per unit mass of ball)
    // J_n = m·(1 + e_n)·|v_nz|
    let j_n = m * (1.0 + E_N) * v_nz.abs();

    // Tangential impulse needed to stop sliding: J_t_stick = -m·v_c / (1 + m·r²/I)
    // The factor (1 + m·r²/I) accounts for spin coupling.
    // For hollow sphere: I = (2/3)·m·r²  → m·r²/I = 3/2  → denominator = 5/2
    let denom = 1.0 + m * r * r / i; // = 5/2 for hollow sphere
    let j_stick_x = -m * vc_x / denom;
    let j_stick_y = -m * vc_y / denom;
    let j_stick_mag = (j_stick_x * j_stick_x + j_stick_y * j_stick_y).sqrt();

    let max_friction = MU * j_n;

    let (j_tx, j_ty) = if j_stick_mag <= max_friction {
        // Sticking (rolling) – enough friction to stop slip
        (j_stick_x, j_stick_y)
    } else {
        // Sliding – kinetic friction caps the impulse
        let scale = max_friction / j_stick_mag;
        (j_stick_x * scale, j_stick_y * scale)
    };

    // Update linear velocity
    let new_vx = v_tx + j_tx / m;
    let new_vy = v_ty + j_ty / m;
    let new_vz = -E_N * v_nz; // restitution

    // Update angular velocity: Δω = r_contact × J_t / I
    // r_contact = (0,0,-r), J_t = (j_tx, j_ty, 0)
    // (0,0,-r) × (j_tx, j_ty, 0) = (0·0 - (-r)·j_ty,  (-r)·j_tx - 0·0,  0·j_ty - 0·j_tx)
    //                             = (r·j_ty, -r·j_tx, 0)
    let d_omega_x = r * j_ty / i;
    let d_omega_y = -r * j_tx / i;

    BallState {
        pos: state.pos,
        vel: Vec3::new(new_vx, new_vy, new_vz),
        omega: Vec3::new(omega.x + d_omega_x, omega.y + d_omega_y, omega.z),
    }
}
