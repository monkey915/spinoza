use super::constants::*;
use super::state::{BallState, Vec3};

/// Total linear acceleration on the ball (m/s²)
pub fn acceleration(state: &BallState) -> Vec3 {
    let vel = state.vel;
    let speed = vel.norm();

    // Gravity
    let gravity = Vec3::new(0.0, 0.0, -G);

    if speed < 1e-9 {
        return gravity;
    }

    // Drag: F_D = -½ · C_D · ρ · A · |v| · v  →  a = F/m
    let drag_factor = -0.5 * CD * AIR_DENSITY * BALL_AREA * speed / BALL_MASS;
    let drag = vel * drag_factor;

    // Magnus: F_M = C_L · ρ · A · (ω × v)  →  a = F/m
    // (Empirical form matching table-tennis experiments: proportional to |ω×v|)
    let magnus_factor = CL * AIR_DENSITY * BALL_AREA * BALL_RADIUS / BALL_MASS;
    let magnus = state.omega.cross(vel) * magnus_factor;

    gravity + drag + magnus
}

/// Angular deceleration due to air friction on spinning ball
/// τ = -k·ω  where k ≈ 8π·μ_air·r³  (Stokes-like torque for sphere)
/// We use a simplified empirical spin-decay constant.
pub fn angular_deceleration(state: &BallState) -> Vec3 {
    // Aerodynamic spin decay: α = -k_spin · ω / I
    // k_spin empirically ~5e-7 N·m·s (very slow decay during flight)
    const K_SPIN: f64 = 5e-7;
    let alpha = -(state.omega * (K_SPIN / BALL_INERTIA));
    alpha
}
