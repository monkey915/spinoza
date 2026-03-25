// ITTF-standard table tennis ball (40mm plastic)
pub const BALL_MASS: f64 = 0.0027;      // kg
pub const BALL_RADIUS: f64 = 0.020;    // m
pub const BALL_AREA: f64 = std::f64::consts::PI * BALL_RADIUS * BALL_RADIUS; // m²

// Aerodynamics
pub const AIR_DENSITY: f64 = 1.2;      // kg/m³
pub const CD: f64 = 0.40;              // drag coefficient (sphere, ~Re 20k-80k)
pub const CL: f64 = 0.60;              // Magnus lift coefficient (empirical)

// Moment of inertia for hollow sphere: I = (2/3)·m·r²
pub const BALL_INERTIA: f64 = (2.0 / 3.0) * BALL_MASS * BALL_RADIUS * BALL_RADIUS;

pub const G: f64 = 9.81;               // m/s²
