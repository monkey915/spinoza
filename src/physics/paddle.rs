use crate::physics::constants::*;
use crate::physics::state::{BallState, Vec3};

/// Paddle coefficient of restitution (slightly lower than table)
const PADDLE_E_N: f64 = 0.85;

/// Paddle friction coefficient (rubber grip is higher than table)
const PADDLE_MU: f64 = 0.45;

/// A simplified paddle: flat disc with position, orientation, and swing velocity.
///
/// The agent controls 7 parameters:
///   - paddle_x: lateral position (m)
///   - paddle_y: depth position — where the player stands (m, along table axis)
///   - paddle_z: height position (m, clamped to above table surface)
///   - tilt_x, tilt_z: paddle face angles (radians)
///   - swing_speed: how hard to hit (m/s)
///   - swing_elevation: swing angle in radians (0 = horizontal, positive = upward)
#[derive(Debug, Clone)]
pub struct PaddleAction {
    pub paddle_x: f64,
    pub paddle_y: f64,
    pub paddle_z: f64,
    pub tilt_x: f64,
    pub tilt_z: f64,
    pub swing_speed: f64,
    pub swing_elevation: f64,
}

/// Compute the paddle face normal from tilt angles.
///
/// Default face normal points toward -Y (hitting the ball back).
/// tilt_x rotates around X axis (forward/backward lean).
/// tilt_z rotates around Z axis (sideways lean).
fn paddle_normal(tilt_x: f64, tilt_z: f64) -> Vec3 {
    let nx = tilt_z.sin();
    let nz = tilt_x.sin();
    let ny = -(1.0 - nx * nx - nz * nz).abs().sqrt();
    Vec3::new(nx, ny, nz).normalized()
}

/// Compute the swing velocity vector from speed and elevation angle.
///
/// Default swing direction is toward -Y (hitting back toward server).
/// Elevation rotates upward from the horizontal swing plane.
///
/// Biomechanical constraint: the effective swing speed decreases with
/// elevation — you can smash hard forward but not swing 12 m/s straight up.
/// Formula: effective_speed = speed × cos(elevation)^0.6
/// At 0° (flat smash):  100% speed
/// At 30° (topspin loop): ~82% speed
/// At 45° (heavy loop):   ~71% speed
/// At 57° (max elevation): ~63% speed
fn swing_velocity(speed: f64, elevation: f64) -> Vec3 {
    let effective_speed = speed * elevation.cos().abs().powf(0.6);
    let vy = -effective_speed * elevation.cos();
    let vz = effective_speed * elevation.sin();
    Vec3::new(0.0, vy, vz)
}

/// Result of a paddle hit attempt
#[derive(Debug, Clone)]
pub enum PaddleResult {
    /// Ball was hit successfully; contains the post-hit ball state
    Hit(BallState),
    /// Ball missed the paddle (too far from paddle center)
    Miss {
        /// Distance between ball path and paddle center (m)
        miss_distance: f64,
    },
}

/// Check if the ball passes close enough to the paddle and apply contact physics.
///
/// The paddle is positioned at (paddle_x, paddle_y, paddle_z).
/// `ball_at_paddle_y` is the ball state when it reaches the paddle's y-plane.
/// `paddle_radius` is the effective reach of the paddle (default: ~0.15 m).
/// `table_surface_z` is used to clamp the paddle above the table.
pub fn apply_paddle_hit(
    ball: &BallState,
    action: &PaddleAction,
    table_surface_z: f64,
    paddle_radius: f64,
) -> PaddleResult {
    // Clamp paddle_z to be above the table surface (with clearance for paddle radius)
    let min_paddle_z = table_surface_z + 0.09;
    let paddle_z = action.paddle_z.max(min_paddle_z);

    let paddle_pos = Vec3::new(action.paddle_x, action.paddle_y, paddle_z);

    // Distance from ball to paddle center (in xz plane, since y is matched)
    // Use paddle_radius + BALL_RADIUS so the ball's edge (not just center) counts
    let dx = ball.pos.x - paddle_pos.x;
    let dz = ball.pos.z - paddle_pos.z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist > paddle_radius + BALL_RADIUS {
        return PaddleResult::Miss {
            miss_distance: dist - paddle_radius - BALL_RADIUS,
        };
    }

    let normal = paddle_normal(action.tilt_x, action.tilt_z);
    let swing_vel = swing_velocity(action.swing_speed, action.swing_elevation);

    // Ball velocity relative to the paddle surface
    let v_rel = ball.vel - swing_vel;

    // Decompose into normal and tangential components
    let v_n = normal * v_rel.dot(normal);
    let v_t = v_rel - v_n;

    // Normal restitution (flip the normal component)
    let v_n_out = v_n * (-PADDLE_E_N);

    // Tangential: friction model (same approach as table bounce)
    let omega = ball.omega;
    let r = BALL_RADIUS;
    let m = BALL_MASS;
    let inertia = BALL_INERTIA;

    // Contact point velocity due to spin: v_contact = v_t + ω × (-r·normal)
    let r_contact = normal * (-r);
    let spin_at_contact = omega.cross(r_contact);
    let vc = v_t + spin_at_contact;
    let vc_mag = vc.norm();

    // Normal impulse magnitude
    let j_n = m * (1.0 + PADDLE_E_N) * v_rel.dot(normal).abs();

    let v_t_out;
    let omega_out;

    if vc_mag < 1e-9 {
        // No tangential slip
        v_t_out = v_t;
        omega_out = omega;
    } else {
        let denom = 1.0 + m * r * r / inertia;
        let j_stick = m * vc_mag / denom;
        let max_friction = PADDLE_MU * j_n;

        let j_t_mag = if j_stick <= max_friction {
            j_stick
        } else {
            max_friction
        };

        // Tangential impulse direction (opposing contact velocity)
        let vc_dir = vc * (1.0 / vc_mag);
        let j_t = vc_dir * (-j_t_mag);

        v_t_out = v_t + j_t * (1.0 / m);

        // Spin update: Δω = (r_contact × J_t) / I
        let d_omega = r_contact.cross(j_t) * (1.0 / inertia);
        omega_out = omega + d_omega;
    }

    // Reconstruct full velocity: add back swing velocity
    let v_out = v_n_out + v_t_out + swing_vel;

    PaddleResult::Hit(BallState {
        pos: ball.pos,
        vel: v_out,
        omega: omega_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_straight_hit_returns_ball() {
        let ball = BallState::new(
            Vec3::new(0.76, 2.74, 0.90),
            Vec3::new(0.0, 2.0, -0.5),
            Vec3::ZERO,
        );
        let action = PaddleAction {
            paddle_x: 0.76,
            paddle_y: 2.74,
            paddle_z: 0.90,
            tilt_x: 0.0,
            tilt_z: 0.0,
            swing_speed: 5.0,
            swing_elevation: 0.15,
        };
        match apply_paddle_hit(&ball, &action, 0.76, 0.15) {
            PaddleResult::Hit(result) => {
                // Ball should now be heading back toward server (vy < 0)
                assert!(result.vel.y < 0.0, "Ball should head back, got vy={}", result.vel.y);
            }
            PaddleResult::Miss { .. } => panic!("Should have hit"),
        }
    }

    #[test]
    fn test_miss_when_too_far() {
        let ball = BallState::new(
            Vec3::new(0.0, 2.74, 0.90),
            Vec3::new(0.0, 2.0, -0.5),
            Vec3::ZERO,
        );
        let action = PaddleAction {
            paddle_x: 1.5,
            paddle_y: 2.74,
            paddle_z: 0.90,
            tilt_x: 0.0,
            tilt_z: 0.0,
            swing_speed: 5.0,
            swing_elevation: 0.15,
        };
        match apply_paddle_hit(&ball, &action, 0.76, 0.15) {
            PaddleResult::Miss { miss_distance } => {
                assert!(miss_distance > 0.0);
            }
            PaddleResult::Hit(_) => panic!("Should have missed"),
        }
    }
}
