use super::forces::{acceleration, angular_deceleration};
use super::state::BallState;

/// Single RK4 step for both linear and angular motion
pub fn rk4_step(state: &BallState, dt: f64) -> BallState {
    let k1_pos = state.vel;
    let k1_vel = acceleration(state);
    let k1_omega = angular_deceleration(state);

    let s2 = BallState {
        pos: state.pos + k1_pos * (dt / 2.0),
        vel: state.vel + k1_vel * (dt / 2.0),
        omega: state.omega + k1_omega * (dt / 2.0),
    };
    let k2_pos = s2.vel;
    let k2_vel = acceleration(&s2);
    let k2_omega = angular_deceleration(&s2);

    let s3 = BallState {
        pos: state.pos + k2_pos * (dt / 2.0),
        vel: state.vel + k2_vel * (dt / 2.0),
        omega: state.omega + k2_omega * (dt / 2.0),
    };
    let k3_pos = s3.vel;
    let k3_vel = acceleration(&s3);
    let k3_omega = angular_deceleration(&s3);

    let s4 = BallState {
        pos: state.pos + k3_pos * dt,
        vel: state.vel + k3_vel * dt,
        omega: state.omega + k3_omega * dt,
    };
    let k4_pos = s4.vel;
    let k4_vel = acceleration(&s4);
    let k4_omega = angular_deceleration(&s4);

    // Weighted sum
    let factor = dt / 6.0;
    BallState {
        pos: state.pos + (k1_pos + k2_pos * 2.0 + k3_pos * 2.0 + k4_pos) * factor,
        vel: state.vel + (k1_vel + k2_vel * 2.0 + k3_vel * 2.0 + k4_vel) * factor,
        omega: state.omega + (k1_omega + k2_omega * 2.0 + k3_omega * 2.0 + k4_omega) * factor,
    }
}
