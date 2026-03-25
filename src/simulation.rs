use crate::physics::bounce::apply_bounce;
use crate::physics::integrator::rk4_step;
use crate::physics::state::{BallState, Vec3};
use crate::table::Table;

const DT: f64 = 0.0005;    // 0.5 ms – fine enough for TT trajectories
const T_MAX: f64 = 5.0;    // 5 s hard limit

#[derive(Debug)]
pub struct SimResult {
    /// Where the ball landed after the first bounce (x, y on table, z = table surface)
    pub landing: Vec3,
    /// Ball state immediately before the bounce
    pub pre_bounce: BallState,
    /// Ball state immediately after the bounce
    pub post_bounce: BallState,
    /// Time of bounce since launch (s)
    pub bounce_time: f64,
    /// Full trajectory: (time, state) samples
    pub trajectory: Vec<(f64, BallState)>,
}

#[derive(Debug)]
pub enum SimError {
    /// Ball never hit the table (missed or flew out)
    MissedTable(String),
    /// Ball hit the floor before landing on table
    HitFloor,
    /// Time limit exceeded
    Timeout,
}

pub fn simulate(initial: BallState, table: &Table) -> Result<SimResult, SimError> {
    let mut state = initial;
    let mut t = 0.0_f64;
    let mut trajectory: Vec<(f64, BallState)> = vec![(t, state)];

    loop {
        if t >= T_MAX {
            return Err(SimError::Timeout);
        }
        // Ball below the floor?
        if state.pos.z < -0.01 {
            return Err(SimError::HitFloor);
        }

        // Is the ball approaching the table surface?
        if state.vel.z < 0.0 && state.pos.z > table.surface_z() {
            // Check if the ball will cross the table surface within the next step
            if let Some(t_hit) = table.time_to_surface(state.pos, state.vel) {
                if t_hit <= DT {
                    // Step exactly to the surface
                    let dt_to_hit = t_hit;
                    if dt_to_hit > 1e-9 {
                        state = rk4_step(&state, dt_to_hit);
                        t += dt_to_hit;
                        trajectory.push((t, state));
                    }

                    // Verify it's actually on the table (x,y coverage)
                    if !table.covers_xy(state.pos.x, state.pos.y) {
                        return Err(SimError::MissedTable(format!(
                            "Ball hit z={:.3} at x={:.3}, y={:.3} (outside table [{:.3},{:.3}]×[{:.3},{:.3}])",
                            state.pos.z, state.pos.x, state.pos.y,
                            0.0, table.width, 0.0, table.length
                        )));
                    }

                    let pre_bounce = state;
                    let bounce_time = t;

                    // Apply bounce model
                    state = apply_bounce(&state, table);
                    // Snap z to surface to avoid floating point drift
                    state.pos.z = table.surface_z();
                    // Ensure ball leaves surface upward
                    if state.vel.z < 0.0 { state.vel.z = 0.001; }

                    let post_bounce = state;
                    trajectory.push((t, state));

                    return Ok(SimResult {
                        landing: pre_bounce.pos,
                        pre_bounce,
                        post_bounce,
                        bounce_time,
                        trajectory,
                    });
                }
            }
        }

        // Normal RK4 step
        state = rk4_step(&state, DT);
        t += DT;
        trajectory.push((t, state));
    }
}
