use crate::physics::bounce::apply_bounce;
use crate::physics::integrator::rk4_step;
use crate::physics::state::{BallState, Vec3};
use crate::table::Table;

pub const DT: f64 = 0.0005;    // 0.5 ms – fine enough for TT trajectories
pub const T_MAX: f64 = 5.0;    // 5 s hard limit

/// Record of a single bounce on the table surface
#[derive(Debug, Clone)]
pub struct BounceEvent {
    pub landing: Vec3,
    pub pre_bounce: BallState,
    pub post_bounce: BallState,
    pub time: f64,
}

/// How the simulation ended
#[derive(Debug, Clone)]
pub enum SimOutcome {
    /// Reached the requested maximum number of bounces
    MaxBounces,
    /// Ball descended below the table surface without hitting it
    HitFloor,
    /// Ball hit the table surface plane outside the table bounds
    MissedTable(String),
    /// Ball flew far from the table (beyond play area)
    LeftPlayArea,
    /// Time limit exceeded
    Timeout,
}

/// Full simulation result with multiple bounces
#[derive(Debug)]
pub struct SimResult {
    pub bounces: Vec<BounceEvent>,
    pub trajectory: Vec<(f64, BallState)>,
    pub final_state: BallState,
    pub final_time: f64,
    pub outcome: SimOutcome,
}

/// Simulate ball flight with up to `max_bounces` table bounces.
///
/// The simulation continues after each bounce until:
/// - `max_bounces` bounces have occurred (SimOutcome::MaxBounces)
/// - The ball hits the floor (z < 0)
/// - The ball lands outside the table on a bounce attempt
/// - The ball leaves the play area (far from table)
/// - Time limit is exceeded
pub fn simulate_full(
    initial: BallState,
    table: &Table,
    max_bounces: usize,
) -> SimResult {
    let mut state = initial;
    let mut t = 0.0_f64;
    let mut trajectory: Vec<(f64, BallState)> = vec![(t, state)];
    let mut bounces: Vec<BounceEvent> = Vec::new();

    let outcome = loop {
        if t >= T_MAX {
            break SimOutcome::Timeout;
        }
        if state.pos.z < -0.01 {
            break SimOutcome::HitFloor;
        }
        // Ball far from table → left play area
        if state.pos.y > table.length + 3.0
            || state.pos.y < -3.0
            || state.pos.x > table.width + 3.0
            || state.pos.x < -3.0
        {
            break SimOutcome::LeftPlayArea;
        }

        // Check for table surface crossing
        if state.vel.z < 0.0 && state.pos.z > table.surface_z() {
            if let Some(t_hit) = table.time_to_surface(state.pos, state.vel) {
                if t_hit <= DT {
                    // Step exactly to the surface
                    if t_hit > 1e-9 {
                        state = rk4_step(&state, t_hit);
                        t += t_hit;
                        trajectory.push((t, state));
                    }

                    // Is the impact point on the table?
                    if !table.covers_xy(state.pos.x, state.pos.y) {
                        break SimOutcome::MissedTable(format!(
                            "Ball hit z={:.3} at x={:.3}, y={:.3} (outside table)",
                            state.pos.z, state.pos.x, state.pos.y,
                        ));
                    }

                    let pre_bounce = state;

                    // Apply bounce model
                    state = apply_bounce(&state, table);
                    state.pos.z = table.surface_z();
                    if state.vel.z < 0.0 {
                        state.vel.z = 0.001;
                    }

                    bounces.push(BounceEvent {
                        landing: pre_bounce.pos,
                        pre_bounce,
                        post_bounce: state,
                        time: t,
                    });
                    trajectory.push((t, state));

                    if bounces.len() >= max_bounces {
                        break SimOutcome::MaxBounces;
                    }

                    // Continue simulation after bounce
                    continue;
                }
            }
        }

        // Normal RK4 step
        state = rk4_step(&state, DT);
        t += DT;
        trajectory.push((t, state));
    };

    SimResult {
        bounces,
        trajectory,
        final_state: state,
        final_time: t,
        outcome,
    }
}

// Legacy API for CLI compatibility
#[derive(Debug)]
pub enum SimError {
    MissedTable(String),
    HitFloor,
    Timeout,
}

#[derive(Debug)]
pub struct LegacySimResult {
    pub landing: Vec3,
    pub pre_bounce: BallState,
    pub post_bounce: BallState,
    pub bounce_time: f64,
    pub trajectory: Vec<(f64, BallState)>,
}

/// Single-bounce simulation (legacy API used by CLI)
pub fn simulate(initial: BallState, table: &Table) -> Result<LegacySimResult, SimError> {
    let result = simulate_full(initial, table, 1);

    if let Some(bounce) = result.bounces.into_iter().next() {
        Ok(LegacySimResult {
            landing: bounce.landing,
            pre_bounce: bounce.pre_bounce,
            post_bounce: bounce.post_bounce,
            bounce_time: bounce.time,
            trajectory: result.trajectory,
        })
    } else {
        match result.outcome {
            SimOutcome::HitFloor => Err(SimError::HitFloor),
            SimOutcome::Timeout => Err(SimError::Timeout),
            SimOutcome::MissedTable(msg) => Err(SimError::MissedTable(msg)),
            SimOutcome::LeftPlayArea => Err(SimError::MissedTable(
                "Ball left the play area without hitting the table".to_string(),
            )),
            SimOutcome::MaxBounces => unreachable!(),
        }
    }
}
