use crate::physics::integrator::rk4_step;
use crate::physics::paddle::{apply_paddle_hit, PaddleAction, PaddleResult};
use crate::physics::state::{BallState, Vec3};
use crate::simulation::{simulate_full, DT};
use crate::table::Table;

/// Observation frame: ball position at a simulated "camera" sample time
pub type ObsFrame = [f64; 3]; // [x, y, z]

/// How many past ball positions the agent can see
pub const OBS_FRAMES: usize = 10;

/// Simulated camera frame interval (~60Hz)
const OBS_DT: f64 = 1.0 / 60.0;

/// Paddle effective reach radius (m)
const PADDLE_RADIUS: f64 = 0.15;

/// How a rally episode ended
#[derive(Debug, Clone)]
pub enum RallyOutcome {
    /// Agent returned the ball and it landed on the server's half
    Success { landing_x: f64, landing_y: f64 },
    /// Agent returned the ball but it missed the server's half
    ReturnMissedTable,
    /// Agent returned the ball but it hit the net
    ReturnHitNet,
    /// Agent's paddle missed the ball entirely
    PaddleMiss { miss_distance: f64 },
    /// Serve itself was invalid (ball didn't reach receiver side)
    BadServe(String),
}

/// Full result of one rally episode
#[derive(Debug)]
pub struct RallyResult {
    pub outcome: RallyOutcome,
    /// Observation frames the agent saw before acting (ball positions at ~60Hz)
    pub observations: Vec<ObsFrame>,
    /// The action the agent took
    pub action: PaddleAction,
    /// Computed reward
    pub reward: f64,
}

/// Run one rally episode:
/// 1. Simulate the serve (ball flight + first bounce on receiver's half)
/// 2. Sample observation frames as the ball approaches the paddle zone
/// 3. Apply the agent's paddle action
/// 4. Simulate the return and check if it lands on the server's half
pub fn run_rally(
    serve_state: BallState,
    action: &PaddleAction,
    table: &Table,
) -> RallyResult {
    // Phase 1: simulate serve until ball has bounced once and is heading
    // toward the receiver (past the net, y > table.length/2)
    let serve_result = simulate_full(serve_state, table, 1);

    // Check serve validity: must have bounced on the table
    let serve_bounce = match serve_result.bounces.first() {
        Some(b) => b.clone(),
        None => {
            return RallyResult {
                outcome: RallyOutcome::BadServe(format!(
                    "Serve never bounced: {:?}",
                    serve_result.outcome
                )),
                observations: Vec::new(),
                action: action.clone(),
                reward: 0.0,
            };
        }
    };

    // Phase 2: continue simulation after bounce, sampling observation frames
    // until ball reaches the paddle plane (y ≈ table.length) or goes out
    let mut state = serve_bounce.post_bounce;
    let mut t = serve_bounce.time;
    let mut obs_frames: Vec<ObsFrame> = Vec::new();
    let mut next_obs_time = t;

    // Collect initial observation at bounce
    obs_frames.push([state.pos.x, state.pos.y, state.pos.z]);
    next_obs_time += OBS_DT;

    loop {
        // Sample observation at ~60Hz
        if t >= next_obs_time {
            obs_frames.push([state.pos.x, state.pos.y, state.pos.z]);
            next_obs_time += OBS_DT;
        }

        // Ball reached paddle plane or went past it?
        if state.pos.y >= table.length {
            break;
        }
        // Ball went below floor or timed out?
        if state.pos.z < 0.0 || t > 5.0 {
            return RallyResult {
                outcome: RallyOutcome::BadServe("Ball never reached receiver".to_string()),
                observations: obs_frames,
                action: action.clone(),
                reward: 0.0,
            };
        }

        state = rk4_step(&state, DT);
        t += DT;
    }

    // Keep only the last OBS_FRAMES observations
    let obs_len = obs_frames.len();
    if obs_len > OBS_FRAMES {
        obs_frames = obs_frames[obs_len - OBS_FRAMES..].to_vec();
    }

    // Phase 3: paddle contact
    let paddle_result = apply_paddle_hit(&state, action, table.length, PADDLE_RADIUS);

    match paddle_result {
        PaddleResult::Miss { miss_distance } => {
            let reward = -0.1 - miss_distance.min(1.0) * 0.2;
            RallyResult {
                outcome: RallyOutcome::PaddleMiss { miss_distance },
                observations: obs_frames,
                action: action.clone(),
                reward,
            }
        }
        PaddleResult::Hit(hit_state) => {
            // Phase 4: simulate the return
            let return_result = simulate_full(hit_state, table, 1);

            // Check net crossing: ball must go from y > length/2 to y < length/2
            let crossed_net = return_result.trajectory.windows(2).any(|w| {
                let prev_y = w[0].1.pos.y;
                let curr_y = w[1].1.pos.y;
                prev_y > table.length / 2.0 && curr_y <= table.length / 2.0
            });

            // Check if ball hit the net (crossed net plane below net height)
            let net_top_z = table.surface_z() + table.net_height;
            let hit_net = return_result.trajectory.windows(2).any(|w| {
                let prev = &w[0].1;
                let curr = &w[1].1;
                if (prev.pos.y > table.length / 2.0) && (curr.pos.y <= table.length / 2.0) {
                    let frac = (table.length / 2.0 - prev.pos.y) / (curr.pos.y - prev.pos.y);
                    let z_at_net = prev.pos.z + frac * (curr.pos.z - prev.pos.z);
                    z_at_net < net_top_z
                } else {
                    false
                }
            });

            if hit_net {
                return RallyResult {
                    outcome: RallyOutcome::ReturnHitNet,
                    observations: obs_frames,
                    action: action.clone(),
                    reward: 0.3 + 0.1, // contact + partial net credit
                };
            }

            if !crossed_net {
                return RallyResult {
                    outcome: RallyOutcome::ReturnMissedTable,
                    observations: obs_frames,
                    action: action.clone(),
                    reward: 0.3, // contact only
                };
            }

            // Check if ball landed on server's half (y < length/2)
            match return_result.bounces.first() {
                Some(b) if b.landing.y < table.length / 2.0 => {
                    // Success! Ball landed on server's half
                    let lx = b.landing.x;
                    let ly = b.landing.y;

                    // Placement bonus: closer to edges = better
                    let edge_x = (lx / table.width - 0.5).abs() * 2.0; // 0..1
                    let edge_y = (1.0 - ly / (table.length / 2.0)).abs(); // 0..1
                    let placement_bonus = (edge_x + edge_y) * 0.1;

                    RallyResult {
                        outcome: RallyOutcome::Success {
                            landing_x: lx,
                            landing_y: ly,
                        },
                        observations: obs_frames,
                        action: action.clone(),
                        reward: 0.3 + 0.2 + 0.5 + placement_bonus,
                    }
                }
                _ => {
                    // Crossed net but missed table
                    RallyResult {
                        outcome: RallyOutcome::ReturnMissedTable,
                        observations: obs_frames,
                        action: action.clone(),
                        reward: 0.3 + 0.2, // contact + net cleared
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_rally_success() {
        let table = Table::standard();
        // A simple serve: medium speed toward opponent
        let serve = BallState::new(
            Vec3::new(0.76, 0.0, 0.90),
            Vec3::new(0.0, 8.0, 0.7),
            Vec3::ZERO,
        );
        // Paddle positioned to hit the ball back
        let action = PaddleAction {
            paddle_x: 0.76,
            paddle_z: 0.86,
            tilt_x: 0.1,
            tilt_z: 0.0,
            swing_speed: 6.0,
            swing_elevation: 0.2,
        };
        let result = run_rally(serve, &action, &table);
        assert!(
            result.observations.len() > 0,
            "Should have observation frames"
        );
        assert!(result.reward >= 0.3, "Should at least make contact, reward={}", result.reward);
    }

    #[test]
    fn test_rally_paddle_miss() {
        let table = Table::standard();
        let serve = BallState::new(
            Vec3::new(0.76, 0.0, 0.90),
            Vec3::new(0.0, 8.0, 0.7),
            Vec3::ZERO,
        );
        // Paddle far from ball path
        let action = PaddleAction {
            paddle_x: 0.0,
            paddle_z: 1.5,
            tilt_x: 0.0,
            tilt_z: 0.0,
            swing_speed: 5.0,
            swing_elevation: 0.1,
        };
        let result = run_rally(serve, &action, &table);
        match result.outcome {
            RallyOutcome::PaddleMiss { .. } => {}
            other => panic!("Expected PaddleMiss, got {:?}", other),
        }
        assert!(result.reward < 0.0, "Miss should have negative reward");
    }
}
