use crate::physics::integrator::rk4_step;
use crate::physics::paddle::{apply_paddle_hit, PaddleAction, PaddleResult};
use crate::physics::state::{BallState, Vec3};
use crate::simulation::{simulate_full, DT};
use crate::table::Table;

/// Observation frame: ball position at a simulated "camera" sample time
pub type ObsFrame = [f64; 3]; // [x, y, z]

/// How many past ball positions the agent can see.
/// 30 frames @ ~60 Hz ≈ 500 ms — covers the complete post-bounce trajectory,
/// giving the network enough curvature data to infer spin strength and direction.
pub const OBS_FRAMES: usize = 30;

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

/// Subsampled trajectory point for replay visualization
#[derive(Debug, Clone)]
pub struct ReplayPoint {
    pub t: f64,
    pub pos: Vec3,
    pub vel: Vec3,
    pub omega: Vec3,
}

/// Full replay data for visualization (serve + return trajectories)
#[derive(Debug)]
pub struct ReplayData {
    pub serve_trajectory: Vec<ReplayPoint>,
    pub serve_bounces: Vec<(f64, Vec3)>,
    pub return_trajectory: Vec<ReplayPoint>,
    pub return_bounces: Vec<(f64, Vec3)>,
    pub paddle_action: PaddleAction,
    pub paddle_contact_pos: Option<Vec3>,
    pub hit_omega: Option<Vec3>,
    pub outcome: RallyOutcome,
    pub reward: f64,
}

/// Subsample interval for replay trajectories (~3ms = every 6 DT steps)
const REPLAY_DT: f64 = 0.003;

/// Result of simulating just the serve up to the paddle zone.
/// Used by the RL env to split observation from action.
#[derive(Debug)]
pub struct ServeObservation {
    /// Ball positions at ~60Hz (agent's "camera" input)
    pub observations: Vec<ObsFrame>,
    /// Post-bounce trajectory sampled at ~60Hz for Y-interpolation
    pub flight_trajectory: Vec<BallState>,
    pub flight_times: Vec<f64>,
    /// Whether the serve was valid
    pub bad_serve: bool,
}

/// Simulate the serve: ball bounces on server's half, crosses net, bounces on
/// receiver's half. Agent observes the ball after the second bounce.
pub fn prepare_serve(serve_state: BallState, table: &Table) -> ServeObservation {
    // Simulate up to 2 bounces: first on server's half, second on receiver's half
    let serve_result = simulate_full(serve_state, table, 2);

    // Need at least 2 bounces for a legal serve
    if serve_result.bounces.len() < 2 {
        return ServeObservation {
            observations: Vec::new(),
            flight_trajectory: Vec::new(),
            flight_times: Vec::new(),
            bad_serve: true,
        };
    }

    let first_bounce = &serve_result.bounces[0];
    let second_bounce = &serve_result.bounces[1];

    // First bounce must be on server's half (y < net_y)
    if first_bounce.landing.y >= table.net_y() {
        return ServeObservation {
            observations: Vec::new(),
            flight_trajectory: Vec::new(),
            flight_times: Vec::new(),
            bad_serve: true,
        };
    }

    // Second bounce must be on receiver's half (y > net_y)
    if second_bounce.landing.y <= table.net_y() {
        return ServeObservation {
            observations: Vec::new(),
            flight_trajectory: Vec::new(),
            flight_times: Vec::new(),
            bad_serve: true,
        };
    }

    // Agent observes the ball AFTER the second bounce (on their side)
    let mut state = second_bounce.post_bounce;
    let mut t = second_bounce.time;
    let mut obs_frames: Vec<ObsFrame> = Vec::new();
    let mut flight_traj: Vec<BallState> = Vec::new();
    let mut flight_times: Vec<f64> = Vec::new();
    let mut next_obs_time = t;

    obs_frames.push([state.pos.x, state.pos.y, state.pos.z]);
    flight_traj.push(state);
    flight_times.push(t);
    next_obs_time += OBS_DT;

    loop {
        if t >= next_obs_time {
            obs_frames.push([state.pos.x, state.pos.y, state.pos.z]);
            flight_traj.push(state);
            flight_times.push(t);
            next_obs_time += OBS_DT;
        }
        // Continue until ball drops below floor or goes far past table
        if state.pos.z < 0.0 || state.pos.y > table.length + 2.0 || t > 5.0 {
            break;
        }
        state = rk4_step(&state, DT);
        t += DT;
    }

    let obs_len = obs_frames.len();
    if obs_len > OBS_FRAMES {
        obs_frames = obs_frames[obs_len - OBS_FRAMES..].to_vec();
    }

    ServeObservation {
        observations: obs_frames,
        flight_trajectory: flight_traj,
        flight_times: flight_times,
        bad_serve: false,
    }
}

/// Find the ball state at a given Y position by interpolating the trajectory.
/// Returns None if the ball never reaches that Y.
fn interpolate_at_y(traj: &[BallState], target_y: f64) -> Option<BallState> {
    for w in traj.windows(2) {
        let a = &w[0];
        let b = &w[1];
        if (a.pos.y <= target_y && b.pos.y >= target_y)
            || (a.pos.y >= target_y && b.pos.y <= target_y)
        {
            let dy = b.pos.y - a.pos.y;
            if dy.abs() < 1e-12 {
                return Some(*a);
            }
            let f = (target_y - a.pos.y) / dy;
            let f = f.clamp(0.0, 1.0);
            return Some(BallState {
                pos: Vec3::new(
                    a.pos.x + f * (b.pos.x - a.pos.x),
                    target_y,
                    a.pos.z + f * (b.pos.z - a.pos.z),
                ),
                vel: Vec3::new(
                    a.vel.x + f * (b.vel.x - a.vel.x),
                    a.vel.y + f * (b.vel.y - a.vel.y),
                    a.vel.z + f * (b.vel.z - a.vel.z),
                ),
                omega: Vec3::new(
                    a.omega.x + f * (b.omega.x - a.omega.x),
                    a.omega.y + f * (b.omega.y - a.omega.y),
                    a.omega.z + f * (b.omega.z - a.omega.z),
                ),
            });
        }
    }
    None
}

/// Apply the paddle action to a ball trajectory and evaluate the return.
/// Interpolates the trajectory to find the ball at the agent's chosen paddle_y.
pub fn evaluate_return(
    flight_trajectory: &[BallState],
    action: &PaddleAction,
    table: &Table,
) -> (RallyOutcome, f64) {
    // Find ball state at paddle_y
    let ball_at_paddle = match interpolate_at_y(flight_trajectory, action.paddle_y) {
        Some(state) => state,
        None => {
            // Ball never reached the paddle_y — miss with distance penalty
            // Estimate miss distance from closest trajectory point
            let min_dist = flight_trajectory
                .iter()
                .map(|s| {
                    let dx = s.pos.x - action.paddle_x;
                    let dy = s.pos.y - action.paddle_y;
                    let dz = s.pos.z - action.paddle_z.max(table.surface_z() + 0.09);
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .fold(f64::MAX, f64::min);
            let reward = -0.1 - min_dist.min(1.0) * 0.2;
            return (RallyOutcome::PaddleMiss { miss_distance: min_dist }, reward);
        }
    };

    // Check if ball is above the table surface at the paddle plane
    if ball_at_paddle.pos.z < table.contact_z() - 0.05 {
        // Ball is under the table — unreachable
        let dz = table.contact_z() - ball_at_paddle.pos.z;
        let reward = -0.1 - dz.min(1.0) * 0.2;
        return (RallyOutcome::PaddleMiss { miss_distance: dz }, reward);
    }

    let paddle_result = apply_paddle_hit(&ball_at_paddle, action, table.surface_z(), PADDLE_RADIUS);

    match paddle_result {
        PaddleResult::Miss { miss_distance } => {
            let reward = -0.1 - miss_distance.min(1.0) * 0.2;
            (RallyOutcome::PaddleMiss { miss_distance }, reward)
        }
        PaddleResult::Hit(hit_state) => {
            let return_result = simulate_full(hit_state, table, 1);
            evaluate_return_result(&return_result, table, &ball_at_paddle, &hit_state)
        }
    }
}

/// Run one rally episode using prepare_serve + evaluate_return.
pub fn run_rally(
    serve_state: BallState,
    action: &PaddleAction,
    table: &Table,
) -> RallyResult {
    let serve_obs = prepare_serve(serve_state, table);

    if serve_obs.bad_serve {
        return RallyResult {
            outcome: RallyOutcome::BadServe("Serve never bounced".to_string()),
            observations: Vec::new(),
            action: action.clone(),
            reward: 0.0,
        };
    }

    let (outcome, reward) = evaluate_return(&serve_obs.flight_trajectory, action, table);

    RallyResult {
        outcome,
        observations: serve_obs.observations,
        action: action.clone(),
        reward,
    }
}

/// Run one rally and return full replay data for visualization.
/// Uses prepare_serve + interpolate_at_y for consistent behavior with training.
pub fn run_rally_replay(
    serve_state: BallState,
    action: &PaddleAction,
    table: &Table,
) -> ReplayData {
    let empty_replay = |outcome: RallyOutcome| ReplayData {
        serve_trajectory: Vec::new(),
        serve_bounces: Vec::new(),
        return_trajectory: Vec::new(),
        return_bounces: Vec::new(),
        paddle_action: action.clone(),
        paddle_contact_pos: None,
        hit_omega: None,
        outcome,
        reward: 0.0,
    };

    // Phase 1: simulate serve until 2 bounces (server half → receiver half)
    let serve_result = simulate_full(serve_state, table, 2);

    if serve_result.bounces.len() < 2 {
        return empty_replay(RallyOutcome::BadServe("Serve didn't complete 2 bounces".to_string()));
    }

    let first_bounce = &serve_result.bounces[0];
    let second_bounce = &serve_result.bounces[1];

    if first_bounce.landing.y >= table.net_y() || second_bounce.landing.y <= table.net_y() {
        return empty_replay(RallyOutcome::BadServe("Illegal serve".to_string()));
    }

    // Subsample the full serve trajectory (both bounces included)
    let mut serve_traj: Vec<ReplayPoint> = Vec::new();
    let mut next_sample = 0.0;
    for (t, st) in &serve_result.trajectory {
        if *t >= next_sample {
            serve_traj.push(ReplayPoint { t: *t, pos: st.pos, vel: st.vel, omega: st.omega });
            next_sample = *t + REPLAY_DT;
        }
    }

    let serve_bounces: Vec<(f64, Vec3)> = serve_result.bounces.iter().map(|b| (b.time, b.landing)).collect();

    // Phase 2: post-second-bounce flight (agent's view)
    let mut state = second_bounce.post_bounce;
    let mut t = second_bounce.time;
    let mut flight_traj: Vec<BallState> = Vec::new();
    let mut next_sample_time = t;

    loop {
        if t >= next_sample_time {
            serve_traj.push(ReplayPoint { t, pos: state.pos, vel: state.vel, omega: state.omega });
            flight_traj.push(state);
            next_sample_time = t + REPLAY_DT;
        }
        if state.pos.z < 0.0 || state.pos.y > table.length + 2.0 || t > 5.0 {
            break;
        }
        state = rk4_step(&state, DT);
        t += DT;
    }

    // Phase 3: find ball at paddle_y and apply paddle
    // Also find the contact time from the replay trajectory for proper timing
    let ball_at_paddle = interpolate_at_y(&flight_traj, action.paddle_y);
    let contact_time = serve_traj.windows(2).find_map(|w| {
        let a = &w[0];
        let b = &w[1];
        if (a.pos.y <= action.paddle_y && b.pos.y >= action.paddle_y)
            || (a.pos.y >= action.paddle_y && b.pos.y <= action.paddle_y)
        {
            let dy = b.pos.y - a.pos.y;
            if dy.abs() < 1e-12 { return Some(a.t); }
            let f = ((action.paddle_y - a.pos.y) / dy).clamp(0.0, 1.0);
            Some(a.t + f * (b.t - a.t))
        } else {
            None
        }
    }).unwrap_or(t);
    let ball_at_paddle = match ball_at_paddle {
        Some(s) => s,
        None => {
            return ReplayData {
                serve_trajectory: serve_traj,
                serve_bounces,
                return_trajectory: Vec::new(),
                return_bounces: Vec::new(),
                paddle_action: action.clone(),
                paddle_contact_pos: None,
                hit_omega: None,
                outcome: RallyOutcome::PaddleMiss { miss_distance: 1.0 },
                reward: -0.3,
            };
        }
    };

    if ball_at_paddle.pos.z < table.contact_z() - 0.05 {
        return ReplayData {
            serve_trajectory: serve_traj,
            serve_bounces,
            return_trajectory: Vec::new(),
            return_bounces: Vec::new(),
            paddle_action: action.clone(),
            paddle_contact_pos: Some(ball_at_paddle.pos),
            hit_omega: None,
            outcome: RallyOutcome::PaddleMiss { miss_distance: table.contact_z() - ball_at_paddle.pos.z },
            reward: -0.2,
        };
    }

    let paddle_result = apply_paddle_hit(&ball_at_paddle, action, table.surface_z(), PADDLE_RADIUS);

    match paddle_result {
        PaddleResult::Miss { miss_distance } => {
            ReplayData {
                serve_trajectory: serve_traj,
                serve_bounces,
                return_trajectory: Vec::new(),
                return_bounces: Vec::new(),
                paddle_action: action.clone(),
                paddle_contact_pos: Some(ball_at_paddle.pos),
                hit_omega: None,
                outcome: RallyOutcome::PaddleMiss { miss_distance },
                reward: -0.1 - miss_distance.min(1.0) * 0.2,
            }
        }
        PaddleResult::Hit(hit_state) => {
            let contact_pos = hit_state.pos;
            let return_result = simulate_full(hit_state, table, 1);

            let mut return_traj: Vec<ReplayPoint> = Vec::new();
            let mut next_sample = 0.0;
            for (rt, st) in &return_result.trajectory {
                if *rt >= next_sample {
                    return_traj.push(ReplayPoint {
                        t: contact_time + *rt, pos: st.pos, vel: st.vel, omega: st.omega,
                    });
                    next_sample = *rt + REPLAY_DT;
                }
            }

            let return_bounces: Vec<(f64, Vec3)> = return_result.bounces.iter()
                .map(|b| (contact_time + b.time, b.landing)).collect();

            let (outcome, reward) = evaluate_return_result(&return_result, table, &ball_at_paddle, &hit_state);

            ReplayData {
                serve_trajectory: serve_traj,
                serve_bounces,
                return_trajectory: return_traj,
                return_bounces,
                paddle_action: action.clone(),
                paddle_contact_pos: Some(contact_pos),
                hit_omega: Some(hit_state.omega),
                outcome,
                reward,
            }
        }
    }
}

/// Evaluate the return simulation result with smooth reward gradients.
///
/// Reward structure — **aggressive profile v6**:
///   Ziel: Balance zwischen "Ball auf Tisch bringen" und "Topspin lernen".
///   Base-Reward hoch genug dass Tischlandung sich lohnt (auch mit Backspin),
///   aber Topspin-Bonus so dominant dass er langfristig die beste Strategie ist.
///
///   Base rewards (solid foundation):
///     Contact:            +0.30  (always, since we got a Hit)
///     Net cleared:        +0.20
///     On table (success): +0.50  → 1.00 total
///
///   Quality bonuses (only on success):
///     Apex timing:        0.00–1.00  ball descending at contact — STARK
///     Net-skim:          -0.50–1.00  sweet spot 2–5cm über Netz
///     Return speed:      -0.20–0.30  fast = bonus, slow = penalty
///     Return spin:       -1.00–3.00  topspin = HAUPTANREIZ, backspin = Strafe
///     Placement:          0.00–0.20  landing near edges/baseline
///
///   Gradient: Topspin(5.50) >>> Neutral(2.50) >> Block(0.50) > Near-miss(0.30)
///   EV@20%: topspin=1.10 > block=0.50 → Agent lernt Topspin zu bevorzugen
fn evaluate_return_result(
    return_result: &crate::simulation::SimResult,
    table: &Table,
    ball_at_contact: &BallState,
    hit_state: &BallState,
) -> (RallyOutcome, f64) {
    let half_y = table.length / 2.0;
    let net_top_z = table.surface_z() + table.net_height;

    // Check net crossing and collision
    let mut z_at_net: Option<f64> = None;
    let crossed_net = return_result.trajectory.windows(2).any(|w| {
        let prev = &w[0].1;
        let curr = &w[1].1;
        if prev.pos.y > half_y && curr.pos.y <= half_y {
            let frac = (half_y - prev.pos.y) / (curr.pos.y - prev.pos.y);
            let z = prev.pos.z + frac * (curr.pos.z - prev.pos.z);
            z_at_net = Some(z);
            true
        } else {
            false
        }
    });

    let hit_net = z_at_net.is_some_and(|z| z < net_top_z);

    // ── Quality bonuses (applied only on success) ────────────────────────
    //
    // Apex timing: STARKER Anreiz, den Ball am höchsten Punkt zu treffen.
    // Profis treffen den Ball kurz nach dem Apex — maximale Kontrolle.
    // Full bonus at apex (vz=0), linearly fades to zero at vz=-2.0 m/s.
    let apex_bonus = {
        let vz = ball_at_contact.vel.z;
        if vz >= 0.0 {
            0.0                                     // still ascending – no bonus
        } else if vz > -2.0 {
            (1.0 + vz / 2.0) * 1.00                // 1.00 at apex → 0.0 at -2 m/s
        } else {
            0.0                                     // well past apex – no bonus
        }
    };

    // Net clearance: Sweet-Spot bei 2–5 cm über dem Netz.
    // Zu knapp (<2cm) ist riskant (Netz!), zu hoch (>10cm) ist ein Lob.
    //   0–2 cm  → +0.50..+1.00 (gut aber riskant, aufsteigend zum sweet spot)
    //   2–5 cm  → +1.00 (perfekt: aggressiv aber sicher)
    //   5–10 cm → +1.00..0.00 (linearer Abfall)
    //   10–30cm → 0.00..-0.50 (Lob-Strafe)
    let net_skim_bonus = match z_at_net {
        Some(z) if z >= net_top_z => {
            let clearance = z - net_top_z;
            if clearance <= 0.02 {
                0.50 + (clearance / 0.02) * 0.50        // 0–2cm: 0.50→1.00
            } else if clearance <= 0.05 {
                1.00                                      // 2–5cm: perfect sweet spot
            } else if clearance <= 0.10 {
                1.00 * (1.0 - (clearance - 0.05) / 0.05) // 5–10cm: 1.00→0.00
            } else {
                -((clearance - 0.10) / 0.20).min(1.0) * 0.50  // 10–30cm: 0.00→-0.50
            }
        }
        _ => 0.0,
    };

    // Return speed bonus: faster return gives opponent less reaction time.
    let speed_bonus = {
        let v = hit_state.vel.norm();
        if v < 5.0 {
            -0.30 * (1.0 - v / 5.0)  // penalty for very slow returns
        } else {
            (v / 20.0).min(1.0) * 0.50
        }
    };

    // Spin bonus: TOPSPIN ist der HAUPTANREIZ (bis +3.0).
    // Backspin wird bestraft (-1.0) aber nicht so brutal dass der Agent
    // lieber den Ball verfehlt als ihn auf den Tisch zu bringen.
    // Gradient: topspin_success(5.5) >>> neutral(2.5) >> backspin(0.5) > miss(0.3)
    // Koordinatensystem: Return in -Y-Richtung → Topspin = omega.x > 0.
    let spin_bonus = {
        let omega_x = hit_state.omega.x;
        if omega_x > 0.0 {
            (omega_x / 50.0).min(1.0) * 3.0        // topspin: up to +3.0
        } else {
            (omega_x / 100.0).max(-1.0) * 1.00     // backspin: up to -1.00
        }
    };

    if hit_net {
        let z = z_at_net.unwrap();
        let clearance_ratio = (z / net_top_z).clamp(0.0, 1.0);
        (RallyOutcome::ReturnHitNet, 0.3 + clearance_ratio * 0.15)
    } else if !crossed_net {
        let first_vel_y = return_result.trajectory.first()
            .map(|(_, s)| s.vel.y)
            .unwrap_or(0.0);
        let direction_bonus = if first_vel_y < 0.0 { 0.1 } else { 0.0 };
        (RallyOutcome::ReturnMissedTable, 0.3 + direction_bonus)
    } else {
        // Crossed net! Check if it landed on the table
        match return_result.bounces.first() {
            Some(b) if b.landing.y < half_y
                && b.landing.x >= 0.0
                && b.landing.x <= table.width =>
            {
                // SUCCESS — base reward + quality bonuses (topspin dominates)
                let lx = b.landing.x;
                let ly = b.landing.y;
                let edge_x = (lx / table.width - 0.5).abs() * 2.0;
                let edge_y = (1.0 - ly / half_y).abs();
                let placement_bonus = (edge_x + edge_y) * 0.1;
                (
                    RallyOutcome::Success { landing_x: lx, landing_y: ly },
                    0.3 + 0.2 + 0.5 + placement_bonus
                        + apex_bonus + net_skim_bonus + speed_bonus + spin_bonus,
                )
            }
            _ => {
                // Crossed net but missed the table — near-miss gradient.
                let ground_landing = return_result.trajectory.windows(2).find_map(|w| {
                    let prev = &w[0].1;
                    let curr = &w[1].1;
                    if prev.pos.z > table.surface_z() && curr.pos.z <= table.surface_z() {
                        let dz = prev.pos.z - curr.pos.z;
                        if dz.abs() < 1e-12 { return None; }
                        let frac = (prev.pos.z - table.surface_z()) / dz;
                        Some(Vec3::new(
                            prev.pos.x + frac * (curr.pos.x - prev.pos.x),
                            prev.pos.y + frac * (curr.pos.y - prev.pos.y),
                            table.surface_z(),
                        ))
                    } else {
                        None
                    }
                });

                let near_miss_bonus = match ground_landing {
                    Some(land) => {
                        let dx = if land.x < 0.0 { -land.x }
                                 else if land.x > table.width { land.x - table.width }
                                 else { 0.0 };
                        let dy = if land.y >= half_y { land.y - half_y } else { 0.0 };
                        let dist = (dx * dx + dy * dy).sqrt();
                        (1.0 - (dist / 2.0).min(1.0)) * 0.4
                    }
                    None => 0.0,
                };

                (RallyOutcome::ReturnMissedTable, 0.3 + 0.2 + near_miss_bonus)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a realistic test serve: launched downward from z=1.05, bounces on
    /// server's half, clears net, bounces on receiver's half.
    fn test_serve() -> BallState {
        let speed = 6.0_f64;
        let elev = (-18.0_f64).to_radians();
        BallState::new(
            Vec3::new(0.76, 0.10, 1.05),
            Vec3::new(0.0, speed * elev.cos(), speed * elev.sin()),
            Vec3::ZERO,
        )
    }

    #[test]
    fn test_basic_rally_success() {
        let table = Table::standard();
        let serve = test_serve();
        let action = PaddleAction {
            paddle_x: 0.76,
            paddle_y: 2.4,
            paddle_z: 0.90,
            tilt_x: 0.1,
            tilt_z: 0.0,
            swing_speed: 6.0,
            swing_elevation: 0.2,
        };
        let result = run_rally(serve, &action, &table);
        assert!(
            !result.observations.is_empty(),
            "Should have observation frames, outcome={:?}", result.outcome
        );
        assert!(result.reward >= 0.3, "Should at least make contact, reward={}", result.reward);
    }

    #[test]
    fn test_rally_paddle_miss() {
        let table = Table::standard();
        let serve = test_serve();
        let action = PaddleAction {
            paddle_x: 0.0,
            paddle_y: 2.5,
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
