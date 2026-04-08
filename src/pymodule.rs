use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::physics::paddle::PaddleAction;
use crate::physics::state::{BallState, Vec3};
use crate::rally::{
    prepare_serve, evaluate_return, run_rally_replay,
    interpolate_at_y as interpolate_at_y_pub,
    RallyOutcome, OBS_FRAMES,
};
use crate::serve::{random_serve, Difficulty, Rng};
use crate::table::Table;

/// The RL environment exposed to Python.
///
/// Two-step Gymnasium flow:
///   reset() → generates serve, simulates to paddle plane, returns ball observations
///   step(action) → applies paddle action, evaluates return, returns reward
#[pyclass]
pub struct SimEnv {
    table: Table,
    rng: Rng,
    difficulty: u8,
    /// Cached flight trajectory for Y-interpolation
    pending_trajectory: Vec<BallState>,
    /// Cached observations from the last reset()
    pending_obs: Vec<f64>,
    /// Cached initial serve state for replay
    pending_serve: BallState,
    /// Serve state from the LAST completed step (survives auto-reset)
    last_serve: BallState,
}

#[pymethods]
impl SimEnv {
    #[new]
    #[pyo3(signature = (seed=42, difficulty=1))]
    fn new(seed: u64, difficulty: u8) -> Self {
        SimEnv {
            table: Table::standard(),
            rng: Rng::new(seed),
            difficulty: difficulty.clamp(1, 3),
            pending_trajectory: Vec::new(),
            pending_obs: vec![0.0; OBS_FRAMES * 3],
            pending_serve: BallState::new(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO),
            last_serve: BallState::new(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO),
        }
    }

    /// Set curriculum difficulty (1=easy, 2=medium, 3=hard)
    fn set_difficulty(&mut self, difficulty: u8) {
        self.difficulty = difficulty.clamp(1, 3);
    }

    /// Number of observation values returned per step
    #[getter]
    fn obs_size(&self) -> usize {
        OBS_FRAMES * 3
    }

    /// Number of action values expected per step
    #[getter]
    fn action_size(&self) -> usize {
        7
    }

    /// Reset: generate a serve, simulate to paddle zone, return ball observations.
    ///
    /// The agent sees the ball trajectory BEFORE choosing its paddle action.
    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyList> {
        let difficulty = match self.difficulty {
            1 => Difficulty::Stage1,
            2 => Difficulty::Stage2,
            _ => Difficulty::Stage3,
        };

        // Keep generating serves until we get a valid one
        loop {
            let serve = random_serve(&mut self.rng, difficulty);
            let obs_result = prepare_serve(serve, &self.table);

            if obs_result.bad_serve {
                continue; // skip invalid serves
            }

            // Flatten observations to 30 floats, right-aligned (pad zeros on left)
            let mut obs_flat = vec![0.0_f64; OBS_FRAMES * 3];
            let n = obs_result.observations.len().min(OBS_FRAMES);
            let offset = OBS_FRAMES - n;
            for (i, frame) in obs_result.observations.iter().rev().take(n).rev().enumerate() {
                obs_flat[(offset + i) * 3] = frame[0];
                obs_flat[(offset + i) * 3 + 1] = frame[1];
                obs_flat[(offset + i) * 3 + 2] = frame[2];
            }

            self.pending_serve = serve;
            self.pending_trajectory = obs_result.flight_trajectory;
            self.pending_obs = obs_flat.clone();
            return PyList::new(py, &obs_flat).unwrap();
        }
    }

    /// Apply the agent's paddle action to the pending serve.
    ///
    /// Action: 7 floats [paddle_x, paddle_y, paddle_z, tilt_x, tilt_z, swing_speed, swing_elevation]
    /// Returns (observation, reward, done, info) — done is always True.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: Vec<f64>,
    ) -> PyResult<(Bound<'py, PyList>, f64, bool, PyObject)> {
        if action.len() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Action must have 7 elements: [paddle_x, paddle_y, paddle_z, tilt_x, tilt_z, swing_speed, swing_elevation]"
            ));
        }

        let paddle_action = PaddleAction {
            paddle_x: action[0],
            paddle_y: action[1],
            paddle_z: action[2],
            tilt_x: action[3],
            tilt_z: action[4],
            swing_speed: action[5],
            swing_elevation: action[6],
        };

        // Apply paddle to cached trajectory
        let trajectory = std::mem::take(&mut self.pending_trajectory);
        let (outcome, reward) = if trajectory.is_empty() {
            (RallyOutcome::BadServe("No pending serve".to_string()), 0.0)
        } else {
            evaluate_return(&trajectory, &paddle_action, &self.table)
        };

        // Build info dict
        let info = pyo3::types::PyDict::new(py);
        let outcome_str = match &outcome {
            RallyOutcome::Success { landing_x, landing_y } => {
                let _ = info.set_item("landing_x", *landing_x);
                let _ = info.set_item("landing_y", *landing_y);
                "success"
            }
            RallyOutcome::ReturnMissedTable => "return_missed_table",
            RallyOutcome::ReturnHitNet => "return_hit_net",
            RallyOutcome::PaddleMiss { miss_distance } => {
                let _ = info.set_item("miss_distance", *miss_distance);
                "paddle_miss"
            }
            RallyOutcome::BadServe(_) => "bad_serve",
        };
        let _ = info.set_item("outcome", outcome_str);

        // Save serve before auto-reset overwrites pending_serve
        self.last_serve = self.pending_serve;

        // Auto-reset: return observation for next episode
        let next_obs = self.reset(py);

        Ok((next_obs, reward, true, info.into_any().unbind()))
    }

    /// Run a replay of the LAST serve from reset() with the given action.
    ///
    /// Returns a dict with serve/return trajectories, paddle action, outcome, etc.
    /// Each trajectory point is [t, x, y, z, vx, vy, vz, ox, oy, oz].
    fn replay<'py>(
        &mut self,
        py: Python<'py>,
        action: Vec<f64>,
    ) -> PyResult<PyObject> {
        if action.len() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Action must have 7 elements",
            ));
        }

        let paddle_action = PaddleAction {
            paddle_x: action[0],
            paddle_y: action[1],
            paddle_z: action[2],
            tilt_x: action[3],
            tilt_z: action[4],
            swing_speed: action[5],
            swing_elevation: action[6],
        };

        // Use the serve from the last completed step() (not pending_serve which was overwritten by auto-reset)
        let replay = run_rally_replay(self.last_serve, &paddle_action, &self.table);

        let result = pyo3::types::PyDict::new(py);

        // Serve trajectory: list of [t, x, y, z, vx, vy, vz, ox, oy, oz]
        let serve_traj: Vec<Vec<f64>> = replay
            .serve_trajectory
            .iter()
            .map(|p| {
                vec![
                    p.t, p.pos.x, p.pos.y, p.pos.z, p.vel.x, p.vel.y, p.vel.z,
                    p.omega.x, p.omega.y, p.omega.z,
                ]
            })
            .collect();
        let _ = result.set_item("serve_trajectory", serve_traj);

        let serve_bounces: Vec<Vec<f64>> = replay
            .serve_bounces
            .iter()
            .map(|(t, p)| vec![*t, p.x, p.y, p.z])
            .collect();
        let _ = result.set_item("serve_bounces", serve_bounces);

        // Return trajectory
        let return_traj: Vec<Vec<f64>> = replay
            .return_trajectory
            .iter()
            .map(|p| {
                vec![
                    p.t, p.pos.x, p.pos.y, p.pos.z, p.vel.x, p.vel.y, p.vel.z,
                    p.omega.x, p.omega.y, p.omega.z,
                ]
            })
            .collect();
        let _ = result.set_item("return_trajectory", return_traj);

        let return_bounces: Vec<Vec<f64>> = replay
            .return_bounces
            .iter()
            .map(|(t, p)| vec![*t, p.x, p.y, p.z])
            .collect();
        let _ = result.set_item("return_bounces", return_bounces);

        // Paddle action
        let paddle = pyo3::types::PyDict::new(py);
        let _ = paddle.set_item("paddle_x", action[0]);
        let _ = paddle.set_item("paddle_y", action[1]);
        let _ = paddle.set_item("paddle_z", action[2]);
        let _ = paddle.set_item("tilt_x", action[3]);
        let _ = paddle.set_item("tilt_z", action[4]);
        let _ = paddle.set_item("swing_speed", action[5]);
        let _ = paddle.set_item("swing_elevation", action[6]);
        let _ = result.set_item("paddle", paddle);

        if let Some(cp) = replay.paddle_contact_pos {
            let _ = result.set_item("contact_pos", vec![cp.x, cp.y, cp.z]);
        }

        if let Some(om) = replay.hit_omega {
            let _ = result.set_item("hit_omega", vec![om.x, om.y, om.z]);
        }

        let outcome_str = match &replay.outcome {
            RallyOutcome::Success { landing_x, landing_y } => {
                let _ = result.set_item("landing", vec![*landing_x, *landing_y]);
                "success"
            }
            RallyOutcome::ReturnMissedTable => "return_missed_table",
            RallyOutcome::ReturnHitNet => "return_hit_net",
            RallyOutcome::PaddleMiss { miss_distance } => {
                let _ = result.set_item("miss_distance", *miss_distance);
                "paddle_miss"
            }
            RallyOutcome::BadServe(_) => "bad_serve",
        };
        let _ = result.set_item("outcome", outcome_str);
        let _ = result.set_item("reward", replay.reward);

        Ok(result.into_any().unbind())
    }

    /// Run a batch of episodes (for vectorized environments).
    /// Returns lists of (observations, rewards, outcomes).
    fn batch_step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<Vec<f64>>,
    ) -> PyResult<(Vec<Bound<'py, PyList>>, Vec<f64>, Vec<String>)> {
        let mut all_obs = Vec::with_capacity(actions.len());
        let mut all_rewards = Vec::with_capacity(actions.len());
        let mut all_outcomes = Vec::with_capacity(actions.len());

        for action in actions {
            let (obs, reward, _, info) = self.step(py, action)?;
            let outcome: String = info
                .bind(py)
                .downcast::<pyo3::types::PyDict>()
                .map(|d| {
                    d.get_item("outcome")
                        .ok()
                        .flatten()
                        .map(|v| v.extract::<String>().unwrap_or_default())
                        .unwrap_or_default()
                })
                .unwrap_or_default();
            all_obs.push(obs);
            all_rewards.push(reward);
            all_outcomes.push(outcome);
        }

        Ok((all_obs, all_rewards, all_outcomes))
    }
    /// Generate a batch of serve trajectories for trajectory prediction training.
    /// Returns list of trajectories, each being 30 frames of [x, y, z] at 60Hz.
    /// Only returns valid serves with exactly 30 frames.
    fn generate_trajectories<'py>(
        &mut self,
        py: Python<'py>,
        count: usize,
        difficulty: u8,
    ) -> PyResult<Bound<'py, PyList>> {
        let diff = match difficulty {
            1 => Difficulty::Stage1,
            2 => Difficulty::Stage2,
            _ => Difficulty::Stage3,
        };

        let result = PyList::empty(py);
        let mut generated = 0;

        while generated < count {
            let serve = random_serve(&mut self.rng, diff);
            let obs_result = prepare_serve(serve, &self.table);

            if obs_result.bad_serve {
                continue;
            }

            // Only use trajectories with exactly OBS_FRAMES observations
            if obs_result.observations.len() < OBS_FRAMES {
                continue;
            }

            // Take last 30 frames
            let n = obs_result.observations.len();
            let start = n - OBS_FRAMES;
            let traj = PyList::empty(py);
            for frame in &obs_result.observations[start..] {
                let point = PyList::new(py, &[frame[0], frame[1], frame[2]])?;
                traj.append(point)?;
            }
            result.append(traj)?;
            generated += 1;
        }

        Ok(result)
    }

    /// Generate trajectories with full metadata for predictor visualization.
    /// Returns list of dicts with positions, velocity, spin, and serve speed.
    fn generate_rich_trajectories<'py>(
        &mut self,
        py: Python<'py>,
        count: usize,
        difficulty: u8,
    ) -> PyResult<Bound<'py, PyList>> {
        let diff = match difficulty {
            1 => Difficulty::Stage1,
            2 => Difficulty::Stage2,
            _ => Difficulty::Stage3,
        };

        let result = PyList::empty(py);
        let mut generated = 0;

        while generated < count {
            let serve = random_serve(&mut self.rng, diff);
            let obs_result = prepare_serve(serve, &self.table);

            if obs_result.bad_serve {
                continue;
            }

            if obs_result.observations.len() < OBS_FRAMES {
                continue;
            }

            let n = obs_result.observations.len();
            let start = n - OBS_FRAMES;
            let traj = PyList::empty(py);
            for frame in &obs_result.observations[start..] {
                let point = PyList::new(py, &[frame[0], frame[1], frame[2]])?;
                traj.append(point)?;
            }

            // Also include full flight trajectory with velocity for visualization
            let full_traj = PyList::empty(py);
            let ft_start = if obs_result.flight_trajectory.len() > OBS_FRAMES {
                obs_result.flight_trajectory.len() - OBS_FRAMES
            } else {
                0
            };
            for state in &obs_result.flight_trajectory[ft_start..] {
                let frame = PyList::new(py, &[
                    state.pos.x, state.pos.y, state.pos.z,
                    state.vel.x, state.vel.y, state.vel.z,
                    state.omega.x, state.omega.y, state.omega.z,
                ])?;
                full_traj.append(frame)?;
            }

            let entry = PyDict::new(py);
            entry.set_item("positions", traj)?;
            entry.set_item("full_states", full_traj)?;
            entry.set_item("serve_speed", serve.vel.norm())?;
            entry.set_item("serve_vx", serve.vel.x)?;
            entry.set_item("serve_vy", serve.vel.y)?;
            entry.set_item("serve_vz", serve.vel.z)?;
            entry.set_item("topspin", -serve.omega.x)?; // positive = topspin
            entry.set_item("backspin", serve.omega.x.max(0.0))?;
            entry.set_item("sidespin", -serve.omega.z)?;
            entry.set_item("trajectory_type", "serve")?;
            result.append(entry)?;
            generated += 1;
        }

        Ok(result)
    }

    /// Generate return-shot (rally) trajectories with random paddle actions.
    /// These are faster balls traveling back toward the server's side.
    fn generate_rally_trajectories<'py>(
        &mut self,
        py: Python<'py>,
        count: usize,
        difficulty: u8,
    ) -> PyResult<Bound<'py, PyList>> {
        use crate::simulation::simulate_full;
        use crate::physics::paddle::apply_paddle_hit;

        let diff = match difficulty {
            1 => Difficulty::Stage1,
            2 => Difficulty::Stage2,
            _ => Difficulty::Stage3,
        };

        let result = PyList::empty(py);
        let mut generated = 0;
        let table = &self.table;

        while generated < count {
            // 1. Generate a valid serve
            let serve = random_serve(&mut self.rng, diff);
            let obs_result = prepare_serve(serve, table);
            if obs_result.bad_serve || obs_result.flight_trajectory.len() < 5 {
                continue;
            }

            // 2. Smart paddle: position WHERE the ball is, random swing
            let paddle_y = self.rng.uniform_range(2.2, 3.0);
            let ball_at_y = match interpolate_at_y_pub(&obs_result.flight_trajectory, paddle_y) {
                Some(s) => s,
                None => continue,
            };

            let paddle_x = ball_at_y.pos.x + self.rng.uniform_range(-0.04, 0.04);
            let paddle_z = (ball_at_y.pos.z + self.rng.uniform_range(-0.02, 0.02))
                .max(table.surface_z() + 0.09);

            let tilt_x = self.rng.uniform_range(-0.3, 0.4);
            let tilt_z = self.rng.uniform_range(-0.15, 0.15);
            let swing_speed = self.rng.uniform_range(5.0, 35.0);
            let swing_elevation = self.rng.uniform_range(0.05, 0.45);

            let action = PaddleAction {
                paddle_x,
                paddle_y,
                paddle_z,
                tilt_x,
                tilt_z,
                swing_speed,
                swing_elevation,
            };

            // 3. Apply paddle hit directly (no landing validation — we want diverse trajectories)
            let paddle_result = apply_paddle_hit(&ball_at_y, &action, table.surface_z(), 0.15);
            let hit_state = match paddle_result {
                crate::physics::paddle::PaddleResult::Hit(s) => s,
                _ => continue,
            };

            // Skip degenerate returns (too slow or going wrong direction)
            if hit_state.vel.norm() < 3.0 {
                continue;
            }

            // 4. Simulate the return flight and sample at 60Hz
            let mut obs_frames: Vec<[f64; 3]> = Vec::new();
            let mut full_states: Vec<BallState> = Vec::new();
            let dt = 0.0005;
            let obs_dt = 1.0 / 60.0;
            let mut state = hit_state;
            let mut t = 0.0;
            let mut next_obs = 0.0;
            let min_frames = 15; // rallies are shorter, need at least 15 frames

            loop {
                if t >= next_obs && obs_frames.len() < OBS_FRAMES {
                    obs_frames.push([state.pos.x, state.pos.y, state.pos.z]);
                    full_states.push(state);
                    next_obs += obs_dt;
                }
                if state.pos.z < 0.0 || state.pos.y < -1.0 || state.pos.y > table.length + 2.0 {
                    break;
                }
                if obs_frames.len() >= OBS_FRAMES {
                    break;
                }
                state = crate::physics::integrator::rk4_step(&state, dt);
                t += dt;
            }

            let n_actual = obs_frames.len();
            if n_actual < min_frames {
                continue;
            }

            // Pad to OBS_FRAMES if needed (repeat last position)
            while obs_frames.len() < OBS_FRAMES {
                obs_frames.push(*obs_frames.last().unwrap());
                full_states.push(*full_states.last().unwrap());
            }

            // 6. Build output
            let traj = PyList::empty(py);
            for frame in &obs_frames {
                let point = PyList::new(py, &[frame[0], frame[1], frame[2]])?;
                traj.append(point)?;
            }

            let full_traj = PyList::empty(py);
            for state in &full_states {
                let frame = PyList::new(py, &[
                    state.pos.x, state.pos.y, state.pos.z,
                    state.vel.x, state.vel.y, state.vel.z,
                    state.omega.x, state.omega.y, state.omega.z,
                ])?;
                full_traj.append(frame)?;
            }

            let entry = PyDict::new(py);
            entry.set_item("positions", traj)?;
            entry.set_item("full_states", full_traj)?;
            entry.set_item("serve_speed", hit_state.vel.norm())?;
            entry.set_item("serve_vx", hit_state.vel.x)?;
            entry.set_item("serve_vy", hit_state.vel.y)?;
            entry.set_item("serve_vz", hit_state.vel.z)?;
            entry.set_item("topspin", -hit_state.omega.x)?;
            entry.set_item("backspin", hit_state.omega.x.max(0.0))?;
            entry.set_item("sidespin", -hit_state.omega.z)?;
            entry.set_item("trajectory_type", "rally")?;
            entry.set_item("n_actual_frames", n_actual)?;
            result.append(entry)?;
            generated += 1;
        }

        Ok(result)
    }
}

/// Python module definition
#[pymodule]
pub fn spinoza(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SimEnv>()?;
    Ok(())
}
