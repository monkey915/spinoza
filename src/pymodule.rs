use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::physics::paddle::PaddleAction;
use crate::physics::state::{BallState, Vec3};
use crate::rally::{
    prepare_serve, evaluate_return, run_rally_replay,
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

        // Use the cached serve from the last reset()
        let replay = run_rally_replay(self.pending_serve, &paddle_action, &self.table);

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
}

/// Python module definition
#[pymodule]
pub fn spinoza(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SimEnv>()?;
    Ok(())
}
