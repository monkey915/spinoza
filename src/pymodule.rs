use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::physics::paddle::PaddleAction;
use crate::physics::state::{BallState, Vec3};
use crate::rally::{run_rally, RallyOutcome, OBS_FRAMES};
use crate::serve::{random_serve, Difficulty, Rng};
use crate::table::Table;

/// The RL environment exposed to Python.
///
/// Each call to `step(action)` runs one full rally episode:
///   serve → flight → observe → paddle hit → return → result
///
/// Returns (observation, reward, done, info) matching Gymnasium API.
#[pyclass]
pub struct SimEnv {
    table: Table,
    rng: Rng,
    difficulty: u8, // 1, 2, or 3
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
        6
    }

    /// Reset the environment. Returns initial observation (zeros).
    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyList> {
        let obs = vec![0.0_f64; OBS_FRAMES * 3];
        PyList::new(py, &obs).unwrap()
    }

    /// Run one rally episode with the given action.
    ///
    /// Args:
    ///   action: list of 6 floats [paddle_x, paddle_z, tilt_x, tilt_z, swing_speed, swing_elevation]
    ///
    /// Returns:
    ///   (observation, reward, done, info_dict)
    ///   - observation: list of 30 floats (10 frames × 3 coords)
    ///   - reward: float
    ///   - done: always True (single-step episodes)
    ///   - info: dict with outcome details
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: Vec<f64>,
    ) -> PyResult<(Bound<'py, PyList>, f64, bool, PyObject)> {
        if action.len() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Action must have 6 elements: [paddle_x, paddle_z, tilt_x, tilt_z, swing_speed, swing_elevation]"
            ));
        }

        let difficulty = match self.difficulty {
            1 => Difficulty::Stage1,
            2 => Difficulty::Stage2,
            _ => Difficulty::Stage3,
        };

        let serve = random_serve(&mut self.rng, difficulty);

        let paddle_action = PaddleAction {
            paddle_x: action[0],
            paddle_z: action[1],
            tilt_x: action[2],
            tilt_z: action[3],
            swing_speed: action[4],
            swing_elevation: action[5],
        };

        let result = run_rally(serve, &paddle_action, &self.table);

        // Flatten observations: pad to OBS_FRAMES if needed
        let mut obs_flat = vec![0.0_f64; OBS_FRAMES * 3];
        let n = result.observations.len().min(OBS_FRAMES);
        let offset = OBS_FRAMES - n;
        for (i, frame) in result.observations.iter().rev().take(n).rev().enumerate() {
            obs_flat[(offset + i) * 3] = frame[0];
            obs_flat[(offset + i) * 3 + 1] = frame[1];
            obs_flat[(offset + i) * 3 + 2] = frame[2];
        }

        let obs_list = PyList::new(py, &obs_flat).unwrap();

        // Build info dict
        let info = pyo3::types::PyDict::new(py);
        let outcome_str = match &result.outcome {
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

        Ok((obs_list, result.reward, true, info.into_any().unbind()))
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
