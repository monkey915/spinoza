"""Gymnasium environment wrapper for spinoza table tennis simulation."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from spinoza import SimEnv

# Must match OBS_FRAMES in src/rally.rs
OBS_FRAMES = 30
OBS_SIZE = OBS_FRAMES * 3  # x,y,z per frame


class TableTennisEnv(gym.Env):
    """Single-step episodic environment for table tennis return.

    Observation: 90 floats (30 frames × 3 coords) — ball positions at ~60Hz,
                 right-aligned with zero-padding for short flights.
                 30 frames ≈ 500 ms covers the complete post-bounce trajectory.
    Action: 7 floats [paddle_x, paddle_y, paddle_z, tilt_x, tilt_z, swing_speed, swing_elevation]
    Reward: shaped (see rally.rs for details)
    """

    metadata = {"render_modes": []}

    def __init__(self, seed=42, difficulty=1):
        super().__init__()
        self.sim = SimEnv(seed=seed, difficulty=difficulty)

        # Observation: 30 frames of (x, y, z) ball positions
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        # Action: [paddle_x, paddle_y, paddle_z, tilt_x, tilt_z, swing_speed, swing_elevation]
        # tilt_x < 0 = offene Fläche (nach hinten gekippt, gegen Backspin)
        # tilt_x > 0 = geschlossene Fläche (nach vorne gekippt, für Topspin)
        # swing_elevation > 0 = Aufwärts-Swing
        self.action_space = spaces.Box(
            low=np.array( [0.0,  1.8,  0.85, -0.8, -0.8,  1.0, -0.3], dtype=np.float32),
            high=np.array([1.525, 3.5,  1.40,  0.8,  0.8, 12.0,  0.7], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.sim.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        action_list = [float(a) for a in action]
        next_obs, reward, done, info = self.sim.step(action_list)
        next_obs_np = np.array(next_obs, dtype=np.float32)
        return next_obs_np, reward, done, False, info

    def set_difficulty(self, difficulty):
        self.sim.set_difficulty(difficulty)


def make_env(seed=0, difficulty=1):
    """Factory function for creating vectorized environments."""
    def _init():
        return TableTennisEnv(seed=seed, difficulty=difficulty)
    return _init
