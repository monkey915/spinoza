#!/usr/bin/env python3
"""Real-time bridge: camera -> trajectory prediction -> IK -> robot arm.

This is the main pipeline that connects the stereo camera ball tracker to
the neural network trajectory predictor and drives the robot arm to the
predicted paddle position.

Usage:
  python bridge.py              # full pipeline (camera + arm)
  python bridge.py --no-arm     # camera + prediction only (no servos)
  python bridge.py --test-arm   # test arm movement without camera
"""

import argparse
import math
import sys
import time
from collections import deque

import numpy as np

from camera.detect import BallTracker
from robot.arm import RobotArm, solve_ik, ik_angles_to_degrees


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Trajectory buffer: 30 frames × 3 coords (matches RL observation space)
TRAJ_BUFFER_SIZE = 30

# Table geometry (from simulation)
TABLE_SURFACE_Z = 0.76   # metres
BALL_RADIUS = 0.02

# Bounce detection
BOUNCE_Z_THRESHOLD = TABLE_SURFACE_Z + BALL_RADIUS + 0.03  # ~0.81 m
MIN_BOUNCE_INTERVAL = 0.15  # seconds between bounces

# Prediction trigger: start predicting after N trajectory points
MIN_POINTS_FOR_PREDICTION = 10

# Loop timing
LOOP_HZ = 30
LOOP_DT = 1.0 / LOOP_HZ


# ---------------------------------------------------------------------------
# Trajectory buffer
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """Collects ball positions and detects bounce events."""

    def __init__(self, max_size=TRAJ_BUFFER_SIZE):
        self.positions = deque(maxlen=max_size)  # (x, y, z) in sim coords
        self.timestamps = deque(maxlen=max_size)
        self.velocities = deque(maxlen=max_size)
        self.bounce_count = 0
        self._last_bounce_time = 0.0
        self._prev_vz = None
        self._active = False  # becomes True after first detection

    def reset(self):
        """Clear the buffer for a new rally."""
        self.positions.clear()
        self.timestamps.clear()
        self.velocities.clear()
        self.bounce_count = 0
        self._last_bounce_time = 0.0
        self._prev_vz = None
        self._active = False

    def add(self, pos, vel, t):
        """Add a position/velocity sample and check for bounces.

        pos: (x, y, z) in sim coords (metres)
        vel: (vx, vy, vz) in m/s
        """
        self.positions.append(pos)
        self.velocities.append(vel)
        self.timestamps.append(t)
        self._active = True

        # Bounce detection: Z velocity sign change near table surface
        vz = vel[2]
        z = pos[2]
        if (self._prev_vz is not None
                and self._prev_vz < 0 and vz > 0
                and z < BOUNCE_Z_THRESHOLD
                and t - self._last_bounce_time > MIN_BOUNCE_INTERVAL):
            self.bounce_count += 1
            self._last_bounce_time = t
            print(f"  Bounce #{self.bounce_count} detected at z={z:.3f}")
        self._prev_vz = vz

    @property
    def ready_for_prediction(self) -> bool:
        """True if we have enough data and 2 bounces (serve completed)."""
        return (len(self.positions) >= MIN_POINTS_FOR_PREDICTION
                and self.bounce_count >= 2)

    def as_observation(self) -> np.ndarray:
        """Convert to the 90-float observation vector (30 frames × 3 coords).

        Right-aligned with zero-padding if fewer than 30 frames.
        """
        obs = np.zeros(TRAJ_BUFFER_SIZE * 3, dtype=np.float32)
        n = len(self.positions)
        for i in range(min(n, TRAJ_BUFFER_SIZE)):
            idx = n - TRAJ_BUFFER_SIZE + i
            if idx >= 0:
                pos = self.positions[idx]
                obs[i * 3] = pos[0]
                obs[i * 3 + 1] = pos[1]
                obs[i * 3 + 2] = pos[2]
        return obs


# ---------------------------------------------------------------------------
# Paddle prediction (placeholder — uses analytical fallback)
# ---------------------------------------------------------------------------

def predict_paddle_position(trajectory: TrajectoryBuffer) -> dict | None:
    """Predict where to place the paddle based on the trajectory.

    This is a simple analytical prediction. For the full NN-based prediction,
    integrate with training/predict.py and training/paddle.py.

    Returns dict with paddle_x, paddle_y, paddle_z, or None.
    """
    if len(trajectory.positions) < 3:
        return None

    # Use last 3 positions to estimate velocity
    p1 = np.array(trajectory.positions[-3])
    p2 = np.array(trajectory.positions[-1])
    t1 = trajectory.timestamps[-3]
    t2 = trajectory.timestamps[-1]
    dt = t2 - t1
    if dt < 0.01:
        return None

    vel = (p2 - p1) / dt

    # Predict where ball crosses paddle_y line (Y ≈ 2.4–2.7m, receiver side)
    target_y = 2.5
    if abs(vel[1]) < 0.1:
        return None

    t_arrive = (target_y - p2[1]) / vel[1]
    if t_arrive < 0 or t_arrive > 2.0:
        return None

    # Simple ballistic prediction (gravity on Z axis)
    pred_x = p2[0] + vel[0] * t_arrive
    pred_z = p2[2] + vel[2] * t_arrive - 0.5 * 9.81 * t_arrive * t_arrive

    # Clamp to table area
    pred_x = max(0.0, min(1.525, pred_x))
    pred_z = max(TABLE_SURFACE_Z + 0.09, min(1.4, pred_z))

    return {
        "paddle_x": pred_x,
        "paddle_y": target_y,
        "paddle_z": pred_z,
        "t_arrive": t_arrive,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(use_arm: bool = True):
    """Run the real-time camera → prediction → arm pipeline."""
    print("=== Spinoza Bridge: Camera -> Prediction -> Arm ===\n")

    # Start camera tracker
    tracker = BallTracker()
    tracker.start()
    print("Camera tracker started")

    # Optionally connect arm
    arm = None
    if use_arm:
        try:
            arm = RobotArm()
            arm.connect()
            arm.enable_torque()
            arm.home()
            time.sleep(1.0)
            print("Robot arm connected and homed")
        except Exception as e:
            print(f"Arm connection failed: {e}")
            print("Continuing without arm (prediction only)")
            arm = None

    traj = TrajectoryBuffer()
    t_start = time.time()
    last_prediction = None
    state = "WAITING"  # WAITING -> TRACKING -> PREDICTED -> RETURNING

    print(f"\nRunning at {LOOP_HZ} Hz. Press Ctrl+C to stop.\n")

    try:
        while True:
            loop_start = time.time()

            pos, vel = tracker.get_position_3d()
            t_now = time.time() - t_start

            if pos is not None:
                if state == "WAITING":
                    state = "TRACKING"
                    traj.reset()
                    print(f"[{t_now:.2f}s] Ball detected, tracking...")

                traj.add(pos, vel, t_now)

                if state == "TRACKING" and traj.ready_for_prediction:
                    pred = predict_paddle_position(traj)
                    if pred is not None:
                        state = "PREDICTED"
                        last_prediction = pred
                        print(f"[{t_now:.2f}s] Prediction: "
                              f"x={pred['paddle_x']:.3f} "
                              f"y={pred['paddle_y']:.3f} "
                              f"z={pred['paddle_z']:.3f} "
                              f"(arrives in {pred['t_arrive']:.3f}s)")

                        if arm is not None:
                            ok = arm.move_to_position(
                                pred["paddle_x"],
                                pred["paddle_y"],
                                pred["paddle_z"],
                            )
                            if ok:
                                print(f"[{t_now:.2f}s] Arm moving to target")
                            else:
                                print(f"[{t_now:.2f}s] Target unreachable for arm")

            else:
                # No ball detected
                if state in ("TRACKING", "PREDICTED"):
                    # Ball lost — wait a bit then reset
                    if not traj._active:
                        continue
                    if (traj.timestamps
                            and t_now - traj.timestamps[-1] > 1.0):
                        state = "WAITING"
                        print(f"[{t_now:.2f}s] Ball lost, returning to wait")
                        if arm is not None:
                            arm.home(speed=400)

            # Status printout every 5 seconds
            if int(t_now) % 5 == 0 and int(t_now * LOOP_HZ) % LOOP_HZ == 0:
                n = len(traj.positions)
                print(f"[{t_now:.1f}s] state={state} "
                      f"buf={n}/{TRAJ_BUFFER_SIZE} "
                      f"bounces={traj.bounce_count}")

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = LOOP_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        tracker.stop()
        if arm is not None:
            arm.disable_torque()
            arm.disconnect()
    print("Done.")


def test_arm():
    """Test arm movement without camera."""
    print("=== Arm Test Mode ===\n")

    with RobotArm() as arm:
        print("Homing...")
        arm.home()
        time.sleep(2.0)

        # Sweep through some test positions
        test_positions = [
            (0.76, 2.5, 1.0),   # center, mid-height
            (0.3, 2.5, 0.9),    # left
            (1.2, 2.5, 0.9),    # right
            (0.76, 2.3, 1.1),   # center, higher
            (0.76, 2.5, 0.85),  # center, table level
        ]

        for x, y, z in test_positions:
            print(f"\nTarget: ({x:.2f}, {y:.2f}, {z:.2f})")
            ok = arm.move_to_position(x, y, z, speed=400)
            if ok:
                time.sleep(1.5)
                angles = arm.read_angles()
                print(f"  Angles: {angles}")
            else:
                print("  Unreachable!")

        print("\nReturning home...")
        arm.home()
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Spinoza real-time bridge")
    parser.add_argument("--no-arm", action="store_true",
                        help="Run without robot arm (prediction only)")
    parser.add_argument("--test-arm", action="store_true",
                        help="Test arm movement without camera")
    args = parser.parse_args()

    if args.test_arm:
        test_arm()
    else:
        run_pipeline(use_arm=not args.no_arm)


if __name__ == "__main__":
    main()
