#!/usr/bin/env python3
"""Real-time bridge: camera -> trajectory prediction -> robot arm.

Pipeline per cycle:
  1. Grab the latest Kalman-filtered ball state from the stereo tracker.
  2. Detect table bounces in the trajectory buffer.
  3. Forward-simulate the flight with the Rust physics core (drag, Magnus,
     bounce) to get future ball positions.
  4. Pick an intercept point the arm can actually reach in time, then stream
     position commands to the arm every cycle (continuous re-planning).
     Planning freezes shortly before predicted contact so the servos settle.

Usage:
  python bridge.py              # full pipeline (camera + arm)
  python bridge.py --no-arm     # camera + prediction only (no servos)
  python bridge.py --test-arm   # test arm movement without camera
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import deque

import numpy as np

from camera.detect import BallTracker
from robot import config as robot_config
from robot.arm import RobotArm

ARM_BASE_Y = robot_config.ARM_BASE_Y


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Trajectory buffer: 30 frames × 3 coords (matches RL observation space)
TRAJ_BUFFER_SIZE = 30

# Table geometry (from simulation)
TABLE_SURFACE_Z = 0.76   # metres
BALL_RADIUS = 0.02

# Bounce detection
# Threshold applies to the PRE-bounce sample height. The filtered vz sign
# change arrives 1-2 frames after physical contact, when the ball has
# already risen a few cm — checking the current height would miss it.
BOUNCE_Z_THRESHOLD = TABLE_SURFACE_Z + BALL_RADIUS + 0.07  # ~0.85 m
MIN_BOUNCE_INTERVAL = 0.15  # seconds between bounces

# Loop timing
LOOP_HZ = 60
LOOP_DT = 1.0 / LOOP_HZ

# Planning
PREDICT_AFTER_BOUNCES = 1    # start predicting after first table bounce
FREEZE_BEFORE_CONTACT = 0.06  # stop re-planning this long before contact (s)
SAFETY_MARGIN = 0.04          # extra time margin for arm travel checks (s)
CONTACT_Z_MIN = TABLE_SURFACE_Z + 0.09  # paddle never below this
COMFORT_Z = 0.95              # preferred intercept height (m)

# Latency stats window (cycles)
LATENCY_WINDOW = 300


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
        prev_pos = self.positions[-1] if self.positions else None
        self.positions.append(pos)
        self.velocities.append(vel)
        self.timestamps.append(t)
        self._active = True

        # Bounce detection: filtered Z velocity sign change. The height
        # check uses the pre-bounce sample — the filter reports the flip
        # only after the ball has left the table.
        vz = vel[2]
        if (prev_pos is not None
                and self._prev_vz is not None
                and self._prev_vz < 0 and vz > 0
                and prev_pos[2] < BOUNCE_Z_THRESHOLD
                and t - self._last_bounce_time > MIN_BOUNCE_INTERVAL):
            self.bounce_count += 1
            self._last_bounce_time = t
        self._prev_vz = vz

    @property
    def ready_for_prediction(self) -> bool:
        """True once enough data exists to seed the forward simulation."""
        return (len(self.positions) >= 3
                and self.bounce_count >= PREDICT_AFTER_BOUNCES)

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
# Forward prediction — Rust physics core with ballistic fallback
# ---------------------------------------------------------------------------

class ForwardPredictor:
    """Predicts future ball positions from the current measured state.

    Uses spinoza.simulate_forward (RK4 with drag, Magnus effect and the
    full bounce model) when the compiled bindings are importable; falls
    back to gravity-only ballistics otherwise.

    Spin is assumed zero — real serve spin estimation is future work; the
    predictor re-runs every frame, so errors shrink as the ball approaches.
    """

    def __init__(self):
        try:
            import spinoza
            self._sim = spinoza.simulate_forward
            self.backend = "rust"
        except ImportError:
            self._sim = None
            self.backend = "ballistic"

    def predict(self, pos, vel, t_now, horizon=1.5, sample_dt=0.005):
        """Return [(t_abs, x, y, z), ...] of future ball positions.

        t_abs are absolute pipeline timestamps. Includes the current
        position as the first sample.
        """
        if self._sim is not None:
            result = self._sim(
                [pos[0], pos[1], pos[2],
                 vel[0], vel[1], vel[2],
                 0.0, 0.0, 0.0],
                max_bounces=3,
                sample_dt=sample_dt,
            )
            return [
                (t_now + s[0], s[1], s[2], s[3])
                for s in result["trajectory"]
                if s[0] <= horizon
            ]

        # Ballistic fallback: gravity-only Euler integration
        samples = []
        p = np.array(pos[:3], dtype=np.float64)
        v = np.array(vel[:3], dtype=np.float64)
        dt = sample_dt
        g = np.array([0.0, 0.0, -9.81])
        for i in range(int(horizon / dt)):
            samples.append((t_now + i * dt, float(p[0]), float(p[1]),
                            float(p[2])))
            v = v + g * dt
            p = p + v * dt
            if p[2] < CONTACT_Z_MIN or p[1] > 5.0:
                break
        return samples


def _intercept_score(z, vx, vy, vz):
    """Higher is better: comfortable height, slow vertical motion."""
    height_score = -abs(z - COMFORT_Z) * 4.0
    apex_score = -abs(vz) * 0.15
    return height_score + apex_score


def select_intercept(samples, t_now, arm=None, min_lead=SAFETY_MARGIN):
    """Pick the best reachable intercept point from predicted samples.

    A candidate is feasible when IK can reach it AND the estimated worst-case
    joint travel time fits into the remaining lead time. Candidates behind
    the arm base are rejected — a ball past the baseline is unreturnable.
    Among feasible candidates the most comfortable one (near COMFORT_Z,
    low |vz|) wins.

    Returns dict(t_contact, x, y, z, travel_time) or None.
    """
    best = None
    best_score = -1e9
    for t_contact, x, y, z in samples:
        if y >= ARM_BASE_Y - 0.02:
            continue
        lead = t_contact - t_now
        if lead < min_lead:
            continue
        travel = 0.0
        if arm is not None:
            travel = arm.travel_time_to(x, y, z)
            if travel is None:
                continue
            if travel + min_lead > lead:
                continue
        score = _intercept_score(z, 0.0, 0.0, 0.0) - travel * 0.5
        if score > best_score:
            best_score = score
            best = {
                "t_contact": t_contact,
                "x": x, "y": y, "z": max(z, CONTACT_Z_MIN),
                "travel_time": travel,
            }
    return best


# ---------------------------------------------------------------------------
# Latency instrumentation
# ---------------------------------------------------------------------------

class LatencyStats:
    """Rolling per-stage latency statistics for the pipeline."""

    STAGES = ("sensor_age_ms", "predict_ms", "plan_ms", "servo_ms", "cycle_ms")

    def __init__(self, window=LATENCY_WINDOW):
        self.window = window
        self.samples = {s: deque(maxlen=window) for s in self.STAGES}

    def record(self, **values):
        for k, v in values.items():
            if k in self.samples and v is not None:
                self.samples[k].append(v)

    def summary(self):
        parts = []
        for stage in self.STAGES:
            data = self.samples[stage]
            if not data:
                continue
            parts.append(
                f"{stage}: median={statistics.median(data):.1f} "
                f"p95={sorted(data)[int(len(data)*0.95)]:.1f}"
            )
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(use_arm: bool = True, latency_log: str | None = None):
    """Run the real-time camera → prediction → arm pipeline."""
    print("=== Spinoza Bridge: Camera -> Prediction -> Arm ===\n")

    predictor = ForwardPredictor()
    print(f"Prediction backend: {predictor.backend}")

    tracker = BallTracker()
    tracker.start()
    print("Camera tracker started")

    arm = None
    if use_arm:
        try:
            arm = RobotArm()
            arm.connect()
            arm.enable_torque()
            arm.home()
            time.sleep(1.0)
            arm.ready()
            time.sleep(0.5)
            print("Robot arm connected, homed, in ready stance")
        except Exception as e:
            print(f"Arm connection failed: {e}")
            print("Continuing without arm (prediction only)")
            arm = None

    traj = TrajectoryBuffer()
    latency = LatencyStats()
    log_file = open(latency_log, "w") if latency_log else None

    t_start = time.time()
    last_frame_ts = None
    frozen_plan = None
    next_status = 5.0
    state = "WAITING"  # WAITING -> TRACKING -> RETURNING -> (frozen) -> ...

    print(f"\nRunning at {LOOP_HZ} Hz. Press Ctrl+C to stop.\n")

    def finish_rally(reason):
        nonlocal state, frozen_plan
        print(f"[{time.time() - t_start:.2f}s] {reason}")
        state = "WAITING"
        frozen_plan = None
        traj.reset()
        if arm is not None:
            arm.ready(speed=400)

    try:
        while True:
            loop_start = time.perf_counter()

            pos, vel = tracker.get_position_3d()
            frame_ts, sensor_age_ms, sensor_proc_ms = tracker.get_timing()

            new_frame = frame_ts != last_frame_ts
            last_frame_ts = frame_ts

            if pos is not None:
                # Exposure time expressed in the bridge clock domain: local
                # receive time minus the sensor-to-here latency.
                t_frame = (time.time() - t_start) - sensor_age_ms / 1000.0
                traj.add(pos, vel, t_frame)

                if state == "WAITING":
                    state = "TRACKING"
                    frozen_plan = None
                    print(f"[{frame_ts:.2f}s] Ball detected, tracking...")

                # --- Continuous re-planning on every new frame ----------
                if new_frame and traj.ready_for_prediction:
                    predict_start = time.perf_counter()
                    samples = predictor.predict(
                        pos, vel, t_frame, horizon=1.5,
                    )
                    plan_start = time.perf_counter()
                    servo_ms = None
                    plan = select_intercept(samples, t_frame, arm=arm)

                    if plan is None and state == "TRACKING" \
                            and traj.bounce_count >= 2:
                        finish_rally("No feasible intercept — resetting")
                        continue

                    commanded = False
                    if plan is not None:
                        lead = plan["t_contact"] - t_frame
                        if lead > FREEZE_BEFORE_CONTACT:
                            # Still time: update the arm target smoothly.
                            frozen_plan = plan
                            if state != "RETURNING":
                                state = "RETURNING"
                            if arm is not None:
                                servo_start = time.perf_counter()
                                arm.move_to_position(
                                    plan["x"], plan["y"], plan["z"],
                                    speed=800,
                                )
                                servo_ms = (time.perf_counter()
                                            - servo_start) * 1000.0
                                commanded = True
                        elif state == "RETURNING":
                            # Frozen window: let the servos settle.
                            pass

                        if plan["t_contact"] + 0.25 < t_frame:
                            finish_rally("Contact window passed, resetting")
                            continue

                    predict_ms = (plan_start - predict_start) * 1000.0
                    plan_ms = (time.perf_counter() - plan_start
                               - (servo_ms or 0.0) / 1000.0) * 1000.0
                    latency.record(predict_ms=predict_ms, plan_ms=plan_ms,
                                   servo_ms=servo_ms)

            else:
                # No ball visible
                if state in ("TRACKING", "RETURNING"):
                    lost = traj.timestamps and \
                        (time.time() - t_start) - traj.timestamps[-1] > 1.0
                    if lost:
                        finish_rally("Ball lost, returning home")

            cycle_ms = (time.perf_counter() - loop_start) * 1000.0
            latency.record(sensor_age_ms=sensor_age_ms, cycle_ms=cycle_ms)

            if log_file is not None and new_frame:
                log_file.write(json.dumps({
                    "t": round(time.time() - t_start, 4),
                    "sensor_age_ms": round(sensor_age_ms, 2),
                    "cycle_ms": round(cycle_ms, 2),
                    "state": state,
                }) + "\n")

            if (time.time() - t_start) >= next_status:
                print(f"[{time.time() - t_start:.1f}s] state={state} "
                      f"buf={len(traj.positions)}/{TRAJ_BUFFER_SIZE} "
                      f"bounces={traj.bounce_count} "
                      f"backend={predictor.backend}")
                summary = latency.summary()
                if summary:
                    print(f"          latency: {summary}")
                next_status += 5.0

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = LOOP_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        if log_file is not None:
            log_file.close()
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
    parser.add_argument("--latency-log", type=str, default=None,
                        help="Write per-cycle latency records to this JSONL file")
    args = parser.parse_args()

    if args.test_arm:
        test_arm()
    else:
        run_pipeline(use_arm=not args.no_arm, latency_log=args.latency_log)


if __name__ == "__main__":
    main()
