"""4-DOF robot arm controller with inverse kinematics.

Provides the ``RobotArm`` class that translates between simulation-space
paddle targets and physical servo positions via IK (ported from the
Three.js web viewer ``solveIK()``).
"""

from __future__ import annotations

import math
import time
from typing import Optional

from . import config
from .servos import FeetechBus


# ---------------------------------------------------------------------------
# Inverse kinematics (ported from web/js/main.js solveIK)
# ---------------------------------------------------------------------------

def solve_ik(
    target_x: float,
    target_y: float,
    target_z: float,
    arm_base: tuple[float, float, float] = (
        config.ARM_BASE_X, config.ARM_BASE_Y, config.ARM_BASE_Z
    ),
    l1: float = config.L_UPPER_ARM,
    l2: float = config.L_FOREARM,
    l3: float = config.L_PADDLE,
    elbow_up: bool = False,
) -> Optional[tuple[float, float, float, float]]:
    """Solve 4-DOF IK for a target position in simulation coordinates.

    Returns (phi1, phi2, phi3, phi4) in **radians** or None if unreachable.

    Joint angles follow the simulation convention:
      phi1 — shoulder yaw (rotation around vertical Z axis)
      phi2 — shoulder pitch (0 = horizontal, pi/2 = straight up)
      phi3 — elbow bend (0 = straight, positive = bend)
      phi4 — wrist tilt (set so paddle faces the incoming ball direction)
    """
    dx = target_x - arm_base[0]
    dy = target_y - arm_base[1]
    dz = target_z - arm_base[2]

    # phi1: shoulder yaw — angle in the XY plane
    phi1 = math.atan2(-dx, -dy)

    # Horizontal distance from base to target
    r_horiz = math.sqrt(dx * dx + dy * dy)
    # Effective reach target for the 2-link arm (subtract paddle length)
    r_eff = r_horiz - l3
    dist_2d = math.sqrt(r_eff * r_eff + dz * dz)

    # Check reachability
    if dist_2d > l1 + l2 or dist_2d < abs(l1 - l2) or r_eff < 0:
        return None

    # 2-link IK in the vertical plane
    cos_q2 = (dist_2d * dist_2d - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))

    if elbow_up:
        q2 = -math.acos(cos_q2)
    else:
        q2 = math.acos(cos_q2)

    alpha = math.atan2(dz, r_eff)
    beta = math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    q1 = alpha + beta

    phi2 = q1    # shoulder pitch
    phi3 = q2    # elbow bend
    # phi4: keep paddle roughly horizontal (compensate for shoulder + elbow)
    phi4 = -(phi2 + phi3)

    return (phi1, phi2, phi3, phi4)


def ik_angles_to_degrees(
    phi1: float, phi2: float, phi3: float, phi4: float,
) -> tuple[float, float, float, float]:
    """Convert IK angles (radians) to degrees."""
    return (
        math.degrees(phi1),
        math.degrees(phi2),
        math.degrees(phi3),
        math.degrees(phi4),
    )


# ---------------------------------------------------------------------------
# Angle <-> raw servo position conversion (per joint)
# ---------------------------------------------------------------------------

def _angle_to_raw(joint_name: str, angle_deg: float) -> int:
    """Convert a simulation angle (degrees) to a raw servo position.

    Accounts for the mechanical zero offset of each joint.
    """
    zero = config.ZERO_OFFSETS_RAW[joint_name]
    delta = config.deg_to_raw(angle_deg)  # fractional turn worth of ticks
    # Signed offset from zero
    raw = zero + delta - config.deg_to_raw(0)  # deg_to_raw(0) == 0
    return raw % (config.POSITION_MAX + 1)


def _raw_to_angle(joint_name: str, raw: int) -> float:
    """Convert a raw servo position to a simulation angle (degrees)."""
    zero = config.ZERO_OFFSETS_RAW[joint_name]
    delta_raw = raw - zero
    return delta_raw / config.POSITION_MAX * config.ANGLE_RANGE_DEG


# ---------------------------------------------------------------------------
# RobotArm controller
# ---------------------------------------------------------------------------

class RobotArm:
    """High-level controller for the 4-DOF table tennis robot arm.

    Wraps ``FeetechBus`` and provides IK-based positioning.
    """

    JOINT_NAMES = ("shoulder_yaw", "shoulder_pitch", "elbow", "wrist")

    def __init__(self, bus: FeetechBus | None = None):
        self.bus = bus or FeetechBus()
        self._connected = False
        # Last commanded joint angles (degrees) — used as the arm-state
        # estimate for travel-time feasibility without bus reads.
        self._commanded_angles: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Open the servo bus and verify all servos respond."""
        self.bus.open()
        status = self.bus.ping_all()
        missing = [name for name, ok in status.items() if not ok]
        if missing:
            self.bus.close()
            raise RuntimeError(f"Servos not responding: {missing}")
        # Apply torque limits
        for name, sid in config.SERVO_IDS.items():
            self.bus.set_torque_limit(sid)
        self._connected = True
        print(f"Robot arm connected: all {len(status)} servos OK")

    def disconnect(self):
        """Disable torque and close the bus."""
        if self._connected:
            self.bus.torque_all(False)
            self._connected = False
        self.bus.close()

    def enable_torque(self):
        """Enable torque on all joints."""
        self.bus.torque_all(True)

    def disable_torque(self):
        """Disable torque on all joints (arm goes limp)."""
        self.bus.torque_all(False)

    def emergency_stop(self):
        """Immediately disable all torque."""
        self.bus.emergency_stop()

    # ------------------------------------------------------------------
    # Joint-level control
    # ------------------------------------------------------------------

    def move_to_angles(
        self,
        phi1_deg: float,
        phi2_deg: float,
        phi3_deg: float,
        phi4_deg: float,
        speed: int | None = None,
    ):
        """Drive all joints to the given angles (degrees) simultaneously.

        Angles are clamped to the configured joint limits. Speeds and
        positions go out as SyncWrite bursts so all joints start moving
        in the same control cycle.
        """
        speed = speed or config.MOVE_SPEED
        targets = {
            "shoulder_yaw":   phi1_deg,
            "shoulder_pitch": phi2_deg,
            "elbow":          phi3_deg,
            "wrist":          phi4_deg,
        }
        raw_positions = {}
        for name in self.JOINT_NAMES:
            lo, hi = config.JOINT_LIMITS_DEG[name]
            angle = max(lo, min(hi, targets[name]))
            raw_positions[config.SERVO_IDS[name]] = _angle_to_raw(name, angle)
            targets[name] = angle

        self.bus.write_speeds_sync(
            {sid: speed for sid in config.SERVO_IDS.values()}
        )
        self.bus.write_positions_sync(raw_positions)
        self._commanded_angles = dict(targets)

    def get_commanded_angles(self) -> dict[str, float] | None:
        """Return the last commanded joint angles (degrees), or None."""
        return None if self._commanded_angles is None \
            else dict(self._commanded_angles)

    def read_angles(self) -> dict[str, float]:
        """Read current joint angles (degrees) from all servos."""
        angles = {}
        for name in self.JOINT_NAMES:
            sid = config.SERVO_IDS[name]
            raw = self.bus.read_position(sid)
            angles[name] = _raw_to_angle(name, raw)
        return angles

    # ------------------------------------------------------------------
    # Cartesian / IK control
    # ------------------------------------------------------------------

    def move_to_position(
        self,
        x: float,
        y: float,
        z: float,
        speed: int | None = None,
    ) -> bool:
        """Move the paddle to a target position in simulation coords (m).

        Returns True if the target is reachable, False otherwise.
        """
        result = solve_ik(x, y, z)
        if result is None:
            print(f"Target unreachable: ({x:.3f}, {y:.3f}, {z:.3f})")
            return False

        phi1, phi2, phi3, phi4 = ik_angles_to_degrees(*result)
        self.move_to_angles(phi1, phi2, phi3, phi4, speed=speed)
        return True

    def travel_time_to(self, x: float, y: float, z: float) -> float | None:
        """Estimated worst-case joint travel time in seconds to reach a
        simulation-coordinate target from the last commanded pose.

        Uses IK + per-joint speed limits from config; no bus reads.
        Returns None if the target is unreachable.
        """
        result = solve_ik(x, y, z)
        if result is None:
            return None

        phi1, phi2, phi3, phi4 = ik_angles_to_degrees(*result)
        target = {
            "shoulder_yaw":   phi1,
            "shoulder_pitch": phi2,
            "elbow":          phi3,
            "wrist":          phi4,
        }
        # Clamp exactly like move_to_angles — unclamped wrist angles
        # (phi4 compensates shoulder+elbow and can exceed the limits)
        # would otherwise report huge impossible travels.
        current = self._commanded_angles or {
            k: v for k, v in config.READY_ANGLES_DEG.items()   # ready stance
        }
        worst = 0.0
        for name in self.JOINT_NAMES:
            lo, hi = config.JOINT_LIMITS_DEG[name]
            goal = max(lo, min(hi, target[name]))
            worst = max(worst, abs(goal - current[name]) /
                        config.MAX_JOINT_SPEED_DEG_S[name])
        return worst

    def home(self, speed: int | None = None):
        """Move arm to the neutral rest position (straight, safe for startup)."""
        h = config.HOME_ANGLES_DEG
        self.move_to_angles(
            h["shoulder_yaw"], h["shoulder_pitch"],
            h["elbow"], h["wrist"], speed=speed or 300,
        )

    def ready(self, speed: int | None = None):
        """Move arm to the ready stance near the paddle plane.

        Waiting here instead of the neutral pose cuts intercept travel
        time to a fraction — the equivalent of a player keeping the
        racket up between shots.
        """
        r = config.READY_ANGLES_DEG
        self.move_to_angles(
            r["shoulder_yaw"], r["shoulder_pitch"],
            r["elbow"], r["wrist"], speed=speed or 400,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Read angles, speed, and load from all servos."""
        info = {}
        for name in self.JOINT_NAMES:
            sid = config.SERVO_IDS[name]
            info[name] = {
                "angle_deg": _raw_to_angle(name, self.bus.read_position(sid)),
                "speed": self.bus.read_speed(sid),
                "load": self.bus.read_load(sid),
            }
        return info

    def __enter__(self):
        self.connect()
        self.enable_torque()
        return self

    def __exit__(self, *exc):
        self.disconnect()
