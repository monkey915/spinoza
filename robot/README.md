# Spinoza Robot Arm Module

4-DOF robot arm control for the spinoza table tennis robot using Feetech STS3215 servos.

## Hardware

- **Servos**: 4× Feetech STS3215 (15 kg·cm, ~19 rad/s, serial TTL bus)
- **Controller**: USB-to-TTL adapter (e.g. FE-URT-1)
- **Baud rate**: 1 Mbps (default)

## Joint Configuration

| Joint | Servo ID | Sim Range | Description |
|-------|----------|-----------|-------------|
| Shoulder yaw (φ1) | 1 | ±180° | Base rotation |
| Shoulder pitch (φ2) | 2 | 0–180° | Shoulder lift |
| Elbow (φ3) | 3 | 0–150° | Elbow bend |
| Wrist (φ4) | 4 | ±90° | Paddle tilt |

## Arm Dimensions

| Segment | Length |
|---------|--------|
| Upper arm | 0.30 m |
| Forearm | 0.25 m |
| Paddle handle | 0.10 m |
| **Total reach** | **0.65 m** |

## Quick Start

```python
from robot.arm import RobotArm

# Context manager handles connect/disconnect and torque
with RobotArm() as arm:
    arm.home()                            # move to rest position
    arm.move_to_position(0.76, 2.5, 1.0)  # IK target in sim coords (m)
    angles = arm.read_angles()            # read current joint angles
```

## Calibration

After assembling the arm, update `robot/config.py`:

1. Power on with torque disabled (`arm.disable_torque()`)
2. Manually move each joint to its simulation zero pose
3. Read the raw positions (`bus.read_all_positions()`)
4. Enter values in `ZERO_OFFSETS_RAW`

## Dependencies

```
scservo_sdk
```
