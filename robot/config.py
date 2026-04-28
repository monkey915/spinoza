"""Configuration for the Feetech STS3215 robot arm."""

import math

# ---------------------------------------------------------------------------
# Serial bus
# ---------------------------------------------------------------------------
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 1_000_000  # STS3215 default

# ---------------------------------------------------------------------------
# Servo IDs (daisy-chained on the same bus)
# ---------------------------------------------------------------------------
SERVO_IDS = {
    "shoulder_yaw":   1,   # phi1 — base rotation
    "shoulder_pitch": 2,   # phi2 — shoulder lift
    "elbow":          3,   # phi3 — elbow bend
    "wrist":          4,   # phi4 — wrist tilt
}

# ---------------------------------------------------------------------------
# STS3215 hardware constants
# ---------------------------------------------------------------------------
POSITION_MIN = 0       # raw servo position minimum
POSITION_MAX = 4095    # raw servo position maximum
ANGLE_RANGE_DEG = 360.0  # full range in degrees

# Conversion helpers
def deg_to_raw(deg):
    """Convert degrees [0, 360) to raw servo position [0, 4095]."""
    return int((deg % 360.0) / ANGLE_RANGE_DEG * POSITION_MAX)

def raw_to_deg(raw):
    """Convert raw servo position [0, 4095] to degrees [0, 360)."""
    return raw / POSITION_MAX * ANGLE_RANGE_DEG

# ---------------------------------------------------------------------------
# Mechanical zero offsets
# ---------------------------------------------------------------------------
# Raw servo position that corresponds to the simulation's 0-degree angle
# for each joint.  Adjust these after assembling the arm:
#   1. Power on with torque disabled
#   2. Manually move each joint to its simulation zero pose
#   3. Read the raw position and enter it here
ZERO_OFFSETS_RAW = {
    "shoulder_yaw":   2048,  # facing forward
    "shoulder_pitch": 2048,  # arm horizontal
    "elbow":          2048,  # forearm straight
    "wrist":          2048,  # paddle neutral
}

# ---------------------------------------------------------------------------
# Joint limits (simulation angles in degrees)
# ---------------------------------------------------------------------------
JOINT_LIMITS_DEG = {
    "shoulder_yaw":   (-180.0, 180.0),
    "shoulder_pitch": (0.0, 180.0),
    "elbow":          (0.0, 150.0),
    "wrist":          (-90.0, 90.0),
}

# ---------------------------------------------------------------------------
# Speed / safety
# ---------------------------------------------------------------------------
MAX_SPEED = 1000       # raw speed units (0-4095, higher = faster)
MOVE_SPEED = 600       # default movement speed
TORQUE_LIMIT = 800     # raw torque limit (0-1000)

# ---------------------------------------------------------------------------
# Arm segment lengths (metres — must match simulation)
# ---------------------------------------------------------------------------
L_UPPER_ARM = 0.30     # shoulder to elbow
L_FOREARM   = 0.25     # elbow to wrist
L_PADDLE    = 0.10     # wrist to paddle center (handle length)

# Robot arm base position in simulation coordinates (metres)
# Receiver side, centered on table width, at table surface height
ARM_BASE_X = 0.7625    # table width / 2
ARM_BASE_Y = 2.74      # far end of table
ARM_BASE_Z = 0.76      # table surface height
