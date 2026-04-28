"""Configuration for the stereo camera setup."""

import os

# Camera device paths (adjust to your system)
CAM_LEFT = "/dev/video6"
CAM_RIGHT = "/dev/video8"

# Capture settings for live tracking
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
CODEC = "MJPG"

# Calibration capture (higher resolution for better accuracy)
CALIB_WIDTH = 1280
CALIB_HEIGHT = 720
CALIB_FPS = 10

# Calibration chessboard
CHESSBOARD_SIZE = (8, 12)  # inner corners (columns, rows)
SQUARE_SIZE_MM = 19.0      # size of one square in mm

# Calibration data paths
_BASE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(_BASE, "calibration_data")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "stereo_calibration.npz")
CALIB_IMAGES_DIR = os.path.join(CALIBRATION_DIR, "captures")

# Ball detection (HSV range for orange table tennis ball)
BALL_HSV_LOWER = (5, 182, 37)
BALL_HSV_UPPER = (97, 238, 255)
BALL_MIN_RADIUS = 10  # minimum detected radius in pixels

# ---------------------------------------------------------------------------
# Camera-to-table extrinsic offset
# ---------------------------------------------------------------------------
# Translation from camera origin to the spinoza table origin (server-side
# left corner of the table surface).  Measured in millimetres in the
# **camera** coordinate frame (X-right, Y-down, Z-forward).
#
# These values define where the spinoza world origin (server-side left
# corner of table surface) appears **in camera coordinates**.
#
# Example: camera mounted 1.2 m above table surface, 20 cm behind edge,
# centred on table width (0.76 m from left corner):
#   CAM_OFFSET_X_MM  =  762.5   (origin is to the left of camera centre)
#   CAM_OFFSET_Y_MM  = 1200.0   (origin is 1.2 m below camera → +Y in cam)
#   CAM_OFFSET_Z_MM  =  200.0   (origin is 20 cm in front → +Z in cam)
#
# Adjust after mounting the camera.
CAM_OFFSET_X_MM = 762.5
CAM_OFFSET_Y_MM = 1200.0
CAM_OFFSET_Z_MM = 200.0
