"""Utility functions for stereo camera operations."""

import cv2
import numpy as np

from . import config


def open_stereo_cameras(width=None, height=None, fps=None):
    """Open both cameras with given or configured settings.

    Returns (cap_left, cap_right).
    """
    width = width or config.FRAME_WIDTH
    height = height or config.FRAME_HEIGHT
    fps = fps or config.FPS

    fourcc = cv2.VideoWriter.fourcc(*config.CODEC)
    caps = []
    for dev in (config.CAM_LEFT, config.CAM_RIGHT):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        # MJPG codec MUST be set before resolution to avoid USB bandwidth issues
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize frame queue lag
        caps.append(cap)

    cap_left, cap_right = caps

    if not cap_left.isOpened() or not cap_right.isOpened():
        raise RuntimeError(
            f"Failed to open cameras. "
            f"Left ({config.CAM_LEFT}): {cap_left.isOpened()}, "
            f"Right ({config.CAM_RIGHT}): {cap_right.isOpened()}"
        )

    actual_w = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap_left.get(cv2.CAP_PROP_FPS)
    print(f"Cameras opened: {actual_w}x{actual_h} @ {actual_fps:.0f} fps")

    return cap_left, cap_right


def load_calibration(target_size=None):
    """Load stereo calibration and compute rectification remap maps.

    If *target_size* differs from the calibration resolution the intrinsic
    matrices are scaled automatically.
    """
    data = np.load(config.CALIBRATION_FILE)
    K1, D1 = data["K1"], data["D1"]
    K2, D2 = data["K2"], data["D2"]
    R, T = data["R"], data["T"]
    calib_size = tuple(data["image_size"])  # (w, h) from calibration

    if target_size is None:
        target_size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)

    # Scale intrinsics from calibration resolution to tracking resolution
    sx = target_size[0] / calib_size[0]
    sy = target_size[1] / calib_size[1]

    K1_scaled = K1.copy()
    K2_scaled = K2.copy()
    K1_scaled[0, :] *= sx
    K1_scaled[1, :] *= sy
    K2_scaled[0, :] *= sx
    K2_scaled[1, :] *= sy

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1_scaled, D1, K2_scaled, D2, target_size, R, T, alpha=0
    )

    map1_left, map2_left = cv2.initUndistortRectifyMap(
        K1_scaled, D1, R1, P1, target_size, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        K2_scaled, D2, R2, P2, target_size, cv2.CV_16SC2
    )

    baseline_mm = np.linalg.norm(T)
    print(f"Calibration loaded: {calib_size[0]}x{calib_size[1]} -> "
          f"scaled to {target_size[0]}x{target_size[1]}, "
          f"baseline: {baseline_mm:.1f} mm")

    return {
        "K1": K1_scaled, "D1": D1,
        "K2": K2_scaled, "D2": D2,
        "R": R, "T": T,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "map1_left": map1_left, "map2_left": map2_left,
        "map1_right": map1_right, "map2_right": map2_right,
    }


def rectify_pair(frame_left, frame_right, calib):
    """Rectify a stereo image pair using precomputed remap maps."""
    left = cv2.remap(frame_left, calib["map1_left"], calib["map2_left"],
                     cv2.INTER_LINEAR)
    right = cv2.remap(frame_right, calib["map1_right"], calib["map2_right"],
                      cv2.INTER_LINEAR)
    return left, right


def camera_to_sim(x_mm, y_mm, z_mm):
    """Convert camera coordinates (mm) to spinoza simulation coordinates (m).

    Camera frame:  X-right, Y-down,    Z-forward   (mm)
    Spinoza frame: X-right, Y-forward, Z-up        (m)

    The camera-to-table offset from config is applied so that the returned
    position is in the spinoza table-origin coordinate system.
    """
    # Apply camera mounting offset (camera coords, mm)
    cx = x_mm - config.CAM_OFFSET_X_MM
    cy = y_mm - config.CAM_OFFSET_Y_MM
    cz = z_mm - config.CAM_OFFSET_Z_MM

    # Axis swap + mm -> m
    sx = cx / 1000.0     # cam X  -> sim X  (right)
    sy = cz / 1000.0     # cam Z  -> sim Y  (forward)
    sz = -cy / 1000.0    # cam -Y -> sim Z  (up)
    return sx, sy, sz
