#!/usr/bin/env python3
"""Real-time 3D table tennis ball tracker using stereo vision.

Standalone usage:
  python -m camera.detect

Programmatic usage:
  from camera.detect import BallTracker
  tracker = BallTracker()
  tracker.start()
  pos, vel = tracker.get_position_3d()  # returns sim coords (m) or (None, None)
  tracker.stop()

Controls (standalone):
  Q     = Quit
  T     = Toggle HSV tuning mode
  S     = Save HSV values (in tuning mode)
  R     = Toggle raw/rectified view
  K     = Toggle Kalman filter
  SPACE = Pause/resume
"""

import re
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np

try:
    from . import config
    from .utils import open_stereo_cameras, load_calibration, rectify_pair, camera_to_sim
except ImportError:
    import config
    from utils import open_stereo_cameras, load_calibration, rectify_pair, camera_to_sim

PLOT_HISTORY = 200
PLOT_HEIGHT = 300
PLOT_WIDTH = 600


# ---------------------------------------------------------------------------
# Ball detection
# ---------------------------------------------------------------------------

def detect_ball(frame, hsv_lower, hsv_upper, min_radius, roi=None):
    """Detect the ball in a single frame.

    roi: optional (x0, y0, x1, y1) search window. Detection runs on the
    crop only and returned coordinates are mapped back to full-frame.
    Returns (center_x, center_y, radius) with sub-pixel precision, or None.
    """
    if roi is not None:
        x0, y0, x1, y1 = roi
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(frame.shape[1], int(x1)); y1 = min(frame.shape[0], int(y1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        frame = frame[y0:y1, x0:x1]

    found = _detect_ball_full(frame, hsv_lower, hsv_upper, min_radius)
    if found is None or roi is None:
        return found
    return (found[0] + x0, found[1] + y0, found[2])


def _detect_ball_full(frame, hsv_lower, hsv_upper, min_radius):
    """Full detection pipeline on the given (sub-)image."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    mask = cv2.GaussianBlur(mask, (9, 9), 2)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * min_radius ** 2:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.4 and area > best_area:
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            if radius >= min_radius:
                best_cnt = cnt
                best_area = area

    if best_cnt is None:
        return None

    M = cv2.moments(best_cnt)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    _, radius = cv2.minEnclosingCircle(best_cnt)
    return (cx, cy, radius)


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def triangulate_point(P1, P2, pt_left, pt_right):
    """Triangulate a 3D point from corresponding 2D points in rectified images."""
    pts_l = np.array([[pt_left[0], pt_left[1]]], dtype=np.float64)
    pts_r = np.array([[pt_right[0], pt_right[1]]], dtype=np.float64)
    points_4d = cv2.triangulatePoints(P1, P2, pts_l.T, pts_r.T)
    point_3d = points_4d[:3] / points_4d[3]
    return point_3d.flatten()


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

GRAVITY_MM = 9810.0  # mm/s²


class BallKalman:
    """9-state constant-acceleration Kalman filter for 3D ball tracking.

    States: [x, y, z, vx, vy, vz, ax, ay, az] (mm, mm/s, mm/s²).
    A gravity feedforward on Z lets the acceleration states stay near zero
    during free flight instead of having to learn -g every rally, which
    removes the systematic vz undershoot of a constant-velocity model.
    Measurements are 3D positions from stereo triangulation (mm).
    """

    def __init__(self, process_noise=500.0, measurement_noise=20.0,
                 gate_distance=250.0):
        # Tuning (units: mm, s)
        q_pos = process_noise * 0.01      # position jitter
        q_vel = process_noise * 0.5       # velocity change per frame
        q_acc = process_noise * 2.0       # acceleration change
        self.q = np.diag([q_pos] * 3 + [q_vel] * 3 + [q_acc] * 3)
        self.r = np.eye(3) * measurement_noise
        # Reject measurements farther than this from the prediction (mm)
        self.gate_distance = gate_distance
        # NIS threshold above which we assume a manoeuvre (table bounce):
        # chi2(3 dof, p=0.01) = 11.34. On trigger the covariance is inflated
        # so the filter snaps to measurements within a frame or two.
        self.nis_threshold = 11.34
        self.cov_inflation = 25.0

        self.x = None          # state vector (9,)
        self.P = np.eye(9) * 1000.0
        self.initialized = False
        self.last_time = None
        self.frames_without_measurement = 0
        self.max_predict_frames = 15

    def _transition(self, dt):
        """Constant-acceleration transition matrix and gravity control."""
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt * dt
        F[3:6, 6:9] = np.eye(3) * dt
        u = np.array([0.0, 0.0, -GRAVITY_MM])
        B = np.zeros((9, 3))
        B[0:3, :] = np.eye(3) * 0.5 * dt * dt
        B[3:6, :] = np.eye(3) * dt
        return F, B @ u

    def update(self, measurement_3d, timestamp):
        """Update with a new 3D measurement (or None to predict only).

        Returns (position, velocity) as 3-element numpy arrays in mm,
        or (None, None).
        """
        if not self.initialized:
            if measurement_3d is None:
                return None, None
            # Gravity enters via the control input, so accel states start at
            # zero and only absorb deviations (drag) from free fall.
            self.x = np.array([measurement_3d[0], measurement_3d[1],
                               measurement_3d[2], 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0])
            self.last_time = timestamp
            self.initialized = True
            self.frames_without_measurement = 0
            return self.x[:3].copy(), self.x[3:6].copy()

        dt = float(np.clip(timestamp - self.last_time, 0.001, 0.5)) \
            if self.last_time is not None else 1.0 / 30.0
        self.last_time = timestamp

        # Predict
        F, gravity = self._transition(dt)
        self.x = F @ self.x + gravity
        self.P = F @ self.P @ F.T + self.q

        # Gate: reject outliers far from the prediction
        accepted = False
        if measurement_3d is not None:
            z = np.asarray(measurement_3d[:3], dtype=np.float64)
            innovation = z - self.x[:3]
            if np.linalg.norm(innovation) <= self.gate_distance:
                H = np.zeros((3, 9))
                H[0, 0] = H[1, 1] = H[2, 2] = 1.0
                S = H @ self.P @ H.T + self.r
                nis = float(innovation @ np.linalg.solve(S, innovation))
                if nis > self.nis_threshold:
                    # Abrupt change (table bounce): inflate covariance and
                    # recompute the gain so the filter follows the manoeuvre.
                    self.P *= self.cov_inflation
                    S = H @ self.P @ H.T + self.r
                K = self.P @ H.T @ np.linalg.inv(S)
                self.x = self.x + K @ innovation
                self.P = (np.eye(9) - K @ H) @ self.P
                self.frames_without_measurement = 0
                accepted = True

        if not accepted:
            self.frames_without_measurement += 1
            if self.frames_without_measurement > self.max_predict_frames:
                self.initialized = False
                return None, None

        return self.x[:3].copy(), self.x[3:6].copy()


# ---------------------------------------------------------------------------
# Search-window tracking
# ---------------------------------------------------------------------------

class RoiWindow:
    """Search window for one camera view.

    Follows the last detection and progressively widens on misses until the
    full frame is searched again. Keeps per-frame detection cost roughly
    constant instead of scanning the whole image.
    """

    def __init__(self, frame_w, frame_h, base_size=160, grow=1.7,
                 max_misses=4):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.base_size = base_size
        self.grow = grow
        self.max_misses = max_misses
        self.size = float(base_size)
        self.center = None       # (cx, cy) of last detection
        self.misses = 0

    def reset(self):
        self.size = float(self.base_size)
        self.center = None
        self.misses = 0

    def on_detection(self, cx, cy, radius):
        self.center = (cx, cy)
        self.misses = 0
        self.size = float(np.clip(radius * 10.0, self.base_size,
                                  min(self.frame_w, self.frame_h)))

    def on_miss(self):
        self.misses += 1
        self.size *= self.grow
        if self.misses > self.max_misses:
            self.reset()

    @property
    def active(self) -> bool:
        return self.center is not None

    def rect(self):
        """Return (x0, y0, x1, y1) or None for a full-frame search."""
        if self.center is None:
            return None
        half = self.size / 2.0
        cx, cy = self.center
        return (int(cx - half), int(cy - half),
                int(cx + half), int(cy + half))


# ---------------------------------------------------------------------------
# BallTracker — programmatic interface for the bridge pipeline
# ---------------------------------------------------------------------------

class BallTracker:
    """Thread-safe 3D ball tracker for real-time use.

    Runs camera capture + detection in a background thread and exposes the
    latest Kalman-filtered position/velocity in spinoza simulation
    coordinates (metres, Z-up).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Latest state (sim coordinates, metres)
        self._position = None   # (x, y, z) or None
        self._velocity = None   # (vx, vy, vz) or None
        self._timestamp = 0.0
        self._position_cam_mm = None  # raw camera coords for debugging
        self._process_latency_ms = 0.0  # grab -> processed

    def start(self):
        """Start capture + detection in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background thread and release cameras."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_position_3d(self):
        """Return (position, velocity) in sim coords, or (None, None).

        position: (x, y, z) in metres — spinoza coordinate system
        velocity: (vx, vy, vz) in m/s
        """
        with self._lock:
            if self._position is None:
                return None, None
            return tuple(self._position), tuple(self._velocity)

    def get_raw_position_mm(self):
        """Return raw camera-frame position in mm, or None."""
        with self._lock:
            return self._position_cam_mm

    def get_timing(self):
        """Return timing diagnostics: (capture_timestamp, observation_age_ms,
        process_latency_ms). observation_age is time since the frame was
        grabbed — the latency budget every downstream stage has to pay."""
        now = time.time()
        with self._lock:
            age_ms = (now - self._timestamp) * 1000.0 if self._timestamp else 0.0
            return self._timestamp, age_ms, self._process_latency_ms

    def _loop(self):
        """Background capture loop."""
        try:
            calib = load_calibration()
        except FileNotFoundError:
            print("BallTracker: no calibration file found!")
            self._running = False
            return

        try:
            cap_left, cap_right = open_stereo_cameras()
        except RuntimeError as e:
            print(f"BallTracker: {e}")
            self._running = False
            return

        P1 = calib["P1"]
        P2 = calib["P2"]
        kalman = BallKalman(process_noise=500.0, measurement_noise=20.0)
        roi_l = RoiWindow(config.FRAME_WIDTH, config.FRAME_HEIGHT)
        roi_r = RoiWindow(config.FRAME_WIDTH, config.FRAME_HEIGHT)
        t_start = time.time()

        try:
            while self._running:
                if not cap_left.grab() or not cap_right.grab():
                    continue
                # Timestamp the capture, not the processing — using the
                # post-processing clock biases velocity estimates.
                t_grab = time.time() - t_start

                ret_l, frame_l = cap_left.retrieve()
                ret_r, frame_r = cap_right.retrieve()
                if not ret_l or not ret_r:
                    continue

                rect_l, rect_r = rectify_pair(frame_l, frame_r, calib)

                ball_l = detect_ball(rect_l, config.BALL_HSV_LOWER,
                                     config.BALL_HSV_UPPER,
                                     config.BALL_MIN_RADIUS,
                                     roi=roi_l.rect())
                ball_r = detect_ball(rect_r, config.BALL_HSV_LOWER,
                                     config.BALL_HSV_UPPER,
                                     config.BALL_MIN_RADIUS,
                                     roi=roi_r.rect())

                pos_3d_mm = None
                if ball_l is not None and ball_r is not None:
                    pos_3d_mm = triangulate_point(
                        P1, P2,
                        (ball_l[0], ball_l[1]),
                        (ball_r[0], ball_r[1]),
                    )

                k_pos, k_vel = kalman.update(pos_3d_mm, t_grab)

                # Update per-view search windows
                if not kalman.initialized:
                    roi_l.reset()
                    roi_r.reset()
                for win, det in ((roi_l, ball_l), (roi_r, ball_r)):
                    if det is not None:
                        win.on_detection(det[0], det[1], det[2])
                    else:
                        win.on_miss()

                t_processed = time.time() - t_start
                with self._lock:
                    self._timestamp = t_grab
                    self._process_latency_ms = (t_processed - t_grab) * 1000.0
                    if k_pos is not None:
                        sx, sy, sz = camera_to_sim(k_pos[0], k_pos[1], k_pos[2])
                        # Velocity: same axis swap, mm/s -> m/s
                        vx = k_vel[0] / 1000.0
                        vy = k_vel[2] / 1000.0
                        vz = -k_vel[1] / 1000.0
                        self._position = (sx, sy, sz)
                        self._velocity = (vx, vy, vz)
                        self._position_cam_mm = tuple(k_pos)
                    else:
                        self._position = None
                        self._velocity = None
                        self._position_cam_mm = None
        finally:
            cap_left.release()
            cap_right.release()


# ---------------------------------------------------------------------------
# Standalone GUI
# ---------------------------------------------------------------------------

def _draw_info(frame, ball, label, position_3d=None):
    """Draw detection info on frame."""
    if ball is not None:
        cx, cy, r = ball
        x, y, r_int = int(cx), int(cy), int(r)
        cv2.circle(frame, (x, y), r_int, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(frame, f"{label} ({cx:.1f},{cy:.1f})", (x - r_int, y - r_int - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if position_3d is not None:
        x_mm, y_mm, z_mm = position_3d
        text = f"X={x_mm:.0f} Y={y_mm:.0f} Z={z_mm:.0f} mm"
        cv2.putText(frame, text, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame


def _draw_plot(times, xs, ys, zs, kxs=None, kys=None, kzs=None):
    """Draw X/Y/Z time-series plot. Raw=thin, Kalman=thick."""
    plot = np.zeros((PLOT_HEIGHT, PLOT_WIDTH, 3), dtype=np.uint8)
    plot[:] = (30, 30, 30)

    if len(times) < 2:
        return plot

    t_arr = np.array(times)
    channels = [
        (np.array(xs), (0, 0, 255), "X"),
        (np.array(ys), (0, 255, 0), "Y"),
        (np.array(zs), (255, 200, 0), "Z"),
    ]

    kalman_channels = None
    if kxs is not None:
        kalman_channels = [
            (np.array(kxs), (0, 0, 255), "X"),
            (np.array(kys), (0, 255, 0), "Y"),
            (np.array(kzs), (255, 200, 0), "Z"),
        ]

    all_vals = np.concatenate([c[0] for c in channels])
    if kalman_channels:
        all_vals = np.concatenate([all_vals] + [c[0] for c in kalman_channels])
    valid = all_vals[~np.isnan(all_vals)]
    if len(valid) == 0:
        return plot

    v_min, v_max = np.min(valid), np.max(valid)
    margin = max((v_max - v_min) * 0.1, 10)
    v_min -= margin
    v_max += margin

    margin_left = 60
    margin_top = 20
    margin_bottom = 25
    draw_w = PLOT_WIDTH - margin_left - 10
    draw_h = PLOT_HEIGHT - margin_top - margin_bottom

    for i in range(5):
        val = v_min + (v_max - v_min) * (4 - i) / 4
        y_px = margin_top + int(draw_h * i / 4)
        cv2.putText(plot, f"{val:.0f}", (2, y_px + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.line(plot, (margin_left, y_px), (PLOT_WIDTH - 10, y_px), (50, 50, 50), 1)

    t_min, t_max = t_arr[0], t_arr[-1]
    t_span = max(t_max - t_min, 0.1)

    def to_points(vals):
        pts = []
        for i in range(len(vals)):
            if np.isnan(vals[i]):
                pts.append(None)
                continue
            px = margin_left + int((t_arr[i] - t_min) / t_span * draw_w)
            py = margin_top + int((1 - (vals[i] - v_min) / (v_max - v_min)) * draw_h)
            py = np.clip(py, margin_top, margin_top + draw_h)
            pts.append((px, py))
        return pts

    raw_alpha = 0.4 if kalman_channels else 1.0
    for vals, color, _ in channels:
        draw_color = tuple(int(c * raw_alpha) for c in color) if kalman_channels else color
        pts = to_points(vals)
        for i in range(1, len(pts)):
            if pts[i - 1] is not None and pts[i] is not None:
                cv2.line(plot, pts[i - 1], pts[i], draw_color, 1, cv2.LINE_AA)

    if kalman_channels:
        for vals, color, _ in kalman_channels:
            pts = to_points(vals)
            for i in range(1, len(pts)):
                if pts[i - 1] is not None and pts[i] is not None:
                    cv2.line(plot, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    for i, (_, color, label) in enumerate(channels):
        lx = margin_left + 5 + i * 70
        cv2.putText(plot, label, (lx + 15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(plot, (lx, 12), (lx + 12, 12), color, 2)

    if kalman_channels:
        cv2.putText(plot, "thick=Kalman thin=raw", (PLOT_WIDTH - 200, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

    cv2.putText(plot, "mm", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return plot


def _create_hsv_trackbars():
    cv2.namedWindow("HSV Tuning", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H min", "HSV Tuning", config.BALL_HSV_LOWER[0], 179, lambda x: None)
    cv2.createTrackbar("S min", "HSV Tuning", config.BALL_HSV_LOWER[1], 255, lambda x: None)
    cv2.createTrackbar("V min", "HSV Tuning", config.BALL_HSV_LOWER[2], 255, lambda x: None)
    cv2.createTrackbar("H max", "HSV Tuning", config.BALL_HSV_UPPER[0], 179, lambda x: None)
    cv2.createTrackbar("S max", "HSV Tuning", config.BALL_HSV_UPPER[1], 255, lambda x: None)
    cv2.createTrackbar("V max", "HSV Tuning", config.BALL_HSV_UPPER[2], 255, lambda x: None)


def _get_hsv_from_trackbars():
    h_min = cv2.getTrackbarPos("H min", "HSV Tuning")
    s_min = cv2.getTrackbarPos("S min", "HSV Tuning")
    v_min = cv2.getTrackbarPos("V min", "HSV Tuning")
    h_max = cv2.getTrackbarPos("H max", "HSV Tuning")
    s_max = cv2.getTrackbarPos("S max", "HSV Tuning")
    v_max = cv2.getTrackbarPos("V max", "HSV Tuning")
    return (h_min, s_min, v_min), (h_max, s_max, v_max)


def main():
    print("=== Spinoza Camera: 3D Ball Tracker ===\n")

    try:
        calib = load_calibration()
    except FileNotFoundError:
        print("No calibration found! Run 'python -m camera.calibrate' first.")
        sys.exit(1)

    cap_left, cap_right = open_stereo_cameras()

    P1, P2 = calib["P1"], calib["P2"]
    hsv_lower = config.BALL_HSV_LOWER
    hsv_upper = config.BALL_HSV_UPPER
    min_radius = config.BALL_MIN_RADIUS

    tuning_mode = False
    use_rectified = True
    use_kalman = True
    paused = False
    frame_count = 0
    fps_timer = time.time()
    fps_display = 0.0

    kalman = BallKalman(process_noise=500.0, measurement_noise=20.0)

    history_x = deque(maxlen=PLOT_HISTORY)
    history_y = deque(maxlen=PLOT_HISTORY)
    history_z = deque(maxlen=PLOT_HISTORY)
    history_kx = deque(maxlen=PLOT_HISTORY)
    history_ky = deque(maxlen=PLOT_HISTORY)
    history_kz = deque(maxlen=PLOT_HISTORY)
    history_t = deque(maxlen=PLOT_HISTORY)
    t_start = time.time()

    print("\nControls: Q=Quit T=HSV-tuning S=Save-HSV R=Rectify K=Kalman SPACE=Pause")

    frame_l = frame_r = None

    try:
        while True:
            if not paused:
                if not cap_left.grab() or not cap_right.grab():
                    print("Grab error!")
                    break
                ret_l, frame_l = cap_left.retrieve()
                ret_r, frame_r = cap_right.retrieve()
                if not ret_l or not ret_r:
                    print("Decode error!")
                    break
                frame_count += 1
                if frame_count % 10 == 0:
                    now = time.time()
                    fps_display = 10.0 / (now - fps_timer)
                    fps_timer = now

            if frame_l is None:
                continue

            if use_rectified:
                rect_l, rect_r = rectify_pair(frame_l, frame_r, calib)
            else:
                rect_l, rect_r = frame_l.copy(), frame_r.copy()

            if tuning_mode:
                hsv_lower, hsv_upper = _get_hsv_from_trackbars()

            ball_l = detect_ball(rect_l, hsv_lower, hsv_upper, min_radius)
            ball_r = detect_ball(rect_r, hsv_lower, hsv_upper, min_radius)

            position_3d = None
            if ball_l is not None and ball_r is not None:
                position_3d = triangulate_point(P1, P2,
                                                (ball_l[0], ball_l[1]),
                                                (ball_r[0], ball_r[1]))

            t_now = time.time() - t_start
            kalman_pos, kalman_vel = kalman.update(position_3d, t_now)

            display_3d = kalman_pos if (use_kalman and kalman_pos is not None) else position_3d

            # History
            if position_3d is not None:
                history_x.append(position_3d[0])
                history_y.append(position_3d[1])
                history_z.append(position_3d[2])
            else:
                history_x.append(np.nan)
                history_y.append(np.nan)
                history_z.append(np.nan)
            if kalman_pos is not None:
                history_kx.append(kalman_pos[0])
                history_ky.append(kalman_pos[1])
                history_kz.append(kalman_pos[2])
            else:
                history_kx.append(np.nan)
                history_ky.append(np.nan)
                history_kz.append(np.nan)
            history_t.append(t_now)

            # Draw
            display_l = _draw_info(rect_l, ball_l, "L", display_3d)
            display_r = _draw_info(rect_r, ball_r, "R")

            if use_kalman and kalman_vel is not None and display_3d is not None:
                speed_kmh = np.linalg.norm(kalman_vel) / 1000.0 * 3.6
                cv2.putText(display_l, f"v={speed_kmh:.1f} km/h",
                            (10, display_l.shape[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            if use_rectified:
                for y in range(0, display_l.shape[0], 80):
                    cv2.line(display_l, (0, y), (display_l.shape[1], y), (50, 50, 50), 1)
                    cv2.line(display_r, (0, y), (display_r.shape[1], y), (50, 50, 50), 1)

            status = "PAUSE" if paused else f"{fps_display:.0f} FPS"
            rect_status = "Rect" if use_rectified else "Raw"
            k_status = "Kalman" if use_kalman else "NoFilt"
            cv2.putText(display_l, f"{status} | {rect_status} | {k_status}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Sim coordinates overlay
            if display_3d is not None:
                sx, sy, sz = camera_to_sim(display_3d[0], display_3d[1], display_3d[2])
                cv2.putText(display_l,
                            f"sim: ({sx:.3f}, {sy:.3f}, {sz:.3f}) m",
                            (10, display_l.shape[0] - 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

            combined = np.hstack((display_l, display_r))
            cv2.imshow("Spinoza Camera: 3D Ball Tracker", combined)

            plot = _draw_plot(history_t, history_x, history_y, history_z,
                              history_kx if use_kalman else None,
                              history_ky if use_kalman else None,
                              history_kz if use_kalman else None)
            cv2.imshow("3D Position (mm)", plot)

            if tuning_mode:
                hsv_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_l, np.array(hsv_lower), np.array(hsv_upper))
                cv2.imshow("HSV Tuning", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                tuning_mode = not tuning_mode
                if tuning_mode:
                    _create_hsv_trackbars()
                    print("HSV tuning enabled")
                else:
                    cv2.destroyWindow("HSV Tuning")
                    print(f"HSV tuning disabled. Values: lower={hsv_lower} upper={hsv_upper}")
            elif key == ord("s") and tuning_mode:
                cfg_path = config.__file__
                with open(cfg_path, "r") as f:
                    cfg = f.read()
                cfg = re.sub(r'BALL_HSV_LOWER = \(.*?\)',
                             f'BALL_HSV_LOWER = {hsv_lower}', cfg)
                cfg = re.sub(r'BALL_HSV_UPPER = \(.*?\)',
                             f'BALL_HSV_UPPER = {hsv_upper}', cfg)
                with open(cfg_path, "w") as f:
                    f.write(cfg)
                print(f"HSV values saved: lower={hsv_lower} upper={hsv_upper}")
            elif key == ord("r"):
                use_rectified = not use_rectified
                print(f"View: {'Rectified' if use_rectified else 'Raw'}")
            elif key == ord("k"):
                use_kalman = not use_kalman
                if use_kalman:
                    kalman = BallKalman()
                print(f"Kalman filter: {'ON' if use_kalman else 'OFF'}")
            elif key == ord(" "):
                paused = not paused

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

    print(f"\nFinished after {frame_count} frames.")


if __name__ == "__main__":
    main()
