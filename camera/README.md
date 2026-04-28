# Spinoza Camera Module

Stereo vision ball tracking for the spinoza table tennis robot.

## Hardware

- **Camera**: ELP Stereo USB camera (rolling shutter, 2× UVC sensors)
- **Baseline**: ~62 mm
- **Tracking resolution**: 640×480 @ 30 FPS (MJPG)
- **Calibration resolution**: 1280×720 @ 10 FPS

## Pipeline

```
Left + Right cameras
    ↓  grab() / retrieve()
Undistort + Rectify
    ↓
HSV threshold → morphology → contour detection
    ↓
Stereo triangulation (cv2.triangulatePoints)
    ↓
Kalman filter (6-DOF: position + velocity)
    ↓
3D position in spinoza simulation coordinates (metres, Z-up)
```

## Coordinate Transform

| Camera frame | Spinoza frame |
|-------------|---------------|
| X → right   | X → right     |
| Y → down    | Z → up (negated) |
| Z → forward | Y → forward   |
| mm          | metres        |

## Quick Start

```bash
# 1. Calibrate (first time only)
python -m camera.calibrate capture     # capture chessboard pairs
python -m camera.calibrate calibrate   # run offline calibration

# 2. Track ball
python -m camera.detect

# 3. Programmatic use
from camera.detect import BallTracker
tracker = BallTracker()
tracker.start()
pos, vel = tracker.get_position_3d()   # (x,y,z) in metres, sim coords
tracker.stop()
```

## Controls (detect.py standalone)

| Key   | Action |
|-------|--------|
| Q     | Quit |
| T     | Toggle HSV tuning trackbars |
| S     | Save HSV values to config.py |
| R     | Toggle rectified/raw view |
| K     | Toggle Kalman filter |
| SPACE | Pause/resume |

## Dependencies

```
opencv-python>=4.8
numpy>=1.24
```
