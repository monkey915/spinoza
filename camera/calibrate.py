#!/usr/bin/env python3
"""Stereo camera calibration script.

Usage:
  python -m camera.calibrate capture    - Capture calibration image pairs
  python -m camera.calibrate calibrate  - Run calibration from saved images
  python -m camera.calibrate            - Capture + calibrate in one go

Controls during capture:
  SPACE = Capture pair | D = Delete last pair | C = Done, calibrate | Q = Quit

Images are saved to camera/calibration_data/captures/ and persist between runs.
"""

import os
import sys
import glob as globmod

import cv2
import numpy as np

# Allow running as `python -m camera.calibrate` or `python camera/calibrate.py`
try:
    from . import config
    from .utils import open_stereo_cameras
except ImportError:
    import config
    from utils import open_stereo_cameras


def find_chessboard(gray, board_size):
    """Find chessboard corners with subpixel refinement and consistent ordering."""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Fix 180-degree ambiguity: ensure first corner is always above last
        if corners[0][0][1] > corners[-1][0][1]:
            corners = corners[::-1].copy()
    return found, corners


def _get_next_pair_index():
    """Find the next available pair index in the captures directory."""
    os.makedirs(config.CALIB_IMAGES_DIR, exist_ok=True)
    existing = globmod.glob(os.path.join(config.CALIB_IMAGES_DIR, "left_*.png"))
    if not existing:
        return 0
    indices = []
    for f in existing:
        base = os.path.basename(f)
        try:
            indices.append(int(base.replace("left_", "").replace(".png", "")))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def _count_saved_pairs():
    """Count existing saved image pairs."""
    if not os.path.isdir(config.CALIB_IMAGES_DIR):
        return 0
    left_files = sorted(globmod.glob(os.path.join(config.CALIB_IMAGES_DIR, "left_*.png")))
    count = 0
    for lf in left_files:
        rf = lf.replace("left_", "right_")
        if os.path.isfile(rf):
            count += 1
    return count


def capture_calibration_images(cap_left, cap_right):
    """Interactive capture — saves each pair to disk immediately."""
    board_size = config.CHESSBOARD_SIZE
    pair_idx = _get_next_pair_index()
    existing = _count_saved_pairs()

    print("\n=== Calibration capture ===")
    print("SPACE = Capture | D = Delete last | C = Calibrate | Q = Quit")
    print(f"Chessboard: {board_size[0]}x{board_size[1]} inner corners")
    print(f"Image folder: {config.CALIB_IMAGES_DIR}/")
    if existing > 0:
        print(f"Previously saved: {existing} pairs (kept)")
    print()

    captured_this_session = []

    while True:
        if not cap_left.grab() or not cap_right.grab():
            print("Grab error!")
            break
        ret_l, frame_l = cap_left.retrieve()
        ret_r, frame_r = cap_right.retrieve()
        if not ret_l or not ret_r:
            print("Frame decode error!")
            break

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = find_chessboard(gray_l, board_size)
        found_r, corners_r = find_chessboard(gray_r, board_size)

        display_l = frame_l.copy()
        display_r = frame_r.copy()
        if found_l:
            cv2.drawChessboardCorners(display_l, board_size, corners_l, found_l)
        if found_r:
            cv2.drawChessboardCorners(display_r, board_size, corners_r, found_r)

        color_l = (0, 255, 0) if found_l else (0, 0, 255)
        color_r = (0, 255, 0) if found_r else (0, 0, 255)
        cv2.putText(display_l, "L: OK" if found_l else "L: --", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_l, 2)
        cv2.putText(display_r, "R: OK" if found_r else "R: --", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_r, 2)

        total = existing + len(captured_this_session)
        cv2.putText(display_l, f"Pairs: {total} ({len(captured_this_session)} new)",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = np.hstack((display_l, display_r))
        cv2.imshow("Stereo Calibration", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if found_l and found_r:
                lpath = os.path.join(config.CALIB_IMAGES_DIR, f"left_{pair_idx:03d}.png")
                rpath = os.path.join(config.CALIB_IMAGES_DIR, f"right_{pair_idx:03d}.png")
                cv2.imwrite(lpath, frame_l)
                cv2.imwrite(rpath, frame_r)
                captured_this_session.append(pair_idx)
                print(f"  Pair {pair_idx:03d} saved ({lpath})")
                pair_idx += 1
            else:
                print("  Chessboard not detected in both frames!")

        elif key == ord("d"):
            if captured_this_session:
                last_idx = captured_this_session.pop()
                lpath = os.path.join(config.CALIB_IMAGES_DIR, f"left_{last_idx:03d}.png")
                rpath = os.path.join(config.CALIB_IMAGES_DIR, f"right_{last_idx:03d}.png")
                for p in (lpath, rpath):
                    if os.path.isfile(p):
                        os.remove(p)
                pair_idx = last_idx
                print(f"  Pair {last_idx:03d} deleted")
            else:
                print("  Nothing to delete in this session")

        elif key == ord("c"):
            total = _count_saved_pairs()
            if total >= 10:
                print(f"\nCalibrating with {total} saved pairs...")
                cv2.destroyAllWindows()
                return True
            else:
                print(f"  Need at least 10 pairs (currently: {total})")

        elif key == ord("q"):
            total = _count_saved_pairs()
            print(f"\n{len(captured_this_session)} new pairs saved. Total: {total}")
            cv2.destroyAllWindows()
            return False

    cv2.destroyAllWindows()
    return False


def load_image_pairs():
    """Load all saved image pairs and detect chessboard corners."""
    board_size = config.CHESSBOARD_SIZE

    if not os.path.isdir(config.CALIB_IMAGES_DIR):
        print(f"No images found in {config.CALIB_IMAGES_DIR}/")
        return []

    left_files = sorted(globmod.glob(os.path.join(config.CALIB_IMAGES_DIR, "left_*.png")))
    if not left_files:
        print(f"No images found in {config.CALIB_IMAGES_DIR}/")
        return []

    image_pairs = []
    skipped = 0

    for lpath in left_files:
        rpath = lpath.replace("left_", "right_")
        if not os.path.isfile(rpath):
            skipped += 1
            continue

        img_l = cv2.imread(lpath)
        img_r = cv2.imread(rpath)
        if img_l is None or img_r is None:
            skipped += 1
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = find_chessboard(gray_l, board_size)
        found_r, corners_r = find_chessboard(gray_r, board_size)

        idx = os.path.basename(lpath).replace("left_", "").replace(".png", "")

        if found_l and found_r:
            image_pairs.append((gray_l, gray_r, corners_l, corners_r))
            print(f"  Pair {idx}: corners found")
        else:
            skipped += 1
            print(f"  Pair {idx}: corners NOT found (skipped)")

    if skipped:
        print(f"  {skipped} images skipped")

    return image_pairs


def run_calibration(image_pairs):
    """Run stereo calibration from image pairs with detected corners."""
    board_size = config.CHESSBOARD_SIZE
    square_size = config.SQUARE_SIZE_MM

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    img_points_l = []
    img_points_r = []

    for gray_l, gray_r, corners_l, corners_r in image_pairs:
        obj_points.append(objp)
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)

    img_size = (image_pairs[0][0].shape[1], image_pairs[0][0].shape[0])
    print(f"\nImage size: {img_size[0]}x{img_size[1]}")
    print(f"Pairs: {len(image_pairs)}")

    # Rational distortion model (14 coefficients) for wide-angle lenses
    calib_flags = cv2.CALIB_RATIONAL_MODEL
    print("\nCalibrating left camera (rational model, 14 coeff.)...")
    ret_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
        obj_points, img_points_l, img_size, None, None, flags=calib_flags
    )
    print(f"  RMS error left: {ret_l:.4f}")

    print("Calibrating right camera (rational model, 14 coeff.)...")
    ret_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(
        obj_points, img_points_r, img_size, None, None, flags=calib_flags
    )
    print(f"  RMS error right: {ret_r:.4f}")

    # Reject outlier pairs (per-image RMS > 2x median)
    errors = []
    for i in range(len(obj_points)):
        proj_l, _ = cv2.projectPoints(obj_points[i], rvecs_l[i], tvecs_l[i], K1, D1)
        err_l = cv2.norm(img_points_l[i], proj_l, cv2.NORM_L2) / len(proj_l)
        proj_r, _ = cv2.projectPoints(obj_points[i], rvecs_r[i], tvecs_r[i], K2, D2)
        err_r = cv2.norm(img_points_r[i], proj_r, cv2.NORM_L2) / len(proj_r)
        errors.append(max(err_l, err_r))
        print(f"  Pair {i:2d}: error L={err_l:.3f} R={err_r:.3f}")

    median_err = np.median(errors)
    threshold = median_err * 2.0
    good = [i for i, e in enumerate(errors) if e <= threshold]
    rejected = len(errors) - len(good)
    if rejected > 0:
        print(f"\n  {rejected} outliers removed (threshold: {threshold:.3f})")
        obj_points = [obj_points[i] for i in good]
        img_points_l = [img_points_l[i] for i in good]
        img_points_r = [img_points_r[i] for i in good]
        print(f"  Recalibrating with {len(good)} pairs...")
        ret_l, K1, D1, _, _ = cv2.calibrateCamera(
            obj_points, img_points_l, img_size, None, None, flags=calib_flags)
        ret_r, K2, D2, _, _ = cv2.calibrateCamera(
            obj_points, img_points_r, img_size, None, None, flags=calib_flags)
        print(f"  RMS after cleanup: left={ret_l:.4f} right={ret_r:.4f}")

    # Stereo calibration
    print("\nStereo calibration...")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K1, D1, K2, D2, img_size,
        criteria=criteria, flags=stereo_flags
    )
    print(f"  Stereo RMS error: {ret_stereo:.4f}")

    baseline_mm = np.linalg.norm(T)
    print(f"  Baseline: {baseline_mm:.1f} mm")
    print(f"  T = {T.flatten()}")

    if ret_stereo > 2.0:
        print(f"\n  WARNING: Stereo RMS={ret_stereo:.2f} is high (target: < 1.0)")
        print("  Tips: hold chessboard steady, vary angles, keep board flat")

    # Save
    os.makedirs(config.CALIBRATION_DIR, exist_ok=True)
    np.savez(
        config.CALIBRATION_FILE,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T, E=E, F=F,
        image_size=np.array(img_size),
    )
    print(f"\nCalibration saved: {config.CALIBRATION_FILE}")

    _show_rectification_preview(K1, D1, K2, D2, R, T, img_size, image_pairs)


def _show_rectification_preview(K1, D1, K2, D2, R, T, img_size, image_pairs):
    """Show rectified sample pairs with epipolar lines."""
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T, alpha=1
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)

    indices = [0, len(image_pairs) // 2, len(image_pairs) - 1]
    for idx in indices:
        gray_l, gray_r = image_pairs[idx][0], image_pairs[idx][1]
        rect_l = cv2.remap(gray_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(gray_r, map1_r, map2_r, cv2.INTER_LINEAR)
        combined = np.hstack((rect_l, rect_r))
        combined_color = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

        for y in range(0, combined_color.shape[0], 40):
            cv2.line(combined_color, (0, y), (combined_color.shape[1], y),
                     (0, 255, 0), 1)

        title = f"Rectification pair {idx} (SPACE=next, Q=done)"
        cv2.imshow(title, combined_color)
        print(f"\nShowing pair {idx} - SPACE=next, Q=done")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord(" "), ord("q")):
                break
        cv2.destroyWindow(title)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    args = sys.argv[1:]
    mode = args[0] if args else "both"

    if mode not in ("capture", "calibrate", "both"):
        print("Usage: python -m camera.calibrate [capture|calibrate]")
        print("  capture   - Capture images only (camera required)")
        print("  calibrate - Offline calibration from saved images")
        print("  (no arg)  - Both in sequence")
        sys.exit(1)

    print("=== Spinoza Camera: Stereo Calibration ===\n")

    if mode in ("capture", "both"):
        cap_left, cap_right = open_stereo_cameras(
            width=config.CALIB_WIDTH,
            height=config.CALIB_HEIGHT,
            fps=config.CALIB_FPS,
        )
        try:
            proceed = capture_calibration_images(cap_left, cap_right)
        finally:
            cap_left.release()
            cap_right.release()
            cv2.destroyAllWindows()

        if not proceed and mode == "both":
            total = _count_saved_pairs()
            if total >= 10:
                print(f"\n{total} pairs available. "
                      "Run calibration with: python -m camera.calibrate calibrate")
            sys.exit(0)

    if mode in ("calibrate", "both"):
        print("\nLoading saved images...")
        image_pairs = load_image_pairs()
        if len(image_pairs) < 10:
            print(f"\nOnly {len(image_pairs)} valid pairs - need at least 10!")
            print("Capture more: python -m camera.calibrate capture")
            sys.exit(1)

        print(f"\n{len(image_pairs)} valid pairs loaded.")
        run_calibration(image_pairs)


if __name__ == "__main__":
    main()
