"""Paddle position calculator — compute optimal return from predicted trajectory.

Given a predicted ball trajectory (positions, velocity, spin), computes the
optimal paddle position, angle, timing, and swing speed for a return shot.

Uses the same physics as the Rust simulator (paddle.rs):
  - Paddle normal from tilt angles
  - Swing velocity with biomechanical speed reduction
  - Friction-based spin transfer
"""
import numpy as np
from dataclasses import dataclass


# Physics constants (match src/physics/paddle.rs)
PADDLE_E_N = 0.85    # coefficient of restitution
PADDLE_MU = 0.45     # friction coefficient (rubber)
BALL_RADIUS = 0.02   # m
TABLE_LENGTH = 2.74  # m
TABLE_WIDTH = 1.525  # m
NET_Y = TABLE_LENGTH / 2.0  # 1.37 m
NET_TOP_Z = 0.76 + 0.1525  # 0.9125 m
TABLE_Z = 0.76       # table surface height


@dataclass
class PaddleAction:
    """Recommended paddle action for returning a ball."""
    paddle_x: float   # lateral position (m)
    paddle_y: float   # depth position (m)
    paddle_z: float   # height (m)
    tilt_x: float     # forward/back lean (rad), positive = closed face (topspin)
    tilt_z: float     # left/right lean (rad)
    swing_speed: float  # m/s
    swing_elevation: float  # rad, positive = upward
    intercept_frame: int    # which frame to intercept
    confidence: float       # 0-1, how confident the calculation is


def compute_paddle_action(
    positions: np.ndarray,
    velocity: np.ndarray = None,
    spin: np.ndarray = None,
    n_observed: int = 0,
    full_states: list = None,
    env=None,
) -> PaddleAction:
    """Compute optimal paddle action using physics-based optimization.
    
    If `env` and `full_states` are provided, uses simulate_hit() to find
    the paddle parameters that produce a successful return with best net
    clearance and table placement. Otherwise falls back to heuristics.
    
    Args:
        positions: (30, 3) ball positions [x, y, z] in meters
        velocity: (30, 3) velocities per frame, or None
        spin: (3,) predicted spin [ωx, ωy, ωz] in rad/s, or None
        n_observed: number of observed (ground truth) frames
        full_states: list of [x,y,z,vx,vy,vz,ωx,ωy,ωz] per frame (from predictor or GT)
        env: SimEnv instance for simulate_hit() calls
    
    Returns:
        PaddleAction with recommended paddle parameters
    """
    n_frames = len(positions)
    
    # Estimate velocity from positions if not provided
    if velocity is None:
        dt = 1.0 / 60.0
        velocity = np.zeros_like(positions)
        velocity[1:] = (positions[1:] - positions[:-1]) / dt
        velocity[0] = velocity[1]
    
    if spin is None:
        spin = np.zeros(3)
    
    # Find best intercept frame: ball near apex at comfortable height
    best_frame = _find_intercept_frame(positions, velocity)
    
    # Ball state at intercept
    bx, by, bz = positions[best_frame]
    paddle_x = float(np.clip(bx, 0.05, TABLE_WIDTH - 0.05))
    paddle_y = float(np.clip(by, 1.5, 4.0))
    paddle_z = float(max(bz, TABLE_Z + 0.09))
    
    # Build ball state vector for simulate_hit
    bvx, bvy, bvz = velocity[best_frame]
    if full_states is not None and best_frame < len(full_states):
        ball_state = list(full_states[best_frame])
    else:
        ball_state = [float(bx), float(by), float(bz),
                      float(bvx), float(bvy), float(bvz),
                      float(spin[0]), float(spin[1]), float(spin[2])]
    
    # Physics-based optimization: try many paddle parameters, pick best
    if env is not None:
        best_action = _optimize_paddle(env, ball_state, paddle_x, paddle_y, paddle_z)
        if best_action is not None:
            return PaddleAction(
                paddle_x=paddle_x, paddle_y=paddle_y, paddle_z=paddle_z,
                tilt_x=best_action[0], tilt_z=best_action[1],
                swing_speed=best_action[2], swing_elevation=best_action[3],
                intercept_frame=best_frame, confidence=best_action[4],
            )
    
    # Fallback: heuristic parameters
    incoming_speed = np.sqrt(bvx**2 + bvy**2 + bvz**2)
    tilt_x = float(np.clip(-0.12 + spin[0] / 400.0, -0.25, 0.05))
    tilt_z = float(np.clip(spin[2] / 500.0, -0.15, 0.15))
    swing_speed = float(np.clip(12.0 - incoming_speed * 0.3, 6.0, 14.0))
    swing_elevation = float(np.radians(45))
    
    return PaddleAction(
        paddle_x=paddle_x, paddle_y=paddle_y, paddle_z=paddle_z,
        tilt_x=tilt_x, tilt_z=tilt_z,
        swing_speed=swing_speed, swing_elevation=swing_elevation,
        intercept_frame=best_frame, confidence=0.3,
    )


def _find_intercept_frame(positions, velocity):
    """Find best frame to intercept the ball (near apex, comfortable height)."""
    best_frame = None
    best_score = -1e9
    
    for i in range(5, len(positions)):
        x, y, z = positions[i]
        vz = velocity[i][2] if i < len(velocity) else 0
        
        if z < TABLE_Z + 0.03 or z > 1.5:
            continue
        if y < TABLE_LENGTH - 0.1 or y > 4.5:
            continue
        
        height_score = -abs(z - 0.92) * 3
        apex_score = -abs(vz) * 0.3
        score = height_score + apex_score
        if score > best_score:
            best_score = score
            best_frame = i
    
    return best_frame if best_frame is not None else min(8, len(positions) - 1)


def _optimize_paddle(env, ball_state, paddle_x, paddle_y, paddle_z):
    """Find optimal tilt/swing parameters via grid search + refinement.
    
    Returns (tilt_x, tilt_z, swing_speed, swing_elevation, confidence) or None.
    """
    # Phase 1: Coarse grid search
    best = None
    best_score = -1e9
    
    for tilt_x in np.arange(-0.25, 0.15, 0.05):
        for swing_elev_deg in range(25, 60, 5):
            for swing_speed in [8, 10, 12, 14]:
                action = [paddle_x, paddle_y, paddle_z,
                          float(tilt_x), 0.0, float(swing_speed),
                          np.radians(float(swing_elev_deg))]
                result = env.simulate_hit(ball_state, action)
                score = _score_result(result)
                if score > best_score:
                    best_score = score
                    best = (float(tilt_x), 0.0, float(swing_speed),
                            np.radians(float(swing_elev_deg)))
    
    if best is None or best_score < 0:
        return None
    
    # Phase 2: Fine refinement around best coarse parameters
    coarse_tilt, _, coarse_speed, coarse_elev = best
    best_fine = best
    best_fine_score = best_score
    
    for dt in np.arange(-0.04, 0.05, 0.01):
        for de in np.arange(-np.radians(4), np.radians(5), np.radians(1)):
            for ds in [-1, 0, 1]:
                for dtz in np.arange(-0.06, 0.07, 0.03):
                    tilt_x = coarse_tilt + dt
                    tilt_z = dtz
                    speed = coarse_speed + ds
                    elev = coarse_elev + de
                    action = [paddle_x, paddle_y, paddle_z,
                              float(tilt_x), float(tilt_z),
                              float(speed), float(elev)]
                    result = env.simulate_hit(ball_state, action)
                    score = _score_result(result)
                    if score > best_fine_score:
                        best_fine_score = score
                        best_fine = (float(tilt_x), float(tilt_z),
                                     float(speed), float(elev))
    
    confidence = min(1.0, max(0.0, best_fine_score))
    return (*best_fine, confidence)


def _score_result(result):
    """Score a simulate_hit result. Higher = better return."""
    outcome = result['outcome']
    if outcome == 'paddle_miss':
        return -2.0
    if outcome == 'hit_net':
        return -1.0 + result.get('net_clearance_z', -0.1) * 5  # closer to clearing = less bad
    if outcome == 'missed_table':
        lx = result.get('landing_x', 0)
        ly = result.get('landing_y', 0)
        # How close to the table?
        dx = max(0, max(-lx, lx - TABLE_WIDTH))
        dy = max(0, max(-ly, ly - TABLE_LENGTH / 2))
        dist = np.sqrt(dx**2 + dy**2)
        return -0.5 - dist * 0.5
    if outcome == 'success':
        # Base score for success
        score = 1.0
        # Prefer good net clearance (2-8cm ideal)
        net_cl = result.get('net_clearance_z', 0)
        if net_cl > 0:
            score += 0.3 * min(net_cl / 0.05, 1.0)  # reward up to 5cm clearance
            if net_cl > 0.15:
                score -= 0.1 * (net_cl - 0.15)  # penalize too high (easy return)
        # Prefer landing deeper (harder to return)
        ly = result.get('landing_y', 0)
        score += 0.2 * (ly / (TABLE_LENGTH / 2))  # deeper = better
        # Prefer landing near center (more margin)
        lx = result.get('landing_x', TABLE_WIDTH / 2)
        score += 0.1 * (1.0 - abs(lx - TABLE_WIDTH / 2) / (TABLE_WIDTH / 2))
        return score
    return -3.0


def evaluate_action(action: PaddleAction, positions: np.ndarray, velocity: np.ndarray = None):
    """Print a human-readable summary of the recommended action."""
    bx, by, bz = positions[action.intercept_frame]
    print(f"=== Paddle Action ===")
    print(f"  Intercept: frame {action.intercept_frame} at ({bx:.3f}, {by:.3f}, {bz:.3f})")
    print(f"  Paddle pos: ({action.paddle_x:.3f}, {action.paddle_y:.3f}, {action.paddle_z:.3f})")
    tilt_label = "closed (topspin)" if action.tilt_x > 0.05 else "open (backspin)" if action.tilt_x < -0.05 else "flat"
    print(f"  Tilt: x={action.tilt_x:.2f} ({tilt_label}), z={action.tilt_z:.2f}")
    print(f"  Swing: {action.swing_speed:.1f} m/s, elevation={np.degrees(action.swing_elevation):.1f}°")
    print(f"  Confidence: {action.confidence:.0%}")


if __name__ == '__main__':
    """End-to-end test: Serve → Predict → Paddle Optimizer → Simulate Return."""
    from spinoza import SimEnv
    import torch
    from predict import TrajectoryPredictor, TOTAL_FRAMES
    import time

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/predictor_rally.pt')
    parser.add_argument('--n-serves', type=int, default=100)
    parser.add_argument('--n-input', type=int, default=15, help='Observed frames before prediction')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ground-truth', action='store_true', help='Use ground truth instead of predictor')
    args = parser.parse_args()

    env = SimEnv(seed=42, difficulty=3)

    model = None
    if not args.ground_truth:
        ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
        model = TrajectoryPredictor(
            hidden=ckpt['hidden'], n_layers=ckpt['n_layers'],
            kernel_size=ckpt.get('kernel_size', 7),
            predict_spin=ckpt.get('predict_spin', False),
            predict_vel=ckpt.get('predict_vel', False),
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

    outcomes = {}
    t0 = time.time()

    for i in range(args.n_serves):
        # Generate a serve trajectory
        trajs = env.generate_rich_trajectories(1, 3)
        t = trajs[0]
        gt_pos = np.array(t['positions'], dtype=np.float32)
        gt_full = [list(s) for s in t['full_states']]
        gt_spin = np.array([gt_full[0][6], gt_full[0][7], gt_full[0][8]])

        if args.ground_truth:
            # Use ground truth positions/velocity/spin
            dt_f = 1.0 / 60.0
            vel = np.zeros_like(gt_pos)
            vel[1:] = (gt_pos[1:] - gt_pos[:-1]) / dt_f
            vel[0] = vel[1]
            action = compute_paddle_action(gt_pos, vel, gt_spin,
                                           full_states=gt_full, env=env)
        else:
            # Use predictor
            n_input = args.n_input
            inp = np.zeros((1, TOTAL_FRAMES, 3), dtype=np.float32)
            inp[0, :n_input] = gt_pos[:n_input]
            mask = np.zeros((1, TOTAL_FRAMES), dtype=np.float32)
            mask[0, :n_input] = 1.0

            with torch.no_grad():
                pos_pred, spin_pred, vel_pred = model(
                    torch.from_numpy(inp), torch.from_numpy(mask)
                )
                pred_pos = pos_pred.numpy()[0]
                pred_spin = spin_pred.numpy()[0] * 150.0 if spin_pred is not None else None
                pred_vel = vel_pred.numpy()[0] * 10.0 if vel_pred is not None else None

            # Merge observed + predicted
            merged_pos = pred_pos.copy()
            merged_pos[:n_input] = gt_pos[:n_input]
            merged_vel = pred_vel.copy() if pred_vel is not None else None
            if merged_vel is not None:
                dt_f = 1.0 / 60.0
                merged_vel[:n_input] = np.gradient(gt_pos[:n_input], dt_f, axis=0)

            # Build full_states from merged data for the optimizer
            merged_full = []
            for j in range(TOTAL_FRAMES):
                if j < n_input:
                    merged_full.append(gt_full[j])
                else:
                    p = merged_pos[j]
                    v = merged_vel[j] if merged_vel is not None else [0, 0, 0]
                    s = pred_spin if pred_spin is not None else [0, 0, 0]
                    merged_full.append([float(p[0]), float(p[1]), float(p[2]),
                                        float(v[0]), float(v[1]), float(v[2]),
                                        float(s[0]), float(s[1]), float(s[2])])

            action = compute_paddle_action(merged_pos, merged_vel, pred_spin,
                                           n_observed=n_input,
                                           full_states=merged_full, env=env)

        # Simulate the return using ground truth ball state at intercept frame
        ball_state = gt_full[action.intercept_frame]
        action_vec = [action.paddle_x, action.paddle_y, action.paddle_z,
                      action.tilt_x, action.tilt_z,
                      action.swing_speed, action.swing_elevation]
        result = env.simulate_hit(ball_state, action_vec)
        out = result['outcome']
        outcomes[out] = outcomes.get(out, 0) + 1

        if args.verbose:
            extra = ''
            if out == 'success':
                extra = f' land=({result["landing_x"]:.2f},{result["landing_y"]:.2f}) net={result["net_clearance_z"]:.3f}'
            print(f'  #{i+1}: {out}{extra} frame={action.intercept_frame} conf={action.confidence:.0%}')

    elapsed = time.time() - t0
    total = sum(outcomes.values())
    mode = "Ground Truth" if args.ground_truth else f"Predictor ({args.n_input} frames)"
    print(f"\n=== Paddle Test: {mode} ({total} serves, {elapsed:.1f}s) ===")
    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        pct = v / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {k:25s}: {v:4d} ({pct:5.1f}%) {bar}")
    print(f"\n  Return rate: {outcomes.get('success', 0) / total:.1%}")
