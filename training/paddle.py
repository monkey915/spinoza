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
    target_y: float = None,
    target_landing: tuple = None,
) -> PaddleAction:
    """Compute optimal paddle action from predicted trajectory.
    
    Args:
        positions: (30, 3) predicted ball positions [x, y, z] in meters
        velocity: (30, 3) predicted velocities per frame, or None
        spin: (3,) predicted spin [ωx, ωy, ωz] in rad/s, or None
        target_y: Y position to intercept (default: auto from trajectory)
        target_landing: (x, y) desired landing spot on opponent's side, or None
    
    Returns:
        PaddleAction with recommended paddle parameters
    """
    n_frames = len(positions)
    
    # Estimate velocity from positions if not provided (finite differences)
    if velocity is None:
        dt = 1.0 / 60.0
        velocity = np.zeros_like(positions)
        velocity[1:] = (positions[1:] - positions[:-1]) / dt
        velocity[0] = velocity[1]
    
    # Default spin: zero
    if spin is None:
        spin = np.zeros(3)
    
    # Find best intercept frame: ball should be at a comfortable height
    # and moving toward the player (positive Y for serves, negative Y for returns)
    best_frame = None
    best_score = -1e9
    
    for i in range(5, n_frames):
        x, y, z = positions[i]
        vx, vy, vz = velocity[i] if i < len(velocity) else velocity[-1]
        
        # Must be reachable height
        if z < TABLE_Z + 0.05 or z > 1.5:
            continue
        # Must be in reasonable Y range
        if y < 1.8 or y > 3.5:
            continue
        
        # Score: prefer ball near apex (vz ≈ 0), comfortable height, reachable
        height_score = -abs(z - 0.95) * 2  # prefer ~0.95m height
        apex_score = -abs(vz) * 0.5         # prefer near apex
        timing_score = -i * 0.01            # slight preference for earlier contact
        
        score = height_score + apex_score + timing_score
        if score > best_score:
            best_score = score
            best_frame = i
    
    # Fallback: use frame where ball crosses Y = 2.5m
    if best_frame is None:
        for i in range(n_frames - 1):
            if positions[i][1] < 2.5 and positions[i + 1][1] >= 2.5:
                best_frame = i + 1
                break
        if best_frame is None:
            best_frame = n_frames // 2
    
    # Ball state at intercept
    bx, by, bz = positions[best_frame]
    bvx, bvy, bvz = velocity[best_frame] if best_frame < len(velocity) else velocity[-1]
    
    # Paddle position: at the ball
    paddle_x = float(np.clip(bx, 0.1, TABLE_WIDTH - 0.1))
    paddle_y = float(np.clip(by, 1.8, 3.5))
    paddle_z = float(max(bz, TABLE_Z + 0.09))
    
    # Compute paddle tilt based on spin
    # Topspin (ωx < 0, i.e. forward spin on incoming ball) → open face (tilt_x < 0)
    # Backspin (ωx > 0) → close face (tilt_x > 0)
    omega_x = spin[0]  # from predictor
    
    # Base tilt: slight upward angle to clear the net
    base_tilt_x = 0.1  # slightly closed
    
    # Spin compensation
    spin_compensation = 0.0
    if abs(omega_x) > 10:
        # Topspin on incoming ball → ball kicks up on paddle → close face more
        # Backspin on incoming ball → ball drops → open face
        spin_compensation = -omega_x / 300.0  # gentle compensation
        spin_compensation = np.clip(spin_compensation, -0.3, 0.3)
    
    tilt_x = float(base_tilt_x + spin_compensation)
    
    # Sidespin compensation
    omega_z = spin[2]
    tilt_z = float(np.clip(omega_z / 500.0, -0.2, 0.2))
    
    # Swing speed: based on desired return speed
    # Faster incoming ball → less swing needed (ball already has energy)
    incoming_speed = np.sqrt(bvx**2 + bvy**2 + bvz**2)
    swing_speed = float(np.clip(15.0 - incoming_speed * 0.3, 8.0, 22.0))
    
    # Swing elevation: aim upward to clear net
    # Calculate needed vertical angle to clear net from contact point
    dy_to_net = abs(paddle_y - NET_Y)
    dz_to_net = NET_TOP_Z + 0.03 - paddle_z  # need to clear by 3cm
    
    if dy_to_net > 0.1:
        needed_angle = np.arctan2(max(dz_to_net, 0), dy_to_net)
        swing_elevation = float(np.clip(needed_angle + 0.05, 0.05, 0.5))
    else:
        swing_elevation = 0.2  # default
    
    # Confidence: higher if we have good data
    confidence = 0.8
    if velocity is None:
        confidence -= 0.2
    if spin is None or np.linalg.norm(spin) < 1:
        confidence -= 0.1
    if best_frame > 25:
        confidence -= 0.2  # late intercept = less certain
    
    return PaddleAction(
        paddle_x=paddle_x,
        paddle_y=paddle_y,
        paddle_z=paddle_z,
        tilt_x=tilt_x,
        tilt_z=tilt_z,
        swing_speed=swing_speed,
        swing_elevation=swing_elevation,
        intercept_frame=best_frame,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


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
    # Demo: compute paddle action from a simulated trajectory
    from spinoza import SimEnv
    import torch
    from predict import TrajectoryPredictor, TOTAL_FRAMES

    # Load predictor
    ckpt = torch.load('models/predictor_full.pt', map_location='cpu', weights_only=False)
    model = TrajectoryPredictor(
        hidden=ckpt['hidden'], n_layers=ckpt['n_layers'],
        kernel_size=ckpt.get('kernel_size', 7),
        predict_spin=ckpt.get('predict_spin', False),
        predict_vel=ckpt.get('predict_vel', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Generate a test serve
    env = SimEnv(seed=42, difficulty=3)
    trajs = env.generate_rich_trajectories(1, 3)
    t = trajs[0]
    positions = np.array(t['positions'], dtype=np.float32)

    # Predict from 15 observed frames
    n_input = 15
    inp = np.zeros((1, TOTAL_FRAMES, 3), dtype=np.float32)
    inp[0, :n_input] = positions[:n_input]
    mask = np.zeros((1, TOTAL_FRAMES), dtype=np.float32)
    mask[0, :n_input] = 1.0

    with torch.no_grad():
        pos_pred, spin_pred, vel_pred = model(
            torch.from_numpy(inp), torch.from_numpy(mask)
        )
        pred_pos = pos_pred.numpy()[0]
        pred_spin = spin_pred.numpy()[0] * 150.0 if spin_pred is not None else None
        pred_vel = vel_pred.numpy()[0] * 10.0 if vel_pred is not None else None

    print(f"Serve: {t['serve_speed']:.1f} m/s, topspin={t['topspin']:.0f} rad/s")
    print(f"Predicted spin: [{pred_spin[0]:.1f}, {pred_spin[1]:.1f}, {pred_spin[2]:.1f}] rad/s")
    print()

    action = compute_paddle_action(pred_pos, pred_vel, pred_spin)
    evaluate_action(action, pred_pos, pred_vel)
