"""Generate replay JSON for the WebUI from paddle optimizer results.

Usage:
    python generate_replays.py [--n-serves 50] [--model models/predictor_rally.pt] [--output ../web/replays_paddle.json]
"""
import numpy as np
import json
import time
import argparse

from spinoza import SimEnv
from paddle import compute_paddle_action

def generate_replays(env, n_serves, model=None, n_input=15):
    """Generate replay data for WebUI visualization.
    
    Args:
        env: SimEnv instance
        n_serves: Number of serves to generate
        model: TrajectoryPredictor model (None = ground truth)
        n_input: Number of observed frames for predictor
    
    Returns:
        List of replay dicts in WebUI format
    """
    import torch
    from predict import TOTAL_FRAMES

    replays = []
    stats = {}

    for i in range(n_serves * 3):  # oversample, skip failed
        if len(replays) >= n_serves:
            break

        trajs = env.generate_rich_trajectories(1, 3)
        t = trajs[0]
        gt_pos = np.array(t['positions'], dtype=np.float32)
        gt_full = [list(s) for s in t['full_states']]
        gt_spin = np.array([gt_full[0][6], gt_full[0][7], gt_full[0][8]])
        hires_serve = t.get('serve_trajectory_hires', [])

        if model is None:
            # Ground truth
            dt_f = 1.0 / 60.0
            vel = np.zeros_like(gt_pos)
            vel[1:] = (gt_pos[1:] - gt_pos[:-1]) / dt_f
            vel[0] = vel[1]
            action = compute_paddle_action(gt_pos, vel, gt_spin,
                                           full_states=gt_full, env=env)
        else:
            # Predictor
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

            merged_pos = pred_pos.copy()
            merged_pos[:n_input] = gt_pos[:n_input]
            merged_vel = pred_vel.copy() if pred_vel is not None else None
            if merged_vel is not None:
                dt_f = 1.0 / 60.0
                merged_vel[:n_input] = np.gradient(gt_pos[:n_input], dt_f, axis=0)

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

        # Simulate the return using ground truth ball state
        ball_state = gt_full[action.intercept_frame]
        action_vec = [action.paddle_x, action.paddle_y, action.paddle_z,
                      action.tilt_x, action.tilt_z,
                      action.swing_speed, action.swing_elevation]
        result = env.simulate_hit(ball_state, action_vec)
        outcome = result['outcome']
        stats[outcome] = stats.get(outcome, 0) + 1

        # Build serve trajectory in WebUI format: [t, x, y, z, vx, vy, vz, ox, oy, oz]
        dt_frame = 1.0 / 60.0
        serve_traj = []
        for j, s in enumerate(hires_serve):
            serve_traj.append([j * dt_frame] + [float(x) for x in s])

        # Detect serve bounces from trajectory (z reaches table height and bounces up)
        table_z = 0.76
        serve_bounces = []
        for j in range(1, len(hires_serve) - 1):
            z_prev = hires_serve[j-1][2]
            z_curr = hires_serve[j][2]
            z_next = hires_serve[j+1][2]
            if z_curr <= table_z + 0.005 and z_next > z_curr and z_prev > z_curr:
                serve_bounces.append([j * dt_frame, float(hires_serve[j][0]),
                                      float(hires_serve[j][1]), table_z])

        # Return trajectory from simulate_hit
        return_traj = [[float(x) for x in pt] for pt in result.get('return_trajectory', [])]
        return_bounces = [[float(x) for x in b] for b in result.get('return_bounces', [])]

        # Contact pos and hit omega
        contact_pos = [float(x) for x in result.get('contact_pos', [0, 0, 0])] if result.get('contact_pos') else None
        hit_omega = [float(x) for x in result.get('hit_omega', [0, 0, 0])] if result.get('hit_omega') else None

        # Landing position
        landing = None
        if outcome == 'success':
            landing = [result['landing_x'], result['landing_y']]

        replay = {
            'serve_trajectory': serve_traj,
            'serve_bounces': serve_bounces,
            'return_trajectory': return_traj,
            'return_bounces': return_bounces,
            'paddle': {
                'paddle_x': action.paddle_x,
                'paddle_y': action.paddle_y,
                'paddle_z': action.paddle_z,
                'tilt_x': action.tilt_x,
                'tilt_z': action.tilt_z,
                'swing_speed': action.swing_speed,
                'swing_elevation': action.swing_elevation,
            },
            'contact_pos': contact_pos,
            'hit_omega': hit_omega,
            'landing': landing,
            'outcome': outcome,
            'reward': 1.0 if outcome == 'success' else -1.0,
            'serve_speed': float(t['serve_speed']),
        }
        replays.append(replay)

    return replays, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-serves', type=int, default=50)
    parser.add_argument('--model', default='models/predictor_rally.pt')
    parser.add_argument('--ground-truth', action='store_true')
    parser.add_argument('--output', default='../web/replays_paddle.json')
    parser.add_argument('--n-input', type=int, default=15)
    args = parser.parse_args()

    env = SimEnv(seed=123, difficulty=3)

    model = None
    if not args.ground_truth:
        import torch
        from predict import TrajectoryPredictor
        ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
        model = TrajectoryPredictor(
            hidden=ckpt['hidden'], n_layers=ckpt['n_layers'],
            kernel_size=ckpt.get('kernel_size', 7),
            predict_spin=ckpt.get('predict_spin', False),
            predict_vel=ckpt.get('predict_vel', False),
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

    t0 = time.time()
    replays, stats = generate_replays(env, args.n_serves, model=model, n_input=args.n_input)
    elapsed = time.time() - t0

    data = {'replays': replays}
    with open(args.output, 'w') as f:
        json.dump(data, f)

    total = sum(stats.values())
    mode = "Ground Truth" if args.ground_truth else f"Predictor ({args.n_input} frames)"
    print(f"Generated {len(replays)} replays ({mode}) in {elapsed:.1f}s → {args.output}")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:4d} ({v/total:.0%})")
    print(f"  Return rate: {stats.get('success', 0) / total:.0%}")
