"""Generate prediction comparison JSON for WebUI visualization.

Creates trajectories with different serve characteristics and compares
ground truth vs predictor output at various input frame counts.
"""
import json
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from predict import TrajectoryPredictor, TOTAL_FRAMES
from spinoza import SimEnv


CATEGORIES = {
    "slow_topspin": {"speed": (6, 7.5), "topspin_min": 30, "backspin_max": 10, "sidespin_max": 30},
    "slow_backspin": {"speed": (6, 7.5), "topspin_max": 10, "backspin_min": 30, "sidespin_max": 30},
    "medium_topspin": {"speed": (7.5, 11), "topspin_min": 50, "backspin_max": 10, "sidespin_max": 30},
    "medium_backspin": {"speed": (7, 11), "topspin_max": 10, "backspin_min": 30, "sidespin_max": 40},
    "heavy_sidespin": {"speed": (6, 11), "sidespin_min": 60},
    "fast_flat": {"speed": (7.5, 11), "topspin_max": 40, "backspin_max": 40, "sidespin_max": 40},
}

N_INPUT_FRAMES = [6, 8, 10, 12, 15, 20, 25]
TRAJS_PER_CATEGORY = 10


def matches_category(traj_info, cat_spec):
    """Check if a trajectory matches category constraints."""
    speed = traj_info['serve_speed']
    ts = traj_info['topspin']
    bs = traj_info['backspin']
    ss = abs(traj_info['sidespin'])

    if not (cat_spec['speed'][0] <= speed <= cat_spec['speed'][1]):
        return False
    if 'topspin_min' in cat_spec and ts < cat_spec['topspin_min']:
        return False
    if 'topspin_max' in cat_spec and ts > cat_spec['topspin_max']:
        return False
    if 'backspin_min' in cat_spec and bs < cat_spec['backspin_min']:
        return False
    if 'backspin_max' in cat_spec and bs > cat_spec['backspin_max']:
        return False
    if 'sidespin_min' in cat_spec and ss < cat_spec['sidespin_min']:
        return False
    if 'sidespin_max' in cat_spec and ss > cat_spec['sidespin_max']:
        return False
    return True


def predict_trajectory(model, positions, n_input):
    """Run predictor on partial input, return predicted positions, spin, velocity."""
    input_padded = np.zeros((1, TOTAL_FRAMES, 3), dtype=np.float32)
    mask = np.zeros((1, TOTAL_FRAMES), dtype=np.float32)
    input_padded[0, :n_input] = positions[:n_input]
    mask[0, :n_input] = 1.0

    with torch.no_grad():
        inp = torch.from_numpy(input_padded)
        m = torch.from_numpy(mask)
        pos_pred, spin_pred, vel_pred = model(inp, m)
        pos_np = pos_pred.numpy()[0]
        spin_np = spin_pred.numpy()[0] * 150.0 if spin_pred is not None else None
        vel_np = vel_pred.numpy()[0] * 10.0 if vel_pred is not None else None
        return pos_np, spin_np, vel_np


def to_json_list(arr):
    """Convert numpy array to JSON-safe nested lists of Python floats."""
    if isinstance(arr, np.ndarray):
        return [[round(float(v), 6) for v in row] for row in arr]
    return [[round(float(v), 6) for v in row] for row in arr]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/predictor_spin.pt', help='Model checkpoint')
    args = parser.parse_args()

    print(f"Loading predictor model from {args.model}...")
    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    predict_spin = ckpt.get('predict_spin', False)
    predict_vel = ckpt.get('predict_vel', False)
    model = TrajectoryPredictor(
        hidden=ckpt.get('hidden', 128),
        n_layers=ckpt.get('n_layers', 4),
        kernel_size=ckpt.get('kernel_size', 7),
        predict_spin=predict_spin,
        predict_vel=predict_vel,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  predict_spin={predict_spin}, predict_vel={predict_vel}")

    print("Generating trajectories...")
    env = SimEnv(seed=123, difficulty=3)
    raw = env.generate_rich_trajectories(10000, 3)
    print(f"  Generated {len(raw)} raw trajectories")

    # Debug: show distribution
    speeds = [t['serve_speed'] for t in raw]
    topspins = [t['topspin'] for t in raw]
    backspins = [t['backspin'] for t in raw]
    sidespins = [abs(t['sidespin']) for t in raw]
    print(f"  Speed range: {min(speeds):.1f} - {max(speeds):.1f} m/s")
    print(f"  Topspin range: {min(topspins):.1f} - {max(topspins):.1f} rad/s")
    print(f"  Backspin range: {min(backspins):.1f} - {max(backspins):.1f} rad/s")
    print(f"  |Sidespin| range: {min(sidespins):.1f} - {max(sidespins):.1f} rad/s")

    # Categorize trajectories
    categorized = {cat: [] for cat in CATEGORIES}
    for traj_info in raw:
        for cat_name, cat_spec in CATEGORIES.items():
            if len(categorized[cat_name]) < TRAJS_PER_CATEGORY and matches_category(traj_info, cat_spec):
                categorized[cat_name].append(traj_info)
                break

    output = {"categories": {}, "n_input_options": N_INPUT_FRAMES, "frame_dt": 1.0 / 60.0}

    for cat_name, trajs in categorized.items():
        print(f"  {cat_name}: {len(trajs)} trajectories")
        cat_data = []

        for traj_info in trajs:
            positions = np.array(traj_info['positions'], dtype=np.float32)
            full_states = [[round(float(v), 6) for v in s] for s in traj_info['full_states']]

            predictions = {}
            for n_input in N_INPUT_FRAMES:
                pred_pos, pred_spin, pred_vel = predict_trajectory(model, positions, n_input)
                errors = []
                for i in range(n_input, TOTAL_FRAMES):
                    dx = float(pred_pos[i][0] - positions[i][0])
                    dy = float(pred_pos[i][1] - positions[i][1])
                    dz = float(pred_pos[i][2] - positions[i][2])
                    err_mm = (dx**2 + dy**2 + dz**2)**0.5 * 1000
                    errors.append(round(err_mm, 1))
                pred_entry = {
                    "predicted": to_json_list(pred_pos),
                    "errors_mm": errors,
                    "avg_error_mm": round(sum(errors) / len(errors), 1) if errors else 0,
                    "max_error_mm": round(max(errors), 1) if errors else 0,
                }
                if pred_spin is not None:
                    pred_entry["predicted_spin"] = [round(float(v), 1) for v in pred_spin]
                if pred_vel is not None:
                    pred_entry["predicted_vel"] = to_json_list(pred_vel)
                predictions[str(n_input)] = pred_entry

            entry = {
                "ground_truth": to_json_list(positions),
                "full_states": full_states,
                "serve_speed": round(float(traj_info['serve_speed']), 1),
                "topspin": round(float(traj_info['topspin']), 1),
                "backspin": round(float(traj_info['backspin']), 1),
                "sidespin": round(float(traj_info['sidespin']), 1),
                "predictions": predictions,
            }
            cat_data.append(entry)

        output["categories"][cat_name] = {
            "label": cat_name.replace("_", " ").title(),
            "trajectories": cat_data,
        }

    out_path = "../web/predictions.json"
    with open(out_path, 'w') as f:
        json.dump(output, f)

    size_mb = len(json.dumps(output)) / 1024 / 1024
    total = sum(len(c["trajectories"]) for c in output["categories"].values())
    print(f"\nExported {total} trajectories to {out_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
