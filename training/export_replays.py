"""Export rally replays from a trained model as JSON for web visualization."""

import argparse
import json
import numpy as np
from stable_baselines3 import PPO
from env import TableTennisEnv


def export_replays(model_path, output_path, n_replays=20, difficulty=1, seed=12345):
    """Run the model on random serves and export full replay data."""
    model = PPO.load(model_path)
    env = TableTennisEnv(seed=seed, difficulty=difficulty)

    replays = []
    outcomes_count = {}

    for i in range(n_replays * 3):  # generate extras, keep best mix
        obs, _ = env.reset()
        obs_np = np.array(obs, dtype=np.float32)
        action, _ = model.predict(obs_np, deterministic=True)
        action_list = [float(a) for a in action]

        replay = env.sim.replay(action_list)
        outcome = replay["outcome"]
        outcomes_count[outcome] = outcomes_count.get(outcome, 0) + 1

        # Round floats to reduce JSON size
        def round_traj(traj):
            return [[round(v, 4) for v in pt] for pt in traj]

        entry = {
            "id": i,
            "outcome": outcome,
            "reward": round(replay["reward"], 3),
            "serve_trajectory": round_traj(replay["serve_trajectory"]),
            "serve_bounces": round_traj(replay["serve_bounces"]),
            "return_trajectory": round_traj(replay["return_trajectory"]),
            "return_bounces": round_traj(replay["return_bounces"]),
            "paddle": {k: round(v, 4) for k, v in replay["paddle"].items()},
            "contact_pos": [round(v, 4) for v in replay.get("contact_pos", [0, 0, 0])],
            "landing": [round(v, 4) for v in replay.get("landing", [])],
        }
        if "hit_omega" in replay:
            entry["hit_omega"] = [round(v, 2) for v in replay["hit_omega"]]
        replays.append(entry)

    # Select a good mix: ~60% successes, ~30% misses, ~10% other
    successes = [r for r in replays if r["outcome"] == "success"]
    misses = [r for r in replays if r["outcome"] == "paddle_miss"]
    others = [r for r in replays if r["outcome"] not in ("success", "paddle_miss")]

    n_success = max(1, int(n_replays * 0.6))
    n_miss = max(1, int(n_replays * 0.3))
    n_other = n_replays - n_success - n_miss

    selected = successes[:n_success] + misses[:n_miss] + others[:n_other]
    selected = selected[:n_replays]

    # Re-index
    for i, r in enumerate(selected):
        r["id"] = i

    data = {
        "model": model_path,
        "difficulty": difficulty,
        "n_replays": len(selected),
        "summary": outcomes_count,
        "replays": selected,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_kb = len(json.dumps(data, separators=(",", ":"))) / 1024
    print(f"Exported {len(selected)} replays to {output_path} ({size_kb:.0f} KB)")
    print(f"  Outcomes: {outcomes_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export rally replays as JSON")
    parser.add_argument("model", help="Path to trained model .zip")
    parser.add_argument("-o", "--output", default="../web/replays.json",
                        help="Output JSON path")
    parser.add_argument("-n", "--n-replays", type=int, default=20)
    parser.add_argument("-d", "--difficulty", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=12345)
    args = parser.parse_args()

    export_replays(args.model, args.output, args.n_replays, args.difficulty, args.seed)
