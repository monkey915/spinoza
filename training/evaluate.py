"""Evaluation suite for trained table tennis agents.

Runs multiple evaluation scenarios and reports detailed statistics:
  - Success rate per difficulty stage
  - Outcome breakdown (success, paddle_miss, hit_net, bad_return, etc.)
  - Reward distribution
  - Action statistics (where does the agent place its paddle?)
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from env import TableTennisEnv


def evaluate_stage(model, difficulty, n_episodes=500, seed_base=100000):
    """Evaluate model on a specific difficulty stage."""
    env = TableTennisEnv(seed=seed_base, difficulty=difficulty)
    outcomes = {}
    rewards = []
    actions_all = []

    for i in range(n_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        outcome = info.get("outcome", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        rewards.append(reward)
        actions_all.append(action.copy())

    rewards = np.array(rewards)
    actions_all = np.array(actions_all)

    return {
        "outcomes": outcomes,
        "rewards": rewards,
        "actions": actions_all,
        "n_episodes": n_episodes,
    }


def print_results(results, difficulty):
    """Pretty-print evaluation results."""
    n = results["n_episodes"]
    outcomes = results["outcomes"]
    rewards = results["rewards"]
    actions = results["actions"]

    stage_names = {1: "Stage 1 (no spin)", 2: "Stage 2 (topspin/backspin)", 3: "Stage 3 (full spin)"}
    print(f"\n{'='*60}")
    print(f"  {stage_names.get(difficulty, f'Stage {difficulty}')}  —  {n} episodes")
    print(f"{'='*60}")

    print("\n  Outcomes:")
    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        pct = v / n * 100
        bar = "█" * int(pct / 2)
        print(f"    {k:<15s} {v:>5d} ({pct:>5.1f}%)  {bar}")

    success = outcomes.get("success", 0)
    print(f"\n  Success rate: {success/n*100:.1f}%")

    print(f"\n  Reward statistics:")
    print(f"    mean:   {rewards.mean():.3f}")
    print(f"    std:    {rewards.std():.3f}")
    print(f"    min:    {rewards.min():.3f}")
    print(f"    max:    {rewards.max():.3f}")
    print(f"    median: {np.median(rewards):.3f}")

    action_names = ["paddle_x", "paddle_z", "tilt_x", "tilt_z", "swing_speed", "swing_elev"]
    print(f"\n  Action statistics:")
    print(f"    {'param':<14s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
    print(f"    {'-'*46}")
    for i, name in enumerate(action_names):
        col = actions[:, i]
        print(f"    {name:<14s} {col.mean():>8.3f} {col.std():>8.3f} {col.min():>8.3f} {col.max():>8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained table tennis agent")
    parser.add_argument("model", type=str, help="Path to trained model .zip file")
    parser.add_argument("--stages", type=int, nargs="+", default=[1],
                        help="Difficulty stages to evaluate (1, 2, 3)")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes per stage")
    args = parser.parse_args()

    model = PPO.load(args.model)
    print(f"Loaded model: {args.model}")
    print(f"  Policy params: {sum(p.numel() for p in model.policy.parameters()):,}")

    for stage in args.stages:
        results = evaluate_stage(model, stage, n_episodes=args.episodes)
        print_results(results, stage)

    print()


if __name__ == "__main__":
    main()
