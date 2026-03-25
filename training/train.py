"""PPO training script for table tennis return agent."""

import argparse
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env import make_env


class ProgressCallback(BaseCallback):
    """Log training progress every N steps."""

    def __init__(self, log_interval=5000, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = None
        self.outcomes = {"success": 0, "paddle_miss": 0, "other": 0}
        self._next_log = log_interval

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            outcome = info.get("outcome", "other")
            if outcome == "success":
                self.outcomes["success"] += 1
            elif outcome == "paddle_miss":
                self.outcomes["paddle_miss"] += 1
            else:
                self.outcomes["other"] += 1

        if self.num_timesteps >= self._next_log:
            self._next_log = self.num_timesteps + self.log_interval
            elapsed = time.time() - self.start_time
            total = sum(self.outcomes.values())
            if total > 0:
                success_rate = self.outcomes["success"] / total * 100
                miss_rate = self.outcomes["paddle_miss"] / total * 100
            else:
                success_rate = miss_rate = 0

            eps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            print(
                f"  step={self.num_timesteps:>8d} | "
                f"eps/s={eps_per_sec:>6.0f} | "
                f"success={success_rate:>5.1f}% | "
                f"miss={miss_rate:>5.1f}% | "
                f"elapsed={elapsed:>5.1f}s",
                flush=True
            )
            self.outcomes = {"success": 0, "paddle_miss": 0, "other": 0}

        return True


def train(args):
    print(f"=== spinoza RL Training ===", flush=True)
    print(f"  n_envs={args.n_envs}, difficulty={args.difficulty}", flush=True)
    print(f"  total_timesteps={args.total_timesteps}", flush=True)
    print(f"  policy=MLP {args.net_arch}", flush=True)
    print(flush=True)

    env = SubprocVecEnv(
        [make_env(seed=i, difficulty=args.difficulty) for i in range(args.n_envs)]
    )

    if args.load:
        print(f"  Loading pretrained model: {args.load}")
        model = PPO.load(args.load, env=env, device="cpu")
        # Allow overriding lr and ent_coef for fine-tuning
        model.learning_rate = args.lr
        model.ent_coef = args.ent_coef
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            ent_coef=args.ent_coef,
            policy_kwargs={"net_arch": args.net_arch},
            verbose=0,
            device="cpu",
        )

    print(f"  model params: {sum(p.numel() for p in model.policy.parameters()):,}", flush=True)
    print(flush=True)

    callback = ProgressCallback(log_interval=args.log_interval)
    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    elapsed = time.time() - t0

    print()
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Avg throughput: {args.total_timesteps / elapsed:.0f} timesteps/sec")

    model.save(args.output)
    print(f"  Model saved to {args.output}")

    print()
    print("=== Evaluation (100 episodes) ===")
    evaluate(model, args.difficulty)

    env.close()


def evaluate(model, difficulty):
    """Run 100 episodes and report success rate."""
    from env import TableTennisEnv

    env = TableTennisEnv(seed=99999, difficulty=difficulty)
    outcomes = {}
    total_reward = 0

    for _ in range(100):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        outcome = info.get("outcome", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        total_reward += reward

    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}%")
    print(f"  avg reward: {total_reward / 100:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train table tennis return agent")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--log-interval", type=int, default=10000)
    parser.add_argument("--load", type=str, default=None,
                        help="Load pretrained model for fine-tuning (curriculum)")
    parser.add_argument("--output", type=str, default="models/ppo_stage1")
    args = parser.parse_args()

    train(args)
