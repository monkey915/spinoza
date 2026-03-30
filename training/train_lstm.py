"""RecurrentPPO (LSTM) training script for table tennis return agent."""

import argparse
import time
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn
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
            success_rate = self.outcomes["success"] / total * 100 if total > 0 else 0
            miss_rate = self.outcomes["paddle_miss"] / total * 100 if total > 0 else 0
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
    print("=== spinoza LSTM Training (RecurrentPPO) ===", flush=True)
    print(f"  n_envs={args.n_envs}, difficulty={args.difficulty}", flush=True)
    print(f"  lstm_hidden={args.lstm_hidden}, total_steps={args.total_timesteps}", flush=True)

    env = SubprocVecEnv([make_env(i, args.difficulty) for i in range(args.n_envs)])

    lr_schedule = (
        get_linear_fn(args.lr, args.lr_final, 1.0)
        if args.lr_final is not None
        else args.lr
    )

    policy_kwargs = {
        "lstm_hidden_size": args.lstm_hidden,
        "n_lstm_layers": args.lstm_layers,
        "shared_lstm": False,
        "enable_critic_lstm": True,
        "net_arch": args.net_arch,
    }

    if args.load:
        print(f"  Lade Modell von {args.load} ...", flush=True)
        model = RecurrentPPO.load(
            args.load,
            env=env,
            learning_rate=lr_schedule,
            ent_coef=args.ent_coef,
            device="cpu",
        )
        # Override n_epochs if set
        model.n_epochs = args.n_epochs
        if args.target_kl is not None:
            model.target_kl = args.target_kl
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="cpu",
        )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  model params: {total_params:,}", flush=True)
    print(flush=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1_000_000 // args.n_envs, 1),
        save_path=args.output + "_checkpoints/",
        name_prefix="ckpt",
        verbose=0,
    )
    callback = ProgressCallback(log_interval=args.log_interval)

    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=[callback, checkpoint_cb])
    elapsed = time.time() - t0

    print()
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Avg throughput: {args.total_timesteps / elapsed:.0f} timesteps/sec")

    model.save(args.output)
    print(f"  Model saved to {args.output}")

    print()
    print("=== Evaluation (100 episodes) ===")
    evaluate_lstm(model, args.difficulty)

    env.close()


def evaluate_lstm(model, difficulty):
    """Run 100 episodes and report success rate."""
    from env import TableTennisEnv

    env = TableTennisEnv(seed=99999, difficulty=difficulty)
    outcomes = {}
    total_reward = 0

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    for _ in range(100):
        obs, _ = env.reset()
        obs_np = obs[np.newaxis]  # add batch dim
        action, lstm_states = model.predict(
            obs_np, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        episode_starts = np.zeros((1,), dtype=bool)
        obs, reward, done, _, info = env.step(action[0])
        outcome = info.get("outcome", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        total_reward += reward
        if done:
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)

    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}%")
    print(f"  avg reward: {total_reward / 100:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train table tennis return agent (LSTM)")
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-final", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--log-interval", type=int, default=10000)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--output", type=str, default="models/lstm_stage1")
    args = parser.parse_args()

    train(args)
