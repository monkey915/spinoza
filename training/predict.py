"""Trajectory prediction model: predict future ball positions from partial observations."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from spinoza import SimEnv

TOTAL_FRAMES = 30  # full trajectory length at 60Hz
MIN_INPUT = 6      # minimum observed frames
MAX_INPUT = 25     # maximum observed frames


class TrajectoryDataset(Dataset):
    """Generate variable-length input / ground-truth output pairs from physics sim.
    
    Pre-computes all tensors for fast DataLoader access.
    """

    def __init__(self, n_trajectories: int, difficulty: int, seed: int = 42):
        env = SimEnv(seed=seed, difficulty=difficulty)
        raw = env.generate_trajectories(n_trajectories, difficulty)
        trajs = np.array(raw, dtype=np.float32)  # (N, 30, 3)

        # Pre-compute all samples: each trajectory × each N-value
        n_variants = MAX_INPUT - MIN_INPUT + 1
        total = n_trajectories * n_variants

        self.inputs = torch.zeros(total, TOTAL_FRAMES, 3)
        self.masks = torch.zeros(total, TOTAL_FRAMES)
        self.targets = torch.from_numpy(trajs).repeat_interleave(n_variants, dim=0)
        self.pred_masks = torch.zeros(total, TOTAL_FRAMES)

        for v, n_input in enumerate(range(MIN_INPUT, MAX_INPUT + 1)):
            start = v
            indices = range(start, total, n_variants)
            for i, idx in enumerate(indices):
                self.inputs[idx, :n_input] = self.targets[idx, :n_input]
                self.masks[idx, :n_input] = 1.0
                self.pred_masks[idx, n_input:] = 1.0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'mask': self.masks[idx],
            'target': self.targets[idx],
            'pred_mask': self.pred_masks[idx],
        }


class TrajectoryPredictor(nn.Module):
    """1D-CNN trajectory prediction model with cross-timestep convolutions.

    Uses Conv1d to propagate information from observed frames to predicted frames.
    With kernel_size=7 and 4 residual blocks (8 conv layers total),
    receptive field = 1 + 8*(7-1) = 49 — covers entire 30-frame sequence.

    Input: (batch, 30, 3) padded positions + (batch, 30) mask
    Output: (batch, 30, 3) predicted full trajectory
    """

    def __init__(self, hidden=128, n_layers=4, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2

        # Input: position (3) + mask (1) = 4 channels → hidden channels
        self.input_conv = nn.Conv1d(4, hidden, kernel_size=kernel_size, padding=pad)

        # Learnable positional encoding (added in channel dim)
        self.pos_embed = nn.Parameter(torch.randn(1, hidden, TOTAL_FRAMES) * 0.02)

        # Residual blocks with actual 1D convolutions across time
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(hidden),
                nn.Conv1d(hidden, hidden * 2, kernel_size=kernel_size, padding=pad),
                nn.GELU(),
                nn.Conv1d(hidden * 2, hidden, kernel_size=kernel_size, padding=pad),
            ))

        self.output_conv = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, 3, kernel_size=1),
        )

    def forward(self, x, mask):
        """
        x: (batch, 30, 3) - input positions (zero-padded)
        mask: (batch, 30) - 1 for observed, 0 for padding
        Returns: (batch, 30, 3) - predicted positions for all frames
        """
        # Concatenate mask as extra channel: (batch, 30, 4)
        x = torch.cat([x, mask.unsqueeze(-1)], dim=-1)

        # Transpose to channels-first for Conv1d: (batch, 4, 30)
        x = x.transpose(1, 2)

        x = self.input_conv(x) + self.pos_embed  # (batch, hidden, 30)

        for layer in self.layers:
            x = x + layer(x)  # residual with cross-timestep convolution

        out = self.output_conv(x)  # (batch, 3, 30)
        return out.transpose(1, 2)  # (batch, 30, 3)


def compute_loss(pred, target, pred_mask):
    """Position MSE on predicted frames + velocity consistency regularizer."""
    # Position loss: only on predicted (unobserved) frames
    pos_diff = (pred - target) ** 2  # (batch, 30, 3)
    pos_diff = pos_diff.sum(dim=-1)  # (batch, 30) — squared distance per frame
    mask_sum = pred_mask.sum(dim=-1).clamp(min=1)  # avoid div by zero
    pos_loss = (pos_diff * pred_mask).sum(dim=-1) / mask_sum  # per-sample mean
    pos_loss = pos_loss.mean()

    # Velocity consistency: differences between consecutive predicted frames
    pred_vel = pred[:, 1:] - pred[:, :-1]    # (batch, 29, 3)
    true_vel = target[:, 1:] - target[:, :-1]  # (batch, 29, 3)
    vel_diff = (pred_vel - true_vel) ** 2
    vel_diff = vel_diff.sum(dim=-1)  # (batch, 29)
    vel_mask = pred_mask[:, 1:]  # shifted mask
    vel_mask_sum = vel_mask.sum(dim=-1).clamp(min=1)
    vel_loss = (vel_diff * vel_mask).sum(dim=-1) / vel_mask_sum
    vel_loss = vel_loss.mean()

    return pos_loss + 0.3 * vel_loss, pos_loss, vel_loss


def train(
    n_trajectories=50000,
    difficulty=3,
    epochs=100,
    batch_size=1024,
    lr=1e-3,
    hidden=128,
    n_layers=4,
    device='cpu',
    output='models/predictor.pt',
):
    import os
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 32)))

    print(f"=== Trajectory Prediction Training ===")
    print(f"  Generating {n_trajectories} trajectories (difficulty={difficulty})...")
    dataset = TrajectoryDataset(n_trajectories, difficulty)
    print(f"  Dataset size: {len(dataset)} samples ({n_trajectories} trajectories × {MAX_INPUT - MIN_INPUT + 1} N-values)")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = TrajectoryPredictor(hidden=hidden, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_pos = 0
        total_vel = 0
        n_batches = 0

        for batch in loader:
            inp = batch['input'].to(device)
            mask = batch['mask'].to(device)
            target = batch['target'].to(device)
            pred_mask = batch['pred_mask'].to(device)

            pred = model(inp, mask)
            loss, pos_loss, vel_loss = compute_loss(pred, target, pred_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pos += pos_loss.item()
            total_vel += vel_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        avg_pos = total_pos / n_batches
        avg_vel = total_vel / n_batches

        import time as _time
        if epoch == 0:
            _epoch_start = _time.time()
        if not hasattr(train, '_t0'):
            train._t0 = _time.time()

        elapsed = _time.time() - train._t0
        eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
        eta_h, eta_m = int(eta // 3600), int((eta % 3600) // 60)

        saved = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'hidden': hidden,
                'n_layers': n_layers,
                'kernel_size': 7,
                'best_loss': best_loss,
                'epoch': epoch + 1,
            }, output)
            saved = " ★"

        print(f"  epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.6f} | pos={avg_pos:.6f} | vel={avg_vel:.6f} | lr={scheduler.get_last_lr()[0]:.2e} | ETA {eta_h}h{eta_m:02d}m{saved}")

    print(f"\n  Best loss: {best_loss:.6f}")
    print(f"  Model saved to {output}")
    return model


def evaluate(model_path='models/predictor.pt', difficulty=3, n_test=1000, device='cpu'):
    """Evaluate prediction error vs number of input frames."""
    checkpoint = torch.load(model_path, map_location=device)
    model = TrajectoryPredictor(
        hidden=checkpoint['hidden'],
        n_layers=checkpoint['n_layers'],
        kernel_size=checkpoint.get('kernel_size', 7),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    env = SimEnv(seed=9999, difficulty=difficulty)
    raw = env.generate_trajectories(n_test, difficulty)
    trajectories = np.array(raw, dtype=np.float32)

    print(f"\n=== Evaluation: prediction error vs input frames ===")
    print(f"{'N input':>8} | {'Avg pos err (mm)':>16} | {'Max pos err (mm)':>16} | {'Final frame err (mm)':>20}")
    print("-" * 70)

    for n_input in range(MIN_INPUT, MAX_INPUT + 1):
        input_padded = np.zeros((n_test, TOTAL_FRAMES, 3), dtype=np.float32)
        input_padded[:, :n_input] = trajectories[:, :n_input]
        mask = np.zeros((n_test, TOTAL_FRAMES), dtype=np.float32)
        mask[:, :n_input] = 1.0

        with torch.no_grad():
            pred = model(
                torch.from_numpy(input_padded).to(device),
                torch.from_numpy(mask).to(device),
            ).cpu().numpy()

        # Error only on predicted (future) frames
        errors = np.sqrt(((pred[:, n_input:] - trajectories[:, n_input:]) ** 2).sum(axis=-1))  # (n_test, 30-n_input)
        avg_err = errors.mean() * 1000  # meters → mm
        max_err = errors.max() * 1000
        final_err = np.sqrt(((pred[:, -1] - trajectories[:, -1]) ** 2).sum(axis=-1)).mean() * 1000

        print(f"{n_input:>8} | {avg_err:>16.1f} | {max_err:>16.1f} | {final_err:>20.1f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--n-trajectories', type=int, default=50000)
    parser.add_argument('--difficulty', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--output', default='models/predictor.pt')
    args = parser.parse_args()

    if args.mode == 'train':
        train(
            n_trajectories=args.n_trajectories,
            difficulty=args.difficulty,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden=args.hidden,
            n_layers=args.n_layers,
            output=args.output,
        )
        evaluate(model_path=args.output, difficulty=args.difficulty)
    else:
        evaluate(model_path=args.output, difficulty=args.difficulty)
