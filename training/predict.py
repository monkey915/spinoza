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
    Uses generate_rich_trajectories to get spin ground truth.
    """

    def __init__(self, n_trajectories: int, difficulty: int, seed: int = 42, noise_mm: float = 0.0):
        env = SimEnv(seed=seed, difficulty=difficulty)
        raw = env.generate_rich_trajectories(n_trajectories, difficulty)

        positions = np.array([t['positions'] for t in raw], dtype=np.float32)  # (N, 30, 3)
        # Extract spin from first full_state frame: [x,y,z, vx,vy,vz, ωx,ωy,ωz]
        spins = np.array([[s[6], s[7], s[8]] for t in raw for s in [t['full_states'][0]]], dtype=np.float32)  # (N, 3)

        n_variants = MAX_INPUT - MIN_INPUT + 1
        total = n_trajectories * n_variants

        self.inputs = torch.zeros(total, TOTAL_FRAMES, 3)
        self.masks = torch.zeros(total, TOTAL_FRAMES)
        self.targets = torch.from_numpy(positions).repeat_interleave(n_variants, dim=0)
        self.pred_masks = torch.zeros(total, TOTAL_FRAMES)
        self.spin_targets = torch.from_numpy(spins).repeat_interleave(n_variants, dim=0)  # (total, 3)
        self.noise_std = noise_mm / 1000.0  # mm → meters

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
        inp = self.inputs[idx].clone()
        mask = self.masks[idx]
        # Add measurement noise to observed frames (different each access)
        if self.noise_std > 0:
            n_observed = int(mask.sum().item())
            noise = torch.randn(n_observed, 3) * self.noise_std
            inp[:n_observed] += noise
        return {
            'input': inp,
            'mask': mask,
            'target': self.targets[idx],
            'pred_mask': self.pred_masks[idx],
            'spin': self.spin_targets[idx],
        }


class TrajectoryPredictor(nn.Module):
    """1D-CNN trajectory prediction model with cross-timestep convolutions.

    Uses Conv1d to propagate information from observed frames to predicted frames.
    With kernel_size=7 and 4 residual blocks (8 conv layers total),
    receptive field = 1 + 8*(7-1) = 49 — covers entire 30-frame sequence.

    Input: (batch, 30, 3) padded positions + (batch, 30) mask
    Output: positions (batch, 30, 3) + spin (batch, 3)
    """

    def __init__(self, hidden=128, n_layers=4, kernel_size=7, predict_spin=False):
        super().__init__()
        self.predict_spin = predict_spin
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

        # Auxiliary spin prediction head: global avg pool → MLP → 3 values (ωx, ωy, ωz)
        if predict_spin:
            self.spin_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3),
            )

    def forward(self, x, mask):
        """
        x: (batch, 30, 3) - input positions (zero-padded)
        mask: (batch, 30) - 1 for observed, 0 for padding
        Returns: (batch, 30, 3) positions [, (batch, 3) spin if predict_spin]
        """
        # Concatenate mask as extra channel: (batch, 30, 4)
        x = torch.cat([x, mask.unsqueeze(-1)], dim=-1)

        # Transpose to channels-first for Conv1d: (batch, 4, 30)
        x = x.transpose(1, 2)

        x = self.input_conv(x) + self.pos_embed  # (batch, hidden, 30)

        for layer in self.layers:
            x = x + layer(x)  # residual with cross-timestep convolution

        pos_out = self.output_conv(x)  # (batch, 3, 30)
        pos_out = pos_out.transpose(1, 2)  # (batch, 30, 3)

        if self.predict_spin:
            # Global average pooling over time → (batch, hidden)
            pooled = x.mean(dim=2)
            spin_out = self.spin_head(pooled)  # (batch, 3)
            return pos_out, spin_out

        return pos_out


def compute_loss(pos_pred, target, pred_mask, spin_pred=None, spin_target=None):
    """Position MSE + velocity consistency + optional spin loss."""
    # Position loss: only on predicted (unobserved) frames
    pos_diff = (pos_pred - target) ** 2  # (batch, 30, 3)
    pos_diff = pos_diff.sum(dim=-1)  # (batch, 30) — squared distance per frame
    mask_sum = pred_mask.sum(dim=-1).clamp(min=1)  # avoid div by zero
    pos_loss = (pos_diff * pred_mask).sum(dim=-1) / mask_sum  # per-sample mean
    pos_loss = pos_loss.mean()

    # Velocity consistency: differences between consecutive predicted frames
    pred_vel = pos_pred[:, 1:] - pos_pred[:, :-1]    # (batch, 29, 3)
    true_vel = target[:, 1:] - target[:, :-1]  # (batch, 29, 3)
    vel_diff = (pred_vel - true_vel) ** 2
    vel_diff = vel_diff.sum(dim=-1)  # (batch, 29)
    vel_mask = pred_mask[:, 1:]  # shifted mask
    vel_mask_sum = vel_mask.sum(dim=-1).clamp(min=1)
    vel_loss = (vel_diff * vel_mask).sum(dim=-1) / vel_mask_sum
    vel_loss = vel_loss.mean()

    total = pos_loss + 0.3 * vel_loss

    # Spin loss: MSE on predicted spin vs ground truth
    spin_loss = torch.tensor(0.0)
    if spin_pred is not None and spin_target is not None:
        # Normalize spin by 150 rad/s so it's roughly same scale as position
        spin_loss = ((spin_pred - spin_target / 150.0) ** 2).mean()
        total = total + 0.1 * spin_loss

    return total, pos_loss, vel_loss, spin_loss


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
    load_backbone=None,
    predict_spin=False,
    noise_mm=0.0,
):
    import os
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 32)))

    print(f"=== Trajectory Prediction Training ===")
    print(f"  predict_spin={predict_spin}, noise={noise_mm}mm")
    print(f"  Generating {n_trajectories} trajectories (difficulty={difficulty})...")
    dataset = TrajectoryDataset(n_trajectories, difficulty, noise_mm=noise_mm)
    print(f"  Dataset size: {len(dataset)} samples ({n_trajectories} trajectories × {MAX_INPUT - MIN_INPUT + 1} N-values)")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = TrajectoryPredictor(hidden=hidden, n_layers=n_layers, predict_spin=predict_spin).to(device)

    # Load pretrained backbone weights (ignoring missing spin head keys)
    if load_backbone:
        print(f"  Loading backbone from: {load_backbone}")
        ckpt = torch.load(load_backbone, map_location=device, weights_only=False)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  New parameters (randomly initialized): {missing}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    import time as _time
    t0 = _time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_pos = 0
        total_vel = 0
        total_spin = 0
        n_batches = 0

        for batch in loader:
            inp = batch['input'].to(device)
            mask = batch['mask'].to(device)
            target = batch['target'].to(device)
            pred_mask = batch['pred_mask'].to(device)

            if predict_spin:
                spin_target = batch['spin'].to(device)
                pos_pred, spin_pred = model(inp, mask)
                loss, pos_loss, vel_loss, spin_loss = compute_loss(
                    pos_pred, target, pred_mask, spin_pred, spin_target)
            else:
                pos_pred = model(inp, mask)
                loss, pos_loss, vel_loss, spin_loss = compute_loss(
                    pos_pred, target, pred_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pos += pos_loss.item()
            total_vel += vel_loss.item()
            total_spin += spin_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        avg_pos = total_pos / n_batches
        avg_vel = total_vel / n_batches
        avg_spin = total_spin / n_batches

        elapsed = _time.time() - t0
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
                'predict_spin': predict_spin,
                'best_loss': best_loss,
                'epoch': epoch + 1,
            }, output)
            saved = " ★"

        spin_str = f" | spin={avg_spin:.6f}" if predict_spin else ""
        print(f"  epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.6f} | pos={avg_pos:.6f} | vel={avg_vel:.6f}{spin_str} | lr={scheduler.get_last_lr()[0]:.2e} | ETA {eta_h}h{eta_m:02d}m{saved}")

    print(f"\n  Best loss: {best_loss:.6f}")
    print(f"  Model saved to {output}")
    return model


def evaluate(model_path='models/predictor.pt', difficulty=3, n_test=1000, device='cpu'):
    """Evaluate prediction error vs number of input frames."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    predict_spin = checkpoint.get('predict_spin', False)
    model = TrajectoryPredictor(
        hidden=checkpoint['hidden'],
        n_layers=checkpoint['n_layers'],
        kernel_size=checkpoint.get('kernel_size', 7),
        predict_spin=predict_spin,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    env = SimEnv(seed=9999, difficulty=difficulty)
    raw = env.generate_rich_trajectories(n_test, difficulty)
    trajectories = np.array([t['positions'] for t in raw], dtype=np.float32)
    spins = np.array([[s[6], s[7], s[8]] for t in raw for s in [t['full_states'][0]]], dtype=np.float32)

    print(f"\n=== Evaluation: prediction error vs input frames ===")
    header = f"{'N input':>8} | {'Avg pos err (mm)':>16} | {'Max pos err (mm)':>16} | {'Final frame err (mm)':>20}"
    if predict_spin:
        header += f" | {'Spin err (rad/s)':>16}"
    print(header)
    print("-" * len(header))

    for n_input in range(MIN_INPUT, MAX_INPUT + 1):
        input_padded = np.zeros((n_test, TOTAL_FRAMES, 3), dtype=np.float32)
        input_padded[:, :n_input] = trajectories[:, :n_input]
        mask = np.zeros((n_test, TOTAL_FRAMES), dtype=np.float32)
        mask[:, :n_input] = 1.0

        with torch.no_grad():
            result = model(
                torch.from_numpy(input_padded).to(device),
                torch.from_numpy(mask).to(device),
            )
            if predict_spin:
                pos_pred, spin_pred = result
                pos_pred = pos_pred.cpu().numpy()
                spin_pred = spin_pred.cpu().numpy()
            else:
                pos_pred = result.cpu().numpy()

        # Position error on predicted (future) frames
        errors = np.sqrt(((pos_pred[:, n_input:] - trajectories[:, n_input:]) ** 2).sum(axis=-1))
        avg_err = errors.mean() * 1000  # meters → mm
        max_err = errors.max() * 1000
        final_err = np.sqrt(((pos_pred[:, -1] - trajectories[:, -1]) ** 2).sum(axis=-1)).mean() * 1000

        line = f"{n_input:>8} | {avg_err:>16.1f} | {max_err:>16.1f} | {final_err:>20.1f}"
        if predict_spin:
            # Spin error in rad/s (spin_pred is normalized by 150)
            spin_err = np.sqrt(((spin_pred * 150.0 - spins) ** 2).sum(axis=-1)).mean()
            line += f" | {spin_err:>16.1f}"
        print(line)


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
    parser.add_argument('--predict-spin', action='store_true', help='Add auxiliary spin prediction head')
    parser.add_argument('--load-backbone', type=str, default=None, help='Load pretrained backbone weights')
    parser.add_argument('--noise-mm', type=float, default=0.0, help='Gaussian measurement noise in mm')
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
            predict_spin=args.predict_spin,
            load_backbone=args.load_backbone,
            noise_mm=args.noise_mm,
        )
        evaluate(model_path=args.output, difficulty=args.difficulty)
    else:
        evaluate(model_path=args.output, difficulty=args.difficulty)
