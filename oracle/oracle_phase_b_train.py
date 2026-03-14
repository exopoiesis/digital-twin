#!/usr/bin/env python3
"""
ORACLE Phase B: Fourier Neural Operator for Membrane PDE Surrogate

Trains an FNO (Fourier Neural Operator) to predict pH profiles and integrated
scalars (J_formate, I_current, tau_transit, A_steady) from 7 membrane parameters.

Input: NPZ file with PDE simulation results
  - samples: (N, 7) membrane parameters
    Order: L_pent, L_mack, L_chamber, delta_pH, D_H_pent, D_H_mack_intra, k_cat
  - ph_profiles: (N, 256) pH(x) on uniform grid
  - scalars: (N, 4) integrated quantities [J_formate, I_current, tau_transit, A_steady]
  - x_grid: (256,) spatial coordinates
  - success: (N,) bool mask for converged simulations

Architecture:
  - FNO1d with 4 Fourier layers (64 hidden channels, 32 modes)
  - Branch: 7 scalar parameters lifted to spatial domain
  - Trunk: x_grid (256 points)
  - Output: pH(x) profile + 4 scalars via MLP head

Training:
  - 80/10/10 train/val/test split
  - Combined loss: MSE(pH) + λ*MSE(scalars)
  - Adam optimizer with cosine annealing
  - Early stopping (patience=20)

Author: Third Matter Research Project
Date: 2026-03-10
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# GPU SETUP AND VRAM CHECK
# ============================================================================

def setup_device():
    """Setup computation device and report VRAM."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / 1024**3
        print(f"Using GPU: {props.name}")
        print(f"Total VRAM: {total_vram_gb:.2f} GB")

        # Check available memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"VRAM allocated: {allocated:.3f} GB, reserved: {reserved:.3f} GB")

        return device, total_vram_gb
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu'), 0.0


# ============================================================================
# FOURIER NEURAL OPERATOR ARCHITECTURE
# ============================================================================

class SpectralConv1d(nn.Module):
    """
    1D Fourier layer: FFT → multiply with learnable weights → IFFT.

    Keeps only the first `modes` Fourier modes.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Learnable weights for Fourier modes (complex-valued)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, n_grid)
        Returns:
            (batch, out_channels, n_grid)
        """
        batch_size = x.shape[0]

        # FFT along spatial dimension
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, n_grid//2 + 1)

        # Multiply relevant modes
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.shape[-1],
                            dtype=torch.cfloat, device=x.device)

        # Only multiply first `modes` modes
        modes_to_use = min(self.modes, x_ft.shape[-1])
        weights_complex = torch.view_as_complex(self.weights[:, :, :modes_to_use, :])  # (in, out, modes)

        # Einstein summation: batch, input_channel, modes → batch, output_channel, modes
        out_ft[:, :, :modes_to_use] = torch.einsum('bix,iox->box', x_ft[:, :, :modes_to_use], weights_complex)

        # IFFT back to spatial domain
        x_out = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)

        return x_out


class FNO1d(nn.Module):
    """
    Fourier Neural Operator for 1D problems.

    Args:
        in_channels: Number of input channels (params + x coordinate)
        out_channels: Number of output channels (pH profile)
        modes: Number of Fourier modes to keep
        width: Hidden channel dimension
        n_layers: Number of Fourier layers
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 32,
        width: int = 64,
        n_layers: int = 4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        # Lift input to hidden dimension
        self.lift = nn.Linear(in_channels, width)

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])

        # Local (non-Fourier) convolutions for skip connections
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)
        ])

        # Project back to output dimension
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, n_grid)
        Returns:
            (batch, out_channels, n_grid)
        """
        # Lift: (batch, in_channels, n_grid) → (batch, width, n_grid)
        x = x.permute(0, 2, 1)  # (batch, n_grid, in_channels)
        x = self.lift(x)  # (batch, n_grid, width)
        x = x.permute(0, 2, 1)  # (batch, width, n_grid)

        # Fourier layers with skip connections
        for fourier, conv in zip(self.fourier_layers, self.conv_layers):
            x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)

        # Project: (batch, width, n_grid) → (batch, out_channels, n_grid)
        x = x.permute(0, 2, 1)  # (batch, n_grid, width)
        x = self.project(x)  # (batch, n_grid, out_channels)
        x = x.permute(0, 2, 1)  # (batch, out_channels, n_grid)

        return x


class MembraneModel(nn.Module):
    """
    Full ORACLE Phase B model: FNO for pH profile + MLP head for scalars.

    Args:
        n_params: Number of membrane parameters (7)
        n_grid: Spatial grid size (256)
        n_scalars: Number of scalar outputs (4: J, I, tau, A_steady)
        modes: Fourier modes
        width: Hidden dimension
        n_layers: Number of Fourier layers
    """
    def __init__(
        self,
        n_params: int = 7,
        n_grid: int = 256,
        n_scalars: int = 4,
        modes: int = 32,
        width: int = 64,
        n_layers: int = 4
    ):
        super().__init__()

        self.n_params = n_params
        self.n_grid = n_grid
        self.n_scalars = n_scalars

        # Input: n_params (lifted to spatial) + 1 (x coordinate)
        self.fno = FNO1d(
            in_channels=n_params + 1,
            out_channels=1,  # pH(x)
            modes=modes,
            width=width,
            n_layers=n_layers
        )

        # MLP head for scalars (from last hidden state)
        self.scalar_head = nn.Sequential(
            nn.Linear(width + n_params, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_scalars)
        )

        # Save width for scalar head
        self.width = width

    def forward(self, params: torch.Tensor, x_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            params: (batch, n_params) membrane parameters
            x_grid: (batch, n_grid) or (n_grid,) spatial coordinates

        Returns:
            ph_profile: (batch, n_grid) pH(x) predictions
            scalars: (batch, n_scalars) [J, I, tau, A_steady]
        """
        batch_size = params.shape[0]

        # Expand x_grid if needed
        if x_grid.dim() == 1:
            x_grid = x_grid.unsqueeze(0).expand(batch_size, -1)  # (batch, n_grid)

        # Lift params to spatial domain by repeating
        params_spatial = params.unsqueeze(-1).expand(-1, -1, self.n_grid)  # (batch, n_params, n_grid)

        # Concatenate with x_grid
        x_input = torch.cat([params_spatial, x_grid.unsqueeze(1)], dim=1)  # (batch, n_params+1, n_grid)

        # FNO forward (get hidden state for scalar head)
        # We need to extract hidden state before final projection
        x = x_input.permute(0, 2, 1)  # (batch, n_grid, n_params+1)
        x = self.fno.lift(x)  # (batch, n_grid, width)
        x = x.permute(0, 2, 1)  # (batch, width, n_grid)

        for fourier, conv in zip(self.fno.fourier_layers, self.fno.conv_layers):
            x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)

        # x: (batch, width, n_grid) — hidden representation

        # Project to pH
        x_for_ph = x.permute(0, 2, 1)  # (batch, n_grid, width)
        ph_profile = self.fno.project(x_for_ph).squeeze(-1)  # (batch, n_grid)

        # Extract global feature for scalars (mean pooling over space)
        x_pooled = x.mean(dim=-1)  # (batch, width)

        # Concatenate with original params for scalar prediction
        x_scalar = torch.cat([x_pooled, params], dim=1)  # (batch, width + n_params)
        scalars = self.scalar_head(x_scalar)  # (batch, n_scalars)

        return ph_profile, scalars


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def _safe_standard_scaler_fit(scaler: StandardScaler):
    """Replace zero-variance features' scale with 1.0 to avoid NaN on transform."""
    zero_var = scaler.scale_ == 0.0
    if np.any(zero_var):
        n_const = int(zero_var.sum())
        print(f"  [WARN] {n_const} constant feature(s) detected (std=0), setting scale=1.0")
        scaler.scale_[zero_var] = 1.0


# Param order (from datagen PARAM_DEFS):
#   L_pent(0), L_mack(1), L_chamber(2), delta_pH(3),
#   D_H_pent(4), D_H_mack_intra(5), k_cat(6)
LOG10_PARAM_COLS = [4, 5, 6]  # D_H_pent, D_H_mack_intra, k_cat — log-sampled, extreme range

# Scalar order: J_formate(0), I_current(1), tau_transit(2), A_steady(3)
# NOTE: I_current is ALL zeros in current dataset — handled by safe scaler (scale=1)
LOG1P_SCALAR_COLS = [0, 2]  # J_formate, tau_transit — extreme dynamic range


def load_and_preprocess_data(
    npz_path: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dict, StandardScaler, MinMaxScaler, StandardScaler]:
    """
    Load NPZ data and split into train/val/test.

    Applies log-transforms to extreme-range features before scaling:
      - params: log10 for D_H_pent, k_cat_alk
      - scalars: log1p for J_formate, tau_transit
      - I_current (all zeros): handled via safe scaler (std=0 → scale=1)

    Returns:
        data_splits: dict with 'train', 'val', 'test' keys
        scaler_params: fitted StandardScaler for params (after log-transform)
        scaler_ph: fitted MinMaxScaler for pH profiles
        scaler_scalars: fitted StandardScaler for scalars (after log1p-transform)
    """
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)

    params_all = data['samples'] if 'samples' in data else data['params']  # (N, 7)
    ph_all = data['ph_profiles']  # (N, 256)
    scalars_all = data['scalars']  # (N, 4)
    x_grid = data['x_grid']  # (256,)

    # Filter out failed simulations (NaN-containing rows)
    if 'success' in data:
        ok = data['success'].astype(bool)
        n_fail = int((~ok).sum())
        print(f"  Total samples: {len(ok)}, failed: {n_fail} → filtering to {ok.sum()} successful")
        params = params_all[ok]
        ph_profiles = ph_all[ok]
        scalars = scalars_all[ok]
    else:
        # Fallback: drop rows with any NaN in ph or scalars
        ok = ~(np.isnan(ph_all).any(axis=1) | np.isnan(scalars_all).any(axis=1))
        n_fail = int((~ok).sum())
        print(f"  Total samples: {len(ok)}, NaN rows: {n_fail} → filtering to {ok.sum()}")
        params = params_all[ok]
        ph_profiles = ph_all[ok]
        scalars = scalars_all[ok]

    N = params.shape[0]
    print(f"  Usable samples: {N}")
    print(f"  Params shape: {params.shape}")
    print(f"  pH profiles shape: {ph_profiles.shape}")
    print(f"  Scalars shape: {scalars.shape}")
    print(f"  x_grid shape: {x_grid.shape}")

    # --- Log-transform extreme-range features BEFORE scaling ---
    params = params.copy()
    scalars = scalars.copy()

    for col in LOG10_PARAM_COLS:
        vmin, vmax = params[:, col].min(), params[:, col].max()
        print(f"  Param col {col}: range [{vmin:.3e}, {vmax:.3e}] → log10")
        params[:, col] = np.log10(np.clip(params[:, col], 1e-30, None))

    for col in LOG1P_SCALAR_COLS:
        vmin, vmax = scalars[:, col].min(), scalars[:, col].max()
        print(f"  Scalar col {col}: range [{vmin:.3e}, {vmax:.3e}] → log1p")
        scalars[:, col] = np.log1p(scalars[:, col])

    # Report constant columns
    for col in range(scalars.shape[1]):
        if scalars[:, col].std() == 0:
            print(f"  Scalar col {col}: CONSTANT (all {scalars[0, col]:.3e}), will be kept as-is")

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(N)

    n_train = int(train_split * N)
    n_val = int(val_split * N)

    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    # Fit scalers on training data
    scaler_params = StandardScaler()
    scaler_ph = MinMaxScaler(feature_range=(0, 1))
    scaler_scalars = StandardScaler()

    scaler_params.fit(params[idx_train])
    _safe_standard_scaler_fit(scaler_params)

    scaler_ph.fit(ph_profiles[idx_train])

    scaler_scalars.fit(scalars[idx_train])
    _safe_standard_scaler_fit(scaler_scalars)

    # Transform all splits
    params_scaled = scaler_params.transform(params)
    ph_scaled = scaler_ph.transform(ph_profiles)
    scalars_scaled = scaler_scalars.transform(scalars)

    # Verify no NaN after scaling
    for name, arr in [('params', params_scaled), ('ph', ph_scaled), ('scalars', scalars_scaled)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            print(f"  [ERROR] {n_nan} NaN values in {name} after scaling!")
        else:
            print(f"  {name}: OK (no NaN)")

    # Normalize x_grid to [0, 1]
    x_grid_norm = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min())

    data_splits = {
        'train': {
            'params': params_scaled[idx_train],
            'ph': ph_scaled[idx_train],
            'scalars': scalars_scaled[idx_train],
            'x_grid': x_grid_norm,
        },
        'val': {
            'params': params_scaled[idx_val],
            'ph': ph_scaled[idx_val],
            'scalars': scalars_scaled[idx_val],
            'x_grid': x_grid_norm,
        },
        'test': {
            'params': params_scaled[idx_test],
            'ph': ph_scaled[idx_test],
            'scalars': scalars_scaled[idx_test],
            'x_grid': x_grid_norm,
        }
    }

    return data_splits, scaler_params, scaler_ph, scaler_scalars


def create_dataloaders(
    data_splits: Dict,
    batch_size: int,
    device: torch.device
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders."""

    def to_loader(split_name: str) -> DataLoader:
        split = data_splits[split_name]

        params_t = torch.tensor(split['params'], dtype=torch.float32, device=device)
        ph_t = torch.tensor(split['ph'], dtype=torch.float32, device=device)
        scalars_t = torch.tensor(split['scalars'], dtype=torch.float32, device=device)
        x_grid_t = torch.tensor(split['x_grid'], dtype=torch.float32, device=device)

        # Dataset: (params, x_grid, ph, scalars)
        # x_grid is same for all samples, but we include it for consistency
        dataset = TensorDataset(
            params_t,
            x_grid_t.unsqueeze(0).expand(len(params_t), -1),
            ph_t,
            scalars_t
        )

        shuffle = (split_name == 'train')
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = to_loader('train')
    val_loader = to_loader('val')
    test_loader = to_loader('test')

    return train_loader, val_loader, test_loader


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_scalar: float,
    device: torch.device
) -> Tuple[float, float, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_ph_loss = 0.0
    total_scalar_loss = 0.0
    n_batches = 0

    for params, x_grid, ph_true, scalars_true in loader:
        optimizer.zero_grad()

        # Forward pass
        ph_pred, scalars_pred = model(params, x_grid)

        # Loss
        loss_ph = F.mse_loss(ph_pred, ph_true)
        loss_scalar = F.mse_loss(scalars_pred, scalars_true)
        loss = loss_ph + lambda_scalar * loss_scalar

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ph_loss += loss_ph.item()
        total_scalar_loss += loss_scalar.item()
        n_batches += 1

    return total_loss / n_batches, total_ph_loss / n_batches, total_scalar_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    lambda_scalar: float,
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluate on validation/test set."""
    model.eval()

    total_loss = 0.0
    total_ph_loss = 0.0
    total_scalar_loss = 0.0
    n_batches = 0

    for params, x_grid, ph_true, scalars_true in loader:
        ph_pred, scalars_pred = model(params, x_grid)

        loss_ph = F.mse_loss(ph_pred, ph_true)
        loss_scalar = F.mse_loss(scalars_pred, scalars_true)
        loss = loss_ph + lambda_scalar * loss_scalar

        total_loss += loss.item()
        total_ph_loss += loss_ph.item()
        total_scalar_loss += loss_scalar.item()
        n_batches += 1

    return total_loss / n_batches, total_ph_loss / n_batches, total_scalar_loss / n_batches


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_scalar: float,
    patience: int,
    output_dir: Path,
    log_dir: Path
) -> Dict:
    """Train FNO model with early stopping."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_ph_loss': [],
        'train_scalar_loss': [],
        'val_loss': [],
        'val_ph_loss': [],
        'val_scalar_loss': [],
    }

    print("\nStarting training...")
    print(f"Epochs: {epochs}, LR: {lr}, λ_scalar: {lambda_scalar}, Patience: {patience}")
    print("=" * 80)

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_ph_loss, train_scalar_loss = train_epoch(
            model, train_loader, optimizer, lambda_scalar, device
        )

        # Validate
        val_loss, val_ph_loss, val_scalar_loss = evaluate(
            model, val_loader, lambda_scalar, device
        )

        # Scheduler step
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_ph_loss'].append(train_ph_loss)
        history['train_scalar_loss'].append(train_scalar_loss)
        history['val_loss'].append(val_loss)
        history['val_ph_loss'].append(val_ph_loss)
        history['val_scalar_loss'].append(val_scalar_loss)

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss_pH/train', train_ph_loss, epoch)
        writer.add_scalar('Loss_pH/val', val_ph_loss, epoch)
        writer.add_scalar('Loss_Scalars/train', train_scalar_loss, epoch)
        writer.add_scalar('Loss_Scalars/val', val_scalar_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} (pH: {train_ph_loss:.6f}, Scalar: {train_scalar_loss:.6f}) | "
                  f"Val Loss: {val_loss:.6f} (pH: {val_ph_loss:.6f}, Scalar: {val_scalar_loss:.6f}) | "
                  f"Time: {elapsed:.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'oracle_membrane_fno.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

    writer.close()

    elapsed_total = time.time() - t_start
    print("=" * 80)
    print(f"Training complete: {elapsed_total/60:.1f} min")
    print(f"Best val loss: {best_val_loss:.6f}")

    return history


# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

@torch.no_grad()
def compute_test_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    scaler_ph: MinMaxScaler,
    scaler_scalars: StandardScaler,
    device: torch.device
) -> Dict:
    """Compute detailed metrics on test set."""
    model.eval()

    all_ph_true = []
    all_ph_pred = []
    all_scalars_true = []
    all_scalars_pred = []

    for params, x_grid, ph_true, scalars_true in test_loader:
        ph_pred, scalars_pred = model(params, x_grid)

        all_ph_true.append(ph_true.cpu().numpy())
        all_ph_pred.append(ph_pred.cpu().numpy())
        all_scalars_true.append(scalars_true.cpu().numpy())
        all_scalars_pred.append(scalars_pred.cpu().numpy())

    # Concatenate
    ph_true = np.vstack(all_ph_true)
    ph_pred = np.vstack(all_ph_pred)
    scalars_true = np.vstack(all_scalars_true)
    scalars_pred = np.vstack(all_scalars_pred)

    # Inverse transform to original scale
    ph_true_orig = scaler_ph.inverse_transform(ph_true)
    ph_pred_orig = scaler_ph.inverse_transform(ph_pred)
    scalars_true_orig = scaler_scalars.inverse_transform(scalars_true)
    scalars_pred_orig = scaler_scalars.inverse_transform(scalars_pred)

    # Undo log1p for extreme-range scalar columns
    for col in LOG1P_SCALAR_COLS:
        scalars_true_orig[:, col] = np.expm1(scalars_true_orig[:, col])
        scalars_pred_orig[:, col] = np.expm1(scalars_pred_orig[:, col])

    # Metrics for pH profile (mean over spatial dimension)
    ph_mae = np.abs(ph_true_orig - ph_pred_orig).mean()
    ph_rmse = np.sqrt(((ph_true_orig - ph_pred_orig)**2).mean())
    ph_r2 = 1 - ((ph_true_orig - ph_pred_orig)**2).sum() / ((ph_true_orig - ph_true_orig.mean())**2).sum()

    # Metrics for each scalar
    scalar_names = ['J_formate', 'I_current', 'tau_transit', 'A_steady']
    scalar_metrics = {}

    for i, name in enumerate(scalar_names):
        true_i = scalars_true_orig[:, i]
        pred_i = scalars_pred_orig[:, i]

        mae = np.abs(true_i - pred_i).mean()
        rmse = np.sqrt(((true_i - pred_i)**2).mean())
        ss_res = ((true_i - pred_i)**2).sum()
        ss_tot = ((true_i - true_i.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        scalar_metrics[name] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2)
        }

    metrics = {
        'pH_profile': {
            'MAE': float(ph_mae),
            'RMSE': float(ph_rmse),
            'R2': float(ph_r2)
        },
        'scalars': scalar_metrics
    }

    return metrics, (ph_true_orig, ph_pred_orig, scalars_true_orig, scalars_pred_orig)


def plot_training_curves(history: Dict, output_path: Path):
    """Plot loss vs epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Total loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(epochs, history['val_loss'], label='Val', color='darkorange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # pH loss
    axes[1].plot(epochs, history['train_ph_loss'], label='Train', color='steelblue')
    axes[1].plot(epochs, history['val_ph_loss'], label='Val', color='darkorange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('pH Loss (MSE)')
    axes[1].set_title('pH Profile Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Scalar loss
    axes[2].plot(epochs, history['train_scalar_loss'], label='Train', color='steelblue')
    axes[2].plot(epochs, history['val_scalar_loss'], label='Val', color='darkorange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Scalar Loss (MSE)')
    axes[2].set_title('Scalar Loss')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved training curves to {output_path}")


def plot_predictions(
    ph_true: np.ndarray,
    ph_pred: np.ndarray,
    x_grid: np.ndarray,
    output_path: Path,
    n_samples: int = 6
):
    """Plot sample predictions vs ground truth."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    indices = np.linspace(0, len(ph_true) - 1, n_samples, dtype=int)

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(x_grid, ph_true[idx], 'o-', label='True', color='steelblue', alpha=0.7, markersize=3)
        ax.plot(x_grid, ph_pred[idx], 's-', label='Pred', color='darkorange', alpha=0.7, markersize=3)
        ax.set_xlabel('x (normalized)')
        ax.set_ylabel('pH')
        ax.set_title(f'Sample {idx}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved prediction plots to {output_path}")


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def predict(
    model: nn.Module,
    params_dict: Dict[str, float],
    x_grid: np.ndarray,
    scaler_params: StandardScaler,
    scaler_ph: MinMaxScaler,
    scaler_scalars: StandardScaler,
    device: torch.device,
    param_order: list
) -> Dict:
    """
    Predict pH profile and scalars for a single parameter set.

    Args:
        model: Trained FNO model
        params_dict: Dictionary of membrane parameters
        x_grid: Spatial grid (original scale)
        scaler_params, scaler_ph, scaler_scalars: Fitted scalers
        device: torch device
        param_order: List of parameter names in correct order

    Returns:
        Dictionary with 'ph_profile', 'J_formate', 'I_current', 'tau_transit', 'A_steady'
    """
    model.eval()

    # Convert params_dict to array
    params_array = np.array([params_dict[name] for name in param_order]).reshape(1, -1)

    # Apply same log-transforms as during training
    params_array = params_array.copy()
    for col in LOG10_PARAM_COLS:
        params_array[0, col] = np.log10(max(params_array[0, col], 1e-30))

    # Scale inputs
    params_scaled = scaler_params.transform(params_array)
    x_grid_norm = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min())

    # To tensors
    params_t = torch.tensor(params_scaled, dtype=torch.float32, device=device)
    x_grid_t = torch.tensor(x_grid_norm, dtype=torch.float32, device=device)

    # Predict
    with torch.no_grad():
        ph_pred, scalars_pred = model(params_t, x_grid_t)

    # Inverse transform
    ph_pred_scaled = ph_pred.cpu().numpy()
    scalars_pred_scaled = scalars_pred.cpu().numpy()

    ph_profile = scaler_ph.inverse_transform(ph_pred_scaled)[0]
    scalars = scaler_scalars.inverse_transform(scalars_pred_scaled)[0]

    # Undo log1p for extreme-range scalar columns
    for col in LOG1P_SCALAR_COLS:
        scalars[col] = np.expm1(scalars[col])

    return {
        'ph_profile': ph_profile,
        'J_formate': float(scalars[0]),
        'I_current': float(scalars[1]),
        'tau_transit': float(scalars[2]),
        'A_steady': float(scalars[3])
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ORACLE Phase B: Train FNO for membrane PDE')
    parser.add_argument('--data', type=str, required=True, help='Input NPZ file')
    parser.add_argument('--output-dir', type=str, default='results/oracle_phase_b', help='Output directory')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda-scalar', type=float, default=0.1, help='Weight for scalar loss')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--modes', type=int, default=32, help='Fourier modes')
    parser.add_argument('--width', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of Fourier layers')

    args = parser.parse_args()

    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'tensorboard'
    log_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("ORACLE PHASE B: FNO Training")
    print("=" * 80)

    # Setup device
    device, vram_gb = setup_device()

    # Load data
    data_splits, scaler_params, scaler_ph, scaler_scalars = load_and_preprocess_data(args.data)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(data_splits, args.batch_size, device)

    # Create model
    model = MembraneModel(
        n_params=7,
        n_grid=256,
        n_scalars=4,
        modes=args.modes,
        width=args.width,
        n_layers=args.n_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Train
    history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_scalar=args.lambda_scalar,
        patience=args.patience,
        output_dir=output_dir,
        log_dir=log_dir
    )

    # Load best model
    best_model_path = output_dir / 'oracle_membrane_fno.pt'
    if not best_model_path.exists():
        print("\n[ERROR] No best model saved (all losses were NaN/inf). Aborting evaluation.")
        sys.exit(1)
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics, predictions = compute_test_metrics(model, test_loader, scaler_ph, scaler_scalars, device)

    print("\nTest Metrics:")
    print(f"  pH profile: MAE={metrics['pH_profile']['MAE']:.4f}, "
          f"RMSE={metrics['pH_profile']['RMSE']:.4f}, R²={metrics['pH_profile']['R2']:.4f}")
    for name, m in metrics['scalars'].items():
        print(f"  {name}: MAE={m['MAE']:.4e}, RMSE={m['RMSE']:.4e}, R²={m['R2']:.4f}")

    # Save metrics
    with open(output_dir / 'oracle_phase_b_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save scalers
    with open(output_dir / 'oracle_phase_b_scalers.pkl', 'wb') as f:
        pickle.dump({
            'scaler_params': scaler_params,
            'scaler_ph': scaler_ph,
            'scaler_scalars': scaler_scalars,
            'param_names': ['L_pent', 'L_mack', 'L_chamber', 'delta_pH', 'D_H_pent', 'D_H_mack_intra', 'k_cat'],
            'log10_param_cols': LOG10_PARAM_COLS,
            'log1p_scalar_cols': LOG1P_SCALAR_COLS,
        }, f)
    print(f"Saved scalers to {output_dir / 'oracle_phase_b_scalers.pkl'}")

    # Plot training curves
    plot_training_curves(history, output_dir / 'training_curves.png')

    # Plot predictions
    ph_true, ph_pred, scalars_true, scalars_pred = predictions
    x_grid_orig = data_splits['test']['x_grid'] * 1e-3  # Convert back to mm (assuming original in µm scaled to mm)
    plot_predictions(ph_true, ph_pred, x_grid_orig, output_dir / 'oracle_phase_b_predictions.png')

    print("\n" + "=" * 80)
    print("ORACLE PHASE B TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model saved: {output_dir / 'oracle_membrane_fno.pt'}")
    print(f"Metrics saved: {output_dir / 'oracle_phase_b_results.json'}")
    print(f"Scalers saved: {output_dir / 'oracle_phase_b_scalers.pkl'}")
    print(f"TensorBoard logs: {log_dir}")


if __name__ == '__main__':
    main()
