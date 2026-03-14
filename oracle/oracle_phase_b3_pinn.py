#!/usr/bin/env python3
"""
ORACLE Phase B.3: Physics-Informed Neural Operator (PINN)

Extends the base FNO model with physics-informed loss components:
  - L_data: Standard MSE on pH profiles and scalars
  - L_pde: PDE residual loss (steady-state diffusion-reaction)
  - L_bc: Boundary condition loss
  - L_scalar: MSE on integrated scalar outputs

Physics: D(x) * d²pH/dx² + R(x) ≈ 0 (steady-state)
  - Zone 1 (chamber 1): D = D_water, R = 0
  - Zone 2 (pentlandite): D = D_H_pent, R = 0
  - Zone 3 (mackinawite): D = D_H_mack, R = k_cat_alk (source)
  - Zone 4 (chamber 2): D = D_water, R = 0

BCs: pH(0) = pH_alk, pH(L_total) = pH_acid

Training strategy:
  - Warm start from pretrained FNO checkpoint
  - Lower learning rate (fine-tuning)
  - Adaptive λ_pde and λ_bc weights

Author: Third Matter Research Project
Date: 2026-03-11
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

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
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None


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
# FOURIER NEURAL OPERATOR ARCHITECTURE (copied from train.py)
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
# DATA LOADING AND PREPROCESSING (copied from train.py)
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
LOG1P_SCALAR_COLS = [0, 2]  # J_formate, tau_transit — extreme dynamic range


def load_and_preprocess_data(
    npz_path: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dict, StandardScaler, MinMaxScaler, StandardScaler]:
    """
    Load NPZ data and split into train/val/test.

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
            'params_physical': params[idx_train],  # Store for PDE loss
        },
        'val': {
            'params': params_scaled[idx_val],
            'ph': ph_scaled[idx_val],
            'scalars': scalars_scaled[idx_val],
            'x_grid': x_grid_norm,
            'params_physical': params[idx_val],
        },
        'test': {
            'params': params_scaled[idx_test],
            'ph': ph_scaled[idx_test],
            'scalars': scalars_scaled[idx_test],
            'x_grid': x_grid_norm,
            'params_physical': params[idx_test],
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
        params_phys_t = torch.tensor(split['params_physical'], dtype=torch.float32, device=device)

        # Dataset: (params, x_grid, ph, scalars, params_physical)
        dataset = TensorDataset(
            params_t,
            x_grid_t.unsqueeze(0).expand(len(params_t), -1),
            ph_t,
            scalars_t,
            params_phys_t
        )

        shuffle = (split_name == 'train')
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = to_loader('train')
    val_loader = to_loader('val')
    test_loader = to_loader('test')

    return train_loader, val_loader, test_loader


# ============================================================================
# PHYSICS-INFORMED LOSS COMPONENTS
# ============================================================================

class PhysicsLoss(nn.Module):
    """
    Physics-informed loss for membrane pH diffusion-reaction system.

    PDE: D(x) * d²pH/dx² + R(x) ≈ 0 (steady-state)

    Zones:
      1. Chamber 1 (0 to L_chamber): D = D_water, R = 0
      2. Pentlandite (L_chamber to L_ch+L_pent): D = D_H_pent, R = 0
      3. Mackinawite (L_ch+L_pent to L_ch+L_pent+L_mack): D = D_H_mack, R = k_cat
      4. Chamber 2 (L_ch+L_pent+L_mack to L_total): D = D_water, R = 0

    BCs:
      - Left (x=0): pH = pH_acid = 9.0 - delta_pH (acidic chamber)
      - Right (x=L_total): pH = pH_alk = 9.0 (alkaline chamber)
    """
    def __init__(
        self,
        n_grid: int = 256,
        D_water: float = 9.3e-9,  # m²/s
        scaler_params: Optional[StandardScaler] = None,
        scaler_ph: Optional[MinMaxScaler] = None
    ):
        super().__init__()
        self.n_grid = n_grid
        self.D_water = D_water
        self.scaler_params = scaler_params
        self.scaler_ph = scaler_ph

        # Param column indices
        self.idx_L_chamber = 2
        self.idx_L_pent = 0
        self.idx_L_mack = 1
        self.idx_delta_pH = 3
        self.idx_D_H_pent = 4
        self.idx_D_H_mack = 5
        self.idx_k_cat = 6

    def compute_pde_loss(
        self,
        ph_pred: torch.Tensor,
        params_phys: torch.Tensor,
        x_grid_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PDE residual: D(x) * d²pH/dx² + R(x).

        Args:
            ph_pred: (batch, n_grid) predicted pH in SCALED space
            params_phys: (batch, 7) physical params (after log-transform)
            x_grid_norm: (batch, n_grid) normalized x in [0, 1]

        Returns:
            pde_loss: scalar tensor
        """
        batch_size = ph_pred.shape[0]

        # Inverse-transform pH to physical scale for gradient computation
        # NOTE: Since scaler_ph is MinMaxScaler, inverse is: pH = pH_scaled * (max - min) + min
        # But for gradient computation, we can work in normalized space and scale residual later
        # For simplicity, we compute in scaled space (gradients are scale-invariant up to constants)

        # Compute d²pH/dx² using finite differences
        # dx = 1.0 / (n_grid - 1) in normalized space
        dx = 1.0 / (self.n_grid - 1)

        # Central difference: d²pH/dx² ≈ (pH[i+1] - 2*pH[i] + pH[i-1]) / dx²
        # For boundaries, use forward/backward differences
        d2ph_dx2 = torch.zeros_like(ph_pred)

        # Interior points (1 to n_grid-2)
        d2ph_dx2[:, 1:-1] = (ph_pred[:, 2:] - 2*ph_pred[:, 1:-1] + ph_pred[:, :-2]) / (dx**2)

        # Boundaries (forward/backward difference)
        # Left: d²pH/dx² ≈ (pH[2] - 2*pH[1] + pH[0]) / dx²
        d2ph_dx2[:, 0] = (ph_pred[:, 2] - 2*ph_pred[:, 1] + ph_pred[:, 0]) / (dx**2)
        # Right: d²pH/dx² ≈ (pH[-1] - 2*pH[-2] + pH[-3]) / dx²
        d2ph_dx2[:, -1] = (ph_pred[:, -1] - 2*ph_pred[:, -2] + ph_pred[:, -3]) / (dx**2)

        # Inverse-transform physical params (undo log10 for D and k_cat)
        L_chamber = params_phys[:, self.idx_L_chamber]  # m (not log-transformed)
        L_pent = params_phys[:, self.idx_L_pent]  # m
        L_mack = params_phys[:, self.idx_L_mack]  # m
        D_H_pent = 10 ** params_phys[:, self.idx_D_H_pent]  # m²/s (was log10)
        D_H_mack = 10 ** params_phys[:, self.idx_D_H_mack]  # m²/s
        k_cat = 10 ** params_phys[:, self.idx_k_cat]  # s⁻¹

        # Total length per sample
        L_total = 2 * L_chamber + L_pent + L_mack  # (batch,)

        # Map normalized x_grid to physical x for each sample
        # x_phys = x_grid_norm * L_total (broadcast)
        x_phys = x_grid_norm * L_total.unsqueeze(-1)  # (batch, n_grid)

        # Zone boundaries for each sample
        # Zone 1: [0, L_chamber]
        # Zone 2: [L_chamber, L_chamber + L_pent]
        # Zone 3: [L_chamber + L_pent, L_chamber + L_pent + L_mack]
        # Zone 4: [L_chamber + L_pent + L_mack, L_total]

        x1 = L_chamber.unsqueeze(-1)  # (batch, 1)
        x2 = (L_chamber + L_pent).unsqueeze(-1)
        x3 = (L_chamber + L_pent + L_mack).unsqueeze(-1)

        # Build D(x) and R(x) tensors
        D_field = torch.zeros_like(x_phys)  # (batch, n_grid)
        R_field = torch.zeros_like(x_phys)

        # Zone 1: 0 <= x < L_chamber
        mask1 = (x_phys < x1)
        D_field[mask1] = self.D_water

        # Zone 2: L_chamber <= x < L_chamber + L_pent
        mask2 = (x_phys >= x1) & (x_phys < x2)
        D_field[mask2] = D_H_pent.unsqueeze(-1).expand_as(x_phys)[mask2]

        # Zone 3: L_chamber + L_pent <= x < L_chamber + L_pent + L_mack
        mask3 = (x_phys >= x2) & (x_phys < x3)
        D_field[mask3] = D_H_mack.unsqueeze(-1).expand_as(x_phys)[mask3]
        R_field[mask3] = k_cat.unsqueeze(-1).expand_as(x_phys)[mask3]  # Source term

        # Zone 4: x >= L_chamber + L_pent + L_mack
        mask4 = (x_phys >= x3)
        D_field[mask4] = self.D_water

        # PDE residual in normalized coordinates:
        # Physical PDE: D(x) * d²pH/dx_phys² + R(x) = 0
        # With x_phys = x_norm * L_total: d²/dx_phys² = (1/L_total²) * d²/dx_norm²
        # So: D(x)/L_total² * d²pH/dx_norm² + R(x) = 0
        L_total_sq = (L_total ** 2).unsqueeze(-1)  # (batch, 1)
        residual = D_field / L_total_sq * d2ph_dx2 + R_field

        # Mean squared residual (ignore boundary points to avoid edge artifacts)
        # Use only interior points (2:-2) for PDE loss
        pde_loss = (residual[:, 2:-2] ** 2).mean()

        return pde_loss

    def compute_bc_loss(
        self,
        ph_pred: torch.Tensor,
        params_phys: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary condition loss.

        BCs:
          - Left (x=0): pH = pH_alk = 9.0
          - Right (x=L_total): pH = pH_acid = 9.0 - delta_pH

        Args:
            ph_pred: (batch, n_grid) predicted pH in SCALED space
            params_phys: (batch, 7) physical params

        Returns:
            bc_loss: scalar tensor
        """
        delta_pH = params_phys[:, self.idx_delta_pH]  # (batch,)

        # Boundary values in physical space
        # datagen: left (x=0) = acidic chamber, right (x=L_total) = alkaline (pH=9.0)
        pH_left = 9.0 - delta_pH  # Acidic side
        pH_right = torch.full_like(delta_pH, 9.0)  # Alkaline side

        # Transform to scaled space using scaler_ph
        # MinMaxScaler: pH_scaled = (pH - pH_min) / (pH_max - pH_min)
        if self.scaler_ph is not None:
            pH_min = self.scaler_ph.data_min_[0]
            pH_max = self.scaler_ph.data_max_[0]

            pH_left_scaled = (pH_left - pH_min) / (pH_max - pH_min)
            pH_right_scaled = (pH_right - pH_min) / (pH_max - pH_min)
        else:
            # Fallback: assume pH in [2, 14], typical range
            pH_left_scaled = (pH_left - 2.0) / 12.0
            pH_right_scaled = (pH_right - 2.0) / 12.0

        # BC loss
        bc_loss = ((ph_pred[:, 0] - pH_left_scaled) ** 2).mean() + \
                  ((ph_pred[:, -1] - pH_right_scaled) ** 2).mean()

        return bc_loss

    def forward(
        self,
        ph_pred: torch.Tensor,
        params_phys: torch.Tensor,
        x_grid_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PDE and BC losses.

        Returns:
            (pde_loss, bc_loss)
        """
        pde_loss = self.compute_pde_loss(ph_pred, params_phys, x_grid_norm)
        bc_loss = self.compute_bc_loss(ph_pred, params_phys)

        return pde_loss, bc_loss


# ============================================================================
# TRAINING LOOP WITH PHYSICS
# ============================================================================

def train_epoch_pinn(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    physics_loss: PhysicsLoss,
    lambda_scalar: float,
    lambda_pde: float,
    lambda_bc: float,
    device: torch.device
) -> Tuple[float, float, float, float, float]:
    """Train for one epoch with physics-informed loss."""
    model.train()

    total_loss = 0.0
    total_ph_loss = 0.0
    total_scalar_loss = 0.0
    total_pde_loss = 0.0
    total_bc_loss = 0.0
    n_batches = 0

    for params, x_grid, ph_true, scalars_true, params_phys in loader:
        optimizer.zero_grad()

        # Forward pass
        ph_pred, scalars_pred = model(params, x_grid)

        # Data loss
        loss_ph = F.mse_loss(ph_pred, ph_true)
        loss_scalar = F.mse_loss(scalars_pred, scalars_true)

        # Physics loss
        loss_pde, loss_bc = physics_loss(ph_pred, params_phys, x_grid)

        # Combined loss
        loss = loss_ph + lambda_scalar * loss_scalar + lambda_pde * loss_pde + lambda_bc * loss_bc

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ph_loss += loss_ph.item()
        total_scalar_loss += loss_scalar.item()
        total_pde_loss += loss_pde.item()
        total_bc_loss += loss_bc.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_ph_loss / n_batches,
        total_scalar_loss / n_batches,
        total_pde_loss / n_batches,
        total_bc_loss / n_batches
    )


@torch.no_grad()
def evaluate_pinn(
    model: nn.Module,
    loader: DataLoader,
    physics_loss: PhysicsLoss,
    lambda_scalar: float,
    lambda_pde: float,
    lambda_bc: float,
    device: torch.device
) -> Tuple[float, float, float, float, float]:
    """Evaluate on validation/test set."""
    model.eval()

    total_loss = 0.0
    total_ph_loss = 0.0
    total_scalar_loss = 0.0
    total_pde_loss = 0.0
    total_bc_loss = 0.0
    n_batches = 0

    for params, x_grid, ph_true, scalars_true, params_phys in loader:
        ph_pred, scalars_pred = model(params, x_grid)

        loss_ph = F.mse_loss(ph_pred, ph_true)
        loss_scalar = F.mse_loss(scalars_pred, scalars_true)
        loss_pde, loss_bc = physics_loss(ph_pred, params_phys, x_grid)

        loss = loss_ph + lambda_scalar * loss_scalar + lambda_pde * loss_pde + lambda_bc * loss_bc

        total_loss += loss.item()
        total_ph_loss += loss_ph.item()
        total_scalar_loss += loss_scalar.item()
        total_pde_loss += loss_pde.item()
        total_bc_loss += loss_bc.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_ph_loss / n_batches,
        total_scalar_loss / n_batches,
        total_pde_loss / n_batches,
        total_bc_loss / n_batches
    )


def train_model_pinn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    physics_loss: PhysicsLoss,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_scalar: float,
    lambda_pde: float,
    lambda_bc: float,
    patience: int,
    output_dir: Path,
    log_dir: Path
) -> Dict:
    """Train PINN model with early stopping."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir=str(log_dir)) if HAS_TENSORBOARD else None

    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_ph_loss': [],
        'train_scalar_loss': [],
        'train_pde_loss': [],
        'train_bc_loss': [],
        'val_loss': [],
        'val_ph_loss': [],
        'val_scalar_loss': [],
        'val_pde_loss': [],
        'val_bc_loss': [],
    }

    print("\nStarting PINN training...")
    print(f"Epochs: {epochs}, LR: {lr}, λ_scalar: {lambda_scalar}, "
          f"λ_pde: {lambda_pde}, λ_bc: {lambda_bc}, Patience: {patience}")
    print("=" * 80)

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_ph, train_scalar, train_pde, train_bc = train_epoch_pinn(
            model, train_loader, optimizer, physics_loss,
            lambda_scalar, lambda_pde, lambda_bc, device
        )

        # Validate
        val_loss, val_ph, val_scalar, val_pde, val_bc = evaluate_pinn(
            model, val_loader, physics_loss,
            lambda_scalar, lambda_pde, lambda_bc, device
        )

        # Scheduler step
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_ph_loss'].append(train_ph)
        history['train_scalar_loss'].append(train_scalar)
        history['train_pde_loss'].append(train_pde)
        history['train_bc_loss'].append(train_bc)
        history['val_loss'].append(val_loss)
        history['val_ph_loss'].append(val_ph)
        history['val_scalar_loss'].append(val_scalar)
        history['val_pde_loss'].append(val_pde)
        history['val_bc_loss'].append(val_bc)

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Loss_pH/train', train_ph, epoch)
            writer.add_scalar('Loss_pH/val', val_ph, epoch)
            writer.add_scalar('Loss_Scalars/train', train_scalar, epoch)
            writer.add_scalar('Loss_Scalars/val', val_scalar, epoch)
            writer.add_scalar('Loss_PDE/train', train_pde, epoch)
            writer.add_scalar('Loss_PDE/val', val_pde, epoch)
            writer.add_scalar('Loss_BC/train', train_bc, epoch)
            writer.add_scalar('Loss_BC/val', val_bc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} (pH:{train_ph:.6f} S:{train_scalar:.6f} PDE:{train_pde:.6f} BC:{train_bc:.6f}) | "
                  f"Val: {val_loss:.6f} (pH:{val_ph:.6f} S:{val_scalar:.6f} PDE:{val_pde:.6f} BC:{val_bc:.6f}) | "
                  f"Time: {elapsed:.1f}s")

        # Early stopping based on validation data loss (not physics loss)
        val_data_loss = val_ph + lambda_scalar * val_scalar
        if val_data_loss < best_val_loss:
            best_val_loss = val_data_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_data_loss': val_data_loss,
            }, output_dir / 'oracle_membrane_pinn.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

    if writer:
        writer.close()

    elapsed_total = time.time() - t_start
    print("=" * 80)
    print(f"Training complete: {elapsed_total/60:.1f} min")
    print(f"Best val data loss: {best_val_loss:.6f}")

    return history


# ============================================================================
# EVALUATION (same as base model)
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

    for params, x_grid, ph_true, scalars_true, params_phys in test_loader:
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

    # Metrics for pH profile
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


def plot_training_curves_pinn(history: Dict, output_path: Path):
    """Plot loss vs epoch including physics losses."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', color='steelblue')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', color='darkorange')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # pH loss
    axes[0, 1].plot(epochs, history['train_ph_loss'], label='Train', color='steelblue')
    axes[0, 1].plot(epochs, history['val_ph_loss'], label='Val', color='darkorange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('pH Loss (MSE)')
    axes[0, 1].set_title('pH Profile Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Scalar loss
    axes[0, 2].plot(epochs, history['train_scalar_loss'], label='Train', color='steelblue')
    axes[0, 2].plot(epochs, history['val_scalar_loss'], label='Val', color='darkorange')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Scalar Loss (MSE)')
    axes[0, 2].set_title('Scalar Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # PDE loss
    axes[1, 0].plot(epochs, history['train_pde_loss'], label='Train', color='steelblue')
    axes[1, 0].plot(epochs, history['val_pde_loss'], label='Val', color='darkorange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PDE Residual Loss')
    axes[1, 0].set_title('PDE Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # BC loss
    axes[1, 1].plot(epochs, history['train_bc_loss'], label='Train', color='steelblue')
    axes[1, 1].plot(epochs, history['val_bc_loss'], label='Val', color='darkorange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('BC Loss')
    axes[1, 1].set_title('Boundary Condition Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Combined data loss (pH + scalar)
    train_data = np.array(history['train_ph_loss']) + 0.1 * np.array(history['train_scalar_loss'])
    val_data = np.array(history['val_ph_loss']) + 0.1 * np.array(history['val_scalar_loss'])
    axes[1, 2].plot(epochs, train_data, label='Train', color='steelblue')
    axes[1, 2].plot(epochs, val_data, label='Val', color='darkorange')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Data Loss')
    axes[1, 2].set_title('Data Loss (pH + 0.1×Scalar)')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved training curves to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ORACLE Phase B.3: PINN Training')
    parser.add_argument('--data', type=str, default='digital-twin/oracle_dataset_50k.h5',
                        help='Input NPZ file')
    parser.add_argument('--pretrained', type=str, default='digital-twin/fno_results_50k/oracle_membrane_fno.pt',
                        help='Pretrained FNO checkpoint')
    parser.add_argument('--scalers', type=str, default='digital-twin/fno_results_50k/oracle_phase_b_scalers.pkl',
                        help='Pretrained scalers')
    parser.add_argument('--output-dir', type=str, default='digital-twin/pinn_results/',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda-pde', type=float, default=0.1, help='PDE loss weight')
    parser.add_argument('--lambda-bc', type=float, default=0.1, help='BC loss weight')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto', help='cuda/cpu/auto')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'tensorboard'
    log_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("ORACLE PHASE B.3: PINN Training")
    print("=" * 80)

    # Setup device
    if args.device == 'auto':
        device, vram_gb = setup_device()
    else:
        device = torch.device(args.device)
        vram_gb = 0.0
        print(f"Using device: {device}")

    # Load data
    if args.data.endswith('.h5'):
        # Convert to npz path (assume same directory)
        npz_path = args.data.replace('.h5', '.npz')
        if not Path(npz_path).exists():
            print(f"[ERROR] NPZ file not found: {npz_path}")
            print("Please run data generation script first.")
            sys.exit(1)
    else:
        npz_path = args.data

    data_splits, scaler_params, scaler_ph, scaler_scalars = load_and_preprocess_data(npz_path, seed=args.seed)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(data_splits, args.batch_size, device)

    # Create model (match pretrained architecture)
    # Hardcoded hyperparams from base training
    model = MembraneModel(
        n_params=7,
        n_grid=256,
        n_scalars=4,
        modes=32,  # Default from base model
        width=64,
        n_layers=4
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Load pretrained weights if available
    pretrained_path = Path(args.pretrained)
    if pretrained_path.exists():
        print(f"\nLoading pretrained weights from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from epoch {checkpoint.get('epoch', '?')}")
    else:
        print(f"\n[WARN] Pretrained checkpoint not found: {pretrained_path}")
        print("Starting from random initialization.")

    # Create physics loss module
    physics_loss = PhysicsLoss(
        n_grid=256,
        D_water=9.3e-9,  # m²/s
        scaler_params=scaler_params,
        scaler_ph=scaler_ph
    ).to(device)

    # Train
    lambda_scalar = 0.1  # Same as base model
    history = train_model_pinn(
        model,
        train_loader,
        val_loader,
        physics_loss,
        device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_scalar=lambda_scalar,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        patience=args.patience,
        output_dir=output_dir,
        log_dir=log_dir
    )

    # Load best model
    best_model_path = output_dir / 'oracle_membrane_pinn.pt'
    if not best_model_path.exists():
        print("\n[ERROR] No best model saved. Aborting evaluation.")
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

    # Compare with base FNO if metrics available
    base_metrics_path = Path(args.scalers).parent / 'oracle_phase_b_results.json'
    if base_metrics_path.exists():
        with open(base_metrics_path, 'r') as f:
            base_metrics = json.load(f)

        print("\nComparison with base FNO:")
        print(f"  pH R²: base={base_metrics['pH_profile']['R2']:.4f}, "
              f"PINN={metrics['pH_profile']['R2']:.4f}, "
              f"Δ={metrics['pH_profile']['R2'] - base_metrics['pH_profile']['R2']:.4f}")
        for name in ['J_formate', 'tau_transit', 'A_steady']:
            if name in base_metrics['scalars'] and name in metrics['scalars']:
                base_r2 = base_metrics['scalars'][name]['R2']
                pinn_r2 = metrics['scalars'][name]['R2']
                if not np.isnan(base_r2) and not np.isnan(pinn_r2):
                    print(f"  {name} R²: base={base_r2:.4f}, PINN={pinn_r2:.4f}, Δ={pinn_r2 - base_r2:.4f}")

    # Save metrics
    metrics['lambda_pde'] = args.lambda_pde
    metrics['lambda_bc'] = args.lambda_bc
    with open(output_dir / 'oracle_phase_b3_pinn_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save scalers (same as base model)
    with open(output_dir / 'oracle_phase_b3_pinn_scalers.pkl', 'wb') as f:
        pickle.dump({
            'scaler_params': scaler_params,
            'scaler_ph': scaler_ph,
            'scaler_scalars': scaler_scalars,
            'param_names': ['L_pent', 'L_mack', 'L_chamber', 'delta_pH', 'D_H_pent', 'D_H_mack_intra', 'k_cat'],
            'log10_param_cols': LOG10_PARAM_COLS,
            'log1p_scalar_cols': LOG1P_SCALAR_COLS,
        }, f)
    print(f"Saved scalers to {output_dir / 'oracle_phase_b3_pinn_scalers.pkl'}")

    # Plot training curves
    plot_training_curves_pinn(history, output_dir / 'pinn_training_curves.png')

    print("\n" + "=" * 80)
    print("ORACLE PHASE B.3 PINN TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model saved: {output_dir / 'oracle_membrane_pinn.pt'}")
    print(f"Metrics saved: {output_dir / 'oracle_phase_b3_pinn_results.json'}")
    print(f"Scalers saved: {output_dir / 'oracle_phase_b3_pinn_scalers.pkl'}")
    print(f"TensorBoard logs: {log_dir}")


if __name__ == '__main__':
    main()
