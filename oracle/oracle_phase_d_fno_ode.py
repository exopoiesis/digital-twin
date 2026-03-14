#!/usr/bin/env python3
"""
ORACLE Phase D: FNO-ODE Integration
====================================
Replaces lumped kinetic parameters (k1, f1_max, Km_f) in TM6v3 ODE with
FNO-predicted J_formate from the trained membrane surrogate (Phase B).

Architecture:
  1. Load trained FNO model (Phase B checkpoint + scalers)
  2. Predict A_steady (mM) for given membrane geometry/chemistry
  3. Derive fno_rate from A_steady via ODE steady-state equation + Newton correction
  4. Integrate 6-variable ODE (A, B, C, M, P, Fe) with fno_rate replacing k1*f1
  5. Validate against original TM6v3-full at nominal G3c parameters
  6. Sweep membrane geometry to find optimal designs

v2 (2026-03-13): Replaced empirical calibration factor (2.275e-6) with
physics-based per-geometry rate derivation from FNO A_steady.

Usage:
  python oracle_phase_d_fno_ode.py --validate
  python oracle_phase_d_fno_ode.py --scan
  python oracle_phase_d_fno_ode.py --validate --scan --output-dir phase_d_results

Author: Third Matter Research Project
Date: 2026-03-10
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp


# ============================================================================
# NUMPY JSON ENCODER
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# v2 (2026-03-13): Calibration factor REMOVED.
# fno_rate is now derived from FNO's A_steady prediction via ODE steady-state
# equation + Newton correction. See compute_fno_rate_from_A_steady().
# Old approach (J_formate / L_chamber * 1e-3 * calibration) had unit confusion
# and required empirical fitting. New approach is physics-based per-geometry.

_builtin_print = print
def print(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ============================================================================
# FOURIER NEURAL OPERATOR ARCHITECTURE (copied from oracle_phase_b_train.py)
# ============================================================================

class SpectralConv1d(nn.Module):
    """
    1D Fourier layer: FFT -> multiply with learnable weights -> IFFT.
    Keeps only the first `modes` Fourier modes.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)

        modes_to_use = min(self.modes, x_ft.shape[-1])
        weights_complex = torch.view_as_complex(self.weights[:, :, :modes_to_use, :])

        out_ft[:, :, :modes_to_use] = torch.einsum(
            'bix,iox->box', x_ft[:, :, :modes_to_use], weights_complex
        )

        x_out = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)
        return x_out


class FNO1d(nn.Module):
    """
    Fourier Neural Operator for 1D problems.
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

        self.lift = nn.Linear(in_channels, width)

        self.fourier_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)
        ])

        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.lift(x)
        x = x.permute(0, 2, 1)

        for fourier, conv in zip(self.fourier_layers, self.conv_layers):
            x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)

        x = x.permute(0, 2, 1)
        x = self.project(x)
        x = x.permute(0, 2, 1)
        return x


class MembraneModel(nn.Module):
    """
    Full ORACLE Phase B model: FNO for pH profile + MLP head for scalars.
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

        self.fno = FNO1d(
            in_channels=n_params + 1,
            out_channels=1,
            modes=modes,
            width=width,
            n_layers=n_layers
        )

        self.scalar_head = nn.Sequential(
            nn.Linear(width + n_params, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_scalars)
        )

        self.width = width

    def forward(
        self, params: torch.Tensor, x_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = params.shape[0]

        if x_grid.dim() == 1:
            x_grid = x_grid.unsqueeze(0).expand(batch_size, -1)

        params_spatial = params.unsqueeze(-1).expand(-1, -1, self.n_grid)
        x_input = torch.cat([params_spatial, x_grid.unsqueeze(1)], dim=1)

        # Forward through FNO layers to get hidden representation
        x = x_input.permute(0, 2, 1)
        x = self.fno.lift(x)
        x = x.permute(0, 2, 1)

        for fourier, conv in zip(self.fno.fourier_layers, self.fno.conv_layers):
            x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)

        # pH profile
        x_for_ph = x.permute(0, 2, 1)
        ph_profile = self.fno.project(x_for_ph).squeeze(-1)

        # Scalars via pooled hidden state + params
        x_pooled = x.mean(dim=-1)
        x_scalar = torch.cat([x_pooled, params], dim=1)
        scalars = self.scalar_head(x_scalar)

        return ph_profile, scalars


# ============================================================================
# SCALER HELPERS (replicate sklearn StandardScaler/MinMaxScaler logic)
# ============================================================================

# Log-transform column indices (same as oracle_phase_b_train.py)
LOG10_PARAM_COLS = [4, 5, 6]   # D_H_pent, D_H_mack_intra, k_cat
LOG1P_SCALAR_COLS = [0, 2]     # J_formate, tau_transit

PARAM_ORDER = [
    'L_pent', 'L_mack', 'L_chamber', 'delta_pH',
    'D_H_pent', 'D_H_mack_intra', 'k_cat'
]

SCALAR_NAMES = ['J_formate', 'I_current', 'tau_transit', 'A_steady']


class SimpleStandardScaler:
    """Minimal StandardScaler replica for inference (no sklearn dependency at runtime)."""

    def __init__(self, mean_: np.ndarray, scale_: np.ndarray):
        self.mean_ = np.asarray(mean_, dtype=np.float64)
        self.scale_ = np.asarray(scale_, dtype=np.float64)
        # Safety: avoid division by zero for constant features
        self.scale_[self.scale_ == 0.0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.mean_


class SimpleMinMaxScaler:
    """Minimal MinMaxScaler replica for inference."""

    def __init__(self, data_min_: np.ndarray, data_max_: np.ndarray,
                 feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.data_min_ = np.asarray(data_min_, dtype=np.float64)
        self.data_max_ = np.asarray(data_max_, dtype=np.float64)
        self.feature_range = feature_range
        rng = data_max_ - data_min_
        rng[rng == 0.0] = 1.0
        self.scale_ = (feature_range[1] - feature_range[0]) / rng
        self.min_ = feature_range[0] - data_min_ * self.scale_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.min_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.scale_


# ============================================================================
# FNO LOADER
# ============================================================================

def load_fno(
    model_path: str,
    scalers_path: Optional[str] = None,
    device: str = 'cpu'
) -> Tuple[MembraneModel, object, object, object, np.ndarray, list]:
    """
    Load trained FNO model and scalers from checkpoint.

    Supports two loading modes:
      A) Combined checkpoint (.pt) that contains model_state_dict + config +
         scaler_params + scaler_ph + scaler_scalars + x_grid + param_order
      B) Separate model checkpoint (.pt) + scalers pickle (.pkl)

    Args:
        model_path: Path to the .pt checkpoint
        scalers_path: Path to the .pkl scalers file (if separate)
        device: 'cpu' or 'cuda:0'

    Returns:
        model, scaler_params, scaler_ph, scaler_scalars, x_grid, param_order
    """
    print(f"Loading FNO model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # --- Determine model config ---
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        modes = cfg.get('modes', cfg.get('n_modes', 32))
        width = cfg.get('width', 64)
        n_layers = cfg.get('n_layers', 4)
        in_channels = cfg.get('in_channels', 8)  # 7 params + 1 x_grid
    else:
        # Infer from model state dict
        modes = 32
        width = 64
        n_layers = 4
        # Try to detect from weights shape
        state = checkpoint.get('model_state_dict', checkpoint)
        for key in state:
            if 'fourier_layers' in key and 'weights' in key:
                # weights shape: (in_ch, out_ch, modes, 2)
                modes = state[key].shape[2]
                width = state[key].shape[1]
                break
        # Count fourier layers
        layer_indices = set()
        for key in state:
            if 'fourier_layers' in key:
                parts = key.split('.')
                for j, part in enumerate(parts):
                    if part == 'fourier_layers' and j + 1 < len(parts):
                        try:
                            layer_indices.add(int(parts[j + 1]))
                        except ValueError:
                            pass
        if layer_indices:
            n_layers = max(layer_indices) + 1

    print(f"  Config: modes={modes}, width={width}, n_layers={n_layers}")

    # --- Reconstruct model ---
    model = MembraneModel(
        n_params=7,
        n_grid=256,
        n_scalars=4,
        modes=modes,
        width=width,
        n_layers=n_layers
    )

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')
    print(f"  Loaded model (epoch={epoch}, val_loss={val_loss})")

    # --- Load scalers ---
    scaler_params = None
    scaler_ph = None
    scaler_scalars = None
    x_grid = None
    param_order = PARAM_ORDER

    # Check if scalers are embedded in checkpoint
    if 'scaler_params' in checkpoint:
        sp = checkpoint['scaler_params']
        if isinstance(sp, dict):
            scaler_params = SimpleStandardScaler(sp['mean_'], sp['scale_'])
        else:
            scaler_params = sp  # Already a scaler object
    if 'scaler_ph' in checkpoint:
        sph = checkpoint['scaler_ph']
        if isinstance(sph, dict):
            scaler_ph = SimpleMinMaxScaler(sph['data_min_'], sph['data_max_'])
        else:
            scaler_ph = sph
    if 'scaler_scalars' in checkpoint:
        ss = checkpoint['scaler_scalars']
        if isinstance(ss, dict):
            scaler_scalars = SimpleStandardScaler(ss['mean_'], ss['scale_'])
        else:
            scaler_scalars = ss
    if 'x_grid' in checkpoint:
        x_grid = np.asarray(checkpoint['x_grid'])
    if 'param_order' in checkpoint:
        param_order = list(checkpoint['param_order'])

    # Fallback: load from separate pickle
    if scaler_params is None and scalers_path is not None:
        import pickle
        print(f"  Loading scalers from {scalers_path}...")
        with open(scalers_path, 'rb') as f:
            scalers_data = pickle.load(f)
        scaler_params = scalers_data['scaler_params']
        scaler_ph = scalers_data['scaler_ph']
        scaler_scalars = scalers_data['scaler_scalars']
        if 'param_names' in scalers_data:
            param_order = scalers_data['param_names']
        print(f"  Scalers loaded (param_order={param_order})")

    # Fallback: auto-detect scalers pickle next to model
    if scaler_params is None and scalers_path is None:
        auto_pkl = Path(model_path).parent / 'oracle_phase_b_scalers.pkl'
        if auto_pkl.exists():
            import pickle
            print(f"  Auto-loading scalers from {auto_pkl}...")
            with open(auto_pkl, 'rb') as f:
                scalers_data = pickle.load(f)
            scaler_params = scalers_data['scaler_params']
            scaler_ph = scalers_data['scaler_ph']
            scaler_scalars = scalers_data['scaler_scalars']
            if 'param_names' in scalers_data:
                param_order = scalers_data['param_names']
            print(f"  Scalers loaded (param_order={param_order})")

    if scaler_params is None:
        raise RuntimeError(
            "Could not load scalers. Provide --scalers-path or embed scalers in checkpoint."
        )

    # x_grid fallback: uniform 256-point grid in [0, L_chamber_max]
    if x_grid is None:
        x_grid = np.linspace(0, 1, 256)
        print("  WARNING: x_grid not found in checkpoint, using uniform [0,1] grid")

    n_params_model = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params_model:,}")

    return model, scaler_params, scaler_ph, scaler_scalars, x_grid, param_order


# ============================================================================
# FNO PREDICTION
# ============================================================================

def fno_predict(
    model: MembraneModel,
    membrane_params: Dict[str, float],
    scaler_params,
    scaler_ph,
    scaler_scalars,
    x_grid: np.ndarray,
    param_order: list,
    device: str = 'cpu'
) -> Dict:
    """
    Predict membrane transport properties from FNO.

    Args:
        model: Trained MembraneModel
        membrane_params: dict with keys matching param_order:
            L_pent (nm), L_mack (nm), L_chamber (um), delta_pH,
            D_H_pent (m2/s), D_H_mack_intra (m2/s), k_cat (s-1)
        scaler_params: StandardScaler for input parameters
        scaler_ph: MinMaxScaler for pH profiles
        scaler_scalars: StandardScaler for scalar outputs
        x_grid: Spatial grid (256 points, original scale)
        param_order: Parameter name ordering
        device: torch device string

    Returns:
        dict with:
            J_formate (mol/m2/s), I_current (A/m2),
            tau_transit (s), A_steady (mM),
            ph_profile (ndarray of shape (256,))
    """
    model.eval()

    # Convert human-friendly units to SI (FNO was trained on SI)
    # L_pent: nm -> m, L_mack: nm -> m, L_chamber: um -> m
    # D_H_*, k_cat, delta_pH stay as-is (already SI)
    mp_si = dict(membrane_params)
    mp_si['L_pent'] = membrane_params['L_pent'] * 1e-9       # nm -> m
    mp_si['L_mack'] = membrane_params['L_mack'] * 1e-9       # nm -> m
    mp_si['L_chamber'] = membrane_params['L_chamber'] * 1e-6  # um -> m

    # Build parameter array in correct order
    params_array = np.array(
        [mp_si[name] for name in param_order],
        dtype=np.float64
    ).reshape(1, -1)

    # Apply log10 to extreme-range columns BEFORE scaling
    params_array = params_array.copy()
    for col in LOG10_PARAM_COLS:
        params_array[0, col] = np.log10(max(params_array[0, col], 1e-30))

    # Scale parameters
    params_scaled = scaler_params.transform(params_array)

    # Normalize x_grid to [0, 1]
    x_min, x_max = x_grid.min(), x_grid.max()
    if x_max > x_min:
        x_grid_norm = (x_grid - x_min) / (x_max - x_min)
    else:
        x_grid_norm = np.zeros_like(x_grid)

    # Convert to tensors
    dev = torch.device(device)
    params_t = torch.tensor(params_scaled, dtype=torch.float32, device=dev)
    x_grid_t = torch.tensor(x_grid_norm, dtype=torch.float32, device=dev)

    # Forward pass
    with torch.no_grad():
        ph_pred_scaled, scalars_pred_scaled = model(params_t, x_grid_t)

    ph_pred_np = ph_pred_scaled.cpu().numpy()
    scalars_pred_np = scalars_pred_scaled.cpu().numpy()

    # Inverse transform pH profile
    ph_profile = scaler_ph.inverse_transform(ph_pred_np)[0]

    # Inverse transform scalars
    scalars_raw = scaler_scalars.inverse_transform(scalars_pred_np)[0]

    # Undo log1p for J_formate (col 0) and tau_transit (col 2)
    for col in LOG1P_SCALAR_COLS:
        scalars_raw[col] = np.expm1(scalars_raw[col])

    return {
        'J_formate': float(scalars_raw[0]),      # mol/m2/s
        'I_current': float(scalars_raw[1]),       # A/m2
        'tau_transit': float(scalars_raw[2]),      # s
        'A_steady': float(scalars_raw[3]),         # mM
        'ph_profile': ph_profile,                  # (256,)
    }


# ============================================================================
# ODE PARAMETERS (Phase D: FNO-derived rate replaces k1, f1_max, Km_f)
# ============================================================================

@dataclass
class PhaseDParams:
    """
    ODE parameters for Phase D FNO-ODE integration.

    The lumped autocatalysis parameters (k1, f1_max, Km_f) are REMOVED.
    Instead, fno_rate (M/s) is computed externally from FNO prediction.
    All other parameters are inherited from TM6v3FullParams.
    """

    # --- Hill autocatalysis ---
    Ka: float = 7e-4              # M, Hill n=2 half-saturation

    # --- FNO-derived formate production rate ---
    fno_rate: float = 0.0         # M/s, set from FNO prediction before ODE

    # --- Membrane growth: Fe2+ + A -> M ---
    km: float = 3e-2              # s-1 M-1, membrane growth rate

    # --- Fe2+ dynamics ---
    k_fe_gen: float = 5e-5        # s-1, A -> Fe (byproduct)
    fe_supply: float = 5e-8       # M/s, Fe0 corrosion

    # --- Degradation ---
    kd_A: float = 1e-4            # s-1, formate
    kd_B: float = 1e-4            # s-1, pyruvate
    kd_C: float = 1e-4            # s-1, glyoxylate
    kd_m: float = 3e-6            # s-1, membrane
    kd_fe: float = 3e-4           # s-1, Fe2+ loss

    # --- Metabolic cycle: B, C ---
    k2: float = 5e-5              # s-1, A -> B
    k3: float = 5e-3              # M-1 s-1, B + C -> A
    c_supply: float = 1e-5        # M/s, glyoxylate influx

    # --- PEDOT (CI-linkage) ---
    kp_on: float = 1e-4           # s-1, PEDOT activation
    kp_off: float = 5e-5          # s-1, PEDOT deactivation
    Kp: float = 1e-3              # M, PEDOT half-saturation for A
    beta_max: float = 1.5         # max km enhancement at P=1


@dataclass
class OriginalTM6v3Params(PhaseDParams):
    """
    Original TM6v3-full parameters (for validation comparison).
    Includes k1, f1_max, Km_f that Phase D replaces with FNO.
    """
    k1: float = 1e-4              # s-1, baseline autocatalysis
    f1_max: float = 5e-3          # M, maximum resource
    Km_f: float = 5e-4            # M, membrane half-saturation


# ============================================================================
# ODE RIGHT-HAND SIDE
# ============================================================================

# Variable names and count
VAR_NAMES = ['a', 'b', 'c', 'm', 'p', 'fe']
N_VAR = 6


def phase_d_rhs(t: float, y: list, p: PhaseDParams) -> list:
    """
    Right-hand side for Phase D ODE: 6 variables (A, B, C, M, P, Fe).

    KEY CHANGE: R1 uses fno_rate (pre-computed from FNO) instead of k1 * f1_eff.
    The fno_rate already encodes the membrane geometry and chemistry.

    dA/dt  = fno_rate * hill(A) + k3*B*C - k2*A - kd_A*A
    dB/dt  = k2*A - k3*B*C - kd_B*B
    dC/dt  = c_supply - k3*B*C - kd_C*C
    dM/dt  = km*beta*Fe*A - kd_m*M
    dP/dt  = kp_on*A/(Kp+A)*(1-P) - kp_off*P
    dFe/dt = fe_supply + k_fe_gen*A - km*beta*Fe*A - kd_fe*Fe
    """
    a, b, c, m, pp, fe = [max(v, 0.0) for v in y]
    pp = min(pp, 1.0)

    # CI-linkage: P -> km enhancement
    beta_factor = 1.0 + (p.beta_max - 1.0) * pp

    # Hill n=2 autocatalysis
    hill_a = a**2 / (p.Ka**2 + a**2) if a > 0 else 0.0

    # Reactions
    r1 = p.fno_rate * hill_a          # FNO-derived formate production
    r2 = p.k2 * a                     # A -> B
    r3 = p.k3 * b * c                 # B + C -> A

    # Membrane growth with CI enhancement
    km_eff = p.km * beta_factor
    r4b = km_eff * fe * a

    # Fe dynamics
    r_fe_gen = p.k_fe_gen * a
    r_fe_supply = p.fe_supply

    # PEDOT switching
    dpp = p.kp_on * a / (p.Kp + a) * (1.0 - pp) - p.kp_off * pp

    # dy/dt
    da = r1 + r3 - r2 - p.kd_A * a
    db = r2 - r3 - p.kd_B * b
    dc = p.c_supply - r3 - p.kd_C * c
    dm = r4b - p.kd_m * m
    dfe = r_fe_gen + r_fe_supply - r4b - p.kd_fe * fe

    return [da, db, dc, dm, dpp, dfe]


def original_rhs(t: float, y: list, p: OriginalTM6v3Params) -> list:
    """
    Original TM6v3-full RHS (for validation comparison).
    Uses k1, f1_max, Km_f instead of fno_rate.
    """
    a, b, c, m, pp, fe = [max(v, 0.0) for v in y]
    pp = min(pp, 1.0)

    beta_factor = 1.0 + (p.beta_max - 1.0) * pp
    f1_eff = p.f1_max * m / (p.Km_f + m) if m > 0 else 0.0
    hill_a = a**2 / (p.Ka**2 + a**2) if a > 0 else 0.0

    r1 = p.k1 * f1_eff * hill_a
    r2 = p.k2 * a
    r3 = p.k3 * b * c

    km_eff = p.km * beta_factor
    r4b = km_eff * fe * a

    r_fe_gen = p.k_fe_gen * a
    r_fe_supply = p.fe_supply

    dpp = p.kp_on * a / (p.Kp + a) * (1.0 - pp) - p.kp_off * pp

    da = r1 + r3 - r2 - p.kd_A * a
    db = r2 - r3 - p.kd_B * b
    dc = p.c_supply - r3 - p.kd_C * c
    dm = r4b - p.kd_m * m
    dfe = r_fe_gen + r_fe_supply - r4b - p.kd_fe * fe

    return [da, db, dc, dm, dpp, dfe]


# ============================================================================
# ODE INTEGRATION
# ============================================================================

# Default initial conditions (from tm6v3_full_verification.py)
DEFAULT_Y0 = [5e-3, 1e-4, 1e-3, 1e-4, 0.0, 5e-4]  # A, B, C, M, P, Fe


def integrate_ode(
    rhs_func,
    params,
    y0: list = None,
    t_end: float = 86400 * 3,
    rtol: float = 1e-10,
    atol: float = 1e-14,
    max_step: float = 100.0
):
    """
    Integrate 6-variable ODE to steady state.

    Args:
        rhs_func: RHS function (phase_d_rhs or original_rhs)
        params: Parameter dataclass
        y0: Initial conditions [A, B, C, M, P, Fe]
        t_end: Integration time in seconds (default: 3 days)
        rtol, atol: Solver tolerances
        max_step: Maximum step size

    Returns:
        ss: dict of steady-state concentrations
        sol: scipy OdeSolution object
    """
    if y0 is None:
        y0 = DEFAULT_Y0.copy()

    sol = solve_ivp(
        lambda t, y: rhs_func(t, y, params),
        [0, t_end],
        y0,
        method='LSODA',
        rtol=rtol,
        atol=atol,
        max_step=max_step
    )

    final = sol.y[:, -1]
    ss = dict(zip(VAR_NAMES, final))

    return ss, sol


def check_stability(ss: dict, rhs_func, params) -> dict:
    """Check linear stability via numerical Jacobian (6x6)."""
    y0 = np.array([ss[n] for n in VAR_NAMES])
    eps = 1e-8
    n = len(y0)
    J = np.zeros((n, n))
    f0 = np.array(rhs_func(0, y0, params))

    for j in range(n):
        y_pert = y0.copy()
        y_pert[j] += eps
        f_pert = np.array(rhs_func(0, y_pert, params))
        J[:, j] = (f_pert - f0) / eps

    eigenvalues = np.linalg.eigvals(J)
    max_real = float(np.max(np.real(eigenvalues)))
    stable = max_real < 0

    return {
        'stable': stable,
        'max_real_eigenvalue': max_real,
        'tau_relax_hours': float(1.0 / (-max_real) / 3600) if stable and max_real < 0 else None
    }


# ============================================================================
# MAIN INTEGRATION: FNO -> ODE
# ============================================================================

# Nominal G3c membrane parameters
NOMINAL_MEMBRANE_PARAMS = {
    'L_pent': 350.0,          # nm
    'L_mack': 35.0,           # nm
    'L_chamber': 10.0,        # um
    'delta_pH': 4.5,          # pH units
    'D_H_pent': 5e-27,        # m2/s
    'D_H_mack_intra': 1e-10,  # m2/s
    'k_cat': 1e-4,            # s-1 (mackinawite CO2RR catalysis)
}


def compute_fno_rate_from_A_steady(
    A_steady_mM: float,
    ode_params: PhaseDParams,
    newton_iters: int = 2,
    t_end: float = 200 * 3600,
) -> float:
    """
    Derive ODE fno_rate from FNO-predicted A_steady (mM).

    Physics: at ODE steady state, dA/dt = 0:
        fno_rate * hill(A) + k3*B*C - k2*A - kd_A*A = 0

    Step 1: Analytical guess (ignoring B+C contribution):
        fno_rate ≈ A * (kd_A + k2) / hill(A)

    Step 2: Newton correction — run ODE with guess, measure actual A*,
        scale fno_rate by (A_target / A_actual).

    Args:
        A_steady_mM: FNO-predicted formate concentration (mM)
        ode_params: ODE parameters (Ka, kd_A, k2, etc.)
        newton_iters: number of Newton refinement steps (default 2)
        t_end: ODE integration time for Newton steps (seconds)

    Returns:
        fno_rate (M/s) that makes ODE converge to A_steady_mM
    """
    A_target = A_steady_mM * 1e-3  # mM → M

    if A_target < 1e-6:
        return 0.0  # system dead in PDE → dead in ODE

    Ka = ode_params.Ka
    kd_A = ode_params.kd_A
    k2 = ode_params.k2

    # Step 1: Analytical guess
    hill_A = A_target**2 / (Ka**2 + A_target**2) if A_target > 0 else 1e-20
    hill_A = max(hill_A, 1e-20)
    fno_rate = A_target * (kd_A + k2) / hill_A

    # Step 2: Newton correction
    for i in range(newton_iters):
        p = PhaseDParams()
        # Copy all fields from ode_params except fno_rate
        for fld in ['Ka', 'km', 'k_fe_gen', 'fe_supply',
                     'kd_A', 'kd_B', 'kd_C', 'kd_m', 'kd_fe',
                     'k2', 'k3', 'c_supply',
                     'kp_on', 'kp_off', 'Kp', 'beta_max']:
            setattr(p, fld, getattr(ode_params, fld))
        p.fno_rate = fno_rate

        ss, _ = integrate_ode(phase_d_rhs, p, t_end=t_end, max_step=300.0)
        A_actual = ss['a']

        if A_actual < 1e-8:
            # ODE died — increase rate significantly
            fno_rate *= 10.0
            continue

        ratio = A_target / A_actual
        fno_rate *= ratio

        if abs(ratio - 1.0) < 0.01:
            break  # converged within 1%

    return fno_rate


def compute_fno_rate(
    J_formate: float,
    L_chamber_um: float
) -> float:
    """
    LEGACY: Convert FNO-predicted J_formate to effective ODE rate.
    Kept for backward compatibility. Use compute_fno_rate_from_A_steady() instead.

    Note: J_formate from datagen has units M·m/s (not mol/m²/s as documented),
    because PDE concentrations are in M and spatial coordinates in m.
    The * 1e-3 in the original formula was an error.
    """
    L_chamber_m = L_chamber_um * 1e-6
    if L_chamber_m <= 0:
        return 0.0
    # (M·m/s) / m = M/s — correct without the old * 1e-3
    fno_rate = J_formate / L_chamber_m
    return fno_rate


def run_phase_d(
    membrane_params: Dict[str, float],
    ode_params: Optional[PhaseDParams] = None,
    model_path: Optional[str] = None,
    scalers_path: Optional[str] = None,
    fno_model: Optional[MembraneModel] = None,
    fno_scalers: Optional[tuple] = None,
    t_end: float = 86400 * 3,
    y0: list = None,
    device: str = 'cpu'
) -> Dict:
    """
    Full Phase D pipeline: FNO predict -> ODE integrate.

    Steps:
      1. Load FNO (or use provided model)
      2. Predict J_formate from membrane_params
      3. Convert: fno_rate = J_formate / L_chamber * 1e-3
      4. Set fno_rate in ODE params
      5. Integrate ODE (3 days default)
      6. Return solution + diagnostics

    Args:
        membrane_params: dict with 7 membrane parameters
        ode_params: PhaseDParams (or None for defaults)
        model_path: Path to FNO checkpoint
        scalers_path: Path to scalers pickle (if separate)
        fno_model: Pre-loaded model (to avoid re-loading for scans)
        fno_scalers: Tuple (scaler_params, scaler_ph, scaler_scalars, x_grid, param_order)
        t_end: ODE integration time (seconds)
        y0: Initial conditions
        device: torch device

    Returns:
        dict with FNO predictions, ODE steady state, stability, diagnostics
    """
    # --- Step 1: Load or reuse FNO ---
    if fno_model is not None and fno_scalers is not None:
        model = fno_model
        scaler_params, scaler_ph, scaler_scalars, x_grid, param_order = fno_scalers
    else:
        if model_path is None:
            model_path = str(
                Path(__file__).parent / 'data' / 'oracle_membrane_fno.pt'
            )
        model, scaler_params, scaler_ph, scaler_scalars, x_grid, param_order = load_fno(
            model_path, scalers_path, device
        )

    # --- Step 2: FNO prediction ---
    fno_result = fno_predict(
        model, membrane_params,
        scaler_params, scaler_ph, scaler_scalars,
        x_grid, param_order, device
    )

    J_formate = fno_result['J_formate']
    A_steady_fno = fno_result['A_steady']

    # --- Step 3: Derive ODE rate from A_steady (v2: physics-based) ---
    if ode_params is None:
        ode_params = PhaseDParams()

    fno_rate = compute_fno_rate_from_A_steady(
        A_steady_fno, ode_params, newton_iters=2, t_end=t_end
    )

    print(f"  FNO prediction:")
    print(f"    J_formate   = {J_formate:.4e} (M·m/s, PDE units)")
    print(f"    A_steady    = {A_steady_fno:.4f} mM (FNO direct → used for rate)")
    print(f"    tau_transit = {fno_result['tau_transit']:.2e} s")
    print(f"    fno_rate    = {fno_rate:.4e} M/s (derived from A_steady)")

    # --- Step 4: Set fno_rate in ODE params ---
    ode_params.fno_rate = fno_rate

    # --- Step 5: Integrate ODE ---
    ss, sol = integrate_ode(phase_d_rhs, ode_params, y0=y0, t_end=t_end)

    alive = ss['a'] > 1e-4  # 0.1 mM threshold

    # --- Step 6: Stability check ---
    stab = check_stability(ss, phase_d_rhs, ode_params)

    print(f"  ODE steady state (Phase D):")
    print(f"    A*  = {ss['a']*1e3:.4f} mM")
    print(f"    B*  = {ss['b']*1e3:.4f} mM")
    print(f"    C*  = {ss['c']*1e3:.4f} mM")
    print(f"    M*  = {ss['m']*1e3:.4f} mM")
    print(f"    P*  = {ss['p']:.6f}")
    print(f"    Fe* = {ss['fe']*1e3:.4f} mM")
    print(f"    Alive: {'YES' if alive else 'NO'}")
    print(f"    Stable: {'YES' if stab['stable'] else 'NO'}")

    return {
        'membrane_params': membrane_params,
        'fno_prediction': {
            'J_formate': J_formate,
            'A_steady_mM': A_steady_fno,
            'tau_transit_s': fno_result['tau_transit'],
            'I_current': fno_result['I_current'],
        },
        'fno_rate': fno_rate,
        'ode_steady_state': {k: float(v) for k, v in ss.items()},
        'ode_steady_state_mM': {
            'A_mM': float(ss['a'] * 1e3),
            'B_mM': float(ss['b'] * 1e3),
            'C_mM': float(ss['c'] * 1e3),
            'M_mM': float(ss['m'] * 1e3),
            'P': float(ss['p']),
            'Fe_mM': float(ss['fe'] * 1e3),
        },
        'alive': alive,
        'stability': stab,
        'ph_profile': fno_result['ph_profile'],
        'sol': sol,
    }


# ============================================================================
# VALIDATION: Phase D vs Original TM6v3-full
# ============================================================================

def validate_against_original(
    model_path: Optional[str] = None,
    scalers_path: Optional[str] = None,
    device: str = 'cpu',
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Compare FNO-ODE (Phase D) vs original TM6v3-full ODE at nominal G3c params.

    The original model uses:
      r1 = k1 * f1_eff * hill_a
      where f1_eff = f1_max * M / (Km_f + M)
      with k1=1e-4, f1_max=5e-3, Km_f=5e-4

    At steady state (M* ~ 16.5 mM from DECISION-043):
      f1_eff = 5e-3 * 0.0165 / (5e-4 + 0.0165) ~ 4.85e-3
      Effective rate constant = k1 * f1_eff = 1e-4 * 4.85e-3 = 4.85e-7 M/s (at hill=1)

    Phase D replaces this with fno_rate * hill_a.

    Expected: A* ~ 4.43 mM (DECISION-043)
    """
    print("\n" + "=" * 70)
    print("  VALIDATION: Phase D (FNO-ODE) vs Original TM6v3-full")
    print("=" * 70)

    # --- Run original TM6v3-full ---
    print("\n[1/3] Running original TM6v3-full ODE...")
    orig_params = OriginalTM6v3Params()
    ss_orig, sol_orig = integrate_ode(
        original_rhs, orig_params,
        t_end=500 * 3600  # 500 hours (same as verification)
    )
    a_star_orig = ss_orig['a'] * 1e3

    print(f"  Original A* = {a_star_orig:.4f} mM")
    print(f"  Original M* = {ss_orig['m']*1e3:.4f} mM")

    # Compute effective rate at steady state for reference
    m_star = ss_orig['m']
    f1_eff_ss = orig_params.f1_max * m_star / (orig_params.Km_f + m_star)
    effective_k1_f1 = orig_params.k1 * f1_eff_ss
    print(f"  Effective k1*f1(M*) = {effective_k1_f1:.4e} M/s")

    # --- Run Phase D (FNO-ODE) ---
    print("\n[2/3] Running Phase D (FNO-ODE) at nominal G3c parameters...")
    phase_d_result = run_phase_d(
        membrane_params=NOMINAL_MEMBRANE_PARAMS,
        model_path=model_path,
        scalers_path=scalers_path,
        device=device,
    )

    a_star_fno = phase_d_result['ode_steady_state_mM']['A_mM']
    fno_rate = phase_d_result['fno_rate']

    # --- Compare ---
    print("\n[3/3] Comparison:")
    print(f"  Original  A* = {a_star_orig:.4f} mM")
    print(f"  Phase D   A* = {a_star_fno:.4f} mM")

    if a_star_orig > 0:
        relative_error = abs(a_star_fno - a_star_orig) / a_star_orig
        calibration_factor = a_star_orig / a_star_fno if a_star_fno > 0 else float('inf')
    else:
        relative_error = float('inf')
        calibration_factor = float('inf')

    print(f"  Relative error = {relative_error:.2%}")
    print(f"  Calibration factor = {calibration_factor:.4f}")
    print(f"  fno_rate = {fno_rate:.4e} M/s")
    print(f"  k1*f1(M*) = {effective_k1_f1:.4e} M/s (original)")

    match_ok = relative_error < 0.10  # 10% tolerance
    print(f"  Match within 10%: {'YES' if match_ok else 'NO'}")
    print(f"  (v2: fno_rate derived from A_steady, no global calibration factor)")

    result = {
        'original': {
            'A_star_mM': float(a_star_orig),
            'M_star_mM': float(ss_orig['m'] * 1e3),
            'effective_k1_f1': float(effective_k1_f1),
        },
        'phase_d': {
            'A_star_mM': float(a_star_fno),
            'fno_rate': float(fno_rate),
            'J_formate': phase_d_result['fno_prediction']['J_formate'],
            'A_steady_fno_mM': phase_d_result['fno_prediction']['A_steady_mM'],
        },
        'comparison': {
            'relative_error': float(relative_error),
            'calibration_factor': float(calibration_factor),
            'match_within_10pct': match_ok,
        },
        'nominal_membrane_params': NOMINAL_MEMBRANE_PARAMS,
    }

    # --- Plot comparison ---
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _plot_validation(sol_orig, phase_d_result['sol'], ss_orig,
                         phase_d_result['ode_steady_state'], output_dir)

        # Save pH profile
        if phase_d_result.get('ph_profile') is not None:
            _plot_ph_profile(phase_d_result['ph_profile'],
                             NOMINAL_MEMBRANE_PARAMS, output_dir)

    return result


def _plot_validation(sol_orig, sol_fno, ss_orig, ss_fno, output_dir: Path):
    """Plot ODE trajectories: original vs Phase D."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    labels = ['A (formate, mM)', 'B (pyruvate, mM)', 'C (glyoxylate, mM)',
              'M (membrane, mM)', 'P (PEDOT)', 'Fe (Fe2+, mM)']
    scales = [1e3, 1e3, 1e3, 1e3, 1.0, 1e3]
    keys = VAR_NAMES

    for i in range(6):
        ax = axes[i // 2][i % 2]
        t_orig_h = sol_orig.t / 3600
        t_fno_h = sol_fno.t / 3600

        ax.plot(t_orig_h, sol_orig.y[i] * scales[i],
                'b-', linewidth=1.5, label='Original', alpha=0.8)
        ax.plot(t_fno_h, sol_fno.y[i] * scales[i],
                'r--', linewidth=1.5, label='Phase D (FNO)', alpha=0.8)

        # Steady state markers
        ax.axhline(ss_orig[keys[i]] * scales[i], color='b', linestyle=':',
                    alpha=0.3, linewidth=0.8)
        ax.axhline(ss_fno[keys[i]] * scales[i], color='r', linestyle=':',
                    alpha=0.3, linewidth=0.8)

        ax.set_xlabel('Time (h)')
        ax.set_ylabel(labels[i])
        ax.set_title(labels[i])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Phase D Validation: Original TM6v3 vs FNO-ODE', fontsize=14)
    plt.tight_layout()
    path = output_dir / 'phase_d_validation_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_ph_profile(ph_profile: np.ndarray, membrane_params: dict, output_dir: Path):
    """Plot FNO-predicted pH profile across the membrane."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x_norm = np.linspace(0, 1, len(ph_profile))
    ax.plot(x_norm, ph_profile, 'b-', linewidth=2)
    ax.set_xlabel('Normalized position (0=acid, 1=alkaline)')
    ax.set_ylabel('pH')
    ax.set_title(
        f'FNO-predicted pH profile\n'
        f'L_pent={membrane_params["L_pent"]:.0f} nm, '
        f'L_mack={membrane_params["L_mack"]:.0f} nm, '
        f'L_ch={membrane_params["L_chamber"]:.0f} um, '
        f'dpH={membrane_params["delta_pH"]:.1f}'
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / 'phase_d_ph_profile_nominal.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# PARAMETER SCAN: Membrane geometry sweep
# ============================================================================

def membrane_geometry_scan(
    model_path: Optional[str] = None,
    scalers_path: Optional[str] = None,
    device: str = 'cpu',
    output_dir: Optional[Path] = None,
    n_grid: int = 10
) -> Dict:
    """
    Sweep membrane geometry while tracking ODE steady state via FNO-ODE.

    Scans:
      1. L_chamber x delta_pH (10x10 grid)
      2. L_pent x L_mack (10x10 grid)

    For each point: FNO predict -> compute fno_rate -> ODE integrate -> record A*, M*.

    Args:
        model_path: Path to FNO checkpoint
        scalers_path: Path to scalers pickle
        device: torch device
        output_dir: Where to save results/plots
        n_grid: Grid resolution per axis

    Returns:
        dict with scan results
    """
    print("\n" + "=" * 70)
    print("  MEMBRANE GEOMETRY SCAN (FNO-ODE)")
    print("=" * 70)

    # Load FNO once
    if model_path is None:
        model_path = str(
            Path(__file__).parent / 'data' / 'oracle_membrane_fno.pt'
        )
    model, scaler_params, scaler_ph, scaler_scalars, x_grid, param_order = load_fno(
        model_path, scalers_path, device
    )
    fno_scalers = (scaler_params, scaler_ph, scaler_scalars, x_grid, param_order)

    results = {}

    # ----------------------------------------------------------------
    # Scan 1: L_chamber x delta_pH
    # ----------------------------------------------------------------
    print("\n[Scan 1] L_chamber x delta_pH...")
    L_ch_values = np.linspace(5.0, 50.0, n_grid)     # um
    dpH_values = np.linspace(2.0, 7.0, n_grid)        # pH units

    A_star_grid_1 = np.zeros((n_grid, n_grid))
    M_star_grid_1 = np.zeros((n_grid, n_grid))
    J_grid_1 = np.zeros((n_grid, n_grid))
    alive_grid_1 = np.zeros((n_grid, n_grid), dtype=bool)

    total = n_grid * n_grid
    count = 0

    for i, L_ch in enumerate(L_ch_values):
        for j, dpH in enumerate(dpH_values):
            count += 1
            if count % 10 == 0:
                print(f"  {count}/{total}...", end=' ')

            mp = NOMINAL_MEMBRANE_PARAMS.copy()
            mp['L_chamber'] = float(L_ch)
            mp['delta_pH'] = float(dpH)

            try:
                fno_result = fno_predict(
                    model, mp,
                    scaler_params, scaler_ph, scaler_scalars,
                    x_grid, param_order, device
                )
                p = PhaseDParams()
                fno_rate = compute_fno_rate_from_A_steady(
                    fno_result['A_steady'], p, newton_iters=1, t_end=200 * 3600
                )
                p.fno_rate = fno_rate
                ss, _ = integrate_ode(phase_d_rhs, p, t_end=200 * 3600)

                A_star_grid_1[i, j] = ss['a'] * 1e3
                M_star_grid_1[i, j] = ss['m'] * 1e3
                J_grid_1[i, j] = fno_result['J_formate']
                alive_grid_1[i, j] = ss['a'] > 1e-4

            except Exception as e:
                print(f"\n  WARNING: Failed at L_ch={L_ch:.1f}, dpH={dpH:.1f}: {e}")
                A_star_grid_1[i, j] = 0.0
                M_star_grid_1[i, j] = 0.0
                J_grid_1[i, j] = 0.0
                alive_grid_1[i, j] = False

    print(f"\n  Alive: {alive_grid_1.sum()}/{total}")

    results['scan1_Lch_dpH'] = {
        'L_chamber_values': L_ch_values.tolist(),
        'delta_pH_values': dpH_values.tolist(),
        'A_star_mM': A_star_grid_1.tolist(),
        'M_star_mM': M_star_grid_1.tolist(),
        'J_formate': J_grid_1.tolist(),
        'alive': alive_grid_1.tolist(),
        'n_alive': int(alive_grid_1.sum()),
    }

    # ----------------------------------------------------------------
    # Scan 2: L_pent x L_mack
    # ----------------------------------------------------------------
    print("\n[Scan 2] L_pent x L_mack...")
    L_pent_values = np.linspace(100.0, 800.0, n_grid)  # nm
    L_mack_values = np.linspace(10.0, 100.0, n_grid)   # nm

    A_star_grid_2 = np.zeros((n_grid, n_grid))
    M_star_grid_2 = np.zeros((n_grid, n_grid))
    J_grid_2 = np.zeros((n_grid, n_grid))
    alive_grid_2 = np.zeros((n_grid, n_grid), dtype=bool)

    count = 0
    for i, L_pent in enumerate(L_pent_values):
        for j, L_mack in enumerate(L_mack_values):
            count += 1
            if count % 10 == 0:
                print(f"  {count}/{total}...", end=' ')

            mp = NOMINAL_MEMBRANE_PARAMS.copy()
            mp['L_pent'] = float(L_pent)
            mp['L_mack'] = float(L_mack)

            try:
                fno_result = fno_predict(
                    model, mp,
                    scaler_params, scaler_ph, scaler_scalars,
                    x_grid, param_order, device
                )
                p = PhaseDParams()
                fno_rate = compute_fno_rate_from_A_steady(
                    fno_result['A_steady'], p, newton_iters=1, t_end=200 * 3600
                )
                p.fno_rate = fno_rate
                ss, _ = integrate_ode(phase_d_rhs, p, t_end=200 * 3600)

                A_star_grid_2[i, j] = ss['a'] * 1e3
                M_star_grid_2[i, j] = ss['m'] * 1e3
                J_grid_2[i, j] = fno_result['J_formate']
                alive_grid_2[i, j] = ss['a'] > 1e-4

            except Exception as e:
                print(f"\n  WARNING: Failed at L_pent={L_pent:.0f}, L_mack={L_mack:.0f}: {e}")
                A_star_grid_2[i, j] = 0.0
                M_star_grid_2[i, j] = 0.0
                J_grid_2[i, j] = 0.0
                alive_grid_2[i, j] = False

    print(f"\n  Alive: {alive_grid_2.sum()}/{total}")

    results['scan2_Lpent_Lmack'] = {
        'L_pent_values': L_pent_values.tolist(),
        'L_mack_values': L_mack_values.tolist(),
        'A_star_mM': A_star_grid_2.tolist(),
        'M_star_mM': M_star_grid_2.tolist(),
        'J_formate': J_grid_2.tolist(),
        'alive': alive_grid_2.tolist(),
        'n_alive': int(alive_grid_2.sum()),
    }

    # ----------------------------------------------------------------
    # Find optimal geometry
    # ----------------------------------------------------------------
    # Best A* from scan 1
    if alive_grid_1.any():
        best_idx_1 = np.unravel_index(
            np.argmax(np.where(alive_grid_1, A_star_grid_1, 0)), A_star_grid_1.shape
        )
        best_Lch = L_ch_values[best_idx_1[0]]
        best_dpH = dpH_values[best_idx_1[1]]
        best_A1 = A_star_grid_1[best_idx_1]
        print(f"\n  Best from Scan 1: L_ch={best_Lch:.1f} um, dpH={best_dpH:.1f}, A*={best_A1:.4f} mM")
        results['optimal_scan1'] = {
            'L_chamber_um': float(best_Lch),
            'delta_pH': float(best_dpH),
            'A_star_mM': float(best_A1),
        }

    if alive_grid_2.any():
        best_idx_2 = np.unravel_index(
            np.argmax(np.where(alive_grid_2, A_star_grid_2, 0)), A_star_grid_2.shape
        )
        best_Lpent = L_pent_values[best_idx_2[0]]
        best_Lmack = L_mack_values[best_idx_2[1]]
        best_A2 = A_star_grid_2[best_idx_2]
        print(f"  Best from Scan 2: L_pent={best_Lpent:.0f} nm, L_mack={best_Lmack:.0f} nm, A*={best_A2:.4f} mM")
        results['optimal_scan2'] = {
            'L_pent_nm': float(best_Lpent),
            'L_mack_nm': float(best_Lmack),
            'A_star_mM': float(best_A2),
        }

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scan 1: A* heatmap (L_chamber x delta_pH)
        _plot_scan_heatmap(
            L_ch_values, dpH_values, A_star_grid_1, alive_grid_1,
            xlabel='L_chamber (um)', ylabel='delta_pH',
            title='Phase D: A* (mM) — L_chamber x delta_pH',
            cbar_label='A* (mM)',
            output_path=output_dir / 'phase_d_scan_Lch_dpH_Astar.png'
        )

        # Scan 1: J_formate heatmap
        _plot_scan_heatmap(
            L_ch_values, dpH_values, J_grid_1, alive_grid_1,
            xlabel='L_chamber (um)', ylabel='delta_pH',
            title='Phase D: J_formate (mol/m2/s) — L_chamber x delta_pH',
            cbar_label='J_formate (mol/m2/s)',
            output_path=output_dir / 'phase_d_scan_Lch_dpH_J.png',
            log_color=True
        )

        # Scan 2: A* heatmap (L_pent x L_mack)
        _plot_scan_heatmap(
            L_pent_values, L_mack_values, A_star_grid_2, alive_grid_2,
            xlabel='L_pent (nm)', ylabel='L_mack (nm)',
            title='Phase D: A* (mM) — L_pent x L_mack',
            cbar_label='A* (mM)',
            output_path=output_dir / 'phase_d_scan_Lpent_Lmack_Astar.png'
        )

        # Scan 2: J_formate heatmap
        _plot_scan_heatmap(
            L_pent_values, L_mack_values, J_grid_2, alive_grid_2,
            xlabel='L_pent (nm)', ylabel='L_mack (nm)',
            title='Phase D: J_formate (mol/m2/s) — L_pent x L_mack',
            cbar_label='J_formate (mol/m2/s)',
            output_path=output_dir / 'phase_d_scan_Lpent_Lmack_J.png',
            log_color=True
        )

        # Save results JSON
        json_path = output_dir / 'phase_d_scan_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\n  Saved scan results: {json_path}")

    return results


def _plot_scan_heatmap(
    x_values: np.ndarray, y_values: np.ndarray,
    z_grid: np.ndarray, alive_grid: np.ndarray,
    xlabel: str, ylabel: str, title: str, cbar_label: str,
    output_path: Path, log_color: bool = False
):
    """Plot 2D heatmap for a parameter scan."""
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(10, 7))

    # Mask dead regions
    z_plot = z_grid.copy()
    z_plot[~alive_grid] = np.nan

    if log_color:
        z_pos = z_plot.copy()
        z_pos[z_pos <= 0] = np.nan
        vmin = np.nanmin(z_pos) if np.any(~np.isnan(z_pos)) else 1e-20
        vmax = np.nanmax(z_pos) if np.any(~np.isnan(z_pos)) else 1e-10
        if vmin <= 0:
            vmin = 1e-20
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(x_values, y_values, z_pos.T,
                           cmap='viridis', norm=norm, shading='auto')
    else:
        im = ax.pcolormesh(x_values, y_values, z_plot.T,
                           cmap='viridis', shading='auto')

    cbar = plt.colorbar(im, ax=ax, label=cbar_label)

    # Mark dead regions with hatching
    dead_count = (~alive_grid).sum()
    if dead_count > 0:
        dead_overlay = np.where(~alive_grid, 1.0, np.nan)
        ax.pcolormesh(x_values, y_values, dead_overlay.T,
                      cmap='Greys', alpha=0.3, shading='auto')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ORACLE Phase D: FNO-ODE Integration for TM6v3'
    )
    parser.add_argument(
        '--model-path', type=str,
        default=None,
        help='Path to trained FNO checkpoint (.pt). '
             'Default: oracle/data/oracle_membrane_fno.pt'
    )
    parser.add_argument(
        '--scalers-path', type=str,
        default=None,
        help='Path to scalers pickle (.pkl). '
             'Default: auto-detect next to model.'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Run validation: compare FNO-ODE vs original TM6v3-full'
    )
    parser.add_argument(
        '--scan', action='store_true',
        help='Run membrane geometry scan (L_chamber x delta_pH + L_pent x L_mack)'
    )
    parser.add_argument(
        '--scan-resolution', type=int, default=10,
        help='Grid resolution for scan (default: 10x10)'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=None,
        help='Output directory for results. '
             'Default: oracle/phase_d_results/'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        choices=['cpu', 'cuda:0', 'cuda'],
        help='Compute device (default: cpu)'
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    digital_twin_dir = Path(__file__).parent

    if args.model_path is None:
        args.model_path = str(digital_twin_dir / 'data' / 'oracle_membrane_fno.pt')

    if args.output_dir is None:
        output_dir = digital_twin_dir / 'phase_d_results'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print("=" * 70)
    print("  ORACLE PHASE D: FNO-ODE Integration")
    print("=" * 70)
    print(f"  Model: {args.model_path}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}")

    all_results = {}

    # v2: No global calibration needed. Validation is optional (for comparison only).
    # Each scan point derives its own fno_rate from A_steady.

    # --- Validate ---
    if args.validate:
        t0 = time.time()
        val_result = validate_against_original(
            model_path=args.model_path,
            scalers_path=args.scalers_path,
            device=device,
            output_dir=output_dir
        )
        val_time = time.time() - t0
        val_result['time_s'] = round(val_time, 1)
        all_results['validation'] = val_result

        # Save validation JSON
        val_json = output_dir / 'phase_d_validation.json'
        with open(val_json, 'w') as f:
            json.dump(val_result, f, indent=2, cls=NumpyEncoder)
        print(f"\n  Saved validation results: {val_json}")

    # --- Scan ---
    if args.scan:
        t0 = time.time()
        scan_result = membrane_geometry_scan(
            model_path=args.model_path,
            scalers_path=args.scalers_path,
            device=device,
            output_dir=output_dir,
            n_grid=args.scan_resolution
        )
        scan_time = time.time() - t0
        scan_result['time_s'] = round(scan_time, 1)
        all_results['scan'] = scan_result

    # --- Default: just predict at nominal ---
    if not args.validate and not args.scan:
        print("\nNo --validate or --scan specified. Running single prediction at nominal G3c...")
        result = run_phase_d(
            membrane_params=NOMINAL_MEMBRANE_PARAMS,
            model_path=args.model_path,
            scalers_path=args.scalers_path,
            device=device,
        )
        all_results['nominal'] = {
            k: v for k, v in result.items()
            if k not in ('sol', 'ph_profile')
        }
        all_results['nominal']['ph_profile_summary'] = {
            'min': float(np.min(result['ph_profile'])),
            'max': float(np.max(result['ph_profile'])),
            'mean': float(np.mean(result['ph_profile'])),
        }

        # Save
        nom_json = output_dir / 'phase_d_nominal.json'
        with open(nom_json, 'w') as f:
            json.dump(all_results['nominal'], f, indent=2, cls=NumpyEncoder)
        print(f"\n  Saved: {nom_json}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  ORACLE PHASE D COMPLETE")
    print("=" * 70)

    if 'validation' in all_results:
        v = all_results['validation']
        print(f"  Validation:")
        print(f"    Original A* = {v['original']['A_star_mM']:.4f} mM")
        print(f"    Phase D  A* = {v['phase_d']['A_star_mM']:.4f} mM")
        print(f"    Error = {v['comparison']['relative_error']:.2%}")
        print(f"    Match < 10%: {'YES' if v['comparison']['match_within_10pct'] else 'NO'}")

    if 'scan' in all_results:
        s = all_results['scan']
        if 'optimal_scan1' in s:
            opt1 = s['optimal_scan1']
            print(f"  Scan 1 optimal: L_ch={opt1['L_chamber_um']:.1f} um, "
                  f"dpH={opt1['delta_pH']:.1f}, A*={opt1['A_star_mM']:.4f} mM")
        if 'optimal_scan2' in s:
            opt2 = s['optimal_scan2']
            print(f"  Scan 2 optimal: L_pent={opt2['L_pent_nm']:.0f} nm, "
                  f"L_mack={opt2['L_mack_nm']:.0f} nm, A*={opt2['A_star_mM']:.4f} mM")

    print(f"\n  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
