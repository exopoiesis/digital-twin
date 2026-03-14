#!/usr/bin/env python3
"""
ORACLE Phase B: PDE Membrane Data Generation for Neural Operator Training

Generates Latin Hypercube Sampling dataset over 7 PDE parameters:
  1. L_pent: [100, 1000] nm (pentlandite thickness)
  2. L_mack: [10, 100] nm (mackinawite thickness)
  3. L_chamber: [1, 100] μm (chamber size, symmetric both sides)
  4. delta_pH: [2, 8] (pH gradient: left acidic, right alkaline)
  5. D_H_pent: [1e-28, 1e-24] m²/s (H+ diffusion in pentlandite, log scale)
  6. D_H_mack_intra: [1e-12, 1e-8] m²/s (H+ diffusion in mackinawite intralayer, log scale)
  7. k_cat: [1e-6, 1e-2] s⁻¹ (mackinawite catalytic rate for R1, log scale)

For each parameter set:
  - Solve 1D PDE to steady state (24h)
  - Extract: pH(x) profile on fixed 256-point grid
  - Extract scalars: J_formate (mol/m²/s), I_current (A/m²), tau_transit (s), A_steady (mM)
  - Save to NPZ: params, ph_profiles, scalars, metadata

Designed for FNO/DeepONet training. Robust to solver failures (NaN for failed samples).
Parallel execution with multiprocessing.

Author: Third Matter Research Project
Date: 2026-03-10
"""

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from tqdm import tqdm


# ============================================================================
# CONSTANTS
# ============================================================================

F_CONST = 96485.0       # Faraday constant, C/mol
R_GAS = 8.314           # Gas constant, J/(mol·K)
T = 298.15              # Temperature, K (25°C)

N_GRID_OUTPUT = 256     # Fixed output grid size for pH profiles

# Parameter definitions: (name, lo, hi, log_scale)
PARAM_DEFS = [
    ('L_pent',           100e-9,   1000e-9,   False),  # m
    ('L_mack',            10e-9,    100e-9,   False),  # m
    ('L_chamber',          1e-6,     50e-6,   False),  # m (reduced from 100 to avoid extreme transit times)
    ('delta_pH',           2.0,       8.0,    False),  # dimensionless
    ('D_H_pent',         1e-27,     1e-18,    True),   # m²/s (wide: NEB ~5e-27 to defective/thin)
    ('D_H_mack_intra',   1e-12,     1e-8,     True),   # m²/s
    ('k_cat',            1e-6,      1e-2,     True),   # s⁻¹
]

PARAM_NAMES = [p[0] for p in PARAM_DEFS]

# Fixed parameters (not varied in LHS)
L_PEDOT = 100e-9        # m (100 nm)
D_H_PEDOT = 1e-6 * 1e-4 # m²/s (1e-6 cm²/s)
D_H_CHAMBER = 9.3e-5 * 1e-4  # m²/s (9.3e-5 cm²/s, water)
D_FE_CHAMBER = 7.2e-6 * 1e-4 # m²/s
D_CO2_CHAMBER = 1.9e-5 * 1e-4
D_HCOO_CHAMBER = 1.5e-5 * 1e-4

K_CAT_ALK = 1e-4        # s⁻¹, electrochemical R1 rate (fixed at nominal)
KM_CO2 = 1e-3           # M, Michaelis constant for CO2


# ============================================================================
# PARAMETER SPACE
# ============================================================================

def get_lhs_samples(n_samples: int, seed: int = 2026) -> np.ndarray:
    """Generate Latin Hypercube samples for the 7-parameter space."""
    sampler = qmc.LatinHypercube(d=len(PARAM_DEFS), seed=seed)
    samples_unit = sampler.random(n_samples)

    # Transform to physical space
    samples = np.zeros_like(samples_unit)
    for i, (name, lo, hi, log_scale) in enumerate(PARAM_DEFS):
        u = samples_unit[:, i]
        if log_scale:
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            samples[:, i] = 10 ** (log_lo + u * (log_hi - log_lo))
        else:
            samples[:, i] = lo + u * (hi - lo)

    return samples


# ============================================================================
# PDE SOLVER
# ============================================================================

def solve_pde_single(params: dict) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Solve 1D PDE for given parameters.

    5 zones: Chamber1 | PEDOT | Pentlandite | Mackinawite | Chamber2
    4 species: H+, Fe2+, CO2, HCOO-

    Args:
        params: Dictionary with keys matching PARAM_NAMES plus pH_left, pH_right

    Returns:
        (success, ph_profile_256, scalars_4)
        ph_profile_256: pH on uniform 256-point grid over total domain [0, L_total]
        scalars_4: [J_formate, I_current, tau_transit, A_steady]
    """
    # Suppress divide/invalid warnings for cleaner output
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        return _solve_pde_single_inner(params)


def _solve_pde_single_inner(params: dict) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Inner solver function (after warning suppression)."""
    # Extract parameters
    L_pent = params['L_pent']
    L_mack = params['L_mack']
    L_chamber = params['L_chamber']
    delta_pH = params['delta_pH']
    D_H_pent = params['D_H_pent']
    D_H_mack_intra = params['D_H_mack_intra']
    k_cat = params['k_cat']

    # Compute pH boundaries
    pH_right = 9.0  # alkaline chamber (fixed)
    pH_left = pH_right - delta_pH  # acidic chamber
    pH_left = max(pH_left, 1.0)  # clamp to physically reasonable range

    # Geometry
    L_ch1 = L_chamber
    L_ch2 = L_chamber
    X0 = 0.0
    X1 = L_ch1
    X2 = X1 + L_PEDOT
    X3 = X2 + L_pent
    X4 = X3 + L_mack
    X5 = X4 + L_ch2
    L_total = X5

    # Boundary conditions [H+, Fe2+, CO2, HCOO-] in M
    H_left = 10 ** (-pH_left)
    H_right = 10 ** (-pH_right)
    BC_left = np.array([H_left, 0.01, 0.0, 0.0])
    BC_right = np.array([H_right, 0.0, 0.033, 0.0])

    # Build nonuniform grid
    # Use moderate resolution to keep solve time ~10-30s per sample
    n_ch = 40
    n_pedot = 15
    n_pent = max(20, int(L_pent / 1e-8))  # at least 1 point per 10 nm
    n_pent = min(n_pent, 50)  # cap at 50
    n_mack = max(10, int(L_mack / 1e-9))  # at least 1 point per nm
    n_mack = min(n_mack, 40)  # cap at 40

    segments = [
        (X0, X1, n_ch, 0),       # Chamber 1
        (X1, X2, n_pedot, 1),    # PEDOT
        (X2, X3, n_pent, 2),     # Pentlandite
        (X3, X4, n_mack, 3),     # Mackinawite
        (X4, X5, n_ch, 4),       # Chamber 2
    ]

    x_list = []
    zone_list = []
    for x_start, x_end, n_pts, zone in segments:
        if n_pts > 0:
            pts = np.linspace(x_start, x_end, n_pts + 2)[1:-1]
            x_list.append(pts)
            zone_list.append(np.full(len(pts), zone, dtype=int))

    x = np.concatenate(x_list)
    zone_id = np.concatenate(zone_list)
    N = len(x)

    # Diffusion coefficients: (5 zones) x (4 species)
    # zone 0: Chamber1, 1: PEDOT, 2: Pentlandite, 3: Mackinawite, 4: Chamber2
    D_matrix = np.array([
        # H+             Fe2+          CO2            HCOO-
        [D_H_CHAMBER,    D_FE_CHAMBER, D_CO2_CHAMBER, D_HCOO_CHAMBER],  # Chamber1
        [D_H_PEDOT,      1e-7*1e-4,    1e-6*1e-4,     1e-7*1e-4      ],  # PEDOT
        [D_H_pent,       0.0,          0.0,           0.0            ],  # Pentlandite
        [D_H_mack_intra, 1e-8*1e-4,    1e-6*1e-4,     1e-7*1e-4      ],  # Mackinawite
        [D_H_CHAMBER,    D_FE_CHAMBER, D_CO2_CHAMBER, D_HCOO_CHAMBER],  # Chamber2
    ])

    # Get node diffusivities
    D_node = np.zeros((N, 4))
    for i in range(N):
        z = zone_id[i]
        D_node[i, :] = D_matrix[z, :]

    # Compute interface diffusivities (harmonic mean)
    dx_half = np.diff(x)
    D_half = np.zeros((N - 1, 4))
    for s in range(4):
        D_l, D_r = D_node[:-1, s], D_node[1:, s]
        mask = (D_l > 0) & (D_r > 0)
        D_half[:, s] = np.where(mask, 2.0 * D_l * D_r / (D_l + D_r), 0.0)

    # Boundary distances and diffusivities
    dx_left = x[0] - X0
    dx_right = X5 - x[-1]

    D_left_bc = np.zeros(4)
    D_right_bc = np.zeros(4)
    for s in range(4):
        D_ch1 = D_matrix[0, s]
        D_ch2 = D_matrix[4, s]
        D_n0, D_nN = D_node[0, s], D_node[-1, s]
        D_left_bc[s] = (2 * D_ch1 * D_n0 / (D_ch1 + D_n0)) if (D_ch1 + D_n0) > 0 else 0.0
        D_right_bc[s] = (2 * D_nN * D_ch2 / (D_nN + D_ch2)) if (D_nN + D_ch2) > 0 else 0.0

    # Central distances for divergence operator
    dx_center = np.zeros(N)
    dx_center[0] = (dx_left + dx_half[0]) / 2.0
    dx_center[-1] = (dx_half[-1] + dx_right) / 2.0
    for i in range(1, N - 1):
        dx_center[i] = (dx_half[i - 1] + dx_half[i]) / 2.0

    # Mask for mackinawite zone (reaction)
    is_mack = (zone_id == 3)

    # Stoichiometry for pH-dependent R1
    # Proton mechanism: CO2 + 2H+ + 2e- -> HCOO- + H2O: [-2, 0, -1, +1]
    # Electrochemical: CO2 + H2O + 2e- -> HCOO- + OH-: [-1, 0, -1, +1]
    stoich_p = np.array([-2.0, 0.0, -1.0, +1.0])
    stoich_e = np.array([-1.0, 0.0, -1.0, +1.0])

    # RHS function
    def rhs(t, y_flat):
        C = y_flat.reshape(N, 4)
        dCdt = np.zeros_like(C)

        # Diffusion for each species
        for s in range(4):
            c = C[:, s]
            flux_int = D_half[:, s] * np.diff(c) / dx_half
            flux_left = D_left_bc[s] * (c[0] - BC_left[s]) / dx_left
            flux_right = D_right_bc[s] * (BC_right[s] - c[-1]) / dx_right

            dCdt[0, s] = (flux_int[0] - flux_left) / dx_center[0]
            dCdt[1:-1, s] = (flux_int[1:] - flux_int[:-1]) / dx_center[1:-1]
            dCdt[-1, s] = (flux_right - flux_int[-1]) / dx_center[-1]

        # Reaction R1 in mackinawite (pH-dependent)
        if np.any(is_mack):
            c_H = np.maximum(C[is_mack, 0], 1e-14)
            c_CO2 = np.maximum(C[is_mack, 2], 0.0)

            pH_local = -np.log10(c_H)
            f_p = 1.0 / (1.0 + np.exp((pH_local - 6.0) / 0.5))  # proton fraction
            f_e = 1.0 - f_p  # electrochemical fraction

            rate_p = k_cat * c_CO2 * c_H / (KM_CO2 + c_CO2)
            rate_e = K_CAT_ALK * c_CO2 / (KM_CO2 + c_CO2)

            rate = f_p * rate_p + f_e * rate_e

            for s in range(4):
                dCdt[is_mack, s] += stoich_p[s] * f_p * rate_p + stoich_e[s] * f_e * rate_e

        return dCdt.flatten()

    # Initial condition (linear interpolation)
    C0 = np.zeros((N, 4))
    for s in range(4):
        C0[:, s] = BC_left[s] + (BC_right[s] - BC_left[s]) * (x - X0) / (X5 - X0)
    C0 = np.maximum(C0, 0.0)
    C0[:, 0] = np.maximum(C0[:, 0], 1e-15)  # H+ minimum
    y0 = C0.flatten()

    # Pre-filter: if pentlandite is effectively impermeable, return analytical solution
    t_end = 86400.0  # 24 hours in seconds
    tau_transit_pent = L_pent**2 / (2 * D_H_pent) if D_H_pent > 0 else np.inf
    if tau_transit_pent > 2 * t_end:
        # Membrane blocks H+ completely on simulation timescale.
        # Analytical solution: each chamber stays at its boundary pH.
        # Mackinawite sees alkaline pH -> electrochemical CO2 reduction only.
        x_uniform = np.linspace(X0, X5, N_GRID_OUTPUT)
        pH_analytical = np.where(x_uniform < X3, pH_left, pH_right)
        # Smooth transition across pentlandite (cosmetic)
        in_pent = (x_uniform >= X2) & (x_uniform <= X3)
        if np.any(in_pent):
            frac = (x_uniform[in_pent] - X2) / max(X3 - X2, 1e-15)
            pH_analytical[in_pent] = pH_left + frac * (pH_right - pH_left)

        # Scalars for impermeable case
        c_CO2_alk = BC_right[2]  # CO2 in alkaline chamber
        rate_e_ss = K_CAT_ALK * c_CO2_alk / (KM_CO2 + c_CO2_alk)
        A_ss = rate_e_ss * L_mack / D_HCOO_CHAMBER * 1e3  # mM, rough estimate
        J_formate_ss = rate_e_ss * L_mack  # mol/m²/s
        I_current_ss = 0.0  # no H+ current through impermeable membrane
        scalars_analytical = np.array([J_formate_ss, I_current_ss, tau_transit_pent, A_ss])
        return True, pH_analytical, scalars_analytical

    # Solve PDE with timeout mechanism
    timeout_seconds = 60.0
    t_start_solve = time.time()

    # Callback to check wall-clock timeout
    def timeout_event(t, y):
        elapsed = time.time() - t_start_solve
        return -1.0 if elapsed < timeout_seconds else 1.0

    timeout_event.terminal = True
    timeout_event.direction = 1

    # Create t_eval for controlled output density
    t_eval = np.linspace(0, t_end, 500)

    try:
        sol = solve_ivp(
            rhs,
            (0, t_end),
            y0,
            method='BDF',
            rtol=1e-5,      # relaxed from 1e-7
            atol=1e-8,      # relaxed from 1e-11
            max_step=300.0,
            t_eval=t_eval,
            events=timeout_event,
            dense_output=False,
        )

        # Check for timeout
        elapsed = time.time() - t_start_solve
        if elapsed >= timeout_seconds or not sol.success:
            return False, np.full(N_GRID_OUTPUT, np.nan), np.full(4, np.nan)

        C_final = sol.y[:, -1].reshape(N, 4)

    except Exception:
        return False, np.full(N_GRID_OUTPUT, np.nan), np.full(4, np.nan)

    # Extract pH profile on uniform 256-point grid
    x_uniform = np.linspace(X0, X5, N_GRID_OUTPUT)
    H_final = np.maximum(C_final[:, 0], 1e-14)
    pH_profile = -np.log10(H_final)

    # Interpolate to uniform grid
    pH_uniform = np.interp(x_uniform, x, pH_profile)

    # Compute scalar outputs
    # 1. J_formate: formate flux from mackinawite to chamber2 (mol/m²/s)
    mack_indices = np.where(is_mack)[0]
    ch2_indices = np.where(zone_id == 4)[0]

    if len(mack_indices) > 0 and len(ch2_indices) > 0:
        i_m = mack_indices[-1]
        i_c = ch2_indices[0]
        dx_mc = x[i_c] - x[i_m]
        D_if = D_matrix[3, 3]  # HCOO- in mackinawite
        D_ch2 = D_matrix[4, 3]
        D_interface = (2 * D_if * D_ch2 / (D_if + D_ch2)) if (D_if + D_ch2) > 0 else 0.0
        J_formate = D_interface * (C_final[i_m, 3] - C_final[i_c, 3]) / dx_mc  # mol/m²/s
    else:
        J_formate = 0.0

    # 2. I_current: proton current through pentlandite (A/m²)
    pent_indices = np.where(zone_id == 2)[0]
    if len(pent_indices) > 1:
        i_p0 = pent_indices[0]
        i_pN = pent_indices[-1]
        dx_pent = x[i_pN] - x[i_p0]
        J_H = D_H_pent * (C_final[i_p0, 0] - C_final[i_pN, 0]) / dx_pent  # mol/m²/s
        I_current = J_H * F_CONST  # A/m²
    else:
        I_current = 0.0

    # 3. tau_transit: time for H+ to diffuse through pentlandite (s)
    tau_transit = L_pent**2 / (2 * D_H_pent) if D_H_pent > 0 else np.inf

    # 4. A_steady: steady-state formate concentration in chamber 2 (mM)
    if len(ch2_indices) > 0:
        A_steady = np.mean(C_final[ch2_indices, 3]) * 1e3  # M -> mM
    else:
        A_steady = 0.0

    scalars = np.array([J_formate, I_current, tau_transit, A_steady])

    return True, pH_uniform, scalars


# ============================================================================
# WORKER
# ============================================================================

def _run_single_wrapper(args):
    """Wrapper for multiprocessing."""
    return run_single_pde(*args)


def run_single_pde(idx: int, param_values: np.ndarray) -> Tuple:
    """Run single PDE simulation."""
    params = {name: val for name, val in zip(PARAM_NAMES, param_values)}

    success, ph_profile, scalars = solve_pde_single(params)

    return idx, success, ph_profile, scalars


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def run_datagen(n_samples: int, workers: int, output: str, seed: int = 2026):
    """Generate PDE dataset."""
    print("ORACLE Phase B: PDE Membrane Data Generation")
    print(f"Samples: {n_samples}")
    print(f"Workers: {workers}")
    print(f"Output: {output}")
    print()

    # Generate LHS samples
    print("Generating Latin Hypercube samples...")
    samples = get_lhs_samples(n_samples, seed=seed)

    # Print parameter ranges
    print("\nParameter ranges:")
    for i, (name, lo, hi, log_scale) in enumerate(PARAM_DEFS):
        scale_str = "log" if log_scale else "linear"
        if name.startswith('L_'):
            unit = "m"
            lo_str = f"{lo*1e9:.1f} nm" if lo < 1e-6 else f"{lo*1e6:.1f} μm"
            hi_str = f"{hi*1e9:.1f} nm" if hi < 1e-6 else f"{hi*1e6:.1f} μm"
        elif name.startswith('D_'):
            unit = "m²/s"
            lo_str = f"{lo:.2e}"
            hi_str = f"{hi:.2e}"
        elif name == 'k_cat':
            unit = "s⁻¹"
            lo_str = f"{lo:.2e}"
            hi_str = f"{hi:.2e}"
        else:
            unit = ""
            lo_str = f"{lo:.1f}"
            hi_str = f"{hi:.1f}"
        print(f"  {name:20s}: [{lo_str:>12s}, {hi_str:>12s}] ({scale_str})")
    print()

    # Run simulations
    results = []
    t_start = time.time()

    print("Running PDE simulations...")
    SIM_TIMEOUT = 90  # seconds per simulation (hard kill)
    with mp.Pool(workers) as pool:
        args = [(idx, samples[idx]) for idx in range(n_samples)]
        async_results = [pool.apply_async(_run_single_wrapper, (a,)) for a in args]

        for ar in tqdm(async_results, total=n_samples, desc="Simulations", unit="sim"):
            try:
                res = ar.get(timeout=SIM_TIMEOUT)
                results.append(res)
            except mp.TimeoutError:
                # Worker hung — record as failed
                idx_failed = len(results)
                results.append((idx_failed, False, np.full(N_GRID_OUTPUT, np.nan), np.full(4, np.nan)))
            except Exception:
                idx_failed = len(results)
                results.append((idx_failed, False, np.full(N_GRID_OUTPUT, np.nan), np.full(4, np.nan)))

    # Sort by index
    results.sort(key=lambda x: x[0])

    # Unpack results
    success = np.array([r[1] for r in results], dtype=bool)
    ph_profiles = np.array([r[2] for r in results], dtype=np.float32)  # (N, 256)
    scalars = np.array([r[3] for r in results], dtype=np.float32)      # (N, 4)

    # Build x_grid (uniform over [0, 1], physical scale stored in metadata)
    x_grid = np.linspace(0, 1, N_GRID_OUTPUT, dtype=np.float32)

    # Save results
    param_ranges = np.array([[p[1], p[2]] for p in PARAM_DEFS], dtype=np.float32)
    param_log_scale = np.array([p[3] for p in PARAM_DEFS], dtype=bool)

    scalar_names = ['J_formate', 'I_current', 'tau_transit', 'A_steady']
    scalar_units = ['mol/m²/s', 'A/m²', 's', 'mM']

    np.savez_compressed(
        output,
        param_names=np.array(PARAM_NAMES),
        param_ranges=param_ranges,
        param_log_scale=param_log_scale,
        scalar_names=np.array(scalar_names),
        scalar_units=np.array(scalar_units),
        samples=samples.astype(np.float32),
        success=success,
        ph_profiles=ph_profiles,
        scalars=scalars,
        x_grid=x_grid,
        n_grid=N_GRID_OUTPUT,
        # Metadata
        T_kelvin=T,
        F_const=F_CONST,
        R_gas=R_GAS,
        K_CAT_ALK=K_CAT_ALK,
        KM_CO2=KM_CO2,
        L_PEDOT=L_PEDOT,
        D_H_PEDOT=D_H_PEDOT,
        D_H_CHAMBER=D_H_CHAMBER,
    )

    # Summary
    elapsed = time.time() - t_start
    n_success = success.sum()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples: {n_samples}")
    print(f"Successful: {n_success} ({100*n_success/n_samples:.2f}%)")
    print(f"Failed: {n_samples - n_success} ({100*(n_samples - n_success)/n_samples:.2f}%)")
    print(f"Total time: {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    print(f"Average rate: {n_samples/elapsed:.2f} sim/s")
    print(f"Output: {output}")

    # Scalar statistics (for successful samples)
    if n_success > 0:
        print()
        print("Scalar output statistics (successful samples):")
        scalars_ok = scalars[success]
        for i, (name, unit) in enumerate(zip(scalar_names, scalar_units)):
            vals = scalars_ok[:, i]
            vals_finite = vals[np.isfinite(vals)]
            if len(vals_finite) > 0:
                print(f"  {name:15s}: min={np.min(vals_finite):.2e}, "
                      f"max={np.max(vals_finite):.2e}, "
                      f"mean={np.mean(vals_finite):.2e} {unit}")
            else:
                print(f"  {name:15s}: (all NaN/inf)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='ORACLE Phase B: PDE Data Generation')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of workers (default: CPU count)')
    parser.add_argument('--output', type=str, default='oracle_membrane_data.npz', help='Output file')
    parser.add_argument('--seed', type=int, default=2026, help='Random seed for LHS')
    parser.add_argument('--test', action='store_true', help='Test mode (10 samples, 1 worker)')

    args = parser.parse_args()

    if args.test:
        print("TEST MODE: 10 samples, 1 worker")
        run_datagen(10, 1, 'oracle_membrane_test.npz', seed=args.seed)
    else:
        workers = args.n_workers if args.n_workers is not None else mp.cpu_count()
        run_datagen(args.n_samples, workers, args.output, seed=args.seed)


if __name__ == '__main__':
    main()
