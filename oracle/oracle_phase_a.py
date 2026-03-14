#!/usr/bin/env python3
"""
ORACLE Phase A: TM6v3-minimal Parameter Space Exploration

Scans 12-dimensional parameter space of the minimal third matter model
using Sobol quasi-random sampling. For each parameter set:
  1. Solve ODE to steady state (500h)
  2. If alive: run Gillespie SSA for 72h from ODE steady state
  3. Record: survival, time-to-death, final concentrations, max A, ODE A*

Model: 3 species (A=formate, M=membrane, Fe=Fe²⁺), 7 reactions
Total Sobol sequence: 2^17 = 131,072 samples
Designed for distributed execution: local (0-65535), gomer (65536-131071)

Author: Third Matter Research Project
Date: 2026-03-09
"""

import argparse
import multiprocessing as mp
import sys
import time
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats.qmc import Sobol


# ============================================================================
# CONSTANTS
# ============================================================================

N_AVOGADRO = 6.022e23
R_GAS = 8.314  # J/(mol·K)
T_REF = 298.15  # K (25°C)

TOTAL_SOBOL = 131072  # 2^17

# Parameter definitions: (name, default, lo, hi, log_scale)
PARAM_DEFS = [
    ('k1',         1e-4,      1e-6,   1e-1,   True),
    ('Ka',         7e-4,      1e-5,   1e-2,   True),
    ('f1',         5e-3,      1e-4,   1.0,    True),
    ('Km_f',       5e-4,      1e-5,   1e-2,   True),
    ('km',         3e-2,      1e-4,   1.0,    True),
    ('k_fe_gen',   5e-5,      1e-7,   1e-3,   True),
    ('fe_supply',  5e-8,      1e-10,  1e-5,   True),
    ('kd_A',       1e-4,      1e-7,   1e-2,   True),
    ('kd_m',       3e-6,      1e-9,   1e-3,   True),
    ('kd_fe',      3e-4,      1e-6,   1e-1,   True),
    ('N_A',        100,       10,     5000,   True),
    ('T_celsius',  25,        5,      150,    False),
]

PARAM_NAMES = [p[0] for p in PARAM_DEFS]

# Activation energies (J/mol)
E_A = {
    'k1': 50000,
    'km': 40000,
    'k_fe_gen': 40000,
    'fe_supply': 60000,
    'kd_A': 30000,
    'kd_m': 80000,
    'kd_fe': 30000,
}

# Stoichiometry matrix: 7 reactions × 3 species [A, M, Fe]
NU = np.array([
    [+1,  0,  0],   # R1: → A
    [ 0, +1, -1],   # R2: Fe,A → M
    [ 0,  0, +1],   # R3: A → Fe
    [ 0,  0, +1],   # R4: → Fe
    [-1,  0,  0],   # D1: A →
    [ 0, -1,  0],   # D2: M →
    [ 0,  0, -1],   # D3: Fe →
], dtype=np.int64)


# ============================================================================
# PARAMETER SPACE
# ============================================================================

def get_sobol_samples(start: int, count: int, seed: int = 2026) -> np.ndarray:
    """Generate Sobol samples for the given range."""
    sampler = Sobol(d=len(PARAM_DEFS), scramble=True, seed=seed)
    if start > 0:
        sampler.fast_forward(start)
    samples_unit = sampler.random(count)

    # Transform to physical space
    samples = np.zeros_like(samples_unit)
    for i, (name, default, lo, hi, log_scale) in enumerate(PARAM_DEFS):
        u = samples_unit[:, i]
        if log_scale:
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            samples[:, i] = 10 ** (log_lo + u * (log_hi - log_lo))
        else:
            samples[:, i] = lo + u * (hi - lo)

    # N_A must be integer
    samples[:, 10] = np.round(samples[:, 10])

    return samples


def apply_temperature_scaling(params: dict, T_celsius: float) -> dict:
    """Apply Arrhenius temperature scaling to rate constants."""
    T = T_celsius + 273.15
    factor_T = 1.0 / T - 1.0 / T_REF

    scaled = params.copy()
    for k, e_a in E_A.items():
        if k in scaled:
            scaled[k] = scaled[k] * np.exp(-e_a / R_GAS * factor_T)

    return scaled


# ============================================================================
# ODE MODEL
# ============================================================================

def ode_rhs(t, y, p):
    """Right-hand side of ODE system."""
    a, m, fe = np.maximum(y, 0)  # clip to non-negative

    f1_eff = p['f1'] * m / (p['Km_f'] + m)
    hill_a = a**2 / (p['Ka']**2 + a**2)

    dadt = p['k1'] * f1_eff * hill_a - p['kd_A'] * a
    dmdt = p['km'] * fe * a - p['kd_m'] * m
    dfedt = p['k_fe_gen'] * a + p['fe_supply'] - p['km'] * fe * a - p['kd_fe'] * fe

    return [dadt, dmdt, dfedt]


def solve_ode_steady_state(params: dict) -> Tuple[bool, float, np.ndarray]:
    """
    Solve ODE to steady state.

    Returns:
        (alive, a_star, final_state)
    """
    y0 = [0.01, 0.01, 0.001]  # M
    t_span = (0, 500 * 3600)  # 500 hours in seconds

    try:
        sol = solve_ivp(
            ode_rhs,
            t_span,
            y0,
            args=(params,),
            method='LSODA',
            rtol=1e-8,
            atol=1e-12,
            dense_output=False,
            max_step=3600,  # max 1 hour steps to limit stiff computation
        )

        if not sol.success:
            return False, 0.0, np.zeros(3)

        final = np.maximum(sol.y[:, -1], 0)
        a_star = final[0]

        alive = a_star > 1e-8

        return alive, a_star, final

    except Exception:
        return False, 0.0, np.zeros(3)


# ============================================================================
# GILLESPIE SSA
# ============================================================================

def gillespie_adaptive(
    n_init: np.ndarray,
    params: dict,
    omega: float,
    t_max_h: float = 72.0,
    seed: int = 42
) -> Tuple[bool, float, np.ndarray, int]:
    """
    Run adaptive tau-leaping Gillespie SSA.

    Args:
        n_init: Initial molecule counts [n_A, n_M, n_Fe]
        params: Rate constants (already temperature-scaled)
        omega: Omega = N_A / A_star_ode for concentration conversion
        t_max_h: Simulation time in hours
        seed: RNG seed

    Returns:
        (survived, t_death_h, final_counts, max_n_A)
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    NU_T = NU.T.copy()  # (3, 7) contiguous for fast matmul

    n = n_init.astype(np.float64)  # use float for speed (avoid int overflow issues)
    t_max = t_max_h * 3600
    max_n_A = n[0]

    # Pre-extract params (avoid dict lookups in hot loop)
    p_k1 = params['k1']
    p_Ka2 = params['Ka'] ** 2
    p_f1 = params['f1']
    p_Km_f = params['Km_f']
    p_km = params['km']
    p_k_fe_gen = params['k_fe_gen']
    p_fe_supply = params['fe_supply']
    p_kd_A = params['kd_A']
    p_kd_m = params['kd_m']
    p_kd_fe = params['kd_fe']

    props = np.empty(7)

    # Fixed tau-leaping: 25920 steps max (tau=10s, 72h).
    # For GP surrogate, small tau-leaping bias is acceptable.
    # Adaptive tau: scale by expected relative change.
    t = 0.0
    while t < t_max:
        if n[0] <= 0:
            return False, t / 3600, n.astype(np.int64), int(max_n_A)

        if n[0] > max_n_A:
            max_n_A = n[0]

        # Propensities
        a = n[0] / omega
        m = n[1] / omega
        denom_m = p_Km_f + m
        f1_eff = p_f1 * m / denom_m if denom_m > 0 else 0.0
        hill_a = a * a / (p_Ka2 + a * a) if (p_Ka2 + a * a) > 0 else 0.0

        props[0] = p_k1 * f1_eff * hill_a * omega
        props[1] = p_km * n[2] * n[0] / omega
        props[2] = p_k_fe_gen * n[0]
        props[3] = p_fe_supply * omega
        props[4] = p_kd_A * n[0]
        props[5] = p_kd_m * n[1]
        props[6] = p_kd_fe * n[2]

        a0 = props.sum()
        if a0 <= 0:
            survived = n[0] > 0
            return survived, (t / 3600 if not survived else t_max_h), n.astype(np.int64), int(max_n_A)

        # Adaptive tau: allow ~20% expected change per step, cap at [10, 120] seconds.
        # For GP surrogate, coarse tau-leaping is fine — we need survival statistics,
        # not exact trajectories. This keeps max iterations at ~25920.
        net_rate = NU_T @ props  # (3,) molecules/s
        tau = 120.0  # default (max)
        for i in range(3):
            if net_rate[i] < 0 and n[i] > 1:
                tau_i = 0.2 * n[i] / abs(net_rate[i])
                if tau_i < tau:
                    tau = tau_i
        tau = max(tau, 10.0)  # minimum 10s
        tau = min(tau, t_max - t)
        if tau <= 0:
            break

        # Tau-leaping step (vectorized)
        firings = rng.poisson(np.maximum(props * tau, 0))
        n += NU_T @ firings
        # Clip to non-negative
        n[0] = max(n[0], 0)
        n[1] = max(n[1], 0)
        n[2] = max(n[2], 0)
        t += tau

    survived = n[0] > 0
    return survived, (t / 3600 if not survived else t_max_h), n.astype(np.int64), int(max_n_A)


# ============================================================================
# SIMULATION WORKER
# ============================================================================

def _run_single_wrapper(args):
    """Wrapper for multiprocessing (top-level function for pickling)."""
    return run_single(*args)


def run_single(idx: int, param_values: np.ndarray) -> Tuple:
    """Run single parameter set: ODE + Gillespie."""
    # Build parameter dict
    params_raw = {name: val for name, val in zip(PARAM_NAMES, param_values)}
    T_celsius = params_raw.pop('T_celsius')
    N_A_molecules = int(params_raw.pop('N_A'))

    # Apply temperature scaling
    params = apply_temperature_scaling(params_raw, T_celsius)

    # Solve ODE
    ode_alive, a_star, final_ode = solve_ode_steady_state(params)

    if not ode_alive or a_star < 1e-8:
        return (
            idx,
            0,        # survived
            0.0,      # t_death_h
            0.0,      # final_A
            0.0,      # final_M
            0.0,      # final_Fe
            0,        # max_A_molecules
            0,        # ode_alive
            0.0,      # a_star_ode
        )

    # Gillespie from ODE steady state
    omega = N_A_molecules / a_star  # molecules per M
    n_init = np.round(final_ode * omega).astype(np.int64)
    n_init = np.maximum(n_init, 1)  # at least 1 molecule each

    seed = idx * 137 + 42
    survived, t_death_h, final_n, max_n_A = gillespie_adaptive(
        n_init, params, omega, t_max_h=72.0, seed=seed
    )

    # Convert back to concentrations
    final_conc = final_n / omega

    return (
        idx,
        int(survived),
        t_death_h,
        final_conc[0],
        final_conc[1],
        final_conc[2],
        max_n_A,
        1,  # ode_alive
        a_star,
    )


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def run_batch(start: int, count: int, workers: int, output: str, progress: int = 500):
    """Run batch of simulations."""
    print(f"ORACLE Phase A: TM6v3-minimal")
    print(f"Sobol range: {start} to {start + count - 1} (total: {count})")
    print(f"Workers: {workers}")
    print(f"Output: {output}")
    print()

    # Generate samples
    samples = get_sobol_samples(start, count)
    indices = np.arange(start, start + count, dtype=np.int32)

    # Run simulations
    results = []
    t_start = time.time()

    with mp.Pool(workers) as pool:
        args = [(idx, samples[j]) for j, idx in enumerate(indices)]

        for i, res in enumerate(pool.imap_unordered(_run_single_wrapper, args), 1):
            results.append(res)

            if i % progress == 0 or i == count:
                elapsed = time.time() - t_start
                rate = i / elapsed
                eta = (count - i) / rate if rate > 0 else 0

                # Compute statistics
                ode_alive_count = sum(r[7] for r in results)
                survived_count = sum(r[1] for r in results)

                print(f"Progress: {i}/{count} | "
                      f"ODE alive: {ode_alive_count}/{i} ({100*ode_alive_count/i:.1f}%) | "
                      f"Gillespie survived: {survived_count}/{i} ({100*survived_count/i:.1f}%) | "
                      f"Rate: {rate:.1f} sim/s | "
                      f"ETA: {eta/60:.1f} min")

    # Sort by index
    results.sort(key=lambda x: x[0])

    # Unpack results
    result_indices = np.array([r[0] for r in results], dtype=np.int32)
    survived = np.array([r[1] for r in results], dtype=np.int8)
    t_death_h = np.array([r[2] for r in results], dtype=np.float32)
    final_A = np.array([r[3] for r in results], dtype=np.float64)
    final_M = np.array([r[4] for r in results], dtype=np.float64)
    final_Fe = np.array([r[5] for r in results], dtype=np.float64)
    max_A_molecules = np.array([r[6] for r in results], dtype=np.int64)
    ode_alive = np.array([r[7] for r in results], dtype=np.int8)
    a_star_ode = np.array([r[8] for r in results], dtype=np.float64)

    # Save results
    param_ranges = np.array([[p[2], p[3]] for p in PARAM_DEFS])
    param_log_scale = np.array([p[4] for p in PARAM_DEFS])
    e_a_values = np.array([E_A.get(name, 0) for name in PARAM_NAMES])

    np.savez_compressed(
        output,
        param_names=np.array(PARAM_NAMES),
        param_ranges=param_ranges,
        param_log_scale=param_log_scale,
        e_a_values=e_a_values,
        total_sobol=TOTAL_SOBOL,
        indices=result_indices,
        samples=samples,
        survived=survived,
        t_death_h=t_death_h,
        final_A=final_A,
        final_M=final_M,
        final_Fe=final_Fe,
        max_A_molecules=max_A_molecules,
        ode_alive=ode_alive,
        a_star_ode=a_star_ode,
    )

    # Final summary
    elapsed_total = time.time() - t_start
    ode_alive_total = ode_alive.sum()
    survived_total = survived.sum()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples: {count}")
    print(f"ODE alive: {ode_alive_total} ({100*ode_alive_total/count:.2f}%)")
    print(f"Gillespie survived: {survived_total} ({100*survived_total/count:.2f}%)")
    print(f"Total time: {elapsed_total/60:.1f} min ({elapsed_total/3600:.2f} h)")
    print(f"Average rate: {count/elapsed_total:.1f} sim/s")
    print(f"Output: {output}")
    print("=" * 70)


def merge_files(files: list, output: str):
    """Merge multiple NPZ files."""
    print(f"Merging {len(files)} files into {output}")

    all_indices = []
    all_samples = []
    all_survived = []
    all_t_death_h = []
    all_final_A = []
    all_final_M = []
    all_final_Fe = []
    all_max_A_molecules = []
    all_ode_alive = []
    all_a_star_ode = []

    metadata = None

    for fname in files:
        data = np.load(fname)

        if metadata is None:
            metadata = {
                'param_names': data['param_names'],
                'param_ranges': data['param_ranges'],
                'param_log_scale': data['param_log_scale'],
                'e_a_values': data['e_a_values'],
                'total_sobol': data['total_sobol'],
            }

        all_indices.append(data['indices'])
        all_samples.append(data['samples'])
        all_survived.append(data['survived'])
        all_t_death_h.append(data['t_death_h'])
        all_final_A.append(data['final_A'])
        all_final_M.append(data['final_M'])
        all_final_Fe.append(data['final_Fe'])
        all_max_A_molecules.append(data['max_A_molecules'])
        all_ode_alive.append(data['ode_alive'])
        all_a_star_ode.append(data['a_star_ode'])

    # Concatenate
    indices = np.concatenate(all_indices)
    samples = np.concatenate(all_samples)
    survived = np.concatenate(all_survived)
    t_death_h = np.concatenate(all_t_death_h)
    final_A = np.concatenate(all_final_A)
    final_M = np.concatenate(all_final_M)
    final_Fe = np.concatenate(all_final_Fe)
    max_A_molecules = np.concatenate(all_max_A_molecules)
    ode_alive = np.concatenate(all_ode_alive)
    a_star_ode = np.concatenate(all_a_star_ode)

    # Sort by index
    sort_idx = np.argsort(indices)

    np.savez_compressed(
        output,
        **metadata,
        indices=indices[sort_idx],
        samples=samples[sort_idx],
        survived=survived[sort_idx],
        t_death_h=t_death_h[sort_idx],
        final_A=final_A[sort_idx],
        final_M=final_M[sort_idx],
        final_Fe=final_Fe[sort_idx],
        max_A_molecules=max_A_molecules[sort_idx],
        ode_alive=ode_alive[sort_idx],
        a_star_ode=a_star_ode[sort_idx],
    )

    print(f"Merged {len(indices)} samples into {output}")
    print(f"ODE alive: {ode_alive.sum()} ({100*ode_alive.sum()/len(indices):.2f}%)")
    print(f"Gillespie survived: {survived.sum()} ({100*survived.sum()/len(indices):.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='ORACLE Phase A: TM6v3-minimal')
    parser.add_argument('--start', type=int, help='Starting Sobol index')
    parser.add_argument('--count', type=int, help='Number of samples')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--output', type=str, default='oracle.npz', help='Output file')
    parser.add_argument('--progress', type=int, default=500, help='Progress interval')
    parser.add_argument('--test', action='store_true', help='Test mode (100 samples, 1 worker)')
    parser.add_argument('--merge', nargs='+', help='Merge multiple NPZ files')

    args = parser.parse_args()

    if args.merge:
        if args.output == 'oracle.npz':
            print("Error: --merge requires explicit -o output.npz")
            sys.exit(1)
        merge_files(args.merge, args.output)
    elif args.test:
        run_batch(0, 100, 1, 'oracle_test.npz', progress=10)
    else:
        if args.start is None or args.count is None:
            print("Error: --start and --count required (or use --test)")
            sys.exit(1)
        run_batch(args.start, args.count, args.workers, args.output, args.progress)


if __name__ == '__main__':
    main()
