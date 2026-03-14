#!/usr/bin/env python3
"""
ORACLE Phase C: Time-Dependent Degradation Analysis (Q-093)
============================================================

Extension of ORACLE Phase A: instead of constant degradation rates,
this simulates linearly increasing degradation:

    kd_A(t) = kd_A0 + delta_A * t     (formate: hydrolysis/oxidation)
    kd_m(t) = kd_m0 + epsilon_m * t   (membrane: pentlandite reconstruction -> NiOOH, Q-088)
    kd_d(t) = kd_d0 + epsilon_d * t   (4F-Azo: hydrolysis, Q-080)

For each sample in Latin Hypercube over (delta_A, epsilon_m, epsilon_d):
  1. Run Gillespie SSA with time-dependent propensities for 72h
  2. Record: survived (A > threshold at t=72h)

Output: critical thresholds delta_A_max, epsilon_m_max, epsilon_d_max
for 72h survival, answering Q-093/Q-088/Q-077/Q-080.

Model: TM6v3-min (A, M, Fe) from ORACLE Phase A
  - Optionally with D-module (TM6v3-DE: A, M, Fe, D)

Author: Third Matter Research Project
Date: 2026-03-09
"""

import sys
import time
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats.qmc import LatinHypercube

# Unbuffered print
_builtin_print = print
def print(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ============================================================================
# DIRECTORIES
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR.parent / "results" / "oracle_phase_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL PARAMETERS (TM6v3-min, from Phase A nominal)
# ============================================================================

# Base parameters at 25 C (from TM6v3MinParams / ORACLE Phase A defaults)
BASE_PARAMS = {
    'k1':        1e-4,     # s^-1      autocatalysis rate
    'Ka':        7e-4,     # M         Hill half-saturation
    'f1':        5e-3,     # M         CO2 supply
    'Km_f':      5e-4,     # M         membrane transport half-sat
    'km':        3e-2,     # s^-1 M^-1 membrane formation
    'k_fe_gen':  5e-5,     # s^-1      A -> Fe generation
    'fe_supply': 5e-8,     # M/s       Fe0 corrosion
    'kd_A':      1e-4,     # s^-1      formate degradation (BASE)
    'kd_m':      3e-6,     # s^-1      membrane degradation (BASE)
    'kd_fe':     3e-4,     # s^-1      Fe2+ loss
}

# D-module parameters (from TM6v3-DE)
D_MODULE_PARAMS = {
    'kp_d':      1e-4,     # s^-1      D photosynthesis rate
    'Ka_p':      1e-3,     # M         half-sat for A on D
    'Km_p':      1e-4,     # M         half-sat for M on D
    'kd_d':      1e-5,     # s^-1      D degradation (BASE)
    'alpha':     10.0,     # dimensionless D->k1 enhancement
    'Kp_alpha':  1e-4,     # M         half-sat for D on alpha
}

# ORACLE Phase A critical thresholds (РЕШЕНИЕ-048)
CRITICAL_KD_A = 2.7e-4    # s^-1  max kd_A for survival at 25C
CRITICAL_KD_M = 1.15e-5   # s^-1  max kd_m for survival at 25C

T_HORIZON = 72 * 3600     # 72 hours in seconds
N_MOLECULES = 1000         # Gillespie molecule count

# Stoichiometry: 7 reactions x 3 species [A, M, Fe]
NU_3SP = np.array([
    [+1,  0,  0],   # R1: -> A (autocatalysis)
    [ 0, +1, -1],   # R2: Fe,A -> M (membrane formation)
    [ 0,  0, +1],   # R3: A -> Fe
    [ 0,  0, +1],   # R4: -> Fe (supply)
    [-1,  0,  0],   # D1: A ->
    [ 0, -1,  0],   # D2: M ->
    [ 0,  0, -1],   # D3: Fe ->
], dtype=np.int64)

# Stoichiometry: 9 reactions x 4 species [A, M, Fe, D] (with D-module)
NU_4SP = np.array([
    [+1,  0,  0,  0],   # R1: -> A (autocatalysis, enhanced by D)
    [ 0, +1, -1,  0],   # R2: Fe,A -> M
    [ 0,  0, +1,  0],   # R3: A -> Fe
    [ 0,  0, +1,  0],   # R4: -> Fe
    [-1,  0,  0,  0],   # D1: A ->
    [ 0, -1,  0,  0],   # D2: M ->
    [ 0,  0, -1,  0],   # D3: Fe ->
    [ 0,  0,  0, +1],   # R_D_prod: -> D (photosynthesis)
    [ 0,  0,  0, -1],   # R_D_deg: D ->
], dtype=np.int64)


# ============================================================================
# ODE MODEL (for steady state finding)
# ============================================================================

def ode_rhs_3sp(t, y, p):
    """3-species ODE: A, M, Fe (constant rates)."""
    a, m, fe = np.maximum(y, 0)
    f1_eff = p['f1'] * m / (p['Km_f'] + m)
    hill_a = a**2 / (p['Ka']**2 + a**2) if a > 0 else 0.0

    da  = p['k1'] * f1_eff * hill_a - p['kd_A'] * a
    dm  = p['km'] * fe * a - p['kd_m'] * m
    dfe = p['k_fe_gen'] * a + p['fe_supply'] - p['km'] * fe * a - p['kd_fe'] * fe
    return [da, dm, dfe]


def ode_rhs_4sp(t, y, p):
    """4-species ODE: A, M, Fe, D (constant rates)."""
    a, m, fe, d = np.maximum(y, 0)
    f1_eff = p['f1'] * m / (p['Km_f'] + m)
    hill_a = a**2 / (p['Ka']**2 + a**2) if a > 0 else 0.0

    # D-enhanced autocatalysis
    alpha_eff = 1.0 + p['alpha'] * d / (p['Kp_alpha'] + d)
    k1_eff = p['k1'] * alpha_eff

    # D production: proportional to A and M availability
    d_prod = p['kp_d'] * a / (p['Ka_p'] + a) * m / (p['Km_p'] + m)

    da  = k1_eff * f1_eff * hill_a - p['kd_A'] * a
    dm  = p['km'] * fe * a - p['kd_m'] * m
    dfe = p['k_fe_gen'] * a + p['fe_supply'] - p['km'] * fe * a - p['kd_fe'] * fe
    dd  = d_prod - p['kd_d'] * d
    return [da, dm, dfe, dd]


def find_steady_state(params, with_d=False):
    """Find ODE steady state by integration (500h)."""
    if with_d:
        y0 = [0.01, 0.01, 0.001, 1e-4]
        rhs = ode_rhs_4sp
    else:
        y0 = [0.01, 0.01, 0.001]
        rhs = ode_rhs_3sp

    try:
        sol = solve_ivp(rhs, [0, 500*3600], y0, args=(params,),
                        method='LSODA', rtol=1e-8, atol=1e-12, max_step=3600)
        if not sol.success:
            return None
        final = np.maximum(sol.y[:, -1], 0)
        if final[0] < 1e-8:  # dead
            return None
        return final
    except Exception:
        return None


# ============================================================================
# GILLESPIE SSA WITH TIME-DEPENDENT DEGRADATION
# ============================================================================

def gillespie_timedep_3sp(n_init, omega, params, delta_A, epsilon_m,
                           t_max, seed=42):
    """
    Gillespie tau-leaping with time-dependent degradation for 3 species.

    kd_A(t) = kd_A0 + delta_A * t
    kd_m(t) = kd_m0 + epsilon_m * t

    Uses adaptive tau-leaping (as in Phase A) but recalculates
    degradation rates at each step.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    NU_T = NU_3SP.T.copy()

    n = n_init.astype(np.float64)

    # Pre-extract constant params
    p_k1 = params['k1']
    p_Ka2 = params['Ka'] ** 2
    p_f1 = params['f1']
    p_Km_f = params['Km_f']
    p_km = params['km']
    p_k_fe_gen = params['k_fe_gen']
    p_fe_supply = params['fe_supply']
    p_kd_A0 = params['kd_A']
    p_kd_m0 = params['kd_m']
    p_kd_fe = params['kd_fe']

    props = np.empty(7)
    t = 0.0

    while t < t_max:
        if n[0] <= 0:
            return False, t

        # Time-dependent degradation
        kd_A_t = p_kd_A0 + delta_A * t
        kd_m_t = p_kd_m0 + epsilon_m * t

        # Concentrations
        a = n[0] / omega
        m = n[1] / omega
        denom_m = p_Km_f + m
        f1_eff = p_f1 * m / denom_m if denom_m > 0 else 0.0
        hill_a = a * a / (p_Ka2 + a * a) if (p_Ka2 + a * a) > 0 else 0.0

        # Propensities
        props[0] = p_k1 * f1_eff * hill_a * omega       # R1
        props[1] = p_km * n[2] * n[0] / omega            # R2
        props[2] = p_k_fe_gen * n[0]                      # R3
        props[3] = p_fe_supply * omega                    # R4
        props[4] = kd_A_t * n[0]                          # D1 (time-dependent!)
        props[5] = kd_m_t * n[1]                          # D2 (time-dependent!)
        props[6] = p_kd_fe * n[2]                         # D3

        a0 = props.sum()
        if a0 <= 0:
            return n[0] > 0, t

        # Adaptive tau
        net_rate = NU_T @ props
        tau = 120.0
        for i in range(3):
            if net_rate[i] < 0 and n[i] > 1:
                tau_i = 0.2 * n[i] / abs(net_rate[i])
                if tau_i < tau:
                    tau = tau_i
        tau = max(tau, 5.0)   # min 5s for better resolution
        tau = min(tau, t_max - t)
        if tau <= 0:
            break

        # Tau-leaping step
        firings = rng.poisson(np.maximum(props * tau, 0))
        n += NU_T @ firings
        n[0] = max(n[0], 0)
        n[1] = max(n[1], 0)
        n[2] = max(n[2], 0)
        t += tau

    return n[0] > 0, t


def gillespie_timedep_4sp(n_init, omega, params, delta_A, epsilon_m, epsilon_d,
                           t_max, seed=42):
    """
    Gillespie tau-leaping with time-dependent degradation for 4 species (with D-module).

    kd_A(t) = kd_A0 + delta_A * t
    kd_m(t) = kd_m0 + epsilon_m * t
    kd_d(t) = kd_d0 + epsilon_d * t
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    NU_T = NU_4SP.T.copy()

    n = n_init.astype(np.float64)

    p_k1 = params['k1']
    p_Ka2 = params['Ka'] ** 2
    p_f1 = params['f1']
    p_Km_f = params['Km_f']
    p_km = params['km']
    p_k_fe_gen = params['k_fe_gen']
    p_fe_supply = params['fe_supply']
    p_kd_A0 = params['kd_A']
    p_kd_m0 = params['kd_m']
    p_kd_fe = params['kd_fe']
    p_kp_d = params['kp_d']
    p_Ka_p = params['Ka_p']
    p_Km_p = params['Km_p']
    p_kd_d0 = params['kd_d']
    p_alpha = params['alpha']
    p_Kp_alpha = params['Kp_alpha']

    props = np.empty(9)
    t = 0.0

    while t < t_max:
        if n[0] <= 0:
            return False, t

        kd_A_t = p_kd_A0 + delta_A * t
        kd_m_t = p_kd_m0 + epsilon_m * t
        kd_d_t = p_kd_d0 + epsilon_d * t

        a = n[0] / omega
        m = n[1] / omega
        d = n[3] / omega

        denom_m = p_Km_f + m
        f1_eff = p_f1 * m / denom_m if denom_m > 0 else 0.0
        hill_a = a * a / (p_Ka2 + a * a) if (p_Ka2 + a * a) > 0 else 0.0

        # D-enhanced k1
        alpha_eff = 1.0 + p_alpha * d / (p_Kp_alpha + d)
        k1_eff = p_k1 * alpha_eff

        # D production rate
        d_prod_rate = p_kp_d * a / (p_Ka_p + a) * m / (p_Km_p + m)

        props[0] = k1_eff * f1_eff * hill_a * omega     # R1 (D-enhanced)
        props[1] = p_km * n[2] * n[0] / omega            # R2
        props[2] = p_k_fe_gen * n[0]                      # R3
        props[3] = p_fe_supply * omega                    # R4
        props[4] = kd_A_t * n[0]                          # D1
        props[5] = kd_m_t * n[1]                          # D2
        props[6] = p_kd_fe * n[2]                         # D3
        props[7] = d_prod_rate * omega                    # R_D_prod
        props[8] = kd_d_t * n[3]                          # R_D_deg

        a0 = props.sum()
        if a0 <= 0:
            return n[0] > 0, t

        net_rate = NU_T @ props
        tau = 120.0
        for i in range(4):
            if net_rate[i] < 0 and n[i] > 1:
                tau_i = 0.2 * n[i] / abs(net_rate[i])
                if tau_i < tau:
                    tau = tau_i
        tau = max(tau, 5.0)
        tau = min(tau, t_max - t)
        if tau <= 0:
            break

        firings = rng.poisson(np.maximum(props * tau, 0))
        n += NU_T @ firings
        for i in range(4):
            n[i] = max(n[i], 0)
        t += tau

    return n[0] > 0, t


# ============================================================================
# WORKER FUNCTION
# ============================================================================

def run_single_3sp(args):
    """Worker: run one 3-species simulation with given degradation rates."""
    idx, delta_A, epsilon_m, ss, omega, params, t_max, n_repeats = args

    n_init = np.round(ss * omega).astype(np.int64)
    n_init = np.maximum(n_init, 1)

    survived_count = 0
    for rep in range(n_repeats):
        seed = idx * 1000 + rep * 137 + 42
        alive, _ = gillespie_timedep_3sp(n_init, omega, params,
                                          delta_A, epsilon_m, t_max, seed)
        if alive:
            survived_count += 1

    return (idx, delta_A, epsilon_m, 0.0, survived_count / n_repeats)


def run_single_4sp(args):
    """Worker: run one 4-species simulation with given degradation rates."""
    idx, delta_A, epsilon_m, epsilon_d, ss, omega, params, t_max, n_repeats = args

    n_init = np.round(ss * omega).astype(np.int64)
    n_init = np.maximum(n_init, 1)

    survived_count = 0
    for rep in range(n_repeats):
        seed = idx * 1000 + rep * 137 + 42
        alive, _ = gillespie_timedep_4sp(n_init, omega, params,
                                          delta_A, epsilon_m, epsilon_d,
                                          t_max, seed)
        if alive:
            survived_count += 1

    return (idx, delta_A, epsilon_m, epsilon_d, survived_count / n_repeats)


# ============================================================================
# SCANNING
# ============================================================================

def generate_lhs_samples_3sp(n_samples, kd_A0, kd_m0, seed=2026):
    """
    Generate LHS samples for (delta_A, epsilon_m).

    Range: 0 to 10 * kd_X0 / T_HORIZON  (so kd doubles over 72h at max)
    Actually: 10x means kd(72h) = 11 * kd0
    """
    sampler = LatinHypercube(d=2, scramble=True, seed=seed)
    samples_unit = sampler.random(n_samples)

    # Max degradation rate increase: 10 * kd_X0 / T_HORIZON
    delta_A_max = 10.0 * kd_A0 / T_HORIZON
    epsilon_m_max = 10.0 * kd_m0 / T_HORIZON

    delta_A = samples_unit[:, 0] * delta_A_max
    epsilon_m = samples_unit[:, 1] * epsilon_m_max

    return delta_A, epsilon_m, delta_A_max, epsilon_m_max


def generate_lhs_samples_4sp(n_samples, kd_A0, kd_m0, kd_d0, seed=2026):
    """Generate LHS samples for (delta_A, epsilon_m, epsilon_d)."""
    sampler = LatinHypercube(d=3, scramble=True, seed=seed)
    samples_unit = sampler.random(n_samples)

    delta_A_max = 10.0 * kd_A0 / T_HORIZON
    epsilon_m_max = 10.0 * kd_m0 / T_HORIZON
    epsilon_d_max = 10.0 * kd_d0 / T_HORIZON

    delta_A = samples_unit[:, 0] * delta_A_max
    epsilon_m = samples_unit[:, 1] * epsilon_m_max
    epsilon_d = samples_unit[:, 2] * epsilon_d_max

    return delta_A, epsilon_m, epsilon_d, delta_A_max, epsilon_m_max, epsilon_d_max


# ============================================================================
# ANALYSIS
# ============================================================================

def find_critical_threshold_1d(values, survival_rates, target=0.5):
    """
    Find critical value where survival drops below target.
    Uses binning + interpolation.
    """
    n_bins = 30
    bins = np.linspace(0, values.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_survival = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for v, s in zip(values, survival_rates):
        idx = min(int(v / values.max() * n_bins), n_bins - 1)
        bin_survival[idx] += s
        bin_count[idx] += 1

    mask = bin_count > 0
    if mask.sum() < 3:
        return None, None, None

    bc = bin_centers[mask]
    bs = bin_survival[mask] / bin_count[mask]

    # Find crossing
    for i in range(len(bs) - 1):
        if bs[i] >= target and bs[i+1] < target:
            # Linear interpolation
            frac = (target - bs[i]) / (bs[i+1] - bs[i])
            critical = bc[i] + frac * (bc[i+1] - bc[i])
            return critical, bc, bs

    # No crossing found
    if bs[-1] >= target:
        return values.max(), bc, bs  # always survives in range
    if bs[0] < target:
        return 0.0, bc, bs  # never survives

    return None, bc, bs


def find_critical_marginal(delta_A, epsilon_m, survival_rates,
                           delta_A_max, epsilon_m_max, target=0.5):
    """Find 1D marginal critical thresholds."""
    # Marginal over epsilon_m: for each delta_A bin, average survival
    crit_dA, bc_dA, bs_dA = find_critical_threshold_1d(delta_A, survival_rates, target)
    crit_eM, bc_eM, bs_eM = find_critical_threshold_1d(epsilon_m, survival_rates, target)

    return {
        'delta_A_critical': crit_dA,
        'epsilon_m_critical': crit_eM,
        'delta_A_bins': bc_dA,
        'delta_A_surv': bs_dA,
        'epsilon_m_bins': bc_eM,
        'epsilon_m_surv': bs_eM,
    }


def compute_max_physical_degradation(critical_rate, kd0):
    """
    Convert critical degradation slope to physical meaning.

    If delta_critical is the max slope, then at t=72h:
      kd(72h) = kd0 + delta_critical * 72h
      kd(72h) / kd0 = 1 + delta_critical * 72h / kd0

    Returns: (kd_at_72h, ratio_72h, halflife_hours)
    halflife_hours = time at which kd doubles
    """
    if critical_rate is None or critical_rate <= 0:
        return None, None, None

    kd_72h = kd0 + critical_rate * T_HORIZON
    ratio_72h = kd_72h / kd0
    # Time to double: kd0 + delta*t = 2*kd0 -> t = kd0/delta
    halflife_s = kd0 / critical_rate
    halflife_h = halflife_s / 3600

    return kd_72h, ratio_72h, halflife_h


# ============================================================================
# PLOTTING
# ============================================================================

def make_plots(results, model_name, out_dir):
    """Generate all analysis plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    delta_A = results['delta_A']
    epsilon_m = results['epsilon_m']
    survival = results['survival_rates']

    kd_A0 = results['kd_A0']
    kd_m0 = results['kd_m0']

    # Convert to physical units: multiples of kd0/72h
    dA_norm = delta_A / (kd_A0 / T_HORIZON)  # 0..10
    eM_norm = epsilon_m / (kd_m0 / T_HORIZON)  # 0..10

    # --- Plot 1: 2D survival heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Bin into 2D grid
    n_bins = 25
    dA_bins = np.linspace(0, 10, n_bins + 1)
    eM_bins = np.linspace(0, 10, n_bins + 1)

    surv_grid = np.full((n_bins, n_bins), np.nan)
    count_grid = np.zeros((n_bins, n_bins))

    for dA, eM, s in zip(dA_norm, eM_norm, survival):
        i = min(int(dA / 10 * n_bins), n_bins - 1)
        j = min(int(eM / 10 * n_bins), n_bins - 1)
        if np.isnan(surv_grid[j, i]):
            surv_grid[j, i] = 0
        surv_grid[j, i] += s
        count_grid[j, i] += 1

    mask = count_grid > 0
    surv_grid[mask] /= count_grid[mask]

    dA_centers = 0.5 * (dA_bins[:-1] + dA_bins[1:])
    eM_centers = 0.5 * (eM_bins[:-1] + eM_bins[1:])

    im = ax.pcolormesh(dA_centers, eM_centers, surv_grid,
                       cmap='RdYlGn', vmin=0, vmax=1, shading='nearest')
    plt.colorbar(im, label='Survival Rate (72h)', ax=ax)

    # 50% contour
    try:
        ax.contour(dA_centers, eM_centers, surv_grid, levels=[0.5],
                   colors='black', linewidths=2, linestyles='--')
    except Exception:
        pass

    ax.set_xlabel(r'$\delta_A$ / ($k_{d,A}^0$ / 72h)  [fold increase]')
    ax.set_ylabel(r'$\epsilon_m$ / ($k_{d,m}^0$ / 72h)  [fold increase]')
    ax.set_title(f'ORACLE Phase C: 72h Survival Map ({model_name})\n'
                 f'$k_{{d,A}}^0$ = {kd_A0:.1e}, $k_{{d,m}}^0$ = {kd_m0:.1e} s$^{{-1}}$')

    plt.tight_layout()
    plt.savefig(out_dir / f'oracle_c_{model_name}_2d_survival.png', dpi=150)
    plt.close()
    print(f"  -> {out_dir / f'oracle_c_{model_name}_2d_survival.png'}")

    # --- Plot 2: 1D marginal survival curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # delta_A marginal
    ax = axes[0]
    crit = results['marginals']
    if crit['delta_A_bins'] is not None:
        ax.plot(crit['delta_A_bins'] / (kd_A0 / T_HORIZON), crit['delta_A_surv'],
                'b-o', markersize=4)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        if crit['delta_A_critical'] is not None:
            ax.axvline(crit['delta_A_critical'] / (kd_A0 / T_HORIZON),
                      color='red', linestyle='--', linewidth=2,
                      label=f'Critical: {crit["delta_A_critical"]/(kd_A0/T_HORIZON):.2f}x')
            ax.legend()
    ax.set_xlabel(r'$\delta_A$ / ($k_{d,A}^0$ / 72h)')
    ax.set_ylabel('Survival Rate')
    ax.set_title(r'Marginal: $\delta_A$ (formate degradation rate)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # epsilon_m marginal
    ax = axes[1]
    if crit['epsilon_m_bins'] is not None:
        ax.plot(crit['epsilon_m_bins'] / (kd_m0 / T_HORIZON), crit['epsilon_m_surv'],
                'g-o', markersize=4)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        if crit['epsilon_m_critical'] is not None:
            ax.axvline(crit['epsilon_m_critical'] / (kd_m0 / T_HORIZON),
                      color='red', linestyle='--', linewidth=2,
                      label=f'Critical: {crit["epsilon_m_critical"]/(kd_m0/T_HORIZON):.2f}x')
            ax.legend()
    ax.set_xlabel(r'$\epsilon_m$ / ($k_{d,m}^0$ / 72h)')
    ax.set_ylabel('Survival Rate')
    ax.set_title(r'Marginal: $\epsilon_m$ (membrane degradation rate)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f'oracle_c_{model_name}_marginals.png', dpi=150)
    plt.close()
    print(f"  -> {out_dir / f'oracle_c_{model_name}_marginals.png'}")

    # --- Plot 3: Scatter colored by survival ---
    fig, ax = plt.subplots(figsize=(10, 8))

    alive_mask = survival >= 0.5
    dead_mask = ~alive_mask

    ax.scatter(dA_norm[dead_mask], eM_norm[dead_mask], c='red', s=3, alpha=0.3, label='Dead')
    ax.scatter(dA_norm[alive_mask], eM_norm[alive_mask], c='green', s=3, alpha=0.3, label='Alive')

    ax.set_xlabel(r'$\delta_A$ / ($k_{d,A}^0$ / 72h)')
    ax.set_ylabel(r'$\epsilon_m$ / ($k_{d,m}^0$ / 72h)')
    ax.set_title(f'ORACLE Phase C: Parameter Space ({model_name})')
    ax.legend(markerscale=3)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f'oracle_c_{model_name}_scatter.png', dpi=150)
    plt.close()
    print(f"  -> {out_dir / f'oracle_c_{model_name}_scatter.png'}")


def make_plots_4sp(results, out_dir):
    """Plots for 4-species model with epsilon_d dimension."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    delta_A = results['delta_A']
    epsilon_m = results['epsilon_m']
    epsilon_d = results['epsilon_d']
    survival = results['survival_rates']

    kd_A0 = results['kd_A0']
    kd_m0 = results['kd_m0']
    kd_d0 = results['kd_d0']

    # 3 marginal 2D slices: (dA, eM), (dA, eD), (eM, eD)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    pairs = [
        (delta_A / (kd_A0/T_HORIZON), epsilon_m / (kd_m0/T_HORIZON),
         r'$\delta_A$ norm', r'$\epsilon_m$ norm'),
        (delta_A / (kd_A0/T_HORIZON), epsilon_d / (kd_d0/T_HORIZON),
         r'$\delta_A$ norm', r'$\epsilon_d$ norm'),
        (epsilon_m / (kd_m0/T_HORIZON), epsilon_d / (kd_d0/T_HORIZON),
         r'$\epsilon_m$ norm', r'$\epsilon_d$ norm'),
    ]

    for ax, (x, y, xlabel, ylabel) in zip(axes, pairs):
        n_bins = 20
        x_bins = np.linspace(0, 10, n_bins + 1)
        y_bins = np.linspace(0, 10, n_bins + 1)

        surv_grid = np.full((n_bins, n_bins), np.nan)
        count_grid = np.zeros((n_bins, n_bins))

        for xi, yi, s in zip(x, y, survival):
            i = min(int(xi / 10 * n_bins), n_bins - 1)
            j = min(int(yi / 10 * n_bins), n_bins - 1)
            if np.isnan(surv_grid[j, i]):
                surv_grid[j, i] = 0
            surv_grid[j, i] += s
            count_grid[j, i] += 1

        mask = count_grid > 0
        surv_grid[mask] /= count_grid[mask]

        x_c = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_c = 0.5 * (y_bins[:-1] + y_bins[1:])

        im = ax.pcolormesh(x_c, y_c, surv_grid, cmap='RdYlGn',
                          vmin=0, vmax=1, shading='nearest')
        plt.colorbar(im, ax=ax, label='Survival')

        try:
            ax.contour(x_c, y_c, surv_grid, levels=[0.5],
                      colors='black', linewidths=2, linestyles='--')
        except Exception:
            pass

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    plt.suptitle('ORACLE Phase C: 4-species (with D-module) 72h Survival', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'oracle_c_4sp_2d_slices.png', dpi=150)
    plt.close()
    print(f"  -> {out_dir / 'oracle_c_4sp_2d_slices.png'}")

    # 1D marginals
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, vals, name, kd0 in [
        (axes[0], delta_A, r'$\delta_A$', kd_A0),
        (axes[1], epsilon_m, r'$\epsilon_m$', kd_m0),
        (axes[2], epsilon_d, r'$\epsilon_d$', kd_d0),
    ]:
        crit_val, bc, bs = find_critical_threshold_1d(vals, survival)
        if bc is not None:
            ax.plot(bc / (kd0/T_HORIZON), bs, 'b-o', markersize=4)
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            if crit_val is not None and crit_val > 0:
                ax.axvline(crit_val / (kd0/T_HORIZON), color='red', linestyle='--',
                          linewidth=2, label=f'Critical: {crit_val/(kd0/T_HORIZON):.2f}x')
                ax.legend()
        ax.set_xlabel(f'{name} / (kd0 / 72h)')
        ax.set_ylabel('Survival Rate')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle('ORACLE Phase C: 4-species Marginal Survival', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'oracle_c_4sp_marginals.png', dpi=150)
    plt.close()
    print(f"  -> {out_dir / 'oracle_c_4sp_marginals.png'}")


# ============================================================================
# MAIN
# ============================================================================

def run_3sp_scan(n_samples=5000, n_repeats=3, workers=None):
    """Run 3-species (TM6v3-min) degradation scan."""
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    print("=" * 70)
    print("  ORACLE Phase C: 3-SPECIES MODEL (TM6v3-min)")
    print(f"  N_samples = {n_samples}, N_repeats = {n_repeats}, Workers = {workers}")
    print("=" * 70)

    params = BASE_PARAMS.copy()
    kd_A0 = params['kd_A']
    kd_m0 = params['kd_m']

    # Find steady state
    print("\n[1] Finding ODE steady state...")
    ss = find_steady_state(params, with_d=False)
    if ss is None:
        print("  FATAL: No living steady state found!")
        return None

    a_star = ss[0]
    omega = N_MOLECULES / a_star
    print(f"  A* = {a_star*1e3:.4f} mM, M* = {ss[1]*1e3:.4f} mM, Fe* = {ss[2]*1e3:.4f} mM")
    print(f"  Omega = {omega:.1f}")

    # Verify baseline survival (no degradation increase)
    print("\n[2] Verifying baseline survival (delta=0, epsilon=0)...")
    n_init_base = np.round(ss * omega).astype(np.int64)
    n_init_base = np.maximum(n_init_base, 1)
    n_alive = 0
    for i in range(10):
        alive, _ = gillespie_timedep_3sp(n_init_base, omega, params, 0.0, 0.0, T_HORIZON, seed=i*137+42)
        if alive:
            n_alive += 1
    print(f"  Baseline survival: {n_alive}/10 ({n_alive*10}%)")
    if n_alive < 7:
        print("  WARNING: Low baseline survival! Results may be noisy.")

    # Generate LHS samples
    print(f"\n[3] Generating {n_samples} LHS samples...")
    delta_A, epsilon_m, dA_max, eM_max = generate_lhs_samples_3sp(
        n_samples, kd_A0, kd_m0)

    print(f"  delta_A range: 0 .. {dA_max:.3e} s^-2 (kd_A doubles in {kd_A0/dA_max/3600:.1f}h at max)")
    print(f"  epsilon_m range: 0 .. {eM_max:.3e} s^-2 (kd_m doubles in {kd_m0/eM_max/3600:.1f}h at max)")

    # Run simulations
    print(f"\n[4] Running {n_samples} simulations (x{n_repeats} repeats each)...")
    t_start = time.time()

    args_list = [
        (i, delta_A[i], epsilon_m[i], ss, omega, params, T_HORIZON, n_repeats)
        for i in range(n_samples)
    ]

    results = []
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single_3sp, args_list), 1):
            results.append(res)
            if i % 500 == 0 or i == n_samples:
                elapsed = time.time() - t_start
                rate = i / elapsed
                eta = (n_samples - i) / rate if rate > 0 else 0
                n_alive_so_far = sum(1 for r in results if r[4] >= 0.5)
                print(f"  Progress: {i}/{n_samples} | "
                      f"Alive: {n_alive_so_far}/{i} ({100*n_alive_so_far/i:.1f}%) | "
                      f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m")

    elapsed_total = time.time() - t_start

    # Sort by index
    results.sort(key=lambda x: x[0])

    survival_rates = np.array([r[4] for r in results])
    delta_A_out = np.array([r[1] for r in results])
    epsilon_m_out = np.array([r[2] for r in results])

    # Analysis
    print(f"\n[5] Analysis...")

    n_alive_total = np.sum(survival_rates >= 0.5)
    print(f"  Total alive (>=50%): {n_alive_total}/{n_samples} ({100*n_alive_total/n_samples:.1f}%)")

    marginals = find_critical_marginal(delta_A_out, epsilon_m_out, survival_rates,
                                        dA_max, eM_max)

    crit_dA = marginals['delta_A_critical']
    crit_eM = marginals['epsilon_m_critical']

    # Physical interpretation
    print(f"\n  CRITICAL THRESHOLDS (marginal, 50% survival at 72h):")

    if crit_dA is not None and crit_dA > 0:
        kd_72h, ratio, halflife = compute_max_physical_degradation(crit_dA, kd_A0)
        print(f"  delta_A_crit = {crit_dA:.3e} s^-2")
        print(f"    kd_A(72h) = {kd_72h:.3e} s^-1 (= {ratio:.2f}x baseline)")
        print(f"    kd_A doubling time: {halflife:.1f} h")
        print(f"    Normalized: {crit_dA / (kd_A0/T_HORIZON):.2f}x (kd_A0/72h)")
    else:
        kd_72h, ratio, halflife = None, None, None
        print(f"  delta_A_crit: NOT FOUND (system always survives or always dies)")

    if crit_eM is not None and crit_eM > 0:
        kd_72h_m, ratio_m, halflife_m = compute_max_physical_degradation(crit_eM, kd_m0)
        print(f"  epsilon_m_crit = {crit_eM:.3e} s^-2")
        print(f"    kd_m(72h) = {kd_72h_m:.3e} s^-1 (= {ratio_m:.2f}x baseline)")
        print(f"    kd_m doubling time: {halflife_m:.1f} h")
        print(f"    Normalized: {crit_eM / (kd_m0/T_HORIZON):.2f}x (kd_m0/72h)")
    else:
        kd_72h_m, ratio_m, halflife_m = None, None, None
        print(f"  epsilon_m_crit: NOT FOUND")

    # Collect all results
    out = {
        'model': 'TM6v3-min (3sp)',
        'n_samples': n_samples,
        'n_repeats': n_repeats,
        'kd_A0': kd_A0,
        'kd_m0': kd_m0,
        'delta_A': delta_A_out,
        'epsilon_m': epsilon_m_out,
        'survival_rates': survival_rates,
        'delta_A_max_scan': dA_max,
        'epsilon_m_max_scan': eM_max,
        'marginals': marginals,
        'ss': ss.tolist(),
        'omega': omega,
        'elapsed_s': elapsed_total,
        'critical': {
            'delta_A_crit': float(crit_dA) if crit_dA is not None else None,
            'epsilon_m_crit': float(crit_eM) if crit_eM is not None else None,
            'delta_A_kd72h': float(kd_72h) if kd_72h is not None else None,
            'delta_A_ratio72h': float(ratio) if ratio is not None else None,
            'delta_A_halflife_h': float(halflife) if halflife is not None else None,
            'epsilon_m_kd72h': float(kd_72h_m) if kd_72h_m is not None else None,
            'epsilon_m_ratio72h': float(ratio_m) if ratio_m is not None else None,
            'epsilon_m_halflife_h': float(halflife_m) if halflife_m is not None else None,
        }
    }

    # Save NPZ
    np.savez_compressed(
        OUT_DIR / 'oracle_c_3sp.npz',
        delta_A=delta_A_out,
        epsilon_m=epsilon_m_out,
        survival_rates=survival_rates,
        ss=ss,
        kd_A0=kd_A0,
        kd_m0=kd_m0,
    )
    print(f"  -> {OUT_DIR / 'oracle_c_3sp.npz'}")

    # Plots
    print(f"\n[6] Generating plots...")
    make_plots(out, '3sp', OUT_DIR)

    return out


def run_4sp_scan(n_samples=5000, n_repeats=3, workers=None):
    """Run 4-species (TM6v3-DE) degradation scan."""
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    print("\n" + "=" * 70)
    print("  ORACLE Phase C: 4-SPECIES MODEL (TM6v3-DE, with D-module)")
    print(f"  N_samples = {n_samples}, N_repeats = {n_repeats}, Workers = {workers}")
    print("=" * 70)

    params = {**BASE_PARAMS, **D_MODULE_PARAMS}
    kd_A0 = params['kd_A']
    kd_m0 = params['kd_m']
    kd_d0 = params['kd_d']

    # Find steady state
    print("\n[1] Finding ODE steady state (4sp)...")
    ss = find_steady_state(params, with_d=True)
    if ss is None:
        print("  FATAL: No living steady state found for 4sp!")
        return None

    a_star = ss[0]
    omega = N_MOLECULES / a_star
    print(f"  A* = {a_star*1e3:.4f} mM, M* = {ss[1]*1e3:.4f} mM, "
          f"Fe* = {ss[2]*1e3:.4f} mM, D* = {ss[3]*1e3:.4f} mM")
    print(f"  Omega = {omega:.1f}")

    # Verify baseline
    print("\n[2] Verifying baseline survival (delta=0, epsilon=0)...")
    n_init_base = np.round(ss * omega).astype(np.int64)
    n_init_base = np.maximum(n_init_base, 1)
    n_alive = 0
    for i in range(10):
        alive, _ = gillespie_timedep_4sp(n_init_base, omega, params, 0, 0, 0, T_HORIZON, seed=i*137+42)
        if alive:
            n_alive += 1
    print(f"  Baseline survival: {n_alive}/10 ({n_alive*10}%)")

    # Generate LHS
    print(f"\n[3] Generating {n_samples} LHS samples (3D)...")
    delta_A, epsilon_m, epsilon_d, dA_max, eM_max, eD_max = generate_lhs_samples_4sp(
        n_samples, kd_A0, kd_m0, kd_d0)

    print(f"  delta_A range: 0 .. {dA_max:.3e}")
    print(f"  epsilon_m range: 0 .. {eM_max:.3e}")
    print(f"  epsilon_d range: 0 .. {eD_max:.3e}")

    # Run
    print(f"\n[4] Running {n_samples} simulations (x{n_repeats} repeats)...")
    t_start = time.time()

    args_list = [
        (i, delta_A[i], epsilon_m[i], epsilon_d[i], ss, omega, params, T_HORIZON, n_repeats)
        for i in range(n_samples)
    ]

    results = []
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single_4sp, args_list), 1):
            results.append(res)
            if i % 500 == 0 or i == n_samples:
                elapsed = time.time() - t_start
                rate = i / elapsed
                eta = (n_samples - i) / rate if rate > 0 else 0
                n_alive_so_far = sum(1 for r in results if r[4] >= 0.5)
                print(f"  Progress: {i}/{n_samples} | "
                      f"Alive: {n_alive_so_far}/{i} ({100*n_alive_so_far/i:.1f}%) | "
                      f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m")

    elapsed_total = time.time() - t_start

    results.sort(key=lambda x: x[0])

    survival_rates = np.array([r[4] for r in results])
    delta_A_out = np.array([r[1] for r in results])
    epsilon_m_out = np.array([r[2] for r in results])
    epsilon_d_out = np.array([r[3] for r in results])

    # Analysis
    print(f"\n[5] Analysis...")
    n_alive_total = np.sum(survival_rates >= 0.5)
    print(f"  Total alive (>=50%): {n_alive_total}/{n_samples} ({100*n_alive_total/n_samples:.1f}%)")

    # 1D marginal for each parameter
    crit_dA, _, _ = find_critical_threshold_1d(delta_A_out, survival_rates)
    crit_eM, _, _ = find_critical_threshold_1d(epsilon_m_out, survival_rates)
    crit_eD, _, _ = find_critical_threshold_1d(epsilon_d_out, survival_rates)

    print(f"\n  CRITICAL THRESHOLDS (marginal, 50% survival at 72h):")

    for name, crit, kd0 in [
        ('delta_A', crit_dA, kd_A0),
        ('epsilon_m', crit_eM, kd_m0),
        ('epsilon_d', crit_eD, kd_d0),
    ]:
        if crit is not None and crit > 0:
            kd72, ratio, hl = compute_max_physical_degradation(crit, kd0)
            print(f"  {name}_crit = {crit:.3e} s^-2")
            print(f"    kd(72h) = {kd72:.3e} (= {ratio:.2f}x baseline)")
            print(f"    Doubling time: {hl:.1f} h")
            print(f"    Normalized: {crit / (kd0/T_HORIZON):.2f}x")
        else:
            print(f"  {name}_crit: NOT FOUND")

    out = {
        'model': 'TM6v3-DE (4sp)',
        'n_samples': n_samples,
        'n_repeats': n_repeats,
        'kd_A0': kd_A0,
        'kd_m0': kd_m0,
        'kd_d0': kd_d0,
        'delta_A': delta_A_out,
        'epsilon_m': epsilon_m_out,
        'epsilon_d': epsilon_d_out,
        'survival_rates': survival_rates,
        'elapsed_s': elapsed_total,
        'critical': {
            'delta_A_crit': float(crit_dA) if crit_dA is not None else None,
            'epsilon_m_crit': float(crit_eM) if crit_eM is not None else None,
            'epsilon_d_crit': float(crit_eD) if crit_eD is not None else None,
        }
    }

    # Physical interpretation for all 3
    for name, crit, kd0 in [('delta_A', crit_dA, kd_A0),
                             ('epsilon_m', crit_eM, kd_m0),
                             ('epsilon_d', crit_eD, kd_d0)]:
        if crit is not None and crit > 0:
            kd72, ratio, hl = compute_max_physical_degradation(crit, kd0)
            out['critical'][f'{name}_kd72h'] = float(kd72)
            out['critical'][f'{name}_ratio72h'] = float(ratio)
            out['critical'][f'{name}_halflife_h'] = float(hl)

    # Save
    np.savez_compressed(
        OUT_DIR / 'oracle_c_4sp.npz',
        delta_A=delta_A_out,
        epsilon_m=epsilon_m_out,
        epsilon_d=epsilon_d_out,
        survival_rates=survival_rates,
        ss=ss,
        kd_A0=kd_A0,
        kd_m0=kd_m0,
        kd_d0=kd_d0,
    )
    print(f"  -> {OUT_DIR / 'oracle_c_4sp.npz'}")

    # Plots
    print(f"\n[6] Generating plots...")
    make_plots_4sp(out, OUT_DIR)

    return out


def main():
    t_global_start = time.time()

    print("=" * 70)
    print("  ORACLE PHASE C: TIME-DEPENDENT DEGRADATION (Q-093)")
    print("  Answers: Q-088 (pentlandite reconstruction)")
    print("           Q-077 (mackinawite dissolution)")
    print("           Q-080 (4F-Azo hydrolysis)")
    print("=" * 70)

    workers = max(1, mp.cpu_count() - 1)
    print(f"\n  CPU workers: {workers}")
    print(f"  Output: {OUT_DIR}")
    print(f"  T_horizon: {T_HORIZON/3600:.0f} h")
    print(f"  N_molecules: {N_MOLECULES}")

    # ----- Phase C.1: 3-species model -----
    results_3sp = run_3sp_scan(n_samples=6000, n_repeats=3, workers=workers)

    # ----- Phase C.2: 4-species model (D-module) -----
    results_4sp = run_4sp_scan(n_samples=6000, n_repeats=3, workers=workers)

    # ----- FINAL SUMMARY -----
    total_time = time.time() - t_global_start

    print("\n" + "=" * 70)
    print("  ORACLE PHASE C: FINAL SUMMARY")
    print("=" * 70)

    summary = {
        'Q-093': 'ORACLE Phase C: time-dependent degradation',
        'date': '2026-03-09',
        'T_horizon_h': 72,
        'N_molecules': N_MOLECULES,
        'total_time_min': round(total_time / 60, 1),
    }

    if results_3sp is not None:
        c3 = results_3sp['critical']
        summary['model_3sp'] = {
            'A_star_mM': round(results_3sp['ss'][0] * 1e3, 4),
            'M_star_mM': round(results_3sp['ss'][1] * 1e3, 4),
            'Fe_star_mM': round(results_3sp['ss'][2] * 1e3, 4),
            'n_samples': results_3sp['n_samples'],
            'alive_fraction': float(np.mean(results_3sp['survival_rates'] >= 0.5)),
            'critical_thresholds': c3,
            'elapsed_s': round(results_3sp['elapsed_s'], 1),
        }
        print(f"\n  3-species (TM6v3-min):")
        print(f"    A* = {results_3sp['ss'][0]*1e3:.4f} mM")
        print(f"    Alive fraction: {np.mean(results_3sp['survival_rates']>=0.5)*100:.1f}%")

        if c3.get('delta_A_crit') and c3['delta_A_crit'] > 0:
            print(f"    delta_A critical: {c3['delta_A_crit']:.3e} s^-2")
            print(f"      -> kd_A can grow {c3.get('delta_A_ratio72h', '?'):.2f}x over 72h")
            print(f"      -> Doubling time: {c3.get('delta_A_halflife_h', '?'):.1f} h")
        else:
            print(f"    delta_A critical: not found")

        if c3.get('epsilon_m_crit') and c3['epsilon_m_crit'] > 0:
            print(f"    epsilon_m critical: {c3['epsilon_m_crit']:.3e} s^-2")
            print(f"      -> kd_m can grow {c3.get('epsilon_m_ratio72h', '?'):.2f}x over 72h")
            print(f"      -> Doubling time: {c3.get('epsilon_m_halflife_h', '?'):.1f} h")
        else:
            print(f"    epsilon_m critical: not found")

    if results_4sp is not None:
        c4 = results_4sp['critical']
        summary['model_4sp'] = {
            'n_samples': results_4sp['n_samples'],
            'alive_fraction': float(np.mean(results_4sp['survival_rates'] >= 0.5)),
            'critical_thresholds': c4,
            'elapsed_s': round(results_4sp['elapsed_s'], 1),
        }
        print(f"\n  4-species (TM6v3-DE, with D-module):")
        print(f"    Alive fraction: {np.mean(results_4sp['survival_rates']>=0.5)*100:.1f}%")

        for name, kd0 in [('delta_A', results_4sp['kd_A0']),
                           ('epsilon_m', results_4sp['kd_m0']),
                           ('epsilon_d', results_4sp['kd_d0'])]:
            crit = c4.get(f'{name}_crit')
            if crit and crit > 0:
                print(f"    {name} critical: {crit:.3e} s^-2")
                ratio = c4.get(f'{name}_ratio72h')
                hl = c4.get(f'{name}_halflife_h')
                if ratio:
                    print(f"      -> kd can grow {ratio:.2f}x over 72h")
                if hl:
                    print(f"      -> Doubling time: {hl:.1f} h")
            else:
                print(f"    {name} critical: not found")

    print(f"\n  Total time: {total_time/60:.1f} min")
    print("=" * 70)

    # Save summary
    summary_path = OUT_DIR / 'oracle_c_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")

    return summary


if __name__ == '__main__':
    main()
