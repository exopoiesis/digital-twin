#!/usr/bin/env python3
"""
Quick runner for ORACLE Phase C with output to file.
Designed to work within a single synchronous process (no multiprocessing).
"""
import sys
import time
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats.qmc import LatinHypercube
from pathlib import Path

# Output file
OUT_DIR = Path(__file__).parent.parent / "results" / "oracle_phase_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / "oracle_c_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg, flush=True)

# Clear log
with open(LOG, 'w') as f:
    f.write('')

log("=" * 70)
log("  ORACLE PHASE C: TIME-DEPENDENT DEGRADATION (Q-093)")
log("=" * 70)

# ============================================================
# MODEL PARAMETERS
# ============================================================
k1 = 1e-4; Ka = 7e-4; f1 = 5e-3; Km_f = 5e-4; km = 3e-2
k_fe_gen = 5e-5; fe_supply = 5e-8; kd_A0 = 1e-4; kd_m0 = 3e-6; kd_fe = 3e-4

# D-module
kp_d = 1e-4; Ka_p = 1e-3; Km_p = 1e-4; kd_d0 = 1e-5
alpha = 10.0; Kp_alpha = 1e-4

T_MAX = 72 * 3600  # 72 hours
N_MOL = 1000

# Stoichiometry
NU_3 = np.array([[+1,0,0],[0,+1,-1],[0,0,+1],[0,0,+1],[-1,0,0],[0,-1,0],[0,0,-1]], dtype=np.int64)
NU_3T = NU_3.T.copy()

NU_4 = np.array([
    [+1,0,0,0],[0,+1,-1,0],[0,0,+1,0],[0,0,+1,0],
    [-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,+1],[0,0,0,-1]
], dtype=np.int64)
NU_4T = NU_4.T.copy()

# ============================================================
# ODE STEADY STATE
# ============================================================
log("\n[1] Finding 3-species ODE steady state...")
def rhs_3(t, y):
    a, m, fe = [max(v, 0) for v in y]
    f1_eff = f1 * m / (Km_f + m)
    hill_a = a**2 / (Ka**2 + a**2) if a > 0 else 0.0
    return [k1*f1_eff*hill_a - kd_A0*a,
            km*fe*a - kd_m0*m,
            k_fe_gen*a + fe_supply - km*fe*a - kd_fe*fe]

sol = solve_ivp(rhs_3, [0, 500*3600], [0.01, 0.01, 0.001],
                method='LSODA', rtol=1e-8, atol=1e-12, max_step=3600)
ss3 = np.maximum(sol.y[:, -1], 0)
log(f"  3sp SS: A*={ss3[0]*1e3:.4f} mM, M*={ss3[1]*1e3:.4f} mM, Fe*={ss3[2]*1e3:.4f} mM")

omega3 = N_MOL / ss3[0]
n_init3 = np.round(ss3 * omega3).astype(np.int64)
n_init3 = np.maximum(n_init3, 1)
log(f"  n_init = {n_init3.tolist()}, omega = {omega3:.1f}")

log("\n[2] Finding 4-species ODE steady state (with D-module)...")
def rhs_4(t, y):
    a, m, fe, d = [max(v, 0) for v in y]
    f1_eff = f1 * m / (Km_f + m)
    hill_a = a**2 / (Ka**2 + a**2) if a > 0 else 0.0
    alpha_eff = 1.0 + alpha * d / (Kp_alpha + d)
    k1_eff = k1 * alpha_eff
    d_prod = kp_d * a / (Ka_p + a) * m / (Km_p + m)
    return [k1_eff*f1_eff*hill_a - kd_A0*a,
            km*fe*a - kd_m0*m,
            k_fe_gen*a + fe_supply - km*fe*a - kd_fe*fe,
            d_prod - kd_d0*d]

sol4 = solve_ivp(rhs_4, [0, 500*3600], [0.01, 0.01, 0.001, 1e-4],
                 method='LSODA', rtol=1e-8, atol=1e-12, max_step=3600)
ss4 = np.maximum(sol4.y[:, -1], 0)
log(f"  4sp SS: A*={ss4[0]*1e3:.4f} mM, M*={ss4[1]*1e3:.4f} mM, "
    f"Fe*={ss4[2]*1e3:.4f} mM, D*={ss4[3]*1e3:.4f} mM")

omega4 = N_MOL / ss4[0]
n_init4 = np.round(ss4 * omega4).astype(np.int64)
n_init4 = np.maximum(n_init4, 1)

# ============================================================
# GILLESPIE WITH TIME-DEPENDENT DEGRADATION
# ============================================================
Ka2 = Ka**2

def gill_3sp(n_in, dA, eM, seed=42):
    """Gillespie tau-leaping for 3sp with time-dependent kd_A, kd_m."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n = n_in.astype(np.float64).copy()
    props = np.empty(7)
    t = 0.0
    while t < T_MAX:
        if n[0] <= 0:
            return False, t
        kd_A_t = kd_A0 + dA * t
        kd_m_t = kd_m0 + eM * t
        a = n[0]/omega3; m = n[1]/omega3
        dm_ = Km_f + m
        f1e = f1*m/dm_ if dm_>0 else 0.0
        ha = a*a/(Ka2+a*a) if (Ka2+a*a)>0 else 0.0
        props[0] = k1*f1e*ha*omega3
        props[1] = km*n[2]*n[0]/omega3
        props[2] = k_fe_gen*n[0]
        props[3] = fe_supply*omega3
        props[4] = kd_A_t*n[0]
        props[5] = kd_m_t*n[1]
        props[6] = kd_fe*n[2]
        a0 = props.sum()
        if a0 <= 0:
            return n[0]>0, t
        nr = NU_3T @ props
        tau = 120.0
        for i in range(3):
            if nr[i]<0 and n[i]>1:
                ti = 0.2*n[i]/abs(nr[i])
                if ti<tau: tau=ti
        tau = max(tau, 5.0)
        tau = min(tau, T_MAX-t)
        if tau <= 0: break
        firings = rng.poisson(np.maximum(props*tau, 0))
        n += NU_3T @ firings
        for i in range(3): n[i]=max(n[i],0)
        t += tau
    return n[0]>0, t

def gill_4sp(n_in, dA, eM, eD, seed=42):
    """Gillespie tau-leaping for 4sp with time-dependent kd_A, kd_m, kd_d."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n = n_in.astype(np.float64).copy()
    props = np.empty(9)
    t = 0.0
    while t < T_MAX:
        if n[0] <= 0:
            return False, t
        kd_A_t = kd_A0 + dA * t
        kd_m_t = kd_m0 + eM * t
        kd_d_t = kd_d0 + eD * t
        a = n[0]/omega4; m = n[1]/omega4; d = n[3]/omega4
        dm_ = Km_f+m
        f1e = f1*m/dm_ if dm_>0 else 0.0
        ha = a*a/(Ka2+a*a) if (Ka2+a*a)>0 else 0.0
        alpha_eff = 1.0 + alpha * d / (Kp_alpha + d)
        k1_eff = k1 * alpha_eff
        d_prod = kp_d * a / (Ka_p+a) * m / (Km_p+m)
        props[0] = k1_eff*f1e*ha*omega4
        props[1] = km*n[2]*n[0]/omega4
        props[2] = k_fe_gen*n[0]
        props[3] = fe_supply*omega4
        props[4] = kd_A_t*n[0]
        props[5] = kd_m_t*n[1]
        props[6] = kd_fe*n[2]
        props[7] = d_prod*omega4
        props[8] = kd_d_t*n[3]
        a0 = props.sum()
        if a0 <= 0:
            return n[0]>0, t
        nr = NU_4T @ props
        tau = 120.0
        for i in range(4):
            if nr[i]<0 and n[i]>1:
                ti = 0.2*n[i]/abs(nr[i])
                if ti<tau: tau=ti
        tau = max(tau, 5.0)
        tau = min(tau, T_MAX-t)
        if tau <= 0: break
        firings = rng.poisson(np.maximum(props*tau, 0))
        n += NU_4T @ firings
        for i in range(4): n[i]=max(n[i],0)
        t += tau
    return n[0]>0, t

# ============================================================
# BASELINE VERIFICATION
# ============================================================
log("\n[3] Baseline verification (no degradation increase)...")
n_alive_base = 0
for i in range(10):
    alive, _ = gill_3sp(n_init3, 0, 0, seed=i*137+42)
    if alive: n_alive_base += 1
log(f"  3sp baseline: {n_alive_base}/10 survived ({n_alive_base*10}%)")

n_alive_base4 = 0
for i in range(10):
    alive, _ = gill_4sp(n_init4, 0, 0, 0, seed=i*137+42)
    if alive: n_alive_base4 += 1
log(f"  4sp baseline: {n_alive_base4}/10 survived ({n_alive_base4*10}%)")

# ============================================================
# PHASE C.1: 3-SPECIES SCAN
# ============================================================
N_SAMPLES = 5000
N_REPEATS = 3
dA_max = 10.0 * kd_A0 / T_MAX
eM_max = 10.0 * kd_m0 / T_MAX

log(f"\n[4] 3-species scan: {N_SAMPLES} samples x {N_REPEATS} repeats")
log(f"  delta_A range: 0 .. {dA_max:.3e} s^-2")
log(f"  epsilon_m range: 0 .. {eM_max:.3e} s^-2")

sampler = LatinHypercube(d=2, scramble=True, seed=2026)
u = sampler.random(N_SAMPLES)
dA_vals = u[:, 0] * dA_max
eM_vals = u[:, 1] * eM_max

t0 = time.time()
surv_3sp = np.zeros(N_SAMPLES)

for i in range(N_SAMPLES):
    alive_count = 0
    for r in range(N_REPEATS):
        alive, _ = gill_3sp(n_init3, dA_vals[i], eM_vals[i], seed=i*1000+r*137+42)
        if alive: alive_count += 1
    surv_3sp[i] = alive_count / N_REPEATS

    if (i+1) % 500 == 0 or (i+1) == N_SAMPLES:
        elapsed = time.time() - t0
        rate = (i+1) / elapsed
        eta = (N_SAMPLES - i - 1) / rate
        n_alive = np.sum(surv_3sp[:i+1] >= 0.5)
        log(f"  3sp: {i+1}/{N_SAMPLES} | alive={n_alive}/{i+1} "
            f"({100*n_alive/(i+1):.1f}%) | {rate:.1f}/s | ETA {eta/60:.1f}m")

elapsed_3sp = time.time() - t0
log(f"  3sp scan done: {elapsed_3sp:.1f}s ({elapsed_3sp/60:.1f}m)")

# ============================================================
# PHASE C.2: 4-SPECIES SCAN
# ============================================================
eD_max = 10.0 * kd_d0 / T_MAX

log(f"\n[5] 4-species scan: {N_SAMPLES} samples x {N_REPEATS} repeats")
log(f"  epsilon_d range: 0 .. {eD_max:.3e} s^-2")

sampler4 = LatinHypercube(d=3, scramble=True, seed=2027)
u4 = sampler4.random(N_SAMPLES)
dA_vals4 = u4[:, 0] * dA_max
eM_vals4 = u4[:, 1] * eM_max
eD_vals = u4[:, 2] * eD_max

t0 = time.time()
surv_4sp = np.zeros(N_SAMPLES)

for i in range(N_SAMPLES):
    alive_count = 0
    for r in range(N_REPEATS):
        alive, _ = gill_4sp(n_init4, dA_vals4[i], eM_vals4[i], eD_vals[i], seed=i*1000+r*137+42)
        if alive: alive_count += 1
    surv_4sp[i] = alive_count / N_REPEATS

    if (i+1) % 500 == 0 or (i+1) == N_SAMPLES:
        elapsed = time.time() - t0
        rate = (i+1) / elapsed
        eta = (N_SAMPLES - i - 1) / rate
        n_alive = np.sum(surv_4sp[:i+1] >= 0.5)
        log(f"  4sp: {i+1}/{N_SAMPLES} | alive={n_alive}/{i+1} "
            f"({100*n_alive/(i+1):.1f}%) | {rate:.1f}/s | ETA {eta/60:.1f}m")

elapsed_4sp = time.time() - t0
log(f"  4sp scan done: {elapsed_4sp:.1f}s ({elapsed_4sp/60:.1f}m)")

# ============================================================
# ANALYSIS
# ============================================================
log("\n[6] Analysis...")

def find_crit_1d(vals, surv, target=0.5, n_bins=30):
    """Find critical threshold via binning."""
    bins = np.linspace(0, vals.max(), n_bins+1)
    bc = 0.5*(bins[:-1]+bins[1:])
    bs = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    for v, s in zip(vals, surv):
        idx = min(int(v/vals.max()*n_bins), n_bins-1)
        bs[idx] += s; cnt[idx] += 1
    mask = cnt > 0
    bc_m = bc[mask]; bs_m = bs[mask]/cnt[mask]
    # Find crossing
    for i in range(len(bs_m)-1):
        if bs_m[i] >= target and bs_m[i+1] < target:
            frac = (target - bs_m[i]) / (bs_m[i+1] - bs_m[i])
            return bc_m[i] + frac*(bc_m[i+1]-bc_m[i]), bc_m, bs_m
    if len(bs_m) > 0 and bs_m[-1] >= target:
        return vals.max(), bc_m, bs_m  # always survives
    if len(bs_m) > 0 and bs_m[0] < target:
        return 0.0, bc_m, bs_m  # always dies
    return None, bc_m, bs_m

def phys_interpret(crit, kd0, name):
    """Physical interpretation of critical degradation slope."""
    if crit is None or crit <= 0:
        log(f"  {name}: NO THRESHOLD FOUND")
        return None
    kd_72h = kd0 + crit * T_MAX
    ratio = kd_72h / kd0
    halflife_h = kd0 / crit / 3600 if crit > 0 else float('inf')
    norm = crit / (kd0 / T_MAX)
    log(f"  {name}:")
    log(f"    Critical slope = {crit:.3e} s^-2")
    log(f"    kd(72h) = {kd_72h:.3e} s^-1 = {ratio:.2f}x baseline")
    log(f"    Doubling time = {halflife_h:.1f} h")
    log(f"    Normalized = {norm:.2f}x (kd0/T)")
    return {'crit': float(crit), 'kd_72h': float(kd_72h), 'ratio_72h': float(ratio),
            'halflife_h': float(halflife_h), 'normalized': float(norm)}

log("\n--- 3-SPECIES MODEL (TM6v3-min) ---")
crit_dA_3, bc_dA_3, bs_dA_3 = find_crit_1d(dA_vals, surv_3sp)
crit_eM_3, bc_eM_3, bs_eM_3 = find_crit_1d(eM_vals, surv_3sp)

r_dA_3 = phys_interpret(crit_dA_3, kd_A0, 'delta_A (formate)')
r_eM_3 = phys_interpret(crit_eM_3, kd_m0, 'epsilon_m (membrane)')

log("\n--- 4-SPECIES MODEL (TM6v3-DE, with D-module) ---")
crit_dA_4, _, _ = find_crit_1d(dA_vals4, surv_4sp)
crit_eM_4, _, _ = find_crit_1d(eM_vals4, surv_4sp)
crit_eD_4, bc_eD_4, bs_eD_4 = find_crit_1d(eD_vals, surv_4sp)

r_dA_4 = phys_interpret(crit_dA_4, kd_A0, 'delta_A (formate)')
r_eM_4 = phys_interpret(crit_eM_4, kd_m0, 'epsilon_m (membrane)')
r_eD_4 = phys_interpret(crit_eD_4, kd_d0, 'epsilon_d (4F-Azo)')

# ============================================================
# SAVE RESULTS
# ============================================================
np.savez_compressed(OUT_DIR / 'oracle_c_3sp.npz',
    delta_A=dA_vals, epsilon_m=eM_vals, survival=surv_3sp,
    ss=ss3, kd_A0=kd_A0, kd_m0=kd_m0)

np.savez_compressed(OUT_DIR / 'oracle_c_4sp.npz',
    delta_A=dA_vals4, epsilon_m=eM_vals4, epsilon_d=eD_vals,
    survival=surv_4sp, ss=ss4, kd_A0=kd_A0, kd_m0=kd_m0, kd_d0=kd_d0)

summary = {
    'Q-093': 'ORACLE Phase C: time-dependent degradation',
    'date': '2026-03-09',
    'T_horizon_h': 72,
    'N_molecules': N_MOL,
    'model_3sp': {
        'ss_mM': [round(x*1e3, 4) for x in ss3],
        'n_samples': N_SAMPLES,
        'n_repeats': N_REPEATS,
        'alive_frac': float(np.mean(surv_3sp >= 0.5)),
        'elapsed_s': round(elapsed_3sp, 1),
        'delta_A': r_dA_3,
        'epsilon_m': r_eM_3,
    },
    'model_4sp': {
        'ss_mM': [round(x*1e3, 4) for x in ss4],
        'n_samples': N_SAMPLES,
        'n_repeats': N_REPEATS,
        'alive_frac': float(np.mean(surv_4sp >= 0.5)),
        'elapsed_s': round(elapsed_4sp, 1),
        'delta_A': r_dA_4,
        'epsilon_m': r_eM_4,
        'epsilon_d': r_eD_4,
    },
    'total_time_min': round((elapsed_3sp + elapsed_4sp) / 60, 1),
}

with open(OUT_DIR / 'oracle_c_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
log(f"\n  Summary: {OUT_DIR / 'oracle_c_summary.json'}")

# ============================================================
# PLOTS
# ============================================================
log("\n[7] Generating plots...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # --- Plot 1: 3sp 2D heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    n_bins = 25
    dA_norm = dA_vals / (kd_A0/T_MAX)
    eM_norm = eM_vals / (kd_m0/T_MAX)
    dA_b = np.linspace(0, 10, n_bins+1)
    eM_b = np.linspace(0, 10, n_bins+1)
    grid = np.full((n_bins, n_bins), np.nan)
    cnt = np.zeros((n_bins, n_bins))
    for d, e, s in zip(dA_norm, eM_norm, surv_3sp):
        i = min(int(d/10*n_bins), n_bins-1)
        j = min(int(e/10*n_bins), n_bins-1)
        if np.isnan(grid[j,i]): grid[j,i]=0
        grid[j,i]+=s; cnt[j,i]+=1
    m = cnt>0; grid[m]/=cnt[m]
    dc = 0.5*(dA_b[:-1]+dA_b[1:]); ec = 0.5*(eM_b[:-1]+eM_b[1:])
    im = ax.pcolormesh(dc, ec, grid, cmap='RdYlGn', vmin=0, vmax=1, shading='nearest')
    plt.colorbar(im, label='Survival Rate (72h)', ax=ax)
    try:
        ax.contour(dc, ec, grid, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    except: pass
    ax.set_xlabel('delta_A / (kd_A0 / 72h)')
    ax.set_ylabel('epsilon_m / (kd_m0 / 72h)')
    ax.set_title(f'ORACLE Phase C: 3sp 72h Survival\nkd_A0={kd_A0:.0e}, kd_m0={kd_m0:.0e} s^-1')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'oracle_c_3sp_heatmap.png', dpi=150)
    plt.close()
    log(f"  -> oracle_c_3sp_heatmap.png")

    # --- Plot 2: 3sp marginals ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, bc, bs, name, kd_0, crit in [
        (axes[0], bc_dA_3, bs_dA_3, 'delta_A (formate)', kd_A0, crit_dA_3),
        (axes[1], bc_eM_3, bs_eM_3, 'epsilon_m (membrane)', kd_m0, crit_eM_3)]:
        if bc is not None:
            ax.plot(bc/(kd_0/T_MAX), bs, 'b-o', markersize=4)
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            if crit and crit > 0:
                ax.axvline(crit/(kd_0/T_MAX), color='red', linestyle='--', linewidth=2,
                          label=f'Critical: {crit/(kd_0/T_MAX):.2f}x')
                ax.legend()
        ax.set_xlabel(f'{name} / (kd0/72h)')
        ax.set_ylabel('Survival Rate')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'Marginal: {name}')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'oracle_c_3sp_marginals.png', dpi=150)
    plt.close()
    log(f"  -> oracle_c_3sp_marginals.png")

    # --- Plot 3: 4sp marginals ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, vals, surv, name, kd_0 in [
        (axes[0], dA_vals4, surv_4sp, 'delta_A', kd_A0),
        (axes[1], eM_vals4, surv_4sp, 'epsilon_m', kd_m0),
        (axes[2], eD_vals, surv_4sp, 'epsilon_d', kd_d0)]:
        crit, bc, bs = find_crit_1d(vals, surv)
        if bc is not None:
            ax.plot(bc/(kd_0/T_MAX), bs, 'b-o', markersize=4)
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            if crit and crit > 0:
                ax.axvline(crit/(kd_0/T_MAX), color='red', linestyle='--', linewidth=2,
                          label=f'Critical: {crit/(kd_0/T_MAX):.2f}x')
                ax.legend()
        ax.set_xlabel(f'{name} / (kd0/72h)')
        ax.set_ylabel('Survival Rate')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'4sp Marginal: {name}')
        ax.grid(alpha=0.3)
    plt.suptitle('ORACLE Phase C: 4sp (with D-module) Marginals', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'oracle_c_4sp_marginals.png', dpi=150)
    plt.close()
    log(f"  -> oracle_c_4sp_marginals.png")

    # --- Plot 4: 4sp 2D slices ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pairs = [
        (dA_vals4/(kd_A0/T_MAX), eM_vals4/(kd_m0/T_MAX), 'dA_norm', 'eM_norm'),
        (dA_vals4/(kd_A0/T_MAX), eD_vals/(kd_d0/T_MAX), 'dA_norm', 'eD_norm'),
        (eM_vals4/(kd_m0/T_MAX), eD_vals/(kd_d0/T_MAX), 'eM_norm', 'eD_norm'),
    ]
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        nb = 20
        xb = np.linspace(0, 10, nb+1); yb = np.linspace(0, 10, nb+1)
        g = np.full((nb,nb), np.nan); c = np.zeros((nb,nb))
        for xi, yi, si in zip(x, y, surv_4sp):
            ii = min(int(xi/10*nb), nb-1); jj = min(int(yi/10*nb), nb-1)
            if np.isnan(g[jj,ii]): g[jj,ii]=0
            g[jj,ii]+=si; c[jj,ii]+=1
        mk = c>0; g[mk]/=c[mk]
        xc = 0.5*(xb[:-1]+xb[1:]); yc = 0.5*(yb[:-1]+yb[1:])
        im = ax.pcolormesh(xc, yc, g, cmap='RdYlGn', vmin=0, vmax=1, shading='nearest')
        plt.colorbar(im, ax=ax, label='Survival')
        try: ax.contour(xc, yc, g, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        except: pass
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    plt.suptitle('ORACLE Phase C: 4sp 2D Survival Slices', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'oracle_c_4sp_2d_slices.png', dpi=150)
    plt.close()
    log(f"  -> oracle_c_4sp_2d_slices.png")

except Exception as e:
    log(f"  Plot error: {e}")

# ============================================================
# FINAL SUMMARY
# ============================================================
total = elapsed_3sp + elapsed_4sp
log(f"\n{'='*70}")
log(f"  ORACLE PHASE C COMPLETE")
log(f"  Total time: {total/60:.1f} min")
log(f"  Results: {OUT_DIR}")
log(f"{'='*70}")
log("DONE")
