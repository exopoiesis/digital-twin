#!/usr/bin/env python3
"""
High-temperature MD simulations of H in pentlandite (Fe,Ni)9S8
for Arrhenius plot of diffusion coefficient D_H(T).

Purpose: Run MD at 700, 800, 900, 1000, 1100 K to extract D_H at each T,
then fit ln(D_H) vs 1/T to get activation energy E_a and compare with
NEB result (E_a = 1.43 eV from 2x2x2 MACE large).

Crystal: Fm3m (225), a=10.07 A, Ni-rich Fe4Ni5S8, 2x2x2 supercell (136 atoms)
Defect: one S vacancy + one interstitial H placed at vacancy site
MD: NVT Langevin thermostat, dt=1 fs, 50000 steps (50 ps) per temperature
     Equilibration: 5 ps (5000 steps), Production: 45 ps (45000 steps)

Output: q071_md_arrhenius_pentlandite.json, q071_md_arrhenius_pentlandite.png
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import numpy as np
from pathlib import Path

import torch
from ase import Atom
from ase.spacegroup import crystal
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import mace_mp
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("/workspace/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# --- VRAM preflight ---
def _check_vram(required_gb):
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - torch.cuda.memory_allocated(0) / 1e9
        print(f"[VRAM] {total:.1f} GB total, {free:.1f} GB free, {required_gb:.1f} GB required")
        if free < required_gb:
            print(f"[VRAM] WARNING: only {free:.1f} GB free, need {required_gb:.1f} GB — risk of OOM")
_check_vram(2.5)

# Temperatures for Arrhenius scan (K)
TEMPERATURES = [700, 800, 900, 1000, 1100]

# MD parameters
DT_FS = 1.0           # timestep in fs
N_STEPS = 50000        # total steps per temperature (50 ps)
N_EQUIL = 5000         # equilibration steps (5 ps)
RECORD_EVERY = 10      # record H position every N steps
FRICTION = 0.01        # Langevin friction in 1/fs

# NEB reference
NEB_E_A_EV = 1.43      # E_a from 2x2x2 NEB MACE large (РЕШЕНИЕ-044/046)
HOP_DISTANCE_A = 2.8   # approximate hop distance in Angstrom

# Boltzmann constant
KB_EV_K = 8.617333e-5  # eV/K


def build_pentlandite_supercell():
    """Build Ni-rich pentlandite Fe4Ni5S8 primitive cell, then 2x2x2 supercell.

    Identical to NEB and MD scripts for consistency.
    """
    atoms = crystal(
        symbols=["Ni", "Fe", "S"],
        basis=[(0, 0, 0), (0.356, 0.356, 0.356), (0.118, 0.118, 0.118)],
        spacegroup=225,
        cellpar=[10.07, 10.07, 10.07, 90, 90, 90],
        primitive_cell=True,
    )
    # Make Ni-rich: replace first 5 Fe with Ni
    symbols = atoms.get_chemical_symbols()
    fe_count = 0
    for i, s in enumerate(symbols):
        if s == "Fe" and fe_count < 5:
            symbols[i] = "Ni"
            fe_count += 1
    atoms.set_chemical_symbols(symbols)
    return atoms.repeat((2, 2, 2))


def find_s_vacancy_site(atoms):
    """Find first S atom index for vacancy creation."""
    for i, s in enumerate(atoms.get_chemical_symbols()):
        if s == "S":
            return i
    raise ValueError("No S atoms found")


def compute_msd(positions, dt_fs):
    """Compute MSD(t) for a single particle trajectory using multiple time origins.

    Args:
        positions: (N_frames, 3) unwrapped positions in Angstrom
        dt_fs: time step between recorded frames in femtoseconds

    Returns:
        times_ps: array of lag times in picoseconds
        msd: array of mean square displacements in Angstrom^2
    """
    n_frames = len(positions)
    max_lag = n_frames // 2
    msd = np.zeros(max_lag)

    for lag in range(1, max_lag):
        displacements = positions[lag:] - positions[:-lag]
        sq_disp = np.sum(displacements**2, axis=1)
        msd[lag] = np.mean(sq_disp)

    times_ps = np.arange(max_lag) * dt_fs / 1000.0  # fs -> ps
    return times_ps, msd


def fit_diffusion_coefficient(times_ps, msd_A2):
    """Fit D from MSD = 6*D*t (3D Einstein relation).

    Uses linear fit on the range [20%, 80%] of total time to avoid
    ballistic regime (short t) and poor statistics (long t).

    Returns:
        D_cm2s: diffusion coefficient in cm^2/s
        slope: MSD slope in A^2/ps
        r_squared: quality of linear fit
    """
    n = len(times_ps)
    i_start = max(1, n // 5)
    i_end = 4 * n // 5

    t_fit = times_ps[i_start:i_end]
    msd_fit = msd_A2[i_start:i_end]

    if len(t_fit) < 3:
        return 0.0, 0.0, 0.0

    # Linear fit: MSD = 6*D*t + b
    coeffs = np.polyfit(t_fit, msd_fit, 1)
    slope = coeffs[0]  # A^2/ps

    # D = slope / 6, convert: 1 A^2/ps = 1e-4 cm^2/s
    D_A2_ps = slope / 6.0
    D_cm2_s = D_A2_ps * 1e-4

    # R^2
    msd_pred = np.polyval(coeffs, t_fit)
    ss_res = np.sum((msd_fit - msd_pred)**2)
    ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return D_cm2_s, slope, r_squared


def arrhenius_D_neb(E_a_eV, hop_A, T_K):
    """Arrhenius estimate from NEB: D = nu_0 * a^2 * exp(-Ea/kT), nu_0 = 10^13 Hz."""
    a_cm = hop_A * 1e-8
    return 1e13 * a_cm**2 * np.exp(-E_a_eV / (KB_EV_K * T_K))


def run_md_at_temperature(atoms_template, calc, T_K, h_idx):
    """Run a single MD simulation at temperature T_K.

    Args:
        atoms_template: relaxed Atoms object with H (will be copied)
        calc: MACE calculator
        T_K: target temperature in Kelvin
        h_idx: index of H atom to track

    Returns:
        dict with D_H, MSD data, temperature stats, timing
    """
    print(f"\n{'='*60}")
    print(f"  MD at T = {T_K} K")
    print(f"{'='*60}")
    t0_run = time.time()

    # Copy structure — IMPORTANT: .copy() does NOT copy calc
    atoms = atoms_template.copy()
    atoms.calc = calc

    # Remove any constraints for MD
    atoms.set_constraint()

    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_K)

    # Langevin thermostat
    dt = DT_FS * units.fs
    friction = FRICTION / units.fs
    dyn = Langevin(atoms, dt, temperature_K=T_K, friction=friction, logfile=None)

    # Storage for H trajectory (unwrapped coordinates)
    n_production = N_STEPS - N_EQUIL
    n_records_equil = N_EQUIL // RECORD_EVERY
    n_records_prod = n_production // RECORD_EVERY
    n_records_total = N_STEPS // RECORD_EVERY

    h_positions_all = []  # store all positions for unwrapping
    h_pos_unwrapped = atoms.positions[h_idx].copy()
    h_pos_prev = atoms.positions[h_idx].copy()
    cell_diag = np.diag(atoms.cell)

    temps_equil = []
    temps_prod = []
    record_count = [0]  # mutable for closure
    phase = ["equil"]   # mutable for closure

    def record_trajectory():
        nonlocal h_pos_unwrapped, h_pos_prev
        # Unwrap PBC: detect jumps > half cell
        h_pos_current = atoms.positions[h_idx].copy()
        delta = h_pos_current - h_pos_prev
        for dim in range(3):
            if delta[dim] > cell_diag[dim] / 2:
                delta[dim] -= cell_diag[dim]
            elif delta[dim] < -cell_diag[dim] / 2:
                delta[dim] += cell_diag[dim]
        h_pos_unwrapped = h_pos_unwrapped + delta
        h_pos_prev = h_pos_current.copy()

        T_inst = atoms.get_temperature()
        if phase[0] == "equil":
            temps_equil.append(T_inst)
        else:
            h_positions_all.append(h_pos_unwrapped.copy())
            temps_prod.append(T_inst)
        record_count[0] += 1

    dyn.attach(record_trajectory, interval=RECORD_EVERY)

    # ── Equilibration ──
    print(f"  Equilibration: {N_EQUIL} steps ({N_EQUIL * DT_FS / 1000:.1f} ps)...")
    t_eq = time.time()
    try:
        dyn.run(N_EQUIL)
    except Exception as exc:
        print(f"  EQUILIBRATION FAILED at T={T_K}K: {exc}")
        return {"T_K": T_K, "success": False, "error": str(exc)}
    t_eq = time.time() - t_eq
    avg_T_eq = np.mean(temps_equil) if temps_equil else 0
    print(f"  Equilibration done in {t_eq:.0f}s, <T>_eq = {avg_T_eq:.1f} K")

    # ── Production ──
    phase[0] = "prod"
    print(f"  Production: {n_production} steps ({n_production * DT_FS / 1000:.1f} ps)...")
    t_prod = time.time()
    try:
        chunk_size = 5000
        n_chunks = n_production // chunk_size
        for chunk in range(n_chunks):
            dyn.run(chunk_size)
            elapsed = time.time() - t_prod
            step_now = (chunk + 1) * chunk_size
            rate = step_now / elapsed if elapsed > 0 else 0
            avg_T_chunk = np.mean(temps_prod[-chunk_size // RECORD_EVERY:]) if temps_prod else 0
            print(f"    Step {step_now}/{n_production} | "
                  f"<T> = {avg_T_chunk:.0f} K | "
                  f"{rate:.0f} steps/s | "
                  f"{elapsed:.0f}s")
    except Exception as exc:
        print(f"  PRODUCTION FAILED at T={T_K}K: {exc}")
        if len(h_positions_all) < 100:
            return {"T_K": T_K, "success": False, "error": str(exc)}
    t_prod = time.time() - t_prod

    # ── MSD analysis ──
    n_actual = len(h_positions_all)
    print(f"  Production frames: {n_actual}")

    if n_actual < 50:
        print(f"  Insufficient frames for MSD at T={T_K}K")
        return {"T_K": T_K, "success": False, "error": "insufficient_frames"}

    h_pos_array = np.array(h_positions_all)
    dt_record_fs = DT_FS * RECORD_EVERY
    times_ps, msd = compute_msd(h_pos_array, dt_record_fs)
    D_cm2s, slope, r_sq = fit_diffusion_coefficient(times_ps, msd)

    avg_T_prod = np.mean(temps_prod)
    std_T_prod = np.std(temps_prod)
    total_time = time.time() - t0_run

    print(f"  MSD slope = {slope:.4f} A^2/ps")
    print(f"  D_H = {D_cm2s:.3e} cm^2/s")
    print(f"  R^2 = {r_sq:.4f}")
    print(f"  <T>_prod = {avg_T_prod:.1f} +/- {std_T_prod:.1f} K")
    print(f"  Total time for T={T_K}K: {total_time:.0f}s")

    return {
        "T_K": T_K,
        "T_actual_K": float(avg_T_prod),
        "T_std_K": float(std_T_prod),
        "success": True,
        "D_cm2s": float(D_cm2s),
        "msd_slope_A2_per_ps": float(slope),
        "msd_fit_R2": float(r_sq),
        "n_production_frames": n_actual,
        "time_s": float(total_time),
        "times_ps": times_ps.tolist(),
        "msd_A2": msd.tolist(),
    }


def main():
    t_total = time.time()

    results = {
        "task": "Arrhenius MD scan for H diffusion in pentlandite",
        "mineral": "pentlandite",
        "model": "MACE-MP-0 large",
        "method": "NVT Langevin MD",
        "cell_type": "2x2x2 supercell (136 atoms + H)",
        "dt_fs": DT_FS,
        "n_steps_total": N_STEPS,
        "n_equil_steps": N_EQUIL,
        "n_production_steps": N_STEPS - N_EQUIL,
        "total_time_per_T_ps": N_STEPS * DT_FS / 1000,
        "production_time_ps": (N_STEPS - N_EQUIL) * DT_FS / 1000,
        "friction_per_fs": FRICTION,
        "record_every": RECORD_EVERY,
        "temperatures_K": TEMPERATURES,
        "neb_reference_E_a_eV": NEB_E_A_EV,
    }

    # ══════════════════════════════════════════════════════════════
    # [1/6] GPU info
    # ══════════════════════════════════════════════════════════════
    print("[1/6] GPU info")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")
        results["gpu"] = gpu_name
        results["vram_gb"] = round(vram_gb, 1)
    else:
        print("  WARNING: CUDA not available, falling back to CPU")
        results["gpu"] = "N/A (CPU fallback)"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ══════════════════════════════════════════════════════════════
    # [2/6] Build supercell + create S vacancy + place H
    # ══════════════════════════════════════════════════════════════
    print("\n[2/6] Build pentlandite 2x2x2 supercell, create S vacancy, place H")
    t0 = time.time()

    atoms = build_pentlandite_supercell()
    n_pristine = len(atoms)
    formula_pristine = atoms.get_chemical_formula()
    print(f"  Pristine: {formula_pristine}, {n_pristine} atoms")
    results["formula_pristine"] = formula_pristine
    results["n_atoms_pristine"] = n_pristine

    # Load MACE calculator
    print("  Loading MACE-MP-0 large...")
    calc = mace_mp(model="large", device=device, default_dtype="float64")
    print(f"  Loaded on {device}")

    # Relax pristine cell
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=0.01, steps=300)
    e_pristine = atoms.get_potential_energy()
    print(f"  E_pristine = {e_pristine:.4f} eV ({opt.nsteps} steps)")
    results["E_pristine_eV"] = float(e_pristine)

    # Create S vacancy and place H
    s_idx = find_s_vacancy_site(atoms)
    pos_s = atoms.positions[s_idx].copy()
    print(f"  Removing S atom at index {s_idx}, position {pos_s}")

    # Remove S atom to create vacancy
    del atoms[s_idx]

    # Place H at vacancy site (offset slightly to avoid exact lattice point)
    h_pos = pos_s + np.array([0.0, 0.0, 0.3])
    atoms.append(Atom("H", position=h_pos))
    h_idx = len(atoms) - 1

    # Relax H position (freeze lattice)
    atoms.calc = calc  # re-assign after structural change
    heavy_idx = [i for i in range(len(atoms)) if atoms[i].symbol != "H"]
    atoms.set_constraint(FixAtoms(indices=heavy_idx))
    opt_h = LBFGS(atoms, logfile=None)
    opt_h.run(fmax=0.02, steps=100)
    e_with_h = atoms.get_potential_energy()

    n_total = len(atoms)
    formula_total = atoms.get_chemical_formula()
    print(f"  With H: {formula_total}, {n_total} atoms, H at index {h_idx}")
    print(f"  E_with_H = {e_with_h:.4f} eV ({opt_h.nsteps} steps)")
    results["formula_total"] = formula_total
    results["n_atoms_total"] = n_total
    results["H_index"] = h_idx
    results["E_with_H_eV"] = float(e_with_h)

    # Remove constraints — will be set per-run
    atoms.set_constraint()
    print(f"  Setup done in {time.time()-t0:.0f}s")

    # ══════════════════════════════════════════════════════════════
    # [3/6] Run MD at each temperature
    # ══════════════════════════════════════════════════════════════
    print(f"\n[3/6] Running MD at {len(TEMPERATURES)} temperatures: {TEMPERATURES}")

    md_results = []
    for T_K in TEMPERATURES:
        res_T = run_md_at_temperature(atoms, calc, T_K, h_idx)
        md_results.append(res_T)

    results["md_runs"] = []
    for r in md_results:
        # Store compact version (without MSD arrays) in main results
        r_compact = {k: v for k, v in r.items() if k not in ("times_ps", "msd_A2")}
        results["md_runs"].append(r_compact)

    # ══════════════════════════════════════════════════════════════
    # [4/6] Arrhenius analysis
    # ══════════════════════════════════════════════════════════════
    print(f"\n[4/6] Arrhenius analysis")

    # Collect successful runs with positive D
    successful = [(r["T_K"], r["D_cm2s"]) for r in md_results
                   if r.get("success") and r.get("D_cm2s", 0) > 0]

    if len(successful) < 2:
        print("  ERROR: fewer than 2 successful runs with D>0, cannot fit Arrhenius")
        results["arrhenius_fit"] = {"success": False, "reason": "insufficient_data"}
    else:
        T_arr = np.array([s[0] for s in successful])
        D_arr = np.array([s[1] for s in successful])

        inv_T = 1.0 / T_arr                    # 1/K
        inv_T_1000 = 1000.0 / T_arr            # 1000/T for plotting
        ln_D = np.log(D_arr)

        # Linear regression: ln(D) = ln(D0) - E_a / (kB * T)
        # ln(D) = A - B * (1/T),  where B = E_a / kB
        slope_arr, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_D)

        E_a_arrhenius = -slope_arr * KB_EV_K    # eV (slope is negative)
        D0 = np.exp(intercept)                   # pre-exponential in cm^2/s
        R_squared = r_value**2

        print(f"  Arrhenius fit: ln(D) = {intercept:.3f} + ({slope_arr:.1f}) * (1/T)")
        print(f"  E_a (Arrhenius) = {E_a_arrhenius:.3f} eV")
        print(f"  D0 = {D0:.3e} cm^2/s")
        print(f"  R^2 = {R_squared:.6f}")
        print(f"  E_a (NEB reference) = {NEB_E_A_EV:.3f} eV")
        print(f"  Ratio E_a(MD) / E_a(NEB) = {E_a_arrhenius / NEB_E_A_EV:.3f}")

        results["arrhenius_fit"] = {
            "success": True,
            "E_a_eV": float(E_a_arrhenius),
            "D0_cm2s": float(D0),
            "ln_D0": float(intercept),
            "slope_K": float(slope_arr),
            "R_squared": float(R_squared),
            "p_value": float(p_value),
            "std_err_slope": float(std_err),
            "E_a_std_err_eV": float(std_err * KB_EV_K),
            "n_points": len(successful),
            "temperatures_used_K": [float(t) for t in T_arr],
            "D_values_cm2s": [float(d) for d in D_arr],
        }

        results["comparison"] = {
            "E_a_MD_arrhenius_eV": float(E_a_arrhenius),
            "E_a_NEB_2x2x2_eV": NEB_E_A_EV,
            "ratio_MD_over_NEB": float(E_a_arrhenius / NEB_E_A_EV),
            "agreement_within_20pct": bool(abs(E_a_arrhenius / NEB_E_A_EV - 1.0) < 0.2),
        }

    # ══════════════════════════════════════════════════════════════
    # [5/6] Save JSON
    # ══════════════════════════════════════════════════════════════
    print(f"\n[5/6] Save results")
    results["total_time_s"] = float(time.time() - t_total)
    results["total_time_hours"] = float((time.time() - t_total) / 3600)

    json_path = RESULTS / "q071_md_arrhenius_pentlandite.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {json_path}")

    # ══════════════════════════════════════════════════════════════
    # [6/6] Plot PNG (two panels)
    # ══════════════════════════════════════════════════════════════
    print(f"\n[6/6] Generate Arrhenius plot")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: MSD vs time for all temperatures ──
    ax1 = axes[0]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(TEMPERATURES)))

    for i, r in enumerate(md_results):
        T_K = r["T_K"]
        if r.get("success") and "times_ps" in r and "msd_A2" in r:
            times_ps = np.array(r["times_ps"])
            msd = np.array(r["msd_A2"])
            D_val = r.get("D_cm2s", 0)
            label = f"{T_K} K (D={D_val:.1e})"
            ax1.plot(times_ps, msd, color=colors[i], linewidth=1.5, label=label)
        else:
            ax1.plot([], [], color=colors[i], linewidth=1.5,
                     label=f"{T_K} K (FAILED)", linestyle="--")

    ax1.set_xlabel("Lag time (ps)", fontsize=12)
    ax1.set_ylabel("MSD ($\\AA^2$)", fontsize=12)
    ax1.set_title("H Mean Square Displacement vs Temperature", fontsize=12)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Arrhenius plot ──
    ax2 = axes[1]

    if len(successful) >= 2:
        T_arr = np.array([s[0] for s in successful])
        D_arr_vals = np.array([s[1] for s in successful])
        inv_T_1000 = 1000.0 / T_arr
        ln_D = np.log(D_arr_vals)

        # Data points
        ax2.scatter(inv_T_1000, ln_D, c="blue", s=80, zorder=5, label="MD data")

        # Fit line
        inv_T_fit = np.linspace(inv_T_1000.min() * 0.95, inv_T_1000.max() * 1.05, 100)
        ln_D_fit = intercept + slope_arr * (inv_T_fit / 1000.0)
        ax2.plot(inv_T_fit, ln_D_fit, "r-", linewidth=2,
                 label=f"Fit: $E_a$ = {E_a_arrhenius:.2f} eV")

        # NEB reference line
        ln_D_neb = np.log(1e13 * (HOP_DISTANCE_A * 1e-8)**2) - NEB_E_A_EV / (KB_EV_K * 1000.0 / inv_T_fit)
        ax2.plot(inv_T_fit, ln_D_neb, "g--", linewidth=1.5, alpha=0.7,
                 label=f"NEB: $E_a$ = {NEB_E_A_EV:.2f} eV")

        # Add temperature labels on top x-axis
        ax2_top = ax2.twiny()
        temp_ticks = inv_T_1000
        ax2_top.set_xlim(ax2.get_xlim())
        ax2_top.set_xticks(temp_ticks)
        ax2_top.set_xticklabels([f"{int(t)} K" for t in T_arr], fontsize=9)

        # Annotation box
        textstr = (
            f"$E_a^{{MD}}$ = {E_a_arrhenius:.3f} $\\pm$ {std_err * KB_EV_K:.3f} eV\n"
            f"$E_a^{{NEB}}$ = {NEB_E_A_EV:.3f} eV\n"
            f"Ratio = {E_a_arrhenius / NEB_E_A_EV:.3f}\n"
            f"$R^2$ = {R_squared:.4f}\n"
            f"$D_0$ = {D0:.2e} cm$^2$/s"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                 verticalalignment="top", horizontalalignment="right", bbox=props)

        ax2.legend(fontsize=10, loc="lower left")
    else:
        ax2.text(0.5, 0.5, "Insufficient data\nfor Arrhenius fit",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=14)

    ax2.set_xlabel("1000 / T  (K$^{-1}$)", fontsize=12)
    ax2.set_ylabel("ln(D$_H$)  [D in cm$^2$/s]", fontsize=12)
    ax2.set_title("Arrhenius Plot: H Diffusion in Pentlandite", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Pentlandite 2x2x2 — H diffusion Arrhenius scan (MACE-MP-0 large)\n"
        f"{len(TEMPERATURES)} temperatures, {N_STEPS * DT_FS / 1000:.0f} ps each "
        f"(equil {N_EQUIL * DT_FS / 1000:.0f} ps + prod {(N_STEPS - N_EQUIL) * DT_FS / 1000:.0f} ps)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    png_path = RESULTS / "q071_md_arrhenius_pentlandite.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {png_path}")

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    total_hrs = (time.time() - t_total) / 3600
    print(f"\n{'='*60}")
    print(f"ARRHENIUS MD SCAN — SUMMARY")
    print(f"{'='*60}")
    print(f"System: pentlandite 2x2x2 + S vacancy + H ({n_total} atoms)")
    print(f"Temperatures: {TEMPERATURES} K")
    print(f"MD per T: {N_STEPS} steps x {DT_FS} fs = {N_STEPS * DT_FS / 1000:.0f} ps")
    print(f"  Equilibration: {N_EQUIL * DT_FS / 1000:.0f} ps, Production: {(N_STEPS - N_EQUIL) * DT_FS / 1000:.0f} ps")
    print()
    print(f"{'T (K)':>8} {'D_H (cm^2/s)':>15} {'MSD slope':>12} {'R^2':>8} {'Status':>8}")
    print("-" * 60)
    for r in md_results:
        T = r["T_K"]
        if r.get("success"):
            D = r.get("D_cm2s", 0)
            sl = r.get("msd_slope_A2_per_ps", 0)
            r2 = r.get("msd_fit_R2", 0)
            print(f"{T:>8d} {D:>15.3e} {sl:>12.4f} {r2:>8.4f} {'OK':>8}")
        else:
            print(f"{T:>8d} {'---':>15} {'---':>12} {'---':>8} {'FAIL':>8}")

    if len(successful) >= 2:
        print()
        print(f"Arrhenius fit:")
        print(f"  E_a (MD)  = {E_a_arrhenius:.3f} +/- {std_err * KB_EV_K:.3f} eV")
        print(f"  E_a (NEB) = {NEB_E_A_EV:.3f} eV")
        print(f"  Ratio     = {E_a_arrhenius / NEB_E_A_EV:.3f}")
        print(f"  R^2       = {R_squared:.6f}")
        print(f"  D0        = {D0:.3e} cm^2/s")

    print(f"[VRAM] Peak: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")

    print(f"\nTotal wall time: {total_hrs:.2f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
