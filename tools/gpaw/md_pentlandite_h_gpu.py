#!/usr/bin/env python3
"""
Molecular dynamics of H in pentlandite (Fe,Ni)9S8 at 300K
using MACE-MP-0 large model on GPU.

Purpose: compute H diffusion coefficient D_H from MSD via Einstein relation
and compare with NEB-derived Arrhenius estimate.

Crystal: Fm3m (225), a=10.07 A, Ni-rich, 2x2x2 supercell (136 atoms + 1 H)
MD: NVT Langevin thermostat, T=300K, dt=0.5 fs, 50000 steps (25 ps)
Output: q071_md_pentlandite_300K.json, q071_md_pentlandite_300K.png
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
from ase import units
from mace.calculators import mace_mp

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

# NEB-derived values for comparison (Arrhenius: D = nu * a^2 * exp(-Ea/kT))
NEB_COMPARISON = {
    "E_a_MACE_large_eV": 1.29,
    "E_a_MACE_medium_eV": 0.96,
    "hop_distance_A": 2.824,
}


def build_pentlandite_supercell():
    """Build Ni-rich pentlandite Fe3Ni6S8 primitive cell, then 2x2x2 supercell."""
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


def find_nearest_s(atoms):
    """Find the S atom with the most neighbors (well-embedded site)."""
    s_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "S"]
    # Pick first S atom (they are all equivalent by symmetry in supercell)
    return s_indices[0]


def compute_msd(positions, dt_fs):
    """Compute MSD(t) for a single particle trajectory.

    positions: (N_frames, 3) unwrapped positions
    dt_fs: time step between frames in femtoseconds
    Returns: times (ps), msd (A^2)
    """
    n_frames = len(positions)
    # Use multiple time origins for better statistics
    max_lag = n_frames // 2
    msd = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    for lag in range(1, max_lag):
        displacements = positions[lag:] - positions[:-lag]
        sq_disp = np.sum(displacements**2, axis=1)
        msd[lag] = np.mean(sq_disp)
        counts[lag] = len(sq_disp)

    times_ps = np.arange(max_lag) * dt_fs / 1000.0  # fs -> ps
    return times_ps, msd


def fit_diffusion_coefficient(times_ps, msd_A2):
    """Fit D from MSD = 6*D*t (3D diffusion).

    Uses linear fit on the range [20%, 80%] of total time to avoid
    ballistic regime (short t) and poor statistics (long t).
    Returns D in cm^2/s.
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

    # D = slope / 6
    D_A2_ps = slope / 6.0
    # Convert: 1 A^2/ps = 1e-16 cm^2 / 1e-12 s = 1e-4 cm^2/s
    D_cm2_s = D_A2_ps * 1e-4

    # R^2 for quality assessment
    msd_pred = np.polyval(coeffs, t_fit)
    ss_res = np.sum((msd_fit - msd_pred)**2)
    ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return D_cm2_s, slope, r_squared


def arrhenius_D(E_a_eV, hop_A, T=298.15):
    """Arrhenius estimate: D = nu_0 * a^2 * exp(-Ea/kT), nu_0 = 10^13 Hz."""
    kB = 8.617e-5  # eV/K
    a_cm = hop_A * 1e-8
    return 1e13 * a_cm**2 * np.exp(-E_a_eV / (kB * T))


def main():
    t_total = time.time()
    results = {
        "mineral": "pentlandite",
        "model": "MACE-MP-0 large",
        "device": "cuda",
        "method": "NVT Langevin MD",
        "T_K": 300,
        "dt_fs": 0.5,
        "n_steps": 50000,
        "total_time_ps": 25.0,
        "cell_type": "2x2x2 supercell",
    }

    # ── [1/8] GPU info ──────────────────────────────────────────────
    print("[1/8] GPU info")
    t0 = time.time()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")
        results["gpu"] = gpu_name
        results["vram_gb"] = round(vram_gb, 1)
    else:
        print("  WARNING: CUDA not available, falling back to CPU")
        results["gpu"] = "N/A (CPU fallback)"
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [2/8] Build supercell + add H ────────────────────────────────
    print("[2/8] Build pentlandite 2x2x2 supercell")
    t0 = time.time()
    atoms = build_pentlandite_supercell()
    n_atoms_pristine = len(atoms)
    formula_pristine = atoms.get_chemical_formula()
    print(f"  Pristine: {formula_pristine}, {n_atoms_pristine} atoms")
    results["formula_pristine"] = formula_pristine
    results["n_atoms_pristine"] = n_atoms_pristine
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [3/8] Load MACE-MP-0 large on GPU ────────────────────────────
    print("[3/8] Load MACE-MP-0 large on GPU")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="large", device=device, default_dtype="float64")
    print(f"  loaded on {device} in {time.time()-t0:.1f}s")

    # ── [4/8] Relax pristine + add H ────────────────────────────────
    print("[4/8] Relax pristine cell, then add H on S site and relax")
    t0 = time.time()

    # Relax pristine
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=0.01, steps=300)
    e_pristine = atoms.get_potential_energy()
    print(f"  E_pristine = {e_pristine:.4f} eV ({opt.nsteps} steps)")
    results["E_pristine_eV"] = float(e_pristine)

    # Find a well-embedded S site and place H there
    s_idx = find_nearest_s(atoms)
    pos_s = atoms.positions[s_idx].copy()

    # Add H ~1.3 A above the S site (typical S-H bond)
    h_pos = pos_s + np.array([0.0, 0.0, 1.34])
    atoms.append(Atom("H", position=h_pos))
    h_idx = len(atoms) - 1

    # Relax only H (freeze everything else)
    atoms.calc = calc
    heavy_idx = [i for i in range(len(atoms)) if atoms[i].symbol != "H"]
    atoms.set_constraint(FixAtoms(indices=heavy_idx))
    opt_h = LBFGS(atoms, logfile=None)
    opt_h.run(fmax=0.02, steps=100)
    e_with_h = atoms.get_potential_energy()
    print(f"  E_with_H = {e_with_h:.4f} eV ({opt_h.nsteps} steps)")
    results["E_with_H_eV"] = float(e_with_h)

    n_atoms_total = len(atoms)
    formula_total = atoms.get_chemical_formula()
    print(f"  System: {formula_total}, {n_atoms_total} atoms, H at index {h_idx}")
    results["formula_total"] = formula_total
    results["n_atoms_total"] = n_atoms_total
    results["H_index"] = h_idx
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [5/8] Setup MD ───────────────────────────────────────────────
    print("[5/8] Setup NVT Langevin MD (T=300K, dt=0.5fs, 50000 steps)")
    t0 = time.time()

    # Remove constraints for MD — all atoms free
    atoms.set_constraint()
    atoms.calc = calc

    # Initialize velocities at target temperature
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    # Langevin thermostat
    dt = 0.5 * units.fs
    T_target = 300  # K
    friction = 0.01 / units.fs  # friction coefficient
    dyn = Langevin(atoms, dt, temperature_K=T_target, friction=friction, logfile=None)

    # Storage for H trajectory (unwrapped)
    n_steps = 50000
    record_every = 10
    n_records = n_steps // record_every
    h_positions = np.zeros((n_records, 3))
    h_pos_unwrapped = atoms.positions[h_idx].copy()
    h_pos_prev = atoms.positions[h_idx].copy()
    cell_diag = np.diag(atoms.cell)  # for unwrapping PBC

    record_idx = 0
    temps = []
    energies = []

    def record_trajectory():
        nonlocal record_idx, h_pos_unwrapped, h_pos_prev
        if record_idx >= n_records:
            return
        # Unwrap PBC: detect jumps > half cell
        h_pos_current = atoms.positions[h_idx].copy()
        delta = h_pos_current - h_pos_prev
        for dim in range(3):
            if delta[dim] > cell_diag[dim] / 2:
                delta[dim] -= cell_diag[dim]
            elif delta[dim] < -cell_diag[dim] / 2:
                delta[dim] += cell_diag[dim]
        h_pos_unwrapped += delta
        h_pos_prev = h_pos_current.copy()
        h_positions[record_idx] = h_pos_unwrapped.copy()
        temps.append(atoms.get_temperature())
        energies.append(atoms.get_potential_energy())
        record_idx += 1

    dyn.attach(record_trajectory, interval=record_every)
    print(f"  MD setup done in {time.time()-t0:.1f}s")
    results["friction_per_fs"] = 0.01
    results["record_every"] = record_every

    # ── [6/8] Run MD ─────────────────────────────────────────────────
    print("[6/8] Running MD (50000 steps = 25 ps)...")
    t0 = time.time()

    try:
        # Run in chunks and report progress
        chunk_size = 5000
        n_chunks = n_steps // chunk_size
        for chunk in range(n_chunks):
            dyn.run(chunk_size)
            elapsed = time.time() - t0
            step_now = (chunk + 1) * chunk_size
            rate = step_now / elapsed if elapsed > 0 else 0
            avg_T = np.mean(temps[-chunk_size//record_every:]) if temps else 0
            print(f"  Step {step_now}/{n_steps} | "
                  f"T_avg = {avg_T:.1f} K | "
                  f"{rate:.0f} steps/s | "
                  f"{elapsed:.0f}s elapsed")

        md_success = True
        results["md_completed"] = True
        results["actual_records"] = record_idx
    except Exception as exc:
        print(f"  MD FAILED at step ~{record_idx * record_every}: {exc}")
        results["md_error"] = str(exc)
        results["md_completed"] = False
        results["actual_records"] = record_idx
        md_success = record_idx > 100  # need at least some data

    print(f"  done in {time.time()-t0:.1f}s")

    # ── [7/8] MSD analysis ───────────────────────────────────────────
    print("[7/8] MSD analysis and diffusion coefficient")
    t0 = time.time()

    D_MD = None
    msd_data = None

    if md_success and record_idx > 10:
        h_pos_actual = h_positions[:record_idx]
        dt_record_fs = 0.5 * record_every  # fs between records

        times_ps, msd = compute_msd(h_pos_actual, dt_record_fs)
        D_MD, slope, r_squared = fit_diffusion_coefficient(times_ps, msd)

        print(f"  MSD slope = {slope:.4f} A^2/ps")
        print(f"  D_MD = {D_MD:.3e} cm^2/s")
        print(f"  R^2 = {r_squared:.4f}")

        results["msd_slope_A2_per_ps"] = float(slope)
        results["D_MD_cm2s"] = float(D_MD)
        results["msd_fit_R2"] = float(r_squared)

        # Arrhenius comparison
        for label, E_a in [("MACE_large", 1.29), ("MACE_medium", 0.96)]:
            D_arr = arrhenius_D(E_a, NEB_COMPARISON["hop_distance_A"])
            ratio = D_MD / D_arr if D_arr > 0 else float("inf")
            print(f"  D_NEB({label}, Ea={E_a}eV) = {D_arr:.3e} cm^2/s | "
                  f"D_MD/D_NEB = {ratio:.2e}")
            results[f"D_NEB_{label}_cm2s"] = float(D_arr)
            results[f"D_MD_over_D_NEB_{label}"] = float(ratio)

        # Effective activation energy from D_MD
        kB = 8.617e-5  # eV/K
        T = 300.0
        a_hop = NEB_COMPARISON["hop_distance_A"] * 1e-8  # cm
        if D_MD > 0:
            E_a_eff = -kB * T * np.log(D_MD / (1e13 * a_hop**2))
            print(f"  Effective E_a from MD = {E_a_eff:.3f} eV")
            results["E_a_effective_eV"] = float(E_a_eff)

        # Temperature statistics
        avg_temp = np.mean(temps) if temps else 0
        std_temp = np.std(temps) if temps else 0
        print(f"  <T> = {avg_temp:.1f} +/- {std_temp:.1f} K")
        results["avg_temperature_K"] = float(avg_temp)
        results["std_temperature_K"] = float(std_temp)

        msd_data = (times_ps, msd)
    else:
        print("  Insufficient data for MSD analysis")

    print(f"  done in {time.time()-t0:.1f}s")

    # ── [8/8] Output ──────────────────────────────────────────────────
    print("[8/8] Save results")
    t0 = time.time()

    results["neb_comparison"] = NEB_COMPARISON
    results["total_time_s"] = float(time.time() - t_total)

    # Save JSON
    json_path = RESULTS / "q071_md_pentlandite_300K.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {json_path}")

    # Save PNG
    if msd_data is not None:
        times_ps, msd = msd_data

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Plot 1: MSD(t)
        ax = axes[0]
        ax.plot(times_ps, msd, "b-", linewidth=1.5, label="MD")
        # Fit line
        n = len(times_ps)
        i_start = max(1, n // 5)
        i_end = 4 * n // 5
        if i_end > i_start + 2:
            coeffs = np.polyfit(times_ps[i_start:i_end], msd[i_start:i_end], 1)
            ax.plot(times_ps[i_start:i_end],
                    np.polyval(coeffs, times_ps[i_start:i_end]),
                    "r--", linewidth=2, label=f"Linear fit (slope={coeffs[0]:.3f})")
        ax.set_xlabel("Time (ps)", fontsize=12)
        ax.set_ylabel("MSD ($\\AA^2$)", fontsize=12)
        ax.set_title("H Mean Square Displacement", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 2: Temperature
        ax = axes[1]
        t_md = np.arange(len(temps)) * record_every * 0.5 / 1000  # ps
        ax.plot(t_md, temps, "gray", linewidth=0.3, alpha=0.5)
        # Running average
        if len(temps) > 50:
            window = 50
            t_avg = np.convolve(temps, np.ones(window)/window, mode="valid")
            ax.plot(t_md[window-1:], t_avg, "r-", linewidth=1.5, label=f"Running avg (w={window})")
        ax.axhline(300, color="blue", linestyle="--", alpha=0.5, label="Target 300 K")
        ax.set_xlabel("Time (ps)", fontsize=12)
        ax.set_ylabel("Temperature (K)", fontsize=12)
        ax.set_title("Temperature stability", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 3: Comparison table
        ax = axes[2]
        ax.axis("off")
        table_data = [
            ["Method", "D (cm$^2$/s)", "E$_a$ (eV)"],
            ["This MD (300K)", f"{D_MD:.2e}", f"{results.get('E_a_effective_eV', 'N/A'):.2f}"
             if 'E_a_effective_eV' in results else "N/A"],
            ["NEB MACE large", f"{results.get('D_NEB_MACE_large_cm2s', 0):.2e}", "1.29"],
            ["NEB MACE medium", f"{results.get('D_NEB_MACE_medium_cm2s', 0):.2e}", "0.96"],
            ["DFT lit (mack.)", "—", "1.12"],
        ]
        table = ax.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        # Bold header
        for j in range(3):
            table[0, j].set_text_props(fontweight="bold")
        ax.set_title("D$_H$ comparison: MD vs NEB", fontsize=12)

        fig.suptitle(
            f"Pentlandite 2x2x2 — H diffusion MD at 300K (MACE-MP-0 large)\n"
            f"25 ps NVT Langevin | D$_{{MD}}$ = {D_MD:.2e} cm$^2$/s",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()
        png_path = RESULTS / "q071_md_pentlandite_300K.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {png_path}")

    print(f"[VRAM] Peak: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")

    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print("=" * 60)
    print("MD Summary:")
    print(f"  System: pentlandite 2x2x2 + H ({n_atoms_total} atoms)")
    print(f"  MD: 50000 steps, 25 ps, NVT 300K")
    if D_MD is not None:
        print(f"  D_MD = {D_MD:.3e} cm^2/s")
        if "E_a_effective_eV" in results:
            print(f"  E_a(effective) = {results['E_a_effective_eV']:.3f} eV")
    print("  NEB comparison:")
    print(f"    MACE large (Ea=1.29 eV): D = {arrhenius_D(1.29, 2.8):.3e} cm^2/s")
    print(f"    MACE medium (Ea=0.96 eV): D = {arrhenius_D(0.96, 2.8):.3e} cm^2/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
