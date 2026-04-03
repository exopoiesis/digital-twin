#!/usr/bin/env python3
"""
HTST (Harmonic Transition State Theory) calculation of attempt frequency nu_0
for H vacancy-hopping diffusion in pentlandite (Fe,Ni)9S8 2x2x2 supercell.

Uses Vineyard formula:
    nu_0 = prod(nu_min, i=1..3N) / prod(nu_saddle_real, i=1..3N-1)

For frozen-lattice (only H moves, 3 DOF):
    nu_0 = prod(nu_min, 3 freqs) / prod(nu_saddle_real, 2 freqs)

NEB result: E_a = 1.43 eV (MACE-MP-0 large, 2x2x2 supercell, RESHENIE-044/046)
Standard approximation: nu_0 = 1e13 Hz
This script: compute real nu_0 from phonon frequencies at minimum and saddle.

Calculator: MACE-MP-0 large on GPU (float64)
Output: results/q071_htst_pentlandite.json, results/q071_htst_pentlandite.png
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
from ase.geometry import get_distances
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
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

# Physical constants
kB_eV = 8.617333262e-5   # eV/K
hbar_eV_s = 6.582119569e-16  # eV*s
AMU_KG = 1.660539066e-27  # kg
EV_J = 1.602176634e-19    # J
ANG_M = 1e-10             # m

# Known result from NEB
E_A_NEB = 1.43  # eV — from 2x2x2 NEB (RESHENIE-044)

# Finite difference step for Hessian
FD_DELTA = 0.01  # Angstrom


def build_pentlandite_supercell():
    """Build Ni-rich pentlandite Fe3Ni6S8 primitive cell, then make 2x2x2 supercell.
    Identical to NEB script for consistency."""
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
    # 2x2x2 supercell
    supercell = atoms.repeat((2, 2, 2))
    return supercell


def find_nearest_ss_pair(atoms):
    """Find nearest S-S pair using PBC distances. Identical to NEB script."""
    s_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "S"]
    s_positions = atoms.positions[s_indices]
    _, d_matrix = get_distances(s_positions, cell=atoms.cell, pbc=True)

    min_d = np.inf
    si, sj = -1, -1
    for a in range(len(s_indices)):
        for b in range(a + 1, len(s_indices)):
            if d_matrix[a, b] < min_d:
                min_d = d_matrix[a, b]
                si, sj = a, b
    return s_indices[si], s_indices[sj], min_d


def compute_hessian_H(atoms, calc, h_index, delta=FD_DELTA):
    """
    Compute 3x3 Hessian matrix for a single H atom by central finite differences.

    H_ij = (F_i(+dj) - F_i(-dj)) / (2 * delta)

    where F_i is force on H in direction i, dj is displacement in direction j.

    All other atoms are fixed via FixAtoms constraint.

    Returns:
        hessian: (3, 3) numpy array in eV/Ang^2
    """
    hessian = np.zeros((3, 3))
    pos0 = atoms.positions.copy()

    for j in range(3):  # displacement direction
        # +delta
        atoms.positions = pos0.copy()
        atoms.positions[h_index, j] += delta
        f_plus = atoms.get_forces()[h_index].copy()  # (3,)

        # -delta
        atoms.positions = pos0.copy()
        atoms.positions[h_index, j] -= delta
        f_minus = atoms.get_forces()[h_index].copy()  # (3,)

        # H_ij = -dF_i/dx_j  (negative of force derivative)
        hessian[:, j] = -(f_plus - f_minus) / (2.0 * delta)

    # Restore original positions
    atoms.positions = pos0.copy()

    # Symmetrize (numerical noise)
    hessian = 0.5 * (hessian + hessian.T)

    return hessian


def hessian_to_frequencies(hessian, mass_amu):
    """
    Convert 3x3 Hessian (eV/Ang^2) to frequencies (Hz) for an atom of given mass.

    eigenvalue lambda = hessian eigenvalue in eV/Ang^2
    omega^2 = lambda / mass  (in SI: eV/Ang^2 -> J/m^2, mass -> kg)
    nu = omega / (2*pi)

    Returns:
        eigenvalues: (3,) in eV/Ang^2 (can be negative for saddle)
        frequencies_Hz: (3,) in Hz (imaginary modes returned as negative values)
        frequencies_THz: (3,) in THz
    """
    eigenvalues = np.linalg.eigvalsh(hessian)  # sorted ascending

    # Convert to SI
    mass_kg = mass_amu * AMU_KG
    conv_factor = EV_J / (ANG_M ** 2)  # eV/Ang^2 -> J/m^2 = kg/s^2

    frequencies_Hz = np.zeros(3)
    for i, lam in enumerate(eigenvalues):
        omega2 = lam * conv_factor / mass_kg  # s^-2
        if omega2 >= 0:
            frequencies_Hz[i] = np.sqrt(omega2) / (2.0 * np.pi)
        else:
            # Imaginary mode: store as negative frequency
            frequencies_Hz[i] = -np.sqrt(abs(omega2)) / (2.0 * np.pi)

    frequencies_THz = frequencies_Hz / 1e12

    return eigenvalues, frequencies_Hz, frequencies_THz


def main():
    t_total = time.time()
    results = {
        "mineral": "pentlandite",
        "model": "MACE-MP-0 large",
        "device": "cuda",
        "cell_type": "2x2x2 supercell",
        "method": "HTST Vineyard formula",
        "purpose": "compute attempt frequency nu_0 for H vacancy-hopping",
        "E_a_NEB_eV": E_A_NEB,
        "fd_delta_Ang": FD_DELTA,
    }

    # ── [1/9] GPU info ──────────────────────────────────────────────
    print("=" * 65)
    print("HTST attempt frequency for H diffusion in pentlandite 2x2x2")
    print("=" * 65)
    print()
    print("[1/9] GPU info")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [2/9] Build structure ───────────────────────────────────────
    print("\n[2/9] Build pentlandite 2x2x2 supercell")
    t0 = time.time()
    atoms_pristine = build_pentlandite_supercell()
    n_atoms = len(atoms_pristine)
    formula = atoms_pristine.get_chemical_formula()
    print(f"  {formula}, {n_atoms} atoms")
    results["formula"] = formula
    results["n_atoms_pristine"] = n_atoms
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [3/9] Load calculator ───────────────────────────────────────
    print("\n[3/9] Load MACE-MP-0 large on GPU")
    t0 = time.time()
    calc = mace_mp(model="large", device=device, default_dtype="float64")
    print(f"  loaded on {device} in {time.time()-t0:.1f}s")

    # ── [4/9] Relax pristine supercell ──────────────────────────────
    print("\n[4/9] Relax pristine supercell (LBFGS, fmax=0.01, 300 steps)")
    t0 = time.time()
    atoms_pristine.calc = calc
    opt = LBFGS(atoms_pristine, logfile=None)
    opt.run(fmax=0.01, steps=300)
    e_pristine = atoms_pristine.get_potential_energy()
    print(f"  E_pristine = {e_pristine:.4f} eV, converged in {opt.nsteps} steps")
    results["E_pristine_eV"] = float(e_pristine)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [5/9] Find S-S pair and build defective cell ────────────────
    print("\n[5/9] Find nearest S-S pair and create vacancies + H")
    t0 = time.time()
    si_idx, sj_idx, hop_dist = find_nearest_ss_pair(atoms_pristine)
    pos_si = atoms_pristine.positions[si_idx].copy()
    pos_sj = atoms_pristine.positions[sj_idx].copy()
    print(f"  S pair: atoms {si_idx} & {sj_idx}, d = {hop_dist:.3f} A")
    results["S_pair_indices"] = [int(si_idx), int(sj_idx)]
    results["hop_distance_A"] = float(hop_dist)

    # Remove both S atoms, place H at site si (minimum = endpoint A of NEB)
    del_indices = sorted([si_idx, sj_idx], reverse=True)

    # Build minimum config (H at vacancy site A)
    atoms_min = atoms_pristine.copy()
    for idx in del_indices:
        del atoms_min[idx]
    atoms_min.append(Atom("H", position=pos_si))
    h_index_min = len(atoms_min) - 1  # H is last atom
    n_defect = len(atoms_min)
    print(f"  Defective cell: {n_defect} atoms ({n_atoms} - 2 S + 1 H)")
    results["n_atoms_defective"] = n_defect
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [6/9] Relax minimum (H at vacancy A) ───────────────────────
    print("\n[6/9] Relax H at minimum (vacancy site A), all heavy atoms fixed")
    t0 = time.time()
    atoms_min.calc = calc  # IMPORTANT: assign calc after .copy()
    heavy_indices = [i for i in range(len(atoms_min)) if atoms_min[i].symbol != "H"]
    atoms_min.set_constraint(FixAtoms(indices=heavy_indices))
    opt_min = LBFGS(atoms_min, logfile=None)
    opt_min.run(fmax=0.01, steps=200)
    e_min = atoms_min.get_potential_energy()
    h_pos_min = atoms_min.positions[h_index_min].copy()
    fmax_min = np.max(np.abs(atoms_min.get_forces()[h_index_min]))
    print(f"  E_min = {e_min:.4f} eV ({opt_min.nsteps} steps)")
    print(f"  H position (min): {h_pos_min}")
    print(f"  |F_max| on H = {fmax_min:.5f} eV/A")
    results["E_min_eV"] = float(e_min)
    results["H_pos_min"] = h_pos_min.tolist()
    results["fmax_min_eVA"] = float(fmax_min)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [7/9] Hessian at minimum ───────────────────────────────────
    print("\n[7/9] Compute Hessian at minimum (6 force evaluations)")
    t0 = time.time()
    hessian_min = compute_hessian_H(atoms_min, calc, h_index_min, delta=FD_DELTA)
    mass_H = 1.00794  # amu

    eig_min, freq_min_Hz, freq_min_THz = hessian_to_frequencies(hessian_min, mass_H)

    print(f"  Hessian eigenvalues (eV/A^2): {eig_min}")
    print(f"  Frequencies (THz): {freq_min_THz}")
    print(f"  Frequencies (Hz):  {freq_min_Hz}")

    # Check all frequencies are real (positive eigenvalues) at minimum
    n_imaginary_min = np.sum(eig_min < -1e-6)
    if n_imaginary_min > 0:
        print(f"  WARNING: {n_imaginary_min} imaginary mode(s) at minimum!")
    else:
        print(f"  OK: all 3 frequencies real at minimum")

    results["hessian_min_eVA2"] = hessian_min.tolist()
    results["eigenvalues_min_eVA2"] = eig_min.tolist()
    results["frequencies_min_THz"] = freq_min_THz.tolist()
    results["frequencies_min_Hz"] = freq_min_Hz.tolist()
    results["n_imaginary_min"] = int(n_imaginary_min)
    t_hess_min = time.time() - t0
    print(f"  done in {t_hess_min:.1f}s")

    # ── [8/9] Build saddle point and compute Hessian ────────────────
    print("\n[8/9] Build saddle point (midpoint interpolation) and compute Hessian")
    t0 = time.time()

    # Saddle point: H at midpoint between two vacancy sites
    # This is the frozen saddle approximation (no relaxation along saddle)
    # Gives upper bound on nu_0
    pos_saddle = 0.5 * (pos_si + pos_sj)
    print(f"  Saddle H position (midpoint): {pos_saddle}")

    # Build saddle config — same defective cell, H at midpoint
    atoms_saddle = atoms_pristine.copy()
    for idx in del_indices:
        del atoms_saddle[idx]
    atoms_saddle.append(Atom("H", position=pos_saddle))
    h_index_saddle = len(atoms_saddle) - 1

    atoms_saddle.calc = calc  # IMPORTANT: assign calc after .copy()
    heavy_saddle = [i for i in range(len(atoms_saddle)) if atoms_saddle[i].symbol != "H"]
    atoms_saddle.set_constraint(FixAtoms(indices=heavy_saddle))

    # Do NOT fully relax — but relax PERPENDICULAR to reaction coordinate
    # The reaction coordinate is the si->sj direction
    # For frozen saddle approximation: skip relaxation entirely
    # For constrained relaxation: project out the reaction coordinate component

    # Option: constrained relaxation in the plane perpendicular to hop direction
    # We implement this by doing a few steps of LBFGS then projecting forces
    rc_dir = pos_sj - pos_si  # reaction coordinate direction
    rc_dir = rc_dir / np.linalg.norm(rc_dir)
    print(f"  Reaction coordinate direction: {rc_dir}")

    # Constrained relaxation: manually relax perpendicular DOFs
    # Do 50 steps of manual gradient descent with force projection
    print("  Constrained relaxation (perpendicular to reaction coordinate)...")
    lr = 0.02  # step size in Angstrom
    for step in range(50):
        forces_H = atoms_saddle.get_forces()[h_index_saddle].copy()
        # Project out reaction coordinate component
        f_par = np.dot(forces_H, rc_dir) * rc_dir
        f_perp = forces_H - f_par
        fmax_perp = np.max(np.abs(f_perp))
        if fmax_perp < 0.01:
            print(f"  Perpendicular relaxation converged at step {step}, |F_perp|={fmax_perp:.5f}")
            break
        # Move H in perpendicular direction
        atoms_saddle.positions[h_index_saddle] += lr * f_perp / np.max(np.abs(f_perp))
    else:
        print(f"  Perpendicular relaxation: 50 steps, |F_perp|={fmax_perp:.5f}")

    e_saddle = atoms_saddle.get_potential_energy()
    h_pos_saddle = atoms_saddle.positions[h_index_saddle].copy()
    forces_saddle_H = atoms_saddle.get_forces()[h_index_saddle].copy()
    print(f"  E_saddle = {e_saddle:.4f} eV")
    print(f"  H position (saddle): {h_pos_saddle}")
    print(f"  Forces on H at saddle: {forces_saddle_H}")
    print(f"  E_a (this calc) = {e_saddle - e_min:.4f} eV  (NEB: {E_A_NEB} eV)")

    results["H_pos_saddle"] = h_pos_saddle.tolist()
    results["E_saddle_eV"] = float(e_saddle)
    results["E_a_htst_eV"] = float(e_saddle - e_min)
    results["forces_saddle_H"] = forces_saddle_H.tolist()

    # Compute Hessian at saddle
    print("  Computing Hessian at saddle (6 force evaluations)...")
    hessian_saddle = compute_hessian_H(atoms_saddle, calc, h_index_saddle, delta=FD_DELTA)
    eig_saddle, freq_saddle_Hz, freq_saddle_THz = hessian_to_frequencies(hessian_saddle, mass_H)

    print(f"  Hessian eigenvalues (eV/A^2): {eig_saddle}")
    print(f"  Frequencies (THz): {freq_saddle_THz}")
    print(f"  Frequencies (Hz):  {freq_saddle_Hz}")

    n_imaginary_saddle = np.sum(eig_saddle < -1e-6)
    n_real_saddle = 3 - n_imaginary_saddle
    print(f"  Imaginary modes at saddle: {n_imaginary_saddle} (expected: 1)")
    if n_imaginary_saddle == 1:
        print(f"  OK: exactly 1 imaginary mode at saddle (unstable mode along RC)")
    elif n_imaginary_saddle == 0:
        print(f"  WARNING: no imaginary modes — saddle may not be at true transition state")
    else:
        print(f"  WARNING: {n_imaginary_saddle} imaginary modes — check structure")

    results["hessian_saddle_eVA2"] = hessian_saddle.tolist()
    results["eigenvalues_saddle_eVA2"] = eig_saddle.tolist()
    results["frequencies_saddle_THz"] = freq_saddle_THz.tolist()
    results["frequencies_saddle_Hz"] = freq_saddle_Hz.tolist()
    results["n_imaginary_saddle"] = int(n_imaginary_saddle)
    results["n_real_saddle"] = int(n_real_saddle)
    t_saddle = time.time() - t0
    print(f"  done in {t_saddle:.1f}s")

    # ── [9/9] Vineyard formula and D_H ──────────────────────────────
    print("\n[9/9] Vineyard formula: nu_0 and D_H")
    print("=" * 65)

    # Vineyard: nu_0 = prod(real freqs at min) / prod(real freqs at saddle)
    # Minimum: all 3 should be real (positive)
    # Saddle: 2 real + 1 imaginary

    # Use absolute values of frequencies, then select real ones
    real_freqs_min = freq_min_Hz[eig_min > -1e-6]
    real_freqs_saddle = freq_saddle_Hz[eig_saddle > -1e-6]

    print(f"\n  Real frequencies at minimum ({len(real_freqs_min)}):")
    for i, f in enumerate(real_freqs_min):
        print(f"    nu_min_{i+1} = {f:.4e} Hz = {f/1e12:.2f} THz")

    print(f"\n  Real frequencies at saddle ({len(real_freqs_saddle)}):")
    for i, f in enumerate(real_freqs_saddle):
        print(f"    nu_saddle_{i+1} = {f:.4e} Hz = {f/1e12:.2f} THz")

    imaginary_freqs = freq_saddle_Hz[eig_saddle < -1e-6]
    if len(imaginary_freqs) > 0:
        print(f"\n  Imaginary frequencies at saddle ({len(imaginary_freqs)}):")
        for i, f in enumerate(imaginary_freqs):
            print(f"    nu_imag_{i+1} = {f:.4e} Hz = {f/1e12:.2f} THz (imaginary)")

    # Compute nu_0
    if len(real_freqs_min) >= 3 and len(real_freqs_saddle) >= 2:
        prod_min = np.prod(real_freqs_min[:3])      # 3 real at minimum
        prod_saddle = np.prod(real_freqs_saddle[:2]) if len(real_freqs_saddle) >= 2 else 1.0

        # If we have 3 real at saddle (no imaginary found), use 2 lowest
        # If 2 real (1 imaginary), use both
        if n_imaginary_saddle == 1:
            # Standard case: 2 real freqs at saddle
            prod_saddle = np.prod(real_freqs_saddle)
        elif n_imaginary_saddle == 0:
            # No imaginary mode found — use 2 highest freqs as "stable" modes
            # and treat the lowest as the "would-be imaginary" mode
            sorted_saddle = np.sort(real_freqs_saddle)
            prod_saddle = np.prod(sorted_saddle[1:])  # exclude lowest
            print(f"\n  NOTE: no imaginary mode found. Excluding lowest freq "
                  f"({sorted_saddle[0]:.4e} Hz) as pseudo-RC mode")
        else:
            # Multiple imaginary — use whatever real modes exist
            prod_saddle = np.prod(real_freqs_saddle) if len(real_freqs_saddle) > 0 else 1.0

        nu_0_vineyard = prod_min / prod_saddle
        print(f"\n  prod(nu_min) = {prod_min:.4e} Hz^3")
        print(f"  prod(nu_saddle_real) = {prod_saddle:.4e} Hz^2")
        print(f"\n  >>> nu_0 (Vineyard) = {nu_0_vineyard:.4e} Hz <<<")
        print(f"  >>> nu_0 / 1e13 = {nu_0_vineyard / 1e13:.3f} <<<")
        print(f"  (standard approximation: 1.0e+13 Hz)")

        results["prod_nu_min_Hz3"] = float(prod_min)
        results["prod_nu_saddle_Hz2"] = float(prod_saddle)
        results["nu_0_vineyard_Hz"] = float(nu_0_vineyard)
        results["nu_0_ratio_to_1e13"] = float(nu_0_vineyard / 1e13)

        # Compute D_H
        T = 298.15  # K
        a_hop_cm = hop_dist * 1e-8  # A -> cm
        a_hop_m = hop_dist * 1e-10  # A -> m

        # D = a^2 * nu_0 * exp(-E_a / kT)
        # With Vineyard nu_0
        D_H_vineyard = a_hop_cm**2 * nu_0_vineyard * np.exp(-E_A_NEB / (kB_eV * T))

        # With standard nu_0 = 1e13
        D_H_standard = a_hop_cm**2 * 1e13 * np.exp(-E_A_NEB / (kB_eV * T))

        # Traversal time through 200 nm membrane
        L_cm = 200e-7  # 200 nm in cm
        tau_vineyard = L_cm**2 / (2 * D_H_vineyard) if D_H_vineyard > 0 else float("inf")
        tau_standard = L_cm**2 / (2 * D_H_standard) if D_H_standard > 0 else float("inf")

        print(f"\n  --- Diffusion at T = {T} K ---")
        print(f"  Hop distance: {hop_dist:.3f} A")
        print(f"  E_a (NEB): {E_A_NEB} eV")
        print(f"\n  With Vineyard nu_0 = {nu_0_vineyard:.3e} Hz:")
        print(f"    D_H = {D_H_vineyard:.3e} cm^2/s")
        print(f"    tau(200 nm) = {tau_vineyard:.3e} s")
        print(f"\n  With standard nu_0 = 1.000e+13 Hz:")
        print(f"    D_H = {D_H_standard:.3e} cm^2/s")
        print(f"    tau(200 nm) = {tau_standard:.3e} s")
        print(f"\n  Ratio D_vineyard / D_standard = {D_H_vineyard / D_H_standard:.3f}")

        results["T_K"] = T
        results["D_H_vineyard_cm2s"] = float(D_H_vineyard)
        results["D_H_standard_cm2s"] = float(D_H_standard)
        results["tau_200nm_vineyard_s"] = float(tau_vineyard)
        results["tau_200nm_standard_s"] = float(tau_standard)
        results["D_ratio_vineyard_standard"] = float(D_H_vineyard / D_H_standard)

        vineyard_valid = True
    else:
        print(f"\n  ERROR: insufficient frequencies for Vineyard formula")
        print(f"  real_freqs_min: {len(real_freqs_min)}, real_freqs_saddle: {len(real_freqs_saddle)}")
        nu_0_vineyard = None
        vineyard_valid = False
        results["vineyard_error"] = "insufficient real frequencies"

    # ── Plot ────────────────────────────────────────────────────────
    print("\n  Generating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart of frequencies
    ax1 = axes[0]
    x_min = np.arange(3)
    x_saddle = np.arange(3)

    colors_min = ["#2196F3"] * 3  # blue for real
    colors_saddle = []
    for ev in eig_saddle:
        if ev < -1e-6:
            colors_saddle.append("#F44336")  # red for imaginary
        else:
            colors_saddle.append("#4CAF50")  # green for real

    bar_width = 0.35
    bars_min = ax1.bar(x_min - bar_width/2, np.abs(freq_min_THz),
                       bar_width, color=colors_min, label="Minimum", alpha=0.8,
                       edgecolor="black", linewidth=0.5)
    bars_saddle = ax1.bar(x_saddle + bar_width/2, np.abs(freq_saddle_THz),
                          bar_width, color=colors_saddle, label="Saddle", alpha=0.8,
                          edgecolor="black", linewidth=0.5)

    ax1.set_xlabel("Mode index", fontsize=12)
    ax1.set_ylabel("|Frequency| (THz)", fontsize=12)
    ax1.set_title("H phonon frequencies in pentlandite\n(vacancy-hopping, 2x2x2)", fontsize=11)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["1", "2", "3"])
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # Mark imaginary modes
    for i, ev in enumerate(eig_saddle):
        if ev < -1e-6:
            ax1.annotate("imaginary",
                         xy=(i + bar_width/2, abs(freq_saddle_THz[i])),
                         xytext=(i + bar_width/2 + 0.3, abs(freq_saddle_THz[i]) * 1.15),
                         fontsize=8, color="red",
                         arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    # Right: summary text
    ax2 = axes[1]
    ax2.axis("off")

    text_lines = [
        "HTST Analysis: H diffusion in pentlandite",
        "=" * 48,
        f"Cell: 2x2x2 supercell ({n_defect} atoms)",
        f"Calculator: MACE-MP-0 large (float64, GPU)",
        f"FD delta: {FD_DELTA} A",
        "",
        "Frequencies at MINIMUM (THz):",
    ]
    for i, f in enumerate(freq_min_THz):
        text_lines.append(f"  nu_{i+1} = {abs(f):.2f} THz {'(imag!)' if f < 0 else ''}")
    text_lines.append("")
    text_lines.append("Frequencies at SADDLE (THz):")
    for i, f in enumerate(freq_saddle_THz):
        tag = "(IMAGINARY)" if eig_saddle[i] < -1e-6 else ""
        text_lines.append(f"  nu_{i+1} = {abs(f):.2f} THz {tag}")

    text_lines.append("")
    text_lines.append("=" * 48)

    if vineyard_valid:
        text_lines.extend([
            f"Vineyard nu_0 = {nu_0_vineyard:.3e} Hz",
            f"Standard nu_0 = 1.000e+13 Hz",
            f"Ratio: {nu_0_vineyard / 1e13:.3f}",
            "",
            f"E_a (NEB) = {E_A_NEB} eV",
            f"Hop distance = {hop_dist:.3f} A",
            f"T = {T} K",
            "",
            f"D_H (Vineyard) = {D_H_vineyard:.3e} cm2/s",
            f"D_H (standard) = {D_H_standard:.3e} cm2/s",
            f"tau(200nm, Vineyard) = {tau_vineyard:.3e} s",
            f"tau(200nm, standard) = {tau_standard:.3e} s",
        ])
    else:
        text_lines.append("Vineyard formula: FAILED (see log)")

    ax2.text(0.05, 0.95, "\n".join(text_lines),
             transform=ax2.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    png_path = RESULTS / "q071_htst_pentlandite.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {png_path}")

    # Save JSON
    results["total_time_s"] = float(time.time() - t_total)
    json_path = RESULTS / "q071_htst_pentlandite.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {json_path}")

    print(f"[VRAM] Peak: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")

    # Final summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    if vineyard_valid:
        print(f"  nu_0 (Vineyard) = {nu_0_vineyard:.3e} Hz")
        print(f"  nu_0 (standard) = 1.000e+13 Hz")
        print(f"  Ratio = {nu_0_vineyard / 1e13:.3f}")
        print(f"  D_H (Vineyard, {T}K) = {D_H_vineyard:.3e} cm2/s")
        print(f"  Impact on diffusion: x{D_H_vineyard / D_H_standard:.2f}")
        if abs(np.log10(nu_0_vineyard) - 13) < 1:
            print(f"\n  CONCLUSION: nu_0 ~ 10^{np.log10(nu_0_vineyard):.1f} Hz")
            print(f"  Standard 10^13 Hz approximation is reasonable (within 1 OoM)")
        else:
            print(f"\n  CONCLUSION: nu_0 = 10^{np.log10(nu_0_vineyard):.1f} Hz")
            print(f"  Standard 10^13 Hz approximation is OFF by "
                  f"{abs(np.log10(nu_0_vineyard) - 13):.1f} orders of magnitude!")
    else:
        print("  Vineyard formula could not be evaluated")
    print("=" * 65)


if __name__ == "__main__":
    main()
