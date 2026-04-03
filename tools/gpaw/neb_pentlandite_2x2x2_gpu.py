#!/usr/bin/env python3
"""
NEB calculation of H vacancy-hopping in pentlandite (Fe,Ni)9S8
using MACE-MP-0 large model on GPU — 2x2x2 supercell convergence test.

Crystal: Fm3m (225), a=10.07 A, Ni-rich (first 5 Fe -> Ni)
Supercell: 2x2x2 (136 atoms) to check size convergence vs primitive cell
Output: q071_neb_pentlandite_2x2x2_large.json, q071_neb_pentlandite_2x2x2_large.png
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
from ase.mep import NEB
from ase.optimize import FIRE, LBFGS
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

COMPARISON = {
    "pentlandite_primitive_MACE_large": "1.29 eV",
    "pentlandite_primitive_MACE_medium": "0.96 eV",
    "mackinawite_DFT_lit": "1.12 eV",
    "Fe3Ni_DFT_lit": "0.70 eV",
}


def build_pentlandite_supercell():
    """Build Ni-rich pentlandite Fe3Ni6S8 primitive cell, then make 2x2x2 supercell."""
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
    """Find nearest S-S pair using PBC distances."""
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


def main():
    t_total = time.time()
    results = {
        "mineral": "pentlandite",
        "model": "MACE-MP-0 large",
        "device": "cuda",
        "cell_type": "2x2x2 supercell",
        "purpose": "size convergence test",
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

    # ── [2/8] Build 2x2x2 supercell ────────────────────────────────
    print("[2/8] Build pentlandite 2x2x2 supercell")
    t0 = time.time()
    atoms = build_pentlandite_supercell()
    n_atoms = len(atoms)
    formula = atoms.get_chemical_formula()
    print(f"  {formula}, {n_atoms} atoms")
    results["formula"] = formula
    results["n_atoms"] = n_atoms
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [3/8] Load MACE-MP-0 large on GPU ────────────────────────────
    print("[3/8] Load MACE-MP-0 large on GPU")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="large", device=device, default_dtype="float64")
    print(f"  loaded on {device} in {time.time()-t0:.1f}s")

    # ── [4/8] Relax supercell ────────────────────────────────────────
    print("[4/8] Relax supercell (LBFGS, fmax=0.01, 300 steps)")
    t0 = time.time()
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=0.01, steps=300)
    e_pristine = atoms.get_potential_energy()
    print(f"  E_pristine = {e_pristine:.4f} eV, converged in {opt.nsteps} steps")
    results["E_pristine_eV"] = float(e_pristine)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [5/8] Find nearest S-S pair ────────────────────────────────
    print("[5/8] Find nearest S-S pair (PBC)")
    t0 = time.time()
    si_idx, sj_idx, min_d = find_nearest_ss_pair(atoms)
    print(f"  S pair: atoms {si_idx} & {sj_idx}, d = {min_d:.3f} A")
    results["S_pair_indices"] = [int(si_idx), int(sj_idx)]
    results["S_pair_distance_A"] = float(min_d)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [6/8] Build and relax endpoints ──────────────────────────────
    print("[6/8] Build and relax endpoint configs (H on S vacancies)")
    t0 = time.time()

    pos_si = atoms.positions[si_idx].copy()
    pos_sj = atoms.positions[sj_idx].copy()

    # Endpoint A: remove both S atoms, place H at site si
    del_indices = sorted([si_idx, sj_idx], reverse=True)

    endA = atoms.copy()
    for idx in del_indices:
        del endA[idx]
    endA.append(Atom("H", position=pos_si))
    endA.calc = calc
    heavy_A = [i for i in range(len(endA)) if endA[i].symbol != "H"]
    endA.set_constraint(FixAtoms(indices=heavy_A))
    optA = LBFGS(endA, logfile=None)
    optA.run(fmax=0.02, steps=100)
    e_A = endA.get_potential_energy()
    print(f"  EndpointA: E = {e_A:.4f} eV ({optA.nsteps} steps)")

    # Endpoint B: remove both S atoms, place H at site sj
    endB = atoms.copy()
    for idx in del_indices:
        del endB[idx]
    endB.append(Atom("H", position=pos_sj))
    endB.calc = calc
    heavy_B = [i for i in range(len(endB)) if endB[i].symbol != "H"]
    endB.set_constraint(FixAtoms(indices=heavy_B))
    optB = LBFGS(endB, logfile=None)
    optB.run(fmax=0.02, steps=100)
    e_B = endB.get_potential_energy()
    print(f"  EndpointB: E = {e_B:.4f} eV ({optB.nsteps} steps)")

    results["E_endpointA_eV"] = float(e_A)
    results["E_endpointB_eV"] = float(e_B)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [7/8] CI-NEB ────────────────────────────────────────────────
    print("[7/8] CI-NEB (11 images, FIRE, fmax=0.03, 500 steps)")
    t0 = time.time()

    neb_energies = None
    neb_converged = False
    E_a = None

    try:
        start = endA.copy()
        start.calc = calc
        images = [start]
        for _ in range(11):
            img = endA.copy()
            img.calc = calc
            heavy_img = [i for i in range(len(img)) if img[i].symbol != "H"]
            img.set_constraint(FixAtoms(indices=heavy_img))
            images.append(img)
        end = endB.copy()
        end.calc = calc
        images.append(end)

        neb = NEB(images, climb=True, allow_shared_calculator=True)
        neb.interpolate()

        opt_neb = FIRE(neb, logfile=None)
        neb_converged = opt_neb.run(fmax=0.03, steps=500)
        neb_energies = [img.get_potential_energy() for img in images]
        e_ref = neb_energies[0]
        neb_energies_rel = [e - e_ref for e in neb_energies]
        E_a = max(neb_energies_rel)
        print(f"  NEB converged: {neb_converged}")
        print(f"  E_a = {E_a:.4f} eV")
        results["neb_converged"] = bool(neb_converged)
        results["E_a_eV"] = float(E_a)
        results["neb_energies_eV"] = [float(e) for e in neb_energies_rel]
    except Exception as exc:
        print(f"  NEB FAILED: {exc}")
        results["neb_error"] = str(exc)

    print(f"  done in {time.time()-t0:.1f}s")

    # ── [8/8] Analysis ───────────────────────────────────────────────
    print("[8/8] Analysis and output")
    t0 = time.time()

    if E_a is not None and E_a > 0:
        kB = 8.617e-5  # eV/K
        T = 298.15  # K
        a_hop = min_d * 1e-8  # cm
        D_H = 1e13 * a_hop**2 * np.exp(-E_a / (kB * T))
        L = 200e-7  # 200 nm in cm
        tau = L**2 / (2 * D_H) if D_H > 0 else float("inf")
        print(f"  D_H = {D_H:.3e} cm^2/s")
        print(f"  tau(200nm) = {tau:.3e} s")
        results["D_H_cm2s"] = float(D_H)
        results["tau_200nm_s"] = float(tau)
        results["hop_distance_A"] = float(min_d)

        # Size convergence comparison
        E_a_primitive = 1.29  # eV, from primitive cell MACE large
        delta = E_a - E_a_primitive
        print(f"  Size convergence: E_a(2x2x2) - E_a(primitive) = {delta:+.4f} eV")
        results["E_a_primitive_eV"] = E_a_primitive
        results["size_convergence_delta_eV"] = float(delta)
    else:
        print("  Skipping diffusion analysis (no valid E_a)")

    results["comparison_table"] = COMPARISON
    results["total_time_s"] = float(time.time() - t_total)

    # Save JSON
    json_path = RESULTS / "q071_neb_pentlandite_2x2x2_large.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {json_path}")

    # Save PNG
    if neb_energies is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(0, 1, len(neb_energies_rel))
        ax.plot(x, neb_energies_rel, "bo-", linewidth=2, markersize=8)
        ax.set_xlabel("Reaction coordinate", fontsize=13)
        ax.set_ylabel("Energy (eV)", fontsize=13)
        ax.set_title(
            f"Pentlandite 2x2x2 H vacancy-hopping NEB\n"
            f"MACE-MP-0 large | {n_atoms} atoms | E_a = {E_a:.3f} eV | "
            f"D_H = {D_H:.2e} cm$^2$/s",
            fontsize=11,
        )
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Comparison annotations
        text_lines = [
            "Size convergence:",
            f"  Primitive (MACE large): {COMPARISON['pentlandite_primitive_MACE_large']}",
            f"  Primitive (MACE medium): {COMPARISON['pentlandite_primitive_MACE_medium']}",
            f"  This 2x2x2 (MACE large): {E_a:.2f} eV",
            f"  Delta: {delta:+.3f} eV",
            "",
            f"  Mackinawite DFT lit: {COMPARISON['mackinawite_DFT_lit']}",
        ]
        ax.text(
            0.02, 0.98, "\n".join(text_lines),
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        fig.tight_layout()
        png_path = RESULTS / "q071_neb_pentlandite_2x2x2_large.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {png_path}")

    print(f"[VRAM] Peak: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")

    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print("=" * 60)
    print("Size convergence comparison:")
    for k, v in COMPARISON.items():
        print(f"  {k}: {v}")
    if E_a is not None:
        print(f"  pentlandite_2x2x2_MACE_large: {E_a:.2f} eV  <-- THIS RUN")
    print("=" * 60)


if __name__ == "__main__":
    main()
