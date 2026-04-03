#!/usr/bin/env python3
"""
NEB calculation of H vacancy-hopping in greigite Fe3S4
using MACE-MP-0 large model on GPU.

Crystal: Fd3m (227), a=9.876 A
Output: q071_neb_greigite_large.json, q071_neb_greigite_large.png
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
_check_vram(1.5)

COMPARISON = {
    "pentlandite_MACE_medium": "0.96 eV",
    "mackinawite_DFT_lit": "1.12 eV",
    "Fe3Ni_DFT_lit": "0.70 eV",
}

def main():
    t_total = time.time()
    results = {"mineral": "greigite", "model": "MACE-MP-0 large", "device": "cuda"}

    # ── [1/8] GPU info ──────────────────────────────────────────────
    print("[1/8] GPU info")
    t0 = time.time()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
        results["gpu"] = gpu_name
    else:
        print("  WARNING: CUDA not available, falling back to CPU")
        results["gpu"] = "N/A (CPU fallback)"
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [2/8] Build greigite structure ───────────────────────────────
    print("[2/8] Build greigite structure")
    t0 = time.time()
    atoms = crystal(
        symbols=["Fe", "Fe", "S"],
        basis=[(0, 0, 0), (5/8, 5/8, 5/8), (0.254, 0.254, 0.254)],
        spacegroup=227,
        cellpar=[9.876, 9.876, 9.876, 90, 90, 90],
        primitive_cell=True,
    )
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

    # ── [4/8] Relax pristine cell ────────────────────────────────────
    print("[4/8] Relax pristine cell (LBFGS, fmax=0.01, 300 steps)")
    t0 = time.time()
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=0.01, steps=300)
    e_pristine = atoms.get_potential_energy()
    print(f"  E_pristine = {e_pristine:.4f} eV, converged in {opt.nsteps} steps")
    results["E_pristine_eV"] = float(e_pristine)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [5/8] Find nearest S-S pair ──────────────────────────────────
    print("[5/8] Find nearest S-S pair (PBC)")
    t0 = time.time()
    s_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "S"]
    s_positions = atoms.positions[s_indices]
    _, d_matrix = get_distances(s_positions, cell=atoms.cell, pbc=True)

    # Find nearest pair
    min_d = np.inf
    si, sj = -1, -1
    for a in range(len(s_indices)):
        for b in range(a + 1, len(s_indices)):
            if d_matrix[a, b] < min_d:
                min_d = d_matrix[a, b]
                si, sj = a, b
    si_idx, sj_idx = s_indices[si], s_indices[sj]
    print(f"  S pair: atoms {si_idx} & {sj_idx}, d = {min_d:.3f} A")
    results["S_pair_indices"] = [int(si_idx), int(sj_idx)]
    results["S_pair_distance_A"] = float(min_d)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [6/8] Build config1 and config2 ──────────────────────────────
    print("[6/8] Build and relax endpoint configs (H on S vacancies)")
    t0 = time.time()

    pos_si = atoms.positions[si_idx].copy()
    pos_sj = atoms.positions[sj_idx].copy()

    # Config1: H at site of S[si_idx]
    config1 = atoms.copy()
    del config1[[si_idx]]
    config1.append(Atom("H", position=pos_si))
    config1.calc = calc
    heavy_idx1 = [i for i in range(len(config1)) if config1[i].symbol != "H"]
    config1.set_constraint(FixAtoms(indices=heavy_idx1))
    opt1 = LBFGS(config1, logfile=None)
    opt1.run(fmax=0.02, steps=100)
    e1 = config1.get_potential_energy()
    print(f"  Config1: E = {e1:.4f} eV ({opt1.nsteps} steps)")

    # Config2: H at site of S[sj_idx]
    config2 = atoms.copy()
    del config2[[sj_idx]]
    config2.append(Atom("H", position=pos_sj))
    config2.calc = calc
    heavy_idx2 = [i for i in range(len(config2)) if config2[i].symbol != "H"]
    config2.set_constraint(FixAtoms(indices=heavy_idx2))
    opt2 = LBFGS(config2, logfile=None)
    opt2.run(fmax=0.02, steps=100)
    e2 = config2.get_potential_energy()
    print(f"  Config2: E = {e2:.4f} eV ({opt2.nsteps} steps)")
    results["E_config1_eV"] = float(e1)
    results["E_config2_eV"] = float(e2)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [7/8] NEB ────────────────────────────────────────────────────
    print("[7/8] CI-NEB (11 images, FIRE, fmax=0.03, 500 steps)")
    t0 = time.time()

    neb_energies = None
    neb_converged = False
    E_a = None
    D_H = None

    try:
        # Build NEB endpoints: remove BOTH S vacancies, re-relax H
        endA = atoms.copy()
        del_indices = sorted([si_idx, sj_idx], reverse=True)
        for idx in del_indices:
            del endA[idx]
        endA.append(Atom("H", position=pos_si))
        endA.calc = calc
        heavy_A = [i for i in range(len(endA)) if endA[i].symbol != "H"]
        endA.set_constraint(FixAtoms(indices=heavy_A))
        optA = LBFGS(endA, logfile=None)
        optA.run(fmax=0.02, steps=100)

        endB = atoms.copy()
        for idx in del_indices:
            del endB[idx]
        endB.append(Atom("H", position=pos_sj))
        endB.calc = calc
        heavy_B = [i for i in range(len(endB)) if endB[i].symbol != "H"]
        endB.set_constraint(FixAtoms(indices=heavy_B))
        optB = LBFGS(endB, logfile=None)
        optB.run(fmax=0.02, steps=100)

        # Create images
        images = [endA.copy()]
        for _ in range(11):
            img = endA.copy()
            img.calc = calc
            heavy_img = [i for i in range(len(img)) if img[i].symbol != "H"]
            img.set_constraint(FixAtoms(indices=heavy_img))
            images.append(img)
        images.append(endB.copy())

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
    else:
        print("  Skipping diffusion analysis (no valid E_a)")

    results["comparison_table"] = COMPARISON
    results["total_time_s"] = float(time.time() - t_total)

    # Save JSON
    json_path = RESULTS / "q071_neb_greigite_large.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {json_path}")

    # Save PNG
    if neb_energies is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(0, 1, len(neb_energies_rel))
        ax.plot(x, neb_energies_rel, "ro-", linewidth=2, markersize=8)
        ax.set_xlabel("Reaction coordinate", fontsize=13)
        ax.set_ylabel("Energy (eV)", fontsize=13)
        ax.set_title(
            f"Greigite H vacancy-hopping NEB\n"
            f"MACE-MP-0 large | E_a = {E_a:.3f} eV | D_H = {D_H:.2e} cm$^2$/s",
            fontsize=12,
        )
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

        text_lines = [
            "Comparison:",
            f"  Pentlandite MACE medium: {COMPARISON['pentlandite_MACE_medium']}",
            f"  Mackinawite DFT lit: {COMPARISON['mackinawite_DFT_lit']}",
            f"  Fe3Ni DFT lit: {COMPARISON['Fe3Ni_DFT_lit']}",
        ]
        ax.text(
            0.02, 0.98, "\n".join(text_lines),
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        fig.tight_layout()
        png_path = RESULTS / "q071_neb_greigite_large.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {png_path}")

    print(f"[VRAM] Peak: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")

    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print("=" * 60)
    print("Comparison table:")
    for k, v in COMPARISON.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
