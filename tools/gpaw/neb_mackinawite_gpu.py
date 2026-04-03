#!/usr/bin/env python3
"""
NEB calculation of H diffusion in mackinawite FeS
using MACE-MP-0 large model on GPU.

Crystal: P4/nmm (129), a=b=3.674 A, c=5.033 A, layered structure.
Two pathways:
  (a) intra-layer: H hop between S atoms in the same layer (Grotthuss in ab-plane)
  (b) inter-layer: H hop between S atoms in different layers (intercalation along c)

Supercell: 3x3x2 (~36 FeS units, 72 atoms) to avoid periodic image artifacts.
Output: q071_neb_mackinawite_large.json, q071_neb_mackinawite_large.png
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
    "mackinawite_DFT_lit": "1.12 eV",
    "pentlandite_MACE_large": "1.29 eV",
    "pentlandite_MACE_medium": "0.96 eV",
    "Fe3Ni_DFT_lit": "0.70 eV",
}


def build_mackinawite_supercell():
    """Build mackinawite FeS (P4/nmm) 3x3x2 supercell.

    Mackinawite: tetragonal, P4/nmm (129)
    a = b = 3.674 A, c = 5.033 A
    Fe at (0, 0, 0), S at (0, 0.5, 0.2602)
    """
    atoms = crystal(
        symbols=["Fe", "S"],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
    )
    # 3x3x2 supercell
    supercell = atoms.repeat((3, 3, 2))
    return supercell


def find_ss_pairs_by_layer(atoms):
    """Classify S-S pairs as intra-layer or inter-layer.

    In mackinawite, S layers are at z ~ 0.26c and z ~ (1-0.26)c = 0.74c.
    With supercell, layers repeat at z + n*c.
    """
    s_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "S"]
    s_positions = atoms.positions[s_indices]
    cell_c = atoms.cell[2, 2]  # total c of supercell

    # Get fractional z coordinates
    frac_z = s_positions[:, 2] / cell_c

    # Cluster S atoms into layers by their fractional z
    # In P4/nmm with basis S at z=0.2602, layers are at z ~ 0.2602 + n*0.5
    # In 3x3x2 supercell, unit c = 5.033, total c = 10.066
    # Layers at: 0.1301, 0.3699, 0.6301, 0.8699 (fractional of supercell)

    _, d_matrix = get_distances(s_positions, cell=atoms.cell, pbc=True)

    intra_pairs = []
    inter_pairs = []

    for a in range(len(s_indices)):
        for b in range(a + 1, len(s_indices)):
            d = d_matrix[a, b]
            # Check if same layer: |dz| < 0.5 A (within layer tolerance)
            dz = abs(s_positions[a, 2] - s_positions[b, 2])
            # Account for PBC: minimum image in z
            dz_pbc = min(dz, cell_c - dz)
            if dz_pbc < 0.5:
                intra_pairs.append((s_indices[a], s_indices[b], d))
            else:
                inter_pairs.append((s_indices[a], s_indices[b], d))

    intra_pairs.sort(key=lambda x: x[2])
    inter_pairs.sort(key=lambda x: x[2])
    return intra_pairs, inter_pairs


def run_neb(atoms_pristine, calc, si_idx, sj_idx, hop_type, results_key, results):
    """Run CI-NEB for a given S-S pair. Returns E_a or None."""
    pos_si = atoms_pristine.positions[si_idx].copy()
    pos_sj = atoms_pristine.positions[sj_idx].copy()
    del_indices = sorted([si_idx, sj_idx], reverse=True)

    # Endpoint A
    endA = atoms_pristine.copy()
    for idx in del_indices:
        del endA[idx]
    endA.append(Atom("H", position=pos_si))
    endA.calc = calc
    heavy_A = [i for i in range(len(endA)) if endA[i].symbol != "H"]
    endA.set_constraint(FixAtoms(indices=heavy_A))
    optA = LBFGS(endA, logfile=None)
    optA.run(fmax=0.02, steps=100)
    e_A = endA.get_potential_energy()
    print(f"    EndpointA ({hop_type}): E = {e_A:.4f} eV ({optA.nsteps} steps)")

    # Endpoint B
    endB = atoms_pristine.copy()
    for idx in del_indices:
        del endB[idx]
    endB.append(Atom("H", position=pos_sj))
    endB.calc = calc
    heavy_B = [i for i in range(len(endB)) if endB[i].symbol != "H"]
    endB.set_constraint(FixAtoms(indices=heavy_B))
    optB = LBFGS(endB, logfile=None)
    optB.run(fmax=0.02, steps=100)
    e_B = endB.get_potential_energy()
    print(f"    EndpointB ({hop_type}): E = {e_B:.4f} eV ({optB.nsteps} steps)")

    results[f"{results_key}_E_endpointA_eV"] = float(e_A)
    results[f"{results_key}_E_endpointB_eV"] = float(e_B)

    # CI-NEB
    neb_energies_rel = None
    E_a = None
    neb_converged = False

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
        print(f"    NEB ({hop_type}) converged: {neb_converged}")
        print(f"    E_a ({hop_type}) = {E_a:.4f} eV")
        results[f"{results_key}_neb_converged"] = bool(neb_converged)
        results[f"{results_key}_E_a_eV"] = float(E_a)
        results[f"{results_key}_neb_energies_eV"] = [float(e) for e in neb_energies_rel]
    except Exception as exc:
        print(f"    NEB ({hop_type}) FAILED: {exc}")
        results[f"{results_key}_neb_error"] = str(exc)

    return E_a, neb_energies_rel


def main():
    t_total = time.time()
    results = {
        "mineral": "mackinawite",
        "model": "MACE-MP-0 large",
        "device": "cuda",
        "structure": "P4/nmm (129), layered FeS",
        "supercell": "3x3x2",
    }

    # ── [1/9] GPU info ──────────────────────────────────────────────
    print("[1/9] GPU info")
    t0 = time.time()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")
        results["gpu"] = gpu_name
    else:
        print("  WARNING: CUDA not available, falling back to CPU")
        results["gpu"] = "N/A (CPU fallback)"
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [2/9] Build mackinawite supercell ────────────────────────────
    print("[2/9] Build mackinawite 3x3x2 supercell")
    t0 = time.time()
    atoms = build_mackinawite_supercell()
    n_atoms = len(atoms)
    formula = atoms.get_chemical_formula()
    print(f"  {formula}, {n_atoms} atoms")
    print(f"  Cell: {atoms.cell.lengths()}")
    results["formula"] = formula
    results["n_atoms"] = n_atoms
    results["cell_A"] = atoms.cell.lengths().tolist()
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [3/9] Load MACE-MP-0 large on GPU ────────────────────────────
    print("[3/9] Load MACE-MP-0 large on GPU")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="large", device=device, default_dtype="float64")
    print(f"  loaded on {device} in {time.time()-t0:.1f}s")

    # ── [4/9] Relax supercell ────────────────────────────────────────
    print("[4/9] Relax supercell (LBFGS, fmax=0.01, 300 steps)")
    t0 = time.time()
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=0.01, steps=300)
    e_pristine = atoms.get_potential_energy()
    print(f"  E_pristine = {e_pristine:.4f} eV, converged in {opt.nsteps} steps")
    results["E_pristine_eV"] = float(e_pristine)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [5/9] Find intra-layer and inter-layer S-S pairs ────────────
    print("[5/9] Find intra-layer and inter-layer S-S pairs")
    t0 = time.time()
    intra_pairs, inter_pairs = find_ss_pairs_by_layer(atoms)

    print(f"  Intra-layer pairs: {len(intra_pairs)}, shortest: {intra_pairs[0][2]:.3f} A")
    print(f"  Inter-layer pairs: {len(inter_pairs)}, shortest: {inter_pairs[0][2]:.3f} A")
    for label, pairs in [("intra", intra_pairs), ("inter", inter_pairs)]:
        for rank in range(min(3, len(pairs))):
            i, j, d = pairs[rank]
            print(f"    {label} #{rank}: atoms {i} & {j}, d = {d:.3f} A")

    intra_si, intra_sj, intra_d = intra_pairs[0]
    inter_si, inter_sj, inter_d = inter_pairs[0]

    results["intra_S_pair_indices"] = [int(intra_si), int(intra_sj)]
    results["intra_S_pair_distance_A"] = float(intra_d)
    results["inter_S_pair_indices"] = [int(inter_si), int(inter_sj)]
    results["inter_S_pair_distance_A"] = float(inter_d)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [6/9] Intra-layer NEB ────────────────────────────────────────
    print("[6/9] Intra-layer CI-NEB (Grotthuss in ab-plane)")
    print(f"  S pair: atoms {intra_si} & {intra_sj}, d = {intra_d:.3f} A")
    t0 = time.time()
    E_a_intra, neb_rel_intra = run_neb(
        atoms, calc, intra_si, intra_sj, "intra-layer", "intra", results
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [7/9] Inter-layer NEB ────────────────────────────────────────
    print("[7/9] Inter-layer CI-NEB (intercalation along c)")
    print(f"  S pair: atoms {inter_si} & {inter_sj}, d = {inter_d:.3f} A")
    t0 = time.time()
    E_a_inter, neb_rel_inter = run_neb(
        atoms, calc, inter_si, inter_sj, "inter-layer", "inter", results
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [8/9] Diffusion analysis ─────────────────────────────────────
    print("[8/9] Diffusion analysis")
    t0 = time.time()

    kB = 8.617e-5  # eV/K
    T = 298.15  # K

    for label, E_a, hop_d in [
        ("intra", E_a_intra, intra_d),
        ("inter", E_a_inter, inter_d),
    ]:
        if E_a is not None and E_a > 0:
            a_hop = hop_d * 1e-8  # cm
            D_H = 1e13 * a_hop**2 * np.exp(-E_a / (kB * T))
            L = 200e-7  # 200 nm in cm
            tau = L**2 / (2 * D_H) if D_H > 0 else float("inf")
            print(f"  {label}: D_H = {D_H:.3e} cm^2/s, tau(200nm) = {tau:.3e} s")
            results[f"{label}_D_H_cm2s"] = float(D_H)
            results[f"{label}_tau_200nm_s"] = float(tau)
        else:
            print(f"  {label}: no valid E_a, skipping")

    # Which pathway dominates?
    if E_a_intra is not None and E_a_inter is not None:
        dominant = "intra-layer" if E_a_intra < E_a_inter else "inter-layer"
        print(f"  Dominant pathway: {dominant} (lower barrier)")
        results["dominant_pathway"] = dominant
        results["E_a_dominant_eV"] = float(min(E_a_intra, E_a_inter))

    print(f"  done in {time.time()-t0:.1f}s")

    # ── [9/9] Output ──────────────────────────────────────────────────
    print("[9/9] Save results")
    t0 = time.time()

    results["comparison_table"] = COMPARISON
    results["total_time_s"] = float(time.time() - t_total)

    # Save JSON
    json_path = RESULTS / "q071_neb_mackinawite_large.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {json_path}")

    # Save PNG (dual plot: intra + inter)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, label, neb_rel, E_a, hop_d, color in [
        (axes[0], "Intra-layer (Grotthuss)", neb_rel_intra, E_a_intra, intra_d, "b"),
        (axes[1], "Inter-layer (intercalation)", neb_rel_inter, E_a_inter, inter_d, "r"),
    ]:
        if neb_rel is not None:
            x = np.linspace(0, 1, len(neb_rel))
            ax.plot(x, neb_rel, f"{color}o-", linewidth=2, markersize=7)
            ax.set_xlabel("Reaction coordinate", fontsize=12)
            ax.set_ylabel("Energy (eV)", fontsize=12)
            ea_str = f"{E_a:.3f}" if E_a is not None else "N/A"
            ax.set_title(f"Mackinawite {label}\nE_a = {ea_str} eV, d = {hop_d:.2f} A",
                         fontsize=11)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"{label}\nNEB FAILED", ha="center", va="center",
                    fontsize=14, transform=ax.transAxes)
            ax.set_title(label, fontsize=11)

    # Comparison table on right plot
    text_lines = [
        "Comparison:",
        f"  Mackinawite DFT lit: {COMPARISON['mackinawite_DFT_lit']}",
        f"  Pentlandite MACE large: {COMPARISON['pentlandite_MACE_large']}",
    ]
    if E_a_intra is not None:
        text_lines.append(f"  This intra-layer: {E_a_intra:.2f} eV")
    if E_a_inter is not None:
        text_lines.append(f"  This inter-layer: {E_a_inter:.2f} eV")
    axes[1].text(
        0.02, 0.98, "\n".join(text_lines),
        transform=axes[1].transAxes, fontsize=8, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )

    fig.suptitle("Mackinawite FeS — H diffusion NEB (MACE-MP-0 large)", fontsize=13, y=1.02)
    fig.tight_layout()
    png_path = RESULTS / "q071_neb_mackinawite_large.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {png_path}")

    print(f"[VRAM] Peak: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")

    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print("=" * 60)
    print("Results summary:")
    if E_a_intra is not None:
        print(f"  Intra-layer E_a = {E_a_intra:.3f} eV (d = {intra_d:.2f} A)")
    if E_a_inter is not None:
        print(f"  Inter-layer E_a = {E_a_inter:.3f} eV (d = {inter_d:.2f} A)")
    print("Comparison table:")
    for k, v in COMPARISON.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
