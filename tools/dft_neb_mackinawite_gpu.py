#!/usr/bin/env python3
"""
DFT (GPAW) NEB validation of H diffusion in mackinawite FeS (intra-layer).

Validates MACE-MP-0 result (E_a = 0.44 eV) with first-principles DFT.

Crystal: P4/nmm (129), a=b=3.674 A, c=5.033 A, layered structure.
Pathway: intra-layer Grotthuss-like H hop between S sites in the same layer.
Mechanism: H on S-vacancy site hops to adjacent S-vacancy site within ab-plane.

Supercell: 2x2x1 (16 Fe + 16 S - 2 S + 1 H = 31 atoms) — fits RTX 4070 12 GB.
Method: GPAW PBE, PW(350 eV), Gamma-point, CI-NEB with 5 images.

Output: q071_dft_neb_mackinawite.json, q071_dft_neb_mackinawite.png

Docker (infra-gpaw):
  MSYS_NO_PATHCONV=1 docker --context gomer run -d --name q071-dft-neb --gpus all \
    -v "C:/Users/Igor/project-third-matter/results:/workspace/results" \
    -v "C:/Users/Igor/project-third-matter/infra/gpu_scripts:/workspace/scripts:ro" \
    -w //workspace infra-gpaw python -u scripts/dft_neb_mackinawite_gpu.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import traceback
import numpy as np
from pathlib import Path

from ase import Atom
from ase.spacegroup import crystal
from ase.geometry import get_distances
from ase.mep import NEB
from ase.optimize import BFGS
from ase.constraints import FixAtoms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("/workspace/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# --- VRAM preflight ---
def _check_vram(required_gb):
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        total, free = [int(x) / 1024 for x in result.stdout.strip().split(', ')]
        print(f"[VRAM] {total:.1f} GB total, {free:.1f} GB free, {required_gb:.1f} GB required")
        if free < required_gb:
            print(f"[VRAM] WARNING: only {free:.1f} GB free, need {required_gb:.1f} GB — risk of OOM")
    except Exception:
        print(f"[VRAM] Could not check VRAM (nvidia-smi not found)")
_check_vram(5.0)

# MACE reference for comparison
E_A_MACE = 0.44  # eV, intra-layer from MACE-MP-0 large (3x3x2)

# NEB parameters
N_IMAGES = 5          # intermediate images (total = N_IMAGES + 2 endpoints)
FMAX_RELAX = 0.03     # eV/A for endpoint relaxation
FMAX_NEB = 0.05       # eV/A for NEB convergence
MAX_STEPS_RELAX = 200
MAX_STEPS_NEB = 300
PW_CUTOFF = 350       # eV, reduced from 400 for NEB feasibility
KPTS = (2, 2, 1)      # k-point mesh
SMEARING = 0.1        # eV, Fermi-Dirac width


def get_gpaw_calc(label="gpaw", kpts=KPTS, txt="-"):
    """Create GPAW calculator for NEB: PBE, PW mode, symmetry off."""
    from gpaw import GPAW, PW, FermiDirac
    calc = GPAW(
        mode=PW(PW_CUTOFF),
        xc='PBE',
        kpts=kpts,
        occupations=FermiDirac(SMEARING),
        convergence={'energy': 0.001},  # 1 meV, relaxed for NEB
        symmetry='off',                 # required for NEB
        txt=txt,
        parallel={'augment_grids': True},
    )
    return calc


def build_mackinawite_supercell(repeat=(2, 2, 1)):
    """Build mackinawite FeS (P4/nmm) supercell.

    Mackinawite: tetragonal, P4/nmm (#129)
    a = b = 3.674 A, c = 5.033 A
    Fe at (0, 0, 0), S at (0, 0.5, 0.2602)

    Unit cell: 2 Fe + 2 S = 4 atoms.
    2x2x1: 8 Fe + 8 S = 16 atoms (but P4/nmm has 2 formula units per cell,
    so actual unit cell = 2 Fe + 2 S; 2x2x1 repeat = 8 Fe + 8 S = 16 atoms).
    """
    atoms = crystal(
        symbols=["Fe", "S"],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
    )
    supercell = atoms.repeat(repeat)
    return supercell


def find_intra_layer_ss_pair(atoms):
    """Find the shortest intra-layer S-S pair for H hop pathway.

    In mackinawite, S layers are at z ~ 0.2602*c and z ~ (1-0.2602)*c = 0.7398*c.
    Intra-layer = both S atoms at same z (within tolerance).

    Returns: (s_idx_A, s_idx_B, distance)
    """
    s_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "S"]
    s_positions = atoms.positions[s_indices]
    cell_c = atoms.cell[2, 2]

    _, d_matrix = get_distances(s_positions, cell=atoms.cell, pbc=True)

    intra_pairs = []
    for a in range(len(s_indices)):
        for b in range(a + 1, len(s_indices)):
            d = d_matrix[a, b]
            dz = abs(s_positions[a, 2] - s_positions[b, 2])
            dz_pbc = min(dz, cell_c - dz)
            if dz_pbc < 0.5:  # same layer
                intra_pairs.append((s_indices[a], s_indices[b], d))

    intra_pairs.sort(key=lambda x: x[2])
    return intra_pairs[0] if intra_pairs else None


def prepare_endpoint(atoms_pristine, si_idx, sj_idx, h_on_idx):
    """Prepare NEB endpoint: remove two S atoms, place H at one S-vacancy site.

    args:
        atoms_pristine: relaxed supercell
        si_idx, sj_idx: indices of two S atoms to remove (create vacancies)
        h_on_idx: index of the S atom where H is placed (si_idx or sj_idx)

    returns: atoms with 2 S removed, 1 H added, heavy atoms constrained
    """
    h_pos = atoms_pristine.positions[h_on_idx].copy()
    del_indices = sorted([si_idx, sj_idx], reverse=True)

    endpoint = atoms_pristine.copy()
    for idx in del_indices:
        del endpoint[idx]

    # Place H at the vacancy site (slightly offset toward interlayer space)
    # S is at z~0.26c or z~0.74c. H sits slightly above/below the S plane.
    endpoint.append(Atom("H", position=h_pos))

    return endpoint


def relax_endpoint(endpoint, label="endpoint"):
    """Relax endpoint with GPAW: fix heavy atoms, relax only H."""
    endpoint.calc = get_gpaw_calc(label=label, txt=f"/workspace/results/gpaw_{label}.txt")

    # Fix all non-H atoms
    heavy = [i for i in range(len(endpoint)) if endpoint[i].symbol != "H"]
    endpoint.set_constraint(FixAtoms(indices=heavy))

    opt = BFGS(endpoint, logfile=None)
    converged = opt.run(fmax=FMAX_RELAX, steps=MAX_STEPS_RELAX)
    energy = endpoint.get_potential_energy()

    print(f"  {label}: E = {energy:.4f} eV, steps = {opt.nsteps}, converged = {converged}")
    return energy, opt.nsteps


def main():
    t_total = time.time()
    results = {
        "system": "mackinawite_intra_layer",
        "method": "DFT_GPAW_PBE",
        "pw_cutoff_eV": PW_CUTOFF,
        "kpts": list(KPTS),
        "smearing_eV": SMEARING,
        "fmax_relax": FMAX_RELAX,
        "fmax_neb": FMAX_NEB,
        "n_images": N_IMAGES,
        "E_a_MACE_eV": E_A_MACE,
    }

    print("=" * 70)
    print("  DFT (GPAW) NEB: H diffusion in mackinawite FeS (intra-layer)")
    print(f"  Method: PBE, PW({PW_CUTOFF} eV), kpts={KPTS}, {N_IMAGES} images")
    print("=" * 70)

    # ── [1/8] GPU/CuPy info ──────────────────────────────────────────
    print("\n[1/8] System info")
    try:
        import cupy
        gpu_name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        vram = cupy.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"] / 1e9
        print(f"  GPU: {gpu_name}, VRAM: {vram:.1f} GB")
        print(f"  CuPy available: GPU-accelerated FFTs enabled")
        results["gpu"] = gpu_name
        results["gpu_fft"] = True
    except Exception:
        print("  CuPy not available: CPU FFTs will be used")
        results["gpu"] = "N/A"
        results["gpu_fft"] = False

    # ── [2/8] Build supercell ─────────────────────────────────────────
    print("\n[2/8] Build mackinawite 2x2x1 supercell")
    t0 = time.time()
    sc_repeat = (2, 2, 1)
    atoms = build_mackinawite_supercell(repeat=sc_repeat)
    n_atoms_pristine = len(atoms)
    formula = atoms.get_chemical_formula()
    cell_lengths = atoms.cell.lengths().tolist()
    print(f"  {formula}, {n_atoms_pristine} atoms")
    print(f"  Cell: a={cell_lengths[0]:.3f}, b={cell_lengths[1]:.3f}, c={cell_lengths[2]:.3f} A")
    results["supercell"] = "x".join(map(str, sc_repeat))
    results["formula_pristine"] = formula
    results["n_atoms_pristine"] = n_atoms_pristine
    results["cell_A"] = cell_lengths
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [3/8] Relax pristine supercell ────────────────────────────────
    print("\n[3/8] Relax pristine supercell (BFGS, fmax=0.03)")
    t0 = time.time()
    atoms.calc = get_gpaw_calc(label="pristine", txt="/workspace/results/gpaw_pristine.txt")
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=FMAX_RELAX, steps=MAX_STEPS_RELAX)
    e_pristine = atoms.get_potential_energy()
    print(f"  E_pristine = {e_pristine:.4f} eV ({opt.nsteps} steps)")
    results["E_pristine_eV"] = float(e_pristine)
    t_relax_pristine = time.time() - t0
    print(f"  done in {t_relax_pristine:.1f}s")
    results["time_relax_pristine_s"] = round(t_relax_pristine, 1)

    # ── [4/8] Find intra-layer S-S pair ───────────────────────────────
    print("\n[4/8] Find intra-layer S-S pair for H hop")
    t0 = time.time()
    pair = find_intra_layer_ss_pair(atoms)
    if pair is None:
        print("  ERROR: no intra-layer S-S pair found!")
        results["error"] = "no intra-layer S-S pair found"
        with open(RESULTS / "q071_dft_neb_mackinawite.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    si_idx, sj_idx, ss_dist = pair
    print(f"  S pair: atoms {si_idx} & {sj_idx}, distance = {ss_dist:.3f} A")
    print(f"  S_i position: {atoms.positions[si_idx]}")
    print(f"  S_j position: {atoms.positions[sj_idx]}")
    results["S_pair_indices"] = [int(si_idx), int(sj_idx)]
    results["S_pair_distance_A"] = float(ss_dist)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [5/8] Prepare and relax endpoints ─────────────────────────────
    print("\n[5/8] Prepare and relax NEB endpoints")
    t0 = time.time()

    # Endpoint A: H at S_i vacancy
    endA = prepare_endpoint(atoms, si_idx, sj_idx, h_on_idx=si_idx)
    n_atoms = len(endA)
    print(f"  System: {endA.get_chemical_formula()}, {n_atoms} atoms")
    results["n_atoms"] = n_atoms
    results["formula"] = endA.get_chemical_formula()

    e_A, steps_A = relax_endpoint(endA, label="endA")
    results["E_endpointA_eV"] = float(e_A)

    # Endpoint B: H at S_j vacancy
    endB = prepare_endpoint(atoms, si_idx, sj_idx, h_on_idx=sj_idx)
    e_B, steps_B = relax_endpoint(endB, label="endB")
    results["E_endpointB_eV"] = float(e_B)

    dE_endpoints = abs(e_A - e_B)
    print(f"  |E_A - E_B| = {dE_endpoints:.4f} eV")
    results["dE_endpoints_eV"] = float(dE_endpoints)

    t_endpoints = time.time() - t0
    print(f"  done in {t_endpoints:.1f}s")
    results["time_endpoints_s"] = round(t_endpoints, 1)

    # ── [6/8] CI-NEB ──────────────────────────────────────────────────
    print(f"\n[6/8] CI-NEB with {N_IMAGES} images")
    t0 = time.time()

    neb_energies_rel = None
    E_a = None
    neb_converged = False
    neb_steps = 0

    try:
        # Build image list: endpoint_A + N_IMAGES intermediates + endpoint_B
        # IMPORTANT: endpoints must keep their calculators from relaxation
        # Intermediates get fresh calculators AFTER interpolation
        images = [endA]
        for i in range(N_IMAGES):
            img = endA.copy()
            # Constraints: fix heavy atoms, relax H
            heavy = [j for j in range(len(img)) if img[j].symbol != "H"]
            img.set_constraint(FixAtoms(indices=heavy))
            images.append(img)
        images.append(endB)

        # Interpolate positions (linear interpolation between endpoints)
        neb = NEB(images, climb=True, allow_shared_calculator=False)
        neb.interpolate()

        # CRITICAL: assign calculator to each intermediate AFTER interpolation
        # (ASE lesson: .copy() does NOT copy calc, and interpolation requires
        # images without calcs initially)
        for i in range(1, len(images) - 1):
            images[i].calc = get_gpaw_calc(
                label=f"neb_img{i}",
                txt=f"/workspace/results/gpaw_neb_img{i}.txt"
            )

        print(f"  Images: {len(images)} total ({N_IMAGES} intermediate + 2 endpoints)")
        print(f"  H positions along path:")
        h_idx_in_image = len(images[0]) - 1  # H is last atom
        for k, img in enumerate(images):
            h_pos = img.positions[h_idx_in_image]
            print(f"    image {k}: H at ({h_pos[0]:.3f}, {h_pos[1]:.3f}, {h_pos[2]:.3f})")

        # Run NEB optimization
        print(f"\n  Running BFGS NEB (fmax={FMAX_NEB}, max_steps={MAX_STEPS_NEB})...")
        opt_neb = BFGS(neb, logfile=None)
        neb_converged = opt_neb.run(fmax=FMAX_NEB, steps=MAX_STEPS_NEB)
        neb_steps = opt_neb.nsteps

        # Extract energies
        neb_energies = [img.get_potential_energy() for img in images]
        e_ref = neb_energies[0]
        neb_energies_rel = [e - e_ref for e in neb_energies]
        E_a = max(neb_energies_rel)

        print(f"\n  NEB converged: {neb_converged} ({neb_steps} steps)")
        print(f"  Energies (relative to endpoint A):")
        for k, e in enumerate(neb_energies_rel):
            marker = " <-- TS" if e == E_a and k > 0 and k < len(neb_energies_rel) - 1 else ""
            print(f"    image {k}: {e:.4f} eV{marker}")
        print(f"\n  E_a (DFT) = {E_a:.4f} eV")
        print(f"  E_a (MACE) = {E_A_MACE:.4f} eV")

        if E_a > 0:
            ratio = E_a / E_A_MACE
            print(f"  DFT/MACE ratio = {ratio:.2f}")
            results["DFT_MACE_ratio"] = float(ratio)

        results["converged"] = bool(neb_converged)
        results["n_neb_steps"] = int(neb_steps)
        results["E_a_eV"] = float(E_a)
        results["energies_per_image"] = [float(e) for e in neb_energies_rel]

    except Exception as exc:
        print(f"  NEB FAILED: {exc}")
        traceback.print_exc()
        results["converged"] = False
        results["neb_error"] = str(exc)

    t_neb = time.time() - t0
    print(f"  done in {t_neb:.1f}s")
    results["time_neb_s"] = round(t_neb, 1)

    # ── [7/8] Diffusion coefficient ───────────────────────────────────
    print("\n[7/8] Diffusion analysis")
    if E_a is not None and E_a > 0:
        kB = 8.617e-5  # eV/K
        T = 298.15     # K
        a_hop = ss_dist * 1e-8  # cm
        D_H = 1e13 * a_hop**2 * np.exp(-E_a / (kB * T))
        L_200nm = 200e-7  # cm
        tau = L_200nm**2 / (2 * D_H) if D_H > 0 else float("inf")

        print(f"  E_a = {E_a:.4f} eV, hop distance = {ss_dist:.3f} A")
        print(f"  D_H (298 K) = {D_H:.3e} cm^2/s")
        print(f"  tau (200 nm) = {tau:.3e} s")

        results["D_H_cm2s"] = float(D_H)
        results["tau_200nm_s"] = float(tau)

        # Compare with MACE
        D_H_mace = 1e13 * a_hop**2 * np.exp(-E_A_MACE / (kB * T))
        tau_mace = L_200nm**2 / (2 * D_H_mace) if D_H_mace > 0 else float("inf")
        print(f"\n  MACE comparison:")
        print(f"    D_H (MACE) = {D_H_mace:.3e} cm^2/s")
        print(f"    tau (MACE, 200 nm) = {tau_mace:.3e} s")
        print(f"    D_H(DFT)/D_H(MACE) = {D_H/D_H_mace:.2e}")
        results["D_H_MACE_cm2s"] = float(D_H_mace)

        # Agreement assessment
        if abs(E_a - E_A_MACE) < 0.1:
            agreement = "excellent (<0.1 eV)"
        elif abs(E_a - E_A_MACE) < 0.2:
            agreement = "good (0.1-0.2 eV)"
        elif abs(E_a - E_A_MACE) < 0.5:
            agreement = "moderate (0.2-0.5 eV)"
        else:
            agreement = f"poor (>{abs(E_a - E_A_MACE):.1f} eV)"
        print(f"\n  Agreement: {agreement}")
        results["agreement"] = agreement
    else:
        print("  No valid E_a, skipping diffusion analysis")
        results["agreement"] = "N/A (NEB failed)"

    # ── [8/8] Save results ────────────────────────────────────────────
    print("\n[8/8] Save results")
    results["total_time_s"] = round(time.time() - t_total, 1)

    # JSON
    json_path = RESULTS / "q071_dft_neb_mackinawite.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {json_path}")

    # PNG: NEB barrier profile
    fig, ax = plt.subplots(figsize=(8, 5))

    if neb_energies_rel is not None:
        n_total = len(neb_energies_rel)
        x = np.linspace(0, 1, n_total)
        ax.plot(x, neb_energies_rel, "bo-", linewidth=2, markersize=8, label="DFT (GPAW PBE)")

        # Highlight transition state
        ts_idx = np.argmax(neb_energies_rel)
        ax.plot(x[ts_idx], neb_energies_rel[ts_idx], "r*", markersize=15, zorder=5,
                label=f"TS: E_a = {E_a:.3f} eV")

        # MACE reference line
        ax.axhline(E_A_MACE, color="orange", linestyle="--", linewidth=1.5, alpha=0.8,
                    label=f"MACE-MP-0: E_a = {E_A_MACE:.3f} eV")

        ax.set_xlabel("Reaction coordinate", fontsize=12)
        ax.set_ylabel("Energy relative to initial (eV)", fontsize=12)
        ax.set_title(
            f"Mackinawite FeS — intra-layer H diffusion NEB\n"
            f"DFT (GPAW PBE, PW {PW_CUTOFF} eV) vs MACE-MP-0",
            fontsize=11,
        )
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotation box
        info_text = (
            f"Supercell: {results['supercell']}\n"
            f"Atoms: {n_atoms}\n"
            f"k-points: {KPTS}\n"
            f"S-S distance: {ss_dist:.3f} A\n"
            f"NEB converged: {neb_converged}\n"
            f"Steps: {neb_steps}\n"
            f"Agreement: {results.get('agreement', 'N/A')}"
        )
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
        )
    else:
        ax.text(0.5, 0.5, "NEB FAILED\nSee JSON for error details",
                ha="center", va="center", fontsize=14, color="red",
                transform=ax.transAxes)
        ax.set_title("Mackinawite FeS — intra-layer H diffusion NEB (FAILED)")

    fig.tight_layout()
    png_path = RESULTS / "q071_dft_neb_mackinawite.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {png_path}")

    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        print(f"[VRAM] Current usage: {int(result.stdout.strip())/1024:.2f} GB")
    except Exception:
        pass

    # Final summary
    total_time = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  System:        mackinawite FeS, intra-layer Grotthuss")
    print(f"  Supercell:     {results['supercell']} ({n_atoms} atoms)")
    print(f"  S-S distance:  {ss_dist:.3f} A")
    if E_a is not None:
        print(f"  E_a (DFT):     {E_a:.4f} eV")
        print(f"  E_a (MACE):    {E_A_MACE:.4f} eV")
        print(f"  Agreement:     {results.get('agreement', 'N/A')}")
        if "D_H_cm2s" in results:
            print(f"  D_H (DFT):     {results['D_H_cm2s']:.3e} cm^2/s")
            print(f"  tau (200 nm):  {results['tau_200nm_s']:.3e} s")
    else:
        print(f"  E_a:           FAILED")
    print(f"  NEB converged: {neb_converged}")
    print(f"  Total time:    {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
