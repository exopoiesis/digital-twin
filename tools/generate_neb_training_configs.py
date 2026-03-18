#!/usr/bin/env python3
"""
Generate DFT training data from NEB transition state images (Tier 3E).

NEB images sample the potential energy surface near transition states —
critical for teaching ML potentials about reaction barriers.

We already have H diffusion in pentlandite from our NEB study (7 images).
This script generates additional NEB paths:

Config breakdown (~28 NEB images → single-point DFT):
  H diffusion in pentlandite (existing):       7  (already have, skip)
  H diffusion in mackinawite (intra-layer):     7
  H diffusion in greigite:                      7
  CO2 → HCOO- on mackinawite (TS estimate):     7
  TOTAL new:                                   21

Strategy: build initial + final states, generate linear interpolation
(IDPP if available), then GPAW single-point on each image.
Full NEB optimization is too expensive for training data — linearly
interpolated images already sample the right region of PES.

Usage:
    python -u generate_neb_training_configs.py --output /workspace/results/neb_training.xyz
    python -u generate_neb_training_configs.py --dry-run
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write, read
from ase.spacegroup import crystal
from gpaw_checkpoint import register_sigterm_handler, is_shutdown_requested


N_IMAGES = 7  # images between endpoints (not counting endpoints)


# ===========================================================================
#  Mineral builders
# ===========================================================================

def build_mackinawite_supercell():
    """Mackinawite 2x2x2 (32 atoms) for H diffusion."""
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
        primitive_cell=True,
    )
    return atoms.repeat((2, 2, 2))


def build_greigite_primitive():
    """Greigite primitive cell (14 atoms)."""
    return crystal(
        symbols=['Fe', 'Fe', 'S'],
        basis=[
            (0.125, 0.125, 0.125),
            (0.5, 0.5, 0.5),
            (0.380, 0.380, 0.380),  # setting 1, ASE default
        ],
        spacegroup=227,
        cellpar=[9.876, 9.876, 9.876, 90, 90, 90],
        primitive_cell=True,
    )


def build_mackinawite_slab():
    """Mackinawite (001) 2x2 slab for CO2 adsorption NEB."""
    from ase.build import surface
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
        primitive_cell=True,
    )
    slab = surface(atoms, (0, 0, 1), layers=2, vacuum=15.0)
    return slab.repeat((2, 2, 1))


# ===========================================================================
#  NEB path builders
# ===========================================================================

def find_interstitial_sites(atoms, element_mask, n_sites=2):
    """Find two interstitial sites for H diffusion.

    Strategy: find voids between atoms by looking at midpoints
    of metal-metal pairs with sufficient space.
    """
    pos = atoms.positions
    syms = np.array(atoms.get_chemical_symbols())
    metal_mask = (syms == 'Fe') | (syms == 'Ni')
    metal_pos = pos[metal_mask]

    # Find pairs of metals that are 2.5-4.0 Å apart
    sites = []
    for i in range(len(metal_pos)):
        for j in range(i + 1, len(metal_pos)):
            d = np.linalg.norm(metal_pos[i] - metal_pos[j])
            if 2.5 < d < 4.0:
                mid = (metal_pos[i] + metal_pos[j]) / 2
                # Check this midpoint is not too close to any atom
                dists = np.linalg.norm(pos - mid, axis=1)
                if np.min(dists) > 1.0:
                    sites.append(mid)
                    if len(sites) >= n_sites:
                        return sites

    # Fallback: use cell fractions
    cell = atoms.cell
    frac_sites = [
        cell @ np.array([0.25, 0.25, 0.25]),
        cell @ np.array([0.75, 0.75, 0.75]),
    ]
    return frac_sites[:n_sites]


def generate_h_diffusion_path(bulk, label_prefix, n_images=N_IMAGES):
    """Generate linearly interpolated NEB images for H diffusion in bulk.

    Returns list of (atoms, label) for each image.
    """
    configs = []

    sites = find_interstitial_sites(bulk, None, n_sites=2)
    if len(sites) < 2:
        print(f"  WARNING: Could not find 2 interstitial sites for {label_prefix}", flush=True)
        return configs

    site_a, site_b = sites[0], sites[1]

    # Create initial and final states with H
    initial = bulk.copy()
    initial += Atoms('H', positions=[site_a])

    final = bulk.copy()
    final += Atoms('H', positions=[site_b])

    # Linear interpolation of H position
    h_idx = len(bulk)  # H is the last atom
    for img_i in range(n_images):
        frac = (img_i + 1) / (n_images + 1)
        image = initial.copy()
        h_pos = site_a * (1 - frac) + site_b * frac
        image.positions[h_idx] = h_pos
        configs.append((image, f"{label_prefix}_neb_{img_i:02d}"))

    return configs


def generate_co2_to_formate_path(slab, label_prefix, n_images=N_IMAGES):
    """Generate interpolated path: CO2 approaching surface → bent CO2 (TS estimate).

    This is NOT a real NEB — it's an approximate path that samples the
    relevant PES region for ML training. The actual TS requires full NEB.

    Path: CO2 far (5 Å) → CO2 close (2.5 Å) with progressive bending.
    """
    from ase.build import molecule

    configs = []

    syms = np.array(slab.get_chemical_symbols())
    fe_mask = syms == 'Fe'
    fe_pos = slab.positions[fe_mask]
    if len(fe_pos) == 0:
        return configs

    top_fe = fe_pos[np.argmax(fe_pos[:, 2])]

    co2 = molecule('CO2')
    co2.positions -= co2.get_center_of_mass()

    for img_i in range(n_images):
        frac = (img_i + 1) / (n_images + 1)

        image = slab.copy()
        co2_img = co2.copy()

        # Height: 5 Å → 2.5 Å
        height = 5.0 - 2.5 * frac

        # Bending: 0° → 30° (CO2 bends on approach to surface)
        bend_angle = 30.0 * frac
        if bend_angle > 0:
            co2_img.rotate(bend_angle, 'y')

        # Tilt: vertical → tilted
        tilt = 45.0 * frac
        co2_img.rotate(tilt, 'x')

        co2_img.positions -= co2_img.get_center_of_mass()
        co2_img.positions += top_fe + np.array([0, 0, height])

        image += co2_img
        configs.append((image, f"{label_prefix}_co2path_{img_i:02d}"))

    return configs


def generate_all_configs():
    configs = []

    print("=" * 60, flush=True)
    print("Generating NEB training configs (Tier 3E)", flush=True)
    print("=" * 60, flush=True)

    # H diffusion in mackinawite (intra-layer)
    print("\n--- H diffusion in mackinawite ---", flush=True)
    mack = build_mackinawite_supercell()
    cfgs = generate_h_diffusion_path(mack, "mack_H_diff")
    configs.extend([(a, l, False) for a, l in cfgs])
    print(f"  {len(cfgs)} images ({len(mack) + 1} atoms each)", flush=True)

    # H diffusion in greigite
    print("--- H diffusion in greigite ---", flush=True)
    greig = build_greigite_primitive()
    cfgs = generate_h_diffusion_path(greig, "greigite_H_diff")
    configs.extend([(a, l, False) for a, l in cfgs])
    print(f"  {len(cfgs)} images ({len(greig) + 1} atoms each)", flush=True)

    # CO2 → HCOO- approach path on mackinawite
    print("--- CO2 approach on mackinawite (001) ---", flush=True)
    mack_slab = build_mackinawite_slab()
    cfgs = generate_co2_to_formate_path(mack_slab, "mack_001_CO2RR")
    configs.extend([(a, l, True) for a, l in cfgs])
    print(f"  {len(cfgs)} images ({len(mack_slab) + 3} atoms each)", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL NEB TRAINING CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


# ===========================================================================
#  DFT + I/O
# ===========================================================================

def set_magnetic_moments(atoms):
    """Set initial magnetic moments (Vaughan 2006, Chang 2008)."""
    magmoms = []
    for sym in atoms.get_chemical_symbols():
        if sym == 'Fe':
            magmoms.append(1.7)
        elif sym == 'Ni':
            magmoms.append(0.3)
        else:
            magmoms.append(0.0)
    atoms.set_initial_magnetic_moments(magmoms)


def run_gpaw_single_point(atoms, config_label, is_slab=False):
    from gpaw import GPAW, PW, FermiDirac

    set_magnetic_moments(atoms)

    n_atoms = len(atoms)

    if is_slab:
        kpts = (1, 1, 1) if n_atoms > 50 else (2, 2, 1)
    elif n_atoms > 30:
        kpts = (2, 2, 2)
    else:
        kpts = (3, 3, 3)

    calc = GPAW(
        mode=PW(400),
        xc='PBE',
        kpts=kpts,
        occupations=FermiDirac(0.1),
        convergence={'energy': 1e-5},
        maxiter=500,
        parallel={'augment_grids': True},
        txt=f'/workspace/results/{config_label}.txt',
    )

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = None
    if all(atoms.pbc) and not is_slab:
        stress = atoms.get_stress(voigt=True)

    return {'energy': energy, 'forces': forces, 'stress': stress, 'config_type': config_label}


def save_to_extxyz(atoms, results, output_path):
    atoms_copy = atoms.copy()
    atoms_copy.info['energy'] = results['energy']
    atoms_copy.info['config_type'] = results['config_type']
    atoms_copy.arrays['forces'] = results['forces']
    if results['stress'] is not None:
        atoms_copy.info['stress'] = results['stress']
    write(output_path, atoms_copy, format='extxyz', append=True)


def load_existing_labels(output_path):
    if not Path(output_path).exists():
        return set()
    try:
        all_atoms = read(output_path, index=':', format='extxyz')
        return {a.info.get('config_type', '') for a in all_atoms}
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(description="Generate NEB training DFT data (Tier 3E)")
    parser.add_argument('--output', type=str, default='/workspace/results/neb_training.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    register_sigterm_handler()

    configs = generate_all_configs()

    if args.dry_run:
        print("\nDRY RUN — config list:")
        for atoms, label, is_slab in configs:
            slab_tag = " [SLAB]" if is_slab else ""
            print(f"  {label}: {len(atoms)} atoms{slab_tag}")
        print(f"\nTotal: {len(configs)} configs")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if args.resume:
        existing = load_existing_labels(output_path)
        print(f"Resuming: found {len(existing)} existing configs", flush=True)

    remaining = [(a, l, s) for a, l, s in configs if l not in existing]
    print(f"Remaining: {len(remaining)} configs to compute\n", flush=True)

    log_path = output_path.parent / 'neb_training_log.txt'

    for i, (atoms, label, is_slab) in enumerate(remaining):
        if is_shutdown_requested():
            print(f"\n[SIGTERM] Graceful shutdown. Resume with --resume.", flush=True)
            break

        t0 = time.time()
        try:
            results = run_gpaw_single_point(atoms, label, is_slab)
            save_to_extxyz(atoms, results, output_path)
            dt = time.time() - t0
            msg = (f"[{i+1}/{len(remaining)}] {label}: "
                   f"E={results['energy']:.4f} eV, "
                   f"max|F|={np.max(np.linalg.norm(results['forces'], axis=1)):.4f} eV/A "
                   f"({dt:.1f}s)")
            print(msg, flush=True)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')
        except Exception as e:
            msg = f"[{i+1}/{len(remaining)}] {label}: FAILED — {e}"
            print(msg, flush=True)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')
            traceback.print_exc()

    print(f"\n{'=' * 60}", flush=True)
    print(f"Done. Output: {output_path}", flush=True)
    if output_path.exists():
        final = read(output_path, index=':', format='extxyz')
        print(f"Total configs in file: {len(final)}", flush=True)


if __name__ == '__main__':
    main()
