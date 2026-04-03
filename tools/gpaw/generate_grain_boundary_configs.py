#!/usr/bin/env python3
"""
Generate DFT training data for grain boundary structures (Tier 4F).

Grain boundaries (GBs) are ubiquitous in polycrystalline sulfides.
They affect proton transport, catalytic activity, and mechanical properties.

Strategy: build simple tilt GBs by mirroring a slab along a chosen axis.
This creates a Σ-type grain boundary with controlled misorientation.

Config breakdown (~20 configs):
  Pentlandite Σ3 (111) tilt GB + rattles:    5
  Pentlandite Σ5 (001) tilt GB + rattles:    5
  Pyrite Σ3 (111) tilt GB + rattles:         5
  Pyrite Σ5 (100) tilt GB + rattles:         5
  TOTAL:                                    20

Usage:
    python -u generate_grain_boundary_configs.py --output /workspace/results/grain_boundary.xyz
    python -u generate_grain_boundary_configs.py --dry-run
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import surface
from ase.io import write, read
from ase.spacegroup import crystal


def build_pyrite():
    return crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.385, 0.385, 0.385)],
        spacegroup=205,
        cellpar=[5.418, 5.418, 5.418, 90, 90, 90],
        primitive_cell=True,
    )


def build_pentlandite():
    return crystal(
        symbols=['Fe', 'Ni', 'S'],
        basis=[
            (0.0, 0.0, 0.0),
            (0.625, 0.625, 0.625),
            (0.25, 0.25, 0.25),
        ],
        spacegroup=225,
        cellpar=[10.07, 10.07, 10.07, 90, 90, 90],
        primitive_cell=False,
    )


def build_tilt_grain_boundary(bulk, miller, label_prefix, layers=2, vacuum=12.0):
    """Build a symmetric tilt grain boundary by mirroring a slab.

    Creates two grains: original slab + mirror image stacked along z.
    The interface between them is the grain boundary.

    Args:
        bulk: bulk crystal
        miller: surface orientation (becomes the GB plane)
        layers: number of layers per grain
        vacuum: vacuum between periodic images (≥12 Å for DFT)
    """
    configs = []

    # Build one grain as a slab (no vacuum)
    grain = surface(bulk, miller, layers=layers, vacuum=0.0)

    # Mirror grain: reflect z positions
    grain_mirror = grain.copy()
    z_max = grain_mirror.positions[:, 2].max()
    grain_mirror.positions[:, 2] = z_max - grain_mirror.positions[:, 2]

    # Stack: grain + gap + mirror_grain
    cell = grain.cell.copy()
    gap = 2.5  # Å between grains (will relax during DFT)

    # Shift mirror grain above original
    grain_mirror.positions[:, 2] += z_max + gap

    # Combine
    gb = grain.copy()
    gb += grain_mirror

    # Adjust cell height
    new_z = 2 * z_max + gap + vacuum
    cell[2, 2] = new_z
    gb.set_cell(cell)
    gb.pbc = True

    # Remove atoms too close to each other at the interface
    dists = gb.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    too_close = set()
    for i in range(len(gb)):
        for j in range(i + 1, len(gb)):
            if dists[i, j] < 1.5:
                too_close.add(j)

    if too_close:
        gb_clean = Atoms()
        gb_clean.set_cell(gb.cell)
        gb_clean.pbc = True
        for i in range(len(gb)):
            if i not in too_close:
                gb_clean += Atoms(gb[i].symbol, positions=[gb.positions[i]])
        gb_clean.set_cell(gb.cell)
        gb_clean.pbc = True
        gb = gb_clean

    configs.append((gb.copy(), f"{label_prefix}_gb_eq", False))

    # Rattled variants
    for i in range(4):
        rattled = gb.copy()
        rattled.rattle(stdev=0.05, seed=42 + hash(label_prefix) % 10000 + i)
        configs.append((rattled, f"{label_prefix}_gb_rattle_{i:02d}", False))

    return configs


def generate_all_configs():
    configs = []

    print("=" * 60, flush=True)
    print("Generating grain boundary configs (Tier 4F)", flush=True)
    print("=" * 60, flush=True)

    # Pentlandite GBs
    print("\n--- Pentlandite ---", flush=True)
    pent = build_pentlandite()

    print("  Building (111) tilt GB...", flush=True)
    cfgs = build_tilt_grain_boundary(pent, (1, 1, 1), "pent_111", layers=2)
    configs.extend(cfgs)
    print(f"    {len(cfgs)} configs ({len(cfgs[0][0])} atoms)", flush=True)

    print("  Building (001) tilt GB...", flush=True)
    cfgs = build_tilt_grain_boundary(pent, (0, 0, 1), "pent_001", layers=2)
    configs.extend(cfgs)
    print(f"    {len(cfgs)} configs ({len(cfgs[0][0])} atoms)", flush=True)

    # Pyrite GBs
    print("\n--- Pyrite ---", flush=True)
    pyr = build_pyrite()

    print("  Building (111) tilt GB...", flush=True)
    cfgs = build_tilt_grain_boundary(pyr, (1, 1, 1), "pyrite_111", layers=2)
    configs.extend(cfgs)
    print(f"    {len(cfgs)} configs ({len(cfgs[0][0])} atoms)", flush=True)

    print("  Building (100) tilt GB...", flush=True)
    cfgs = build_tilt_grain_boundary(pyr, (1, 0, 0), "pyrite_100", layers=2)
    configs.extend(cfgs)
    print(f"    {len(cfgs)} configs ({len(cfgs[0][0])} atoms)", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL GRAIN BOUNDARY CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


def run_gpaw_single_point(atoms, config_label):
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)

    if n_atoms > 80:
        kpts = (1, 1, 1)
    elif n_atoms > 40:
        kpts = (2, 2, 1)
    else:
        kpts = (2, 2, 2)

    calc = GPAW(
        mode=PW(400),
        xc='PBE',
        kpts=kpts,
        occupations=FermiDirac(0.1),
        convergence={'energy': 1e-5},
        parallel={'augment_grids': True},
        txt=None,
    )

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True) if all(atoms.pbc) else None

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
    parser = argparse.ArgumentParser(description="Generate grain boundary DFT data (Tier 4F)")
    parser.add_argument('--output', type=str, default='/workspace/results/grain_boundary.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    configs = generate_all_configs()

    if args.dry_run:
        print("\nDRY RUN — config list:")
        for atoms, label, _ in configs:
            print(f"  {label}: {len(atoms)} atoms")
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

    log_path = output_path.parent / 'grain_boundary_log.txt'

    for i, (atoms, label, _) in enumerate(remaining):
        t0 = time.time()
        try:
            results = run_gpaw_single_point(atoms, label)
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
