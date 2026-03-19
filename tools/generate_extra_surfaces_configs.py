#!/usr/bin/env python3
"""
Generate DFT training data for additional surface facets (Tier 2F).

Real crystals expose multiple facets. The base dataset only has the
primary cleavage planes. This script adds secondary surfaces.

Config breakdown (~24 configs):
  Mackinawite (100) slab + rattles:     6
  Pyrite (111) slab + rattles:          6
  Pyrite (110) slab + rattles:          6
  Pentlandite (001) slab + rattles:     6
  TOTAL:                               24

Usage:
    python -u generate_extra_surfaces_configs.py --output /workspace/results/extra_surfaces.xyz
    python -u generate_extra_surfaces_configs.py --output /workspace/results/extra_surfaces.xyz --resume
    python -u generate_extra_surfaces_configs.py --dry-run
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from ase.build import surface
from ase.io import write, read
from ase.spacegroup import crystal


def build_mackinawite():
    return crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
        primitive_cell=True,
    )


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


def make_slab_set(bulk, miller, label_prefix, layers=2, vacuum=12.0,
                  repeat=None, n_rattles=5, rattle_stdev=0.05):
    """Build a slab and generate eq + rattled variants."""
    configs = []

    slab = surface(bulk, miller, layers=layers, vacuum=vacuum)
    if repeat is not None:
        slab = slab.repeat(repeat)

    configs.append((slab.copy(), f"{label_prefix}_slab", True))

    for i in range(n_rattles):
        rattled = slab.copy()
        rattled.rattle(stdev=rattle_stdev, seed=100 + hash(label_prefix) % 10000 + i)
        configs.append((rattled, f"{label_prefix}_slab_rattle_{i:02d}", True))

    return configs


def generate_all_configs():
    configs = []

    print("=" * 60, flush=True)
    print("Generating extra surface configs (Tier 2F)", flush=True)
    print("=" * 60, flush=True)

    # Mackinawite (100) — edge face of layered structure
    print("\n--- Mackinawite (100) ---", flush=True)
    mack = build_mackinawite()
    cfgs = make_slab_set(mack, (1, 0, 0), "mack_100", repeat=(2, 2, 1))
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs ({len(cfgs[0][0])} atoms)", flush=True)

    # Pyrite (111) — octahedral termination
    print("--- Pyrite (111) ---", flush=True)
    pyr = build_pyrite()
    cfgs = make_slab_set(pyr, (1, 1, 1), "pyrite_111", repeat=(2, 2, 1))
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs ({len(cfgs[0][0])} atoms)", flush=True)

    # Pyrite (110) — edge face
    print("--- Pyrite (110) ---", flush=True)
    cfgs = make_slab_set(pyr, (1, 1, 0), "pyrite_110", repeat=(2, 2, 1))
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs ({len(cfgs[0][0])} atoms)", flush=True)

    # Pentlandite (001) — alternative to (111)
    print("--- Pentlandite (001) ---", flush=True)
    pent = build_pentlandite()
    cfgs = make_slab_set(pent, (0, 0, 1), "pent_001")
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs ({len(cfgs[0][0])} atoms)", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL EXTRA SURFACE CONFIGS: {len(configs)}", flush=True)
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
        kpts = (2, 2, 1)

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

    return {'energy': energy, 'forces': forces, 'stress': None, 'config_type': config_label}


def save_to_extxyz(atoms, results, output_path):
    atoms_copy = atoms.copy()
    atoms_copy.info['energy'] = results['energy']
    atoms_copy.info['config_type'] = results['config_type']
    atoms_copy.arrays['forces'] = results['forces']
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
    parser = argparse.ArgumentParser(description="Generate extra surface DFT data (Tier 2F)")
    parser.add_argument('--output', type=str, default='/workspace/results/extra_surfaces.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Just count configs, no DFT')
    args = parser.parse_args()

    configs = generate_all_configs()

    if args.dry_run:
        print("\nDRY RUN — config list:")
        for atoms, label, is_slab in configs:
            print(f"  {label}: {len(atoms)} atoms [SLAB]")
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

    log_path = output_path.parent / 'extra_surfaces_log.txt'

    for i, (atoms, label, is_slab) in enumerate(remaining):
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
