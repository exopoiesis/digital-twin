#!/usr/bin/env python3
"""
Generate DFT training data for millerite NiS (Tier 2C).

Millerite is a pure nickel sulfide important for serpentinization
(Ni released from olivine). Rhombohedral structure R3m (#160).

Structure: each Ni is square-pyramidal coordinated by 5 S.
a = 9.616 Å, c = 3.143 Å (hexagonal setting).
Ni at 9b: (0.083, 0.917, 0.25) [approx]
S  at 9b: (0.375, 0.625, 0.25) [approx]

Config breakdown (~39 configs):
  Bulk eq + rattles (σ=0.03-0.20):   16
  Bulk strains (±1-5%):              10
  (001) slab + rattles:                5
  (100) slab + rattles:                5
  H adsorption (4 sites):             4
  TOTAL:                             ~40

Usage:
    python -u generate_millerite_configs.py --output /workspace/results/millerite_train.xyz
    python -u generate_millerite_configs.py --output /workspace/results/millerite_train.xyz --resume
    python -u generate_millerite_configs.py --dry-run
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
from gpaw_checkpoint import register_sigterm_handler, is_shutdown_requested


def build_millerite():
    """Build millerite NiS (R3m, #160, hexagonal setting).

    Experimental: a = 9.616 Å, c = 3.143 Å
    Ni at (0.0829, 0.9171, 0.25) — 9b site
    S  at (0.3748, 0.6252, 0.25) — 9b site

    Hexagonal cell contains 9 NiS formula units = 18 atoms.
    """
    atoms = crystal(
        symbols=['Ni', 'S'],
        basis=[
            (0.0829, 0.9171, 0.25),
            (0.3748, 0.6252, 0.25),
        ],
        spacegroup=160,
        cellpar=[9.616, 9.616, 3.143, 90, 90, 120],
        primitive_cell=False,  # keep hexagonal cell (18 atoms)
    )
    return atoms


def build_millerite_primitive():
    """Build primitive millerite cell (6 atoms, rhombohedral)."""
    atoms = crystal(
        symbols=['Ni', 'S'],
        basis=[
            (0.0829, 0.9171, 0.25),
            (0.3748, 0.6252, 0.25),
        ],
        spacegroup=160,
        cellpar=[9.616, 9.616, 3.143, 90, 90, 120],
        primitive_cell=True,
    )
    return atoms


def rattle_atoms(atoms, stdev, label_prefix, count=3):
    configs = []
    for i in range(count):
        rattled = atoms.copy()
        rattled.rattle(stdev=stdev, seed=42 + int(stdev * 100) + i)
        configs.append((rattled, f"{label_prefix}_rattle_{stdev:.2f}_{i:02d}"))
    return configs


def strain_atoms(atoms, label_prefix):
    configs = []
    for strain_pct in [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]:
        strained = atoms.copy()
        factor = 1.0 + strain_pct / 100.0
        strained.set_cell(strained.cell * factor, scale_atoms=True)
        configs.append((strained, f"{label_prefix}_strain_{strain_pct:+d}pct"))
    return configs


def generate_all_configs():
    configs = []

    print("=" * 60, flush=True)
    print("Generating millerite NiS configs (Tier 2C)", flush=True)
    print("=" * 60, flush=True)

    mill = build_millerite()
    mill_prim = build_millerite_primitive()
    print(f"  Hexagonal cell: {len(mill)} atoms", flush=True)
    print(f"  Primitive cell: {len(mill_prim)} atoms", flush=True)

    # Use primitive cell for bulk configs (smaller, faster)
    bulk = mill_prim

    # Bulk equilibrium
    configs.append((bulk.copy(), "millerite_bulk_eq", False))

    # Rattles
    for stdev in [0.03, 0.05, 0.08, 0.10, 0.20]:
        configs.extend([(a, l, False) for a, l in rattle_atoms(bulk, stdev, "millerite_bulk", 3)])

    # Strains
    configs.extend([(a, l, False) for a, l in strain_atoms(bulk, "millerite_bulk")])

    n_bulk = len(configs)
    print(f"  Bulk configs: {n_bulk}", flush=True)

    # (001) slab — basal plane, layered termination
    print("\n  Building (001) surface...", flush=True)
    slab_001 = surface(mill, (0, 0, 1), layers=2, vacuum=12.0)
    configs.append((slab_001.copy(), "millerite_001_slab", True))
    for i in range(4):
        rattled = slab_001.copy()
        rattled.rattle(stdev=0.05, seed=500 + i)
        configs.append((rattled, f"millerite_001_slab_rattle_{i:02d}", True))
    print(f"  (001) slab: {len(slab_001)} atoms", flush=True)

    # (100) slab — side face
    print("  Building (100) surface...", flush=True)
    slab_100 = surface(mill, (1, 0, 0), layers=2, vacuum=12.0)
    configs.append((slab_100.copy(), "millerite_100_slab", True))
    for i in range(4):
        rattled = slab_100.copy()
        rattled.rattle(stdev=0.05, seed=600 + i)
        configs.append((rattled, f"millerite_100_slab_rattle_{i:02d}", True))
    print(f"  (100) slab: {len(slab_100)} atoms", flush=True)

    # H adsorption on (001)
    print("\n  Building H adsorption configs on (001)...", flush=True)
    syms = np.array(slab_001.get_chemical_symbols())
    ni_mask = syms == 'Ni'
    s_mask = syms == 'S'
    ni_pos = slab_001.positions[ni_mask]
    s_pos = slab_001.positions[s_mask]

    if len(ni_pos) > 0 and len(s_pos) > 0:
        top_ni = ni_pos[np.argmax(ni_pos[:, 2])]
        top_s = s_pos[np.argmax(s_pos[:, 2])]

        # On-top Ni
        s1 = slab_001.copy()
        s1 += Atoms('H', positions=[top_ni + [0, 0, 1.6]])
        configs.append((s1, "millerite_001_H_ontop_Ni", True))

        # On-top S
        s2 = slab_001.copy()
        s2 += Atoms('H', positions=[top_s + [0, 0, 1.5]])
        configs.append((s2, "millerite_001_H_ontop_S", True))

        # Bridge Ni-S
        h_bridge = (top_ni + top_s) / 2 + [0, 0, 1.8]
        dists = np.linalg.norm(slab_001.positions - h_bridge, axis=1)
        if np.min(dists) < 1.0:
            h_bridge[2] += 0.5
        s3 = slab_001.copy()
        s3 += Atoms('H', positions=[h_bridge])
        configs.append((s3, "millerite_001_H_bridge_NiS", True))

        # Hollow
        ni_dists = np.linalg.norm(ni_pos - top_ni, axis=1)
        ni_dists[np.argmax(ni_pos[:, 2])] = np.inf
        second_ni = ni_pos[np.argmin(ni_dists)]
        h_hollow = (top_ni + top_s + second_ni) / 3 + [0, 0, 1.5]
        dists = np.linalg.norm(slab_001.positions - h_hollow, axis=1)
        if np.min(dists) < 1.0:
            h_hollow[2] += 0.5
        s4 = slab_001.copy()
        s4 += Atoms('H', positions=[h_hollow])
        configs.append((s4, "millerite_001_H_hollow", True))

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL MILLERITE CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


def run_gpaw_single_point(atoms, config_label, is_slab=False):
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)
    mode = PW(400) if is_slab or n_atoms > 20 else PW(500)

    if is_slab:
        kpts = (1, 1, 1) if n_atoms > 60 else (2, 2, 1)
    elif n_atoms > 30:
        kpts = (2, 2, 2)
    elif n_atoms > 12:
        kpts = (3, 3, 3)
    else:
        kpts = (4, 4, 4)

    calc = GPAW(
        mode=mode,
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
    stress = None
    if all(atoms.pbc):
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
    parser = argparse.ArgumentParser(description="Generate millerite NiS DFT training data (Tier 2C)")
    parser.add_argument('--output', type=str, default='/workspace/results/millerite_train.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Just count configs, no DFT')
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

    log_path = output_path.parent / 'millerite_log.txt'

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
