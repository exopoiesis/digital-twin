#!/usr/bin/env python3
"""
Generate DFT training data for violarite FeNi2S4 (Tier 2B).

Violarite is an inverse thiospinel (Fd-3m, #227) — the oxidation product
of pentlandite. Active for OER. Same structure type as greigite, but with
Ni on octahedral sites.

Structure: Fd-3m inverse spinel
  8a  (1/8, 1/8, 1/8):   Fe2+ (tetrahedral A-site)
  16d (1/2, 1/2, 1/2):   Ni3+ (octahedral B-site)
  32e (u, u, u), u≈0.254: S2-

Config breakdown (~48 configs):
  Bulk eq + rattles (σ=0.03-0.20):   16
  Bulk strains (±1-5%):              10
  2x2x2 supercell rattles:            5
  (001) slab + rattles:                6
  (111) slab + rattles:                6
  H adsorption (3 sites × 2 surf):    6
  TOTAL:                             ~49

Usage:
    python -u generate_violarite_configs.py --output /workspace/results/violarite_train.xyz
    python -u generate_violarite_configs.py --output /workspace/results/violarite_train.xyz --resume
    python -u generate_violarite_configs.py --dry-run
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


def build_violarite():
    """Build violarite FeNi2S4 primitive cell (Fd-3m, #227).

    Same structure as greigite but Fe on 8a, Ni on 16d.
    a = 9.464 Å (experimental).
    """
    atoms = crystal(
        symbols=['Fe', 'Ni', 'S'],
        basis=[
            (0.125, 0.125, 0.125),   # 8a tetrahedral Fe
            (0.5, 0.5, 0.5),         # 16d octahedral Ni
            (0.254, 0.254, 0.254),   # 32e sulfur
        ],
        spacegroup=227,
        cellpar=[9.464, 9.464, 9.464, 90, 90, 90],
        primitive_cell=True,
    )
    return atoms


def build_violarite_conventional():
    """Build full conventional cell (56 atoms)."""
    atoms = crystal(
        symbols=['Fe', 'Ni', 'S'],
        basis=[
            (0.125, 0.125, 0.125),
            (0.5, 0.5, 0.5),
            (0.254, 0.254, 0.254),
        ],
        spacegroup=227,
        cellpar=[9.464, 9.464, 9.464, 90, 90, 90],
        primitive_cell=False,
    )
    return atoms


def rattle_atoms(atoms, stdev, label_prefix, count=5):
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
    print("Generating violarite FeNi2S4 configs (Tier 2B)", flush=True)
    print("=" * 60, flush=True)

    viol = build_violarite()
    viol_conv = build_violarite_conventional()
    print(f"  Primitive cell: {len(viol)} atoms", flush=True)
    print(f"  Conventional cell: {len(viol_conv)} atoms", flush=True)

    # Bulk equilibrium
    configs.append((viol.copy(), "violarite_bulk_eq", False))

    # Rattles
    for stdev in [0.03, 0.05, 0.08, 0.10, 0.20]:
        n = 3
        configs.extend([(a, l, False) for a, l in rattle_atoms(viol, stdev, "violarite_bulk", n)])

    # Strains
    configs.extend([(a, l, False) for a, l in strain_atoms(viol, "violarite_bulk")])

    n_bulk = len(configs)
    print(f"  Bulk configs: {n_bulk}", flush=True)

    # Supercell rattles (conventional cell = ~56 atoms)
    for i in range(5):
        rattled = viol_conv.copy()
        rattled.rattle(stdev=0.05, seed=900 + i)
        configs.append((rattled, f"violarite_conv_rattle_0.05_{i:02d}", False))

    # (001) slab
    print("\n  Building (001) surface...", flush=True)
    slab_001 = surface(viol_conv, (0, 0, 1), layers=2, vacuum=12.0)
    configs.append((slab_001.copy(), "violarite_001_slab", True))
    for i in range(5):
        rattled = slab_001.copy()
        rattled.rattle(stdev=0.05, seed=500 + i)
        configs.append((rattled, f"violarite_001_slab_rattle_{i:02d}", True))
    print(f"  (001) slab: {len(slab_001)} atoms", flush=True)

    # (111) slab
    print("  Building (111) surface...", flush=True)
    slab_111 = surface(viol_conv, (1, 1, 1), layers=2, vacuum=12.0)
    configs.append((slab_111.copy(), "violarite_111_slab", True))
    for i in range(5):
        rattled = slab_111.copy()
        rattled.rattle(stdev=0.05, seed=600 + i)
        configs.append((rattled, f"violarite_111_slab_rattle_{i:02d}", True))
    print(f"  (111) slab: {len(slab_111)} atoms", flush=True)

    # H adsorption on both surfaces
    print("\n  Building H adsorption configs...", flush=True)
    for slab, miller_str, seed_base in [(slab_001, '001', 700), (slab_111, '111', 800)]:
        syms = np.array(slab.get_chemical_symbols())
        fe_mask = syms == 'Fe'
        ni_mask = syms == 'Ni'
        s_mask = syms == 'S'
        metal_mask = fe_mask | ni_mask

        metal_pos = slab.positions[metal_mask]
        s_pos = slab.positions[s_mask]
        ni_pos = slab.positions[ni_mask]

        if len(metal_pos) == 0 or len(s_pos) == 0:
            continue

        top_metal = metal_pos[np.argmax(metal_pos[:, 2])]
        top_s = s_pos[np.argmax(s_pos[:, 2])]

        # On-top metal
        s1 = slab.copy()
        s1 += Atoms('H', positions=[top_metal + [0, 0, 1.6]])
        configs.append((s1, f"violarite_{miller_str}_H_ontop_metal", True))

        # Bridge metal-S
        h_bridge = (top_metal + top_s) / 2 + [0, 0, 1.8]
        dists = np.linalg.norm(slab.positions - h_bridge, axis=1)
        if np.min(dists) < 1.0:
            h_bridge[2] += 0.5
        s2 = slab.copy()
        s2 += Atoms('H', positions=[h_bridge])
        configs.append((s2, f"violarite_{miller_str}_H_bridge", True))

        # On-top Ni (if present)
        if len(ni_pos) > 0:
            top_ni = ni_pos[np.argmax(ni_pos[:, 2])]
            s3 = slab.copy()
            s3 += Atoms('H', positions=[top_ni + [0, 0, 1.6]])
            configs.append((s3, f"violarite_{miller_str}_H_ontop_Ni", True))

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL VIOLARITE CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


def run_gpaw_single_point(atoms, config_label, is_slab=False):
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)
    mode = PW(400) if is_slab or n_atoms > 30 else PW(500)

    if is_slab:
        kpts = (1, 1, 1) if n_atoms > 60 else (2, 2, 1)
    elif n_atoms > 30:
        kpts = (2, 2, 2)
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
    parser = argparse.ArgumentParser(description="Generate violarite FeNi2S4 DFT training data (Tier 2B)")
    parser.add_argument('--output', type=str, default='/workspace/results/violarite_train.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Just count configs, no DFT')
    args = parser.parse_args()

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

    log_path = output_path.parent / 'violarite_log.txt'

    for i, (atoms, label, is_slab) in enumerate(remaining):
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
