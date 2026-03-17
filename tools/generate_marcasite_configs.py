#!/usr/bin/env python3
"""
Generate DFT training data for marcasite FeS2 (Tier 3D).

Marcasite is the orthorhombic polymorph of pyrite FeS2.
Structure: Pnnm (#58), a=4.443 Å, b=5.425 Å, c=3.387 Å
Fe at 2a: (0, 0, 0)
S  at 4g: (0.200, 0.378, 0)

Primitive cell: 6 atoms (2 Fe + 4 S)

Config breakdown (~35 configs):
  Bulk eq:                         1
  Bulk rattles (σ=0.03-0.20):     15
  Bulk strains (±1-5%):           10
  (010) slab + rattles:            6
  H adsorption (3 sites):          3
  TOTAL:                         ~35

Usage:
    python -u generate_marcasite_configs.py --output /workspace/results/marcasite_train.xyz
    python -u generate_marcasite_configs.py --output /workspace/results/marcasite_train.xyz --resume
    python -u generate_marcasite_configs.py --dry-run
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


def build_marcasite():
    """Build marcasite FeS2 primitive cell (Pnnm, #58, orthorhombic).

    Structure:
      a = 4.443 Å, b = 5.425 Å, c = 3.387 Å
      Fe at 2a: (0, 0, 0)
      S  at 4g: (0.200, 0.378, 0)

    Primitive cell: 6 atoms (2 Fe + 4 S)
    """
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[
            (0.0, 0.0, 0.0),       # 2a Fe
            (0.200, 0.378, 0.0),   # 4g S
        ],
        spacegroup=58,
        cellpar=[4.443, 5.425, 3.387, 90, 90, 90],
        primitive_cell=True,
    )
    return atoms


def rattle_atoms(atoms, stdev, label_prefix, count=3):
    """Generate rattled structures."""
    configs = []
    for i in range(count):
        rattled = atoms.copy()
        rattled.rattle(stdev=stdev, seed=42 + int(stdev * 100) + i)
        configs.append((rattled, f"{label_prefix}_rattle_{stdev:.2f}_{i:02d}"))
    return configs


def strain_atoms(atoms, label_prefix):
    """Generate volumetrically strained structures."""
    configs = []
    for strain_pct in [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]:
        strained = atoms.copy()
        factor = 1.0 + strain_pct / 100.0
        strained.set_cell(strained.cell * factor, scale_atoms=True)
        configs.append((strained, f"{label_prefix}_strain_{strain_pct:+d}pct"))
    return configs


def safe_place_h(base_positions, h_target, min_dist=1.0, max_raise=2.0):
    """Safely place H atom, raising it if too close to any atom."""
    h_pos = h_target.copy()
    dists = np.linalg.norm(base_positions - h_pos, axis=1)

    if np.min(dists) < min_dist:
        raise_z = 0.0
        while np.min(dists) < min_dist and raise_z < max_raise:
            raise_z += 0.2
            h_pos[2] = h_target[2] + raise_z
            dists = np.linalg.norm(base_positions - h_pos, axis=1)

    return h_pos


def build_marcasite_surface_010():
    """Build marcasite (010) slab + rattled variants.

    (010) is the b-axis normal surface, common for layered structures.
    """
    marc = build_marcasite()
    # Repeat 2x2x1 to get adequate slab size
    slab = surface(marc, (0, 1, 0), layers=2, vacuum=12.0)
    slab = slab.repeat((2, 1, 2))

    configs = [(slab.copy(), "marcasite_010_slab", True)]
    for i in range(5):
        rattled = slab.copy()
        rattled.rattle(stdev=0.05, seed=500 + i)
        configs.append((rattled, f"marcasite_010_slab_rattle_{i:02d}", True))

    return configs


def build_marcasite_H_adsorption():
    """Build marcasite (010) surface with H at different adsorption sites."""
    marc = build_marcasite()
    slab = surface(marc, (0, 1, 0), layers=2, vacuum=12.0)
    slab = slab.repeat((2, 1, 2))

    configs = []

    syms = np.array(slab.get_chemical_symbols())
    fe_mask = syms == 'Fe'
    s_mask = syms == 'S'

    fe_pos = slab.positions[fe_mask]
    s_pos = slab.positions[s_mask]

    if len(fe_pos) == 0 or len(s_pos) == 0:
        print("Warning: Could not find Fe or S atoms in slab for H adsorption")
        return configs

    # Find topmost Fe and S
    top_fe = fe_pos[np.argmax(fe_pos[:, 1])]  # y-axis is normal for (010)
    top_s = s_pos[np.argmax(s_pos[:, 1])]

    # Site 1: H on-top Fe
    s1 = slab.copy()
    h_ontop_fe = top_fe.copy()
    h_ontop_fe[1] += 1.5  # y-axis normal
    h_ontop_fe = safe_place_h(slab.positions, h_ontop_fe, min_dist=1.0)
    s1 += Atoms('H', positions=[h_ontop_fe])
    configs.append((s1, "marcasite_010_H_ontop_Fe", True))

    # Site 2: H on-top S
    s2 = slab.copy()
    h_ontop_s = top_s.copy()
    h_ontop_s[1] += 1.4  # y-axis normal
    h_ontop_s = safe_place_h(slab.positions, h_ontop_s, min_dist=1.0)
    s2 += Atoms('H', positions=[h_ontop_s])
    configs.append((s2, "marcasite_010_H_ontop_S", True))

    # Site 3: H bridge Fe-S
    s3 = slab.copy()
    h_bridge = (top_fe + top_s) / 2
    h_bridge[1] += 1.2  # y-axis normal
    h_bridge = safe_place_h(slab.positions, h_bridge, min_dist=1.0)
    s3 += Atoms('H', positions=[h_bridge])
    configs.append((s3, "marcasite_010_H_bridge_FeS", True))

    return configs


def generate_all_configs():
    """Generate all marcasite configurations."""
    configs = []  # List of (atoms, label, is_slab)

    print("=" * 60, flush=True)
    print("Generating marcasite FeS2 configs (Tier 3D)", flush=True)
    print("=" * 60, flush=True)

    marc = build_marcasite()
    print(f"  Primitive cell: {len(marc)} atoms", flush=True)
    print(f"  Cell parameters: a={marc.cell[0,0]:.3f}, b={marc.cell[1,1]:.3f}, c={marc.cell[2,2]:.3f} Å", flush=True)

    # Bulk equilibrium
    configs.append((marc.copy(), "marcasite_bulk_eq", False))

    # Rattles
    for stdev in [0.03, 0.05, 0.08, 0.10, 0.20]:
        configs.extend([(a, l, False) for a, l in rattle_atoms(marc, stdev, "marcasite_bulk", 3)])

    # Strains
    configs.extend([(a, l, False) for a, l in strain_atoms(marc, "marcasite_bulk")])

    n_bulk = len(configs)
    print(f"  Bulk configs: {n_bulk}", flush=True)

    # (010) surface
    print("\n  Building (010) surface...", flush=True)
    surf_configs = build_marcasite_surface_010()
    configs.extend([(a, l, s) for a, l, s in surf_configs])
    if surf_configs:
        print(f"  (010) slab: {len(surf_configs[0][0])} atoms", flush=True)

    # H adsorption
    print("  Building H adsorption configs...", flush=True)
    h_configs = build_marcasite_H_adsorption()
    configs.extend([(a, l, s) for a, l, s in h_configs])

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL MARCASITE CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


def run_gpaw_single_point(atoms, config_label, is_slab=False):
    """Run GPAW single-point calculation.

    Settings:
      - PW(400) for slabs, PW(500) for small bulk
      - PBE functional
      - kpts: (4,4,4) for 6-atom bulk, (2,2,1) for slabs
      - FermiDirac(0.1)
      - convergence energy 1e-5
    """
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)
    mode = PW(400) if is_slab or n_atoms > 30 else PW(500)

    if is_slab:
        kpts = (2, 2, 1)
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

    return {
        'energy': energy,
        'forces': forces,
        'stress': stress,
        'config_type': config_label
    }


def save_to_extxyz(atoms, results, output_path):
    """Save to extended XYZ (append mode)."""
    atoms_copy = atoms.copy()
    atoms_copy.info['energy'] = results['energy']
    atoms_copy.info['config_type'] = results['config_type']
    atoms_copy.arrays['forces'] = results['forces']
    if results['stress'] is not None:
        atoms_copy.info['stress'] = results['stress']
    write(output_path, atoms_copy, format='extxyz', append=True)


def load_existing_labels(output_path):
    """Load already computed config labels for resume functionality."""
    if not Path(output_path).exists():
        return set()
    try:
        all_atoms = read(output_path, index=':', format='extxyz')
        return {a.info.get('config_type', '') for a in all_atoms}
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(description="Generate marcasite DFT training data (Tier 3D)")
    parser.add_argument('--output', type=str, default='/workspace/results/marcasite_train.xyz')
    parser.add_argument('--resume', action='store_true', help='Skip already computed configs')
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

    log_path = output_path.parent / 'marcasite_log.txt'

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

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print(f"Done. Output: {output_path}", flush=True)
    if output_path.exists():
        final = read(output_path, index=':', format='extxyz')
        print(f"Total configs in file: {len(final)}", flush=True)


if __name__ == '__main__':
    main()
