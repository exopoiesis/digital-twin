#!/usr/bin/env python3
"""
Generate DFT training data for pentlandite Fe-Ni composition series (Tier 3B).

Tetzlaff 2021 showed Fe3Ni6S8 is optimal for electrocatalysis.
We need training data across the composition range.

Base dataset already has Fe4.5Ni4.5S8 (equal Fe:Ni).
This script adds Fe-rich and Ni-rich endpoints.

Config breakdown (~30 configs):
  Fe3Ni6S8 (Ni-rich, Tetzlaff optimum):
    Bulk eq + rattles + strains:      10
    (111) slab + H ads:                5
  Fe6Ni3S8 (Fe-rich):
    Bulk eq + rattles + strains:      10
    (111) slab + H ads:                5
  TOTAL:                              30

Usage:
    python -u generate_pent_composition_configs.py --output /workspace/results/pent_composition.xyz
    python -u generate_pent_composition_configs.py --dry-run
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


def build_pentlandite_conventional():
    """Build pentlandite conventional cell (68 atoms) with equal Fe:Ni."""
    return crystal(
        symbols=['Fe', 'Ni', 'S'],
        basis=[
            (0.0, 0.0, 0.0),       # 4a
            (0.625, 0.625, 0.625),  # 8c
            (0.25, 0.25, 0.25),     # 8c: S
        ],
        spacegroup=225,
        cellpar=[10.07, 10.07, 10.07, 90, 90, 90],
        primitive_cell=False,
    )


def swap_composition(atoms, target_fe_frac, seed=42):
    """Swap Fe/Ni atoms to achieve target Fe fraction among metals.

    target_fe_frac: fraction of metal sites that should be Fe.
    In (Fe,Ni)9S8: 9 metal sites per formula unit.
    For Fe3Ni6: target_fe_frac = 3/9 = 0.333
    For Fe6Ni3: target_fe_frac = 6/9 = 0.667
    """
    result = atoms.copy()
    syms = list(result.get_chemical_symbols())
    metal_indices = [i for i, s in enumerate(syms) if s in ('Fe', 'Ni')]

    n_metals = len(metal_indices)
    n_fe_target = int(round(target_fe_frac * n_metals))
    n_ni_target = n_metals - n_fe_target

    rng = np.random.RandomState(seed)
    rng.shuffle(metal_indices)

    for i, idx in enumerate(metal_indices):
        if i < n_fe_target:
            syms[idx] = 'Fe'
        else:
            syms[idx] = 'Ni'

    result.set_chemical_symbols(syms)
    return result


def generate_composition_set(fe_frac, label_prefix, seed_base=0):
    """Generate bulk + slab + H adsorption configs for a given composition."""
    configs = []

    # Build and swap
    conv = build_pentlandite_conventional()
    bulk = swap_composition(conv, fe_frac, seed=seed_base)

    n_fe = sum(1 for s in bulk.get_chemical_symbols() if s == 'Fe')
    n_ni = sum(1 for s in bulk.get_chemical_symbols() if s == 'Ni')
    n_s = sum(1 for s in bulk.get_chemical_symbols() if s == 'S')
    print(f"  Composition: Fe{n_fe}Ni{n_ni}S{n_s} ({len(bulk)} atoms)", flush=True)

    # Use primitive-like cell for bulk (conventional is too big for many configs)
    # Actually, keep conventional for consistency with adsorption
    prim = crystal(
        symbols=['Fe', 'Ni', 'S'],
        basis=[
            (0.0, 0.0, 0.0),
            (0.625, 0.625, 0.625),
            (0.25, 0.25, 0.25),
        ],
        spacegroup=225,
        cellpar=[10.07, 10.07, 10.07, 90, 90, 90],
        primitive_cell=True,
    )
    prim = swap_composition(prim, fe_frac, seed=seed_base + 100)

    # Bulk eq
    configs.append((prim.copy(), f"{label_prefix}_bulk_eq", False))

    # Rattles
    for i, stdev in enumerate([0.03, 0.05, 0.10]):
        rattled = prim.copy()
        rattled.rattle(stdev=stdev, seed=seed_base + 200 + i)
        configs.append((rattled, f"{label_prefix}_bulk_rattle_{stdev:.2f}", False))

    # Strains
    for strain_pct in [2, 4, -2, -4]:
        strained = prim.copy()
        factor = 1.0 + strain_pct / 100.0
        strained.set_cell(strained.cell * factor, scale_atoms=True)
        configs.append((strained, f"{label_prefix}_bulk_strain_{strain_pct:+d}pct", False))

    # Shears
    for i, (c1, c2) in enumerate([(0, 1), (0, 2)]):
        sheared = prim.copy()
        cell = sheared.cell.copy()
        cell[c1, c2] += 0.02 * np.linalg.norm(cell[c1])
        sheared.set_cell(cell, scale_atoms=True)
        configs.append((sheared, f"{label_prefix}_bulk_shear_{i:02d}", False))

    # (111) slab
    slab = surface(bulk, (1, 1, 1), layers=2, vacuum=12.0)
    configs.append((slab.copy(), f"{label_prefix}_111_slab", True))

    # Slab rattle
    rattled = slab.copy()
    rattled.rattle(stdev=0.05, seed=seed_base + 300)
    configs.append((rattled, f"{label_prefix}_111_slab_rattle", True))

    # H adsorption on (111)
    syms = np.array(slab.get_chemical_symbols())
    fe_mask = syms == 'Fe'
    ni_mask = syms == 'Ni'

    fe_pos = slab.positions[fe_mask]
    ni_pos = slab.positions[ni_mask]

    if len(fe_pos) > 0:
        top_fe = fe_pos[np.argmax(fe_pos[:, 2])]
        s_h = slab.copy()
        s_h += Atoms('H', positions=[top_fe + [0, 0, 1.6]])
        configs.append((s_h, f"{label_prefix}_111_H_ontop_Fe", True))

    if len(ni_pos) > 0:
        top_ni = ni_pos[np.argmax(ni_pos[:, 2])]
        s_h = slab.copy()
        s_h += Atoms('H', positions=[top_ni + [0, 0, 1.6]])
        configs.append((s_h, f"{label_prefix}_111_H_ontop_Ni", True))

    if len(fe_pos) > 0 and len(ni_pos) > 0:
        h_bridge = (top_fe + top_ni) / 2 + [0, 0, 1.8]
        dists = np.linalg.norm(slab.positions - h_bridge, axis=1)
        if np.min(dists) < 1.0:
            h_bridge[2] += 0.5
        s_h = slab.copy()
        s_h += Atoms('H', positions=[h_bridge])
        configs.append((s_h, f"{label_prefix}_111_H_bridge_FeNi", True))

    return configs


def generate_all_configs():
    configs = []

    print("=" * 60, flush=True)
    print("Generating pentlandite composition series (Tier 3B)", flush=True)
    print("=" * 60, flush=True)

    # Fe3Ni6S8 — Ni-rich (Tetzlaff optimum)
    print("\n--- Fe3Ni6S8 (Ni-rich) ---", flush=True)
    cfgs = generate_composition_set(3/9, "pent_Fe3Ni6", seed_base=1000)
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs", flush=True)

    # Fe6Ni3S8 — Fe-rich
    print("\n--- Fe6Ni3S8 (Fe-rich) ---", flush=True)
    cfgs = generate_composition_set(6/9, "pent_Fe6Ni3", seed_base=2000)
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL COMPOSITION CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


def run_gpaw_single_point(atoms, config_label, is_slab=False):
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)

    if is_slab:
        kpts = (1, 1, 1) if n_atoms > 60 else (2, 2, 1)
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
        parallel={'augment_grids': True},
        txt=None,
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
    parser = argparse.ArgumentParser(description="Generate pentlandite composition DFT data (Tier 3B)")
    parser.add_argument('--output', type=str, default='/workspace/results/pent_composition.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    configs = generate_all_configs()

    if args.dry_run:
        print("\nDRY RUN — config list:")
        for atoms, label, is_slab in configs:
            slab_tag = " [SLAB]" if is_slab else ""
            elems = {}
            for s in atoms.get_chemical_symbols():
                elems[s] = elems.get(s, 0) + 1
            elem_str = " ".join(f"{k}{v}" for k, v in sorted(elems.items()))
            print(f"  {label}: {len(atoms)} atoms ({elem_str}){slab_tag}")
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

    log_path = output_path.parent / 'pent_composition_log.txt'

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
