#!/usr/bin/env python3
"""
Generate DFT training data for H2O adsorption on sulfide surfaces (Tier 2D).

Water is the real solvent for all sulfide applications. Training with explicit
H2O molecules teaches the ML potential about mineral-water interactions.

Config breakdown (~28 configs):
  Mackinawite (001) + 1-3 H2O:     6
  Greigite (001) + 1-3 H2O:        6
  Pyrite (100) + 1-3 H2O:          6
  Pentlandite (111) + 1-3 H2O:     6
  Pyrrhotite (001) + 1-2 H2O:      4
  TOTAL:                           28

Usage:
    python -u generate_h2o_adsorption_configs.py --output /workspace/results/h2o_ads_train.xyz
    python -u generate_h2o_adsorption_configs.py --output /workspace/results/h2o_ads_train.xyz --resume
    python -u generate_h2o_adsorption_configs.py --dry-run
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from ase.build import surface, molecule
from ase.io import write, read
from ase.spacegroup import crystal


# ===========================================================================
#  Mineral builders
# ===========================================================================

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


def build_greigite_conventional():
    return crystal(
        symbols=['Fe', 'Fe', 'S'],
        basis=[
            (0.125, 0.125, 0.125),
            (0.5, 0.5, 0.5),
            (0.254, 0.254, 0.254),
        ],
        spacegroup=227,
        cellpar=[9.876, 9.876, 9.876, 90, 90, 90],
        primitive_cell=False,
    )


def build_pyrrhotite():
    """Build pyrrhotite Fe7S8 (simplified: troilite 2x2x2 − 2 Fe)."""
    troilite = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (1/3, 2/3, 0.25)],
        spacegroup=194,
        cellpar=[3.446, 3.446, 5.877, 90, 90, 120],
        primitive_cell=True,
    )
    supercell = troilite.repeat((2, 2, 2))
    syms = np.array(supercell.get_chemical_symbols())
    fe_indices = np.where(syms == 'Fe')[0]
    fe_z = supercell.positions[fe_indices, 2]
    z_median = np.median(fe_z)
    layer_lo = fe_indices[fe_z < z_median]
    layer_hi = fe_indices[fe_z >= z_median]
    rng = np.random.RandomState(42)
    remove_lo = rng.choice(layer_lo)
    lo_pos = supercell.positions[remove_lo, :2]
    hi_dists = np.linalg.norm(supercell.positions[layer_hi, :2] - lo_pos, axis=1)
    remove_hi = layer_hi[np.argmax(hi_dists)]
    pyrrhotite = supercell.copy()
    for idx in sorted([remove_lo, remove_hi], reverse=True):
        del pyrrhotite[idx]
    return pyrrhotite


# ===========================================================================
#  H2O placement
# ===========================================================================

def make_h2o():
    """Build H2O molecule from ASE database."""
    water = molecule('H2O')
    water.positions -= water.get_center_of_mass()
    return water


def get_top_metal(slab):
    """Find topmost metal atom position on slab."""
    syms = np.array(slab.get_chemical_symbols())
    metal_mask = (syms == 'Fe') | (syms == 'Ni')
    metal_pos = slab.positions[metal_mask]
    if len(metal_pos) == 0:
        return None
    return metal_pos[np.argmax(metal_pos[:, 2])]


def place_h2o_cluster(slab, n_water, site_pos, height=2.8, spread=2.5, seed=42):
    """Place 1-3 H2O molecules above a surface site.

    First H2O goes directly above site. Additional H2O are offset laterally.
    """
    result = slab.copy()
    rng = np.random.RandomState(seed)

    for w in range(n_water):
        h2o = make_h2o()

        # Random rotation
        angle = rng.uniform(0, 360)
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        h2o.rotate(angle, axis)

        if w == 0:
            offset = np.array([0.0, 0.0, height])
        else:
            # Lateral offset for additional waters
            dx = rng.uniform(-spread, spread)
            dy = rng.uniform(-spread, spread)
            dz = rng.uniform(0, 1.5)
            offset = np.array([dx, dy, height + dz])

        h2o.positions += site_pos + offset

        # Safety: check no atom overlap
        for _ in range(10):
            all_pos = result.positions
            h2o_center = h2o.get_center_of_mass()
            dists = np.linalg.norm(all_pos - h2o_center, axis=1)
            if np.min(dists) >= 1.5:
                break
            h2o.positions[:, 2] += 0.5

        result += h2o

    return result


def generate_h2o_on_mineral(bulk, miller, label_prefix, layers=2,
                            vacuum=15.0, repeat=None, n_waters=[1, 2, 3]):
    """Generate H2O adsorption configs for a given mineral surface."""
    configs = []
    slab = surface(bulk, miller, layers=layers, vacuum=vacuum)
    if repeat is not None:
        slab = slab.repeat(repeat)

    top = get_top_metal(slab)
    if top is None:
        return configs

    for n_w in n_waters:
        for orient_seed in [42, 137]:
            s = place_h2o_cluster(slab, n_w, top, height=2.8, spread=2.5, seed=orient_seed)
            label = f"{label_prefix}_{n_w}H2O_s{orient_seed}"
            configs.append((s, label))

    return configs


def generate_all_configs():
    configs = []

    print("=" * 60, flush=True)
    print("Generating H2O adsorption configs (Tier 2D)", flush=True)
    print("=" * 60, flush=True)

    # Mackinawite (001) + 1-3 H2O
    print("\n--- Mackinawite (001) ---", flush=True)
    mack = build_mackinawite()
    cfgs = generate_h2o_on_mineral(mack, (0, 0, 1), "mack_001",
                                   repeat=(2, 2, 1))
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs", flush=True)

    # Greigite (001) + 1-3 H2O
    print("--- Greigite (001) ---", flush=True)
    greig = build_greigite_conventional()
    cfgs = generate_h2o_on_mineral(greig, (0, 0, 1), "greigite_001")
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs", flush=True)

    # Pyrite (100) + 1-3 H2O
    print("--- Pyrite (100) ---", flush=True)
    pyr = build_pyrite()
    cfgs = generate_h2o_on_mineral(pyr, (1, 0, 0), "pyrite_100",
                                   repeat=(2, 2, 1))
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs", flush=True)

    # Pentlandite (111) + 1-3 H2O
    print("--- Pentlandite (111) ---", flush=True)
    pent = build_pentlandite()
    cfgs = generate_h2o_on_mineral(pent, (1, 1, 1), "pent_111")
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs", flush=True)

    # Pyrrhotite (001) + 1-2 H2O (smaller set)
    print("--- Pyrrhotite (001) ---", flush=True)
    pyrrh = build_pyrrhotite()
    cfgs = generate_h2o_on_mineral(pyrrh, (0, 0, 1), "pyrrhotite_001",
                                   n_waters=[1, 2])
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL H2O ADSORPTION CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


# ===========================================================================
#  DFT + I/O
# ===========================================================================

def run_gpaw_single_point(atoms, config_label):
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)

    if n_atoms > 80:
        kpts = (1, 1, 1)
    elif n_atoms > 50:
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
    parser = argparse.ArgumentParser(description="Generate H2O adsorption DFT data (Tier 2D)")
    parser.add_argument('--output', type=str, default='/workspace/results/h2o_ads_train.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Just count configs, no DFT')
    args = parser.parse_args()

    configs = generate_all_configs()

    if args.dry_run:
        print("\nDRY RUN — config list:")
        for atoms, label in configs:
            elements = sorted(set(atoms.get_chemical_symbols()))
            print(f"  {label}: {len(atoms)} atoms, elements={elements}")
        print(f"\nTotal: {len(configs)} configs")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if args.resume:
        existing = load_existing_labels(output_path)
        print(f"Resuming: found {len(existing)} existing configs", flush=True)

    remaining = [(a, l) for a, l in configs if l not in existing]
    print(f"Remaining: {len(remaining)} configs to compute\n", flush=True)

    log_path = output_path.parent / 'h2o_ads_log.txt'

    for i, (atoms, label) in enumerate(remaining):
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
