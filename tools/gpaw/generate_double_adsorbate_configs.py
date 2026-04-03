#!/usr/bin/env python3
"""
Generate DFT training data for double adsorbates on sulfide surfaces (Tier 4E).

Real catalytic conditions involve co-adsorption. These configs teach the
ML potential about adsorbate-adsorbate interactions on surfaces.

Config breakdown (~60 configs):
  CO2 + H2O on mackinawite (001):      6
  CO2 + H2O on greigite (001):         6
  CO2 + H2O on greigite (111):         6
  CO2 + H2O on pentlandite (111):      6
  H + CO2 on mackinawite (001):        6
  H + CO2 on greigite (001):           6
  H + CO2 on pentlandite (111):        6
  H2S + CO2 on mackinawite (001):      6
  H2S + CO2 on greigite (001):         6
  H2S + CO2 on pentlandite (111):      6
  TOTAL:                              60

Usage:
    python -u generate_double_adsorbate_configs.py --output /workspace/results/double_ads.xyz
    python -u generate_double_adsorbate_configs.py --dry-run
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
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


# ===========================================================================
#  Adsorbate helpers
# ===========================================================================

def get_top_metal(slab):
    syms = np.array(slab.get_chemical_symbols())
    metal_mask = (syms == 'Fe') | (syms == 'Ni')
    metal_pos = slab.positions[metal_mask]
    if len(metal_pos) == 0:
        return None
    return metal_pos[np.argmax(metal_pos[:, 2])]


def get_top_s(slab):
    syms = np.array(slab.get_chemical_symbols())
    s_pos = slab.positions[syms == 'S']
    if len(s_pos) == 0:
        return None
    return s_pos[np.argmax(s_pos[:, 2])]


def make_co2():
    co2 = molecule('CO2')
    co2.positions -= co2.get_center_of_mass()
    return co2


def make_h2o():
    h2o = molecule('H2O')
    h2o.positions -= h2o.get_center_of_mass()
    return h2o


def make_h2s():
    """Build H2S molecule. H-S bond 1.34 Å, H-S-H angle 92°."""
    d = 1.34
    angle = np.radians(92.0 / 2)
    positions = np.array([
        [0.0, 0.0, 0.0],                        # S
        [d * np.sin(angle), 0.0, d * np.cos(angle)],   # H1
        [-d * np.sin(angle), 0.0, d * np.cos(angle)],  # H2
    ])
    h2s = Atoms('SHH', positions=positions)
    h2s.positions -= h2s.get_center_of_mass()
    return h2s


def check_min_distance(atoms, min_dist=1.2, adsorbate_sizes=None):
    """Check if any two atoms are closer than min_dist (Å).

    adsorbate_sizes: list of ints — sizes of adsorbate molecules appended
                     at the end of atoms. Intra-molecular distances within
                     each adsorbate are excluded from the check.
                     Example: [2] for CO, [3] for CO2, [1, 3] for H + CO2.

    Returns (ok, min_d).
    """
    if len(atoms) < 2:
        return True, float('inf')
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)

    if adsorbate_sizes:
        n = len(atoms)
        idx = n
        for mol_size in reversed(adsorbate_sizes):
            start = idx - mol_size
            dists[start:idx, start:idx] = np.inf
            idx = start

    min_d = np.min(dists)
    return min_d >= min_dist, min_d


def place_pair(slab, mol_a, mol_b, site_pos, height_a=2.5, height_b=2.5,
               lateral_sep=3.0, seed=42):
    """Place two adsorbate molecules near a surface site.

    mol_a goes directly above site, mol_b is offset laterally.
    Multiple orientations via seed.
    Checks min distance and raises adsorbates if needed.
    """
    configs = []
    rng = np.random.RandomState(seed)
    n_a = len(mol_a)
    n_b = len(mol_b)

    for trial in range(6):
        result = slab.copy()

        # Random rotation for each molecule
        a = mol_a.copy()
        b = mol_b.copy()

        angle_a = rng.uniform(0, 360)
        angle_b = rng.uniform(0, 360)
        axis_a = rng.randn(3); axis_a /= np.linalg.norm(axis_a)
        axis_b = rng.randn(3); axis_b /= np.linalg.norm(axis_b)
        a.rotate(angle_a, axis_a)
        b.rotate(angle_b, axis_b)

        a.positions -= a.get_center_of_mass()
        b.positions -= b.get_center_of_mass()

        # Place mol_a above site
        a.positions += site_pos + np.array([0, 0, height_a])

        # Place mol_b laterally offset
        dx = lateral_sep * np.cos(rng.uniform(0, 2 * np.pi))
        dy = lateral_sep * np.sin(rng.uniform(0, 2 * np.pi))
        b.positions += site_pos + np.array([dx, dy, height_b])

        result += a
        result += b

        # Safety: raise all adsorbate atoms if too close
        # Exclude intra-molecular distances within each adsorbate
        for attempt in range(5):
            ok, min_d = check_min_distance(result, adsorbate_sizes=[n_a, n_b])
            if ok:
                break
            print(f"    WARNING: trial {trial} min dist {min_d:.2f} Å < 1.2 Å, raising adsorbates by 0.3 Å (attempt {attempt+1})", flush=True)
            result.positions[-(n_a + n_b):, 2] += 0.3

        configs.append(result)

    return configs


def generate_pair_configs(bulk, miller, label_prefix, pair_name,
                          mol_a_func, mol_b_func, repeat=None,
                          height_a=2.5, height_b=2.5, lateral_sep=3.0):
    """Generate 6 co-adsorption configs for a mineral surface."""
    configs = []

    slab = surface(bulk, miller, layers=2, vacuum=15.0)
    if repeat is not None:
        slab = slab.repeat(repeat)

    site = get_top_metal(slab)
    if site is None:
        return configs

    mol_a = mol_a_func()
    mol_b = mol_b_func()

    structures = place_pair(slab, mol_a, mol_b, site,
                            height_a=height_a, height_b=height_b,
                            lateral_sep=lateral_sep, seed=hash(label_prefix) % 10000)

    for i, s in enumerate(structures):
        configs.append((s, f"{label_prefix}_{pair_name}_{i:02d}"))

    return configs


def make_h_atom():
    """Return a single H atom (for H + CO2 pair)."""
    return Atoms('H', positions=[[0, 0, 0]])


def generate_h_co2_pair(bulk, miller, label_prefix, repeat=None):
    """H atom + CO2 molecule on surface — 6 configs."""
    configs = []

    slab = surface(bulk, miller, layers=2, vacuum=15.0)
    if repeat is not None:
        slab = slab.repeat(repeat)

    site = get_top_metal(slab)
    if site is None:
        return configs

    rng = np.random.RandomState(hash(label_prefix) % 10000)

    for trial in range(6):
        result = slab.copy()

        # H atom on surface (1.6 Å above metal)
        result += Atoms('H', positions=[site + [0, 0, 1.6]])

        # CO2 nearby (different orientation each time)
        co2 = make_co2()
        angle = rng.uniform(0, 360)
        co2.rotate(angle, 'z')
        co2.rotate(rng.uniform(0, 90), 'y')
        co2.positions -= co2.get_center_of_mass()

        dx = 3.0 * np.cos(rng.uniform(0, 2 * np.pi))
        dy = 3.0 * np.sin(rng.uniform(0, 2 * np.pi))
        co2.positions += site + np.array([dx, dy, 2.8])

        result += co2

        # Safety: raise adsorbates if too close
        # [1, 3] = H atom (1) + CO2 molecule (3), exclude intra-CO2 bonds
        for attempt in range(5):
            ok, min_d = check_min_distance(result, adsorbate_sizes=[1, 3])
            if ok:
                break
            print(f"    WARNING: H+CO2 trial {trial} min dist {min_d:.2f} Å, raising by 0.3 Å", flush=True)
            result.positions[-4:, 2] += 0.3  # last 4 = H + CO2(3)

        configs.append((result, f"{label_prefix}_H_CO2_{trial:02d}"))

    return configs


# ===========================================================================
#  Main generator
# ===========================================================================

def generate_all_configs():
    configs = []

    print("=" * 60, flush=True)
    print("Generating double adsorbate configs (Tier 4E)", flush=True)
    print("=" * 60, flush=True)

    minerals = [
        ("mack_001", build_mackinawite(), (0, 0, 1), (2, 2, 1)),
        ("greigite_001", build_greigite_conventional(), (0, 0, 1), None),
        ("greigite_111", build_greigite_conventional(), (1, 1, 1), None),
        ("pent_111", build_pentlandite(), (1, 1, 1), None),
    ]

    # CO2 + H2O pairs
    print("\n--- CO2 + H2O ---", flush=True)
    for label, bulk, miller, repeat in minerals:
        cfgs = generate_pair_configs(bulk, miller, label, "CO2_H2O",
                                     make_co2, make_h2o, repeat=repeat)
        configs.extend(cfgs)
        print(f"  {label}: {len(cfgs)} configs", flush=True)

    # H + CO2 pairs (subset of minerals)
    print("\n--- H + CO2 ---", flush=True)
    for label, bulk, miller, repeat in [minerals[0], minerals[1], minerals[3]]:
        cfgs = generate_h_co2_pair(bulk, miller, label, repeat=repeat)
        configs.extend(cfgs)
        print(f"  {label}: {len(cfgs)} configs", flush=True)

    # H2S + CO2 pairs (subset)
    print("\n--- H2S + CO2 ---", flush=True)
    for label, bulk, miller, repeat in [minerals[0], minerals[1], minerals[3]]:
        cfgs = generate_pair_configs(bulk, miller, label, "H2S_CO2",
                                     make_h2s, make_co2, repeat=repeat)
        configs.extend(cfgs)
        print(f"  {label}: {len(cfgs)} configs", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL DOUBLE ADSORBATE CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


# ===========================================================================
#  DFT + I/O
# ===========================================================================

def run_gpaw_single_point(atoms, config_label):
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)
    kpts = (1, 1, 1) if n_atoms > 50 else (2, 2, 1)

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
    parser = argparse.ArgumentParser(description="Generate double adsorbate DFT data (Tier 4E)")
    parser.add_argument('--output', type=str, default='/workspace/results/double_ads.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
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

    log_path = output_path.parent / 'double_ads_log.txt'

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
