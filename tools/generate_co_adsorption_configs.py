#!/usr/bin/env python3
"""
Generate DFT training data for CO adsorption on sulfide surfaces (Tier 2E).

CO is the primary CO2RR product on pentlandite (Tetzlaff 2021).
Important for electrocatalysis community.

Config breakdown (~10 configs):
  Pentlandite (111) + CO (3 sites × 2 orient):  6
  Greigite (111) + CO (2 sites × 2 orient):     4
  TOTAL:                                        10

Usage:
    python -u generate_co_adsorption_configs.py --output /workspace/results/co_ads_train.xyz
    python -u generate_co_adsorption_configs.py --output /workspace/results/co_ads_train.xyz --resume
    python -u generate_co_adsorption_configs.py --dry-run
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


def make_co():
    """Build CO molecule. C-O bond = 1.128 Å."""
    co = molecule('CO')
    co.positions -= co.get_center_of_mass()
    return co


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

    # Mask out intra-molecular distances within each adsorbate
    if adsorbate_sizes:
        n = len(atoms)
        idx = n
        for mol_size in reversed(adsorbate_sizes):
            start = idx - mol_size
            dists[start:idx, start:idx] = np.inf
            idx = start

    min_d = np.min(dists)
    return min_d >= min_dist, min_d


def place_co_on_site(slab, site_pos, orientation='C_down', height=2.2):
    """Place CO on surface site.

    Orientations:
        'C_down':  C closest to surface (typical on metals)
        'O_down':  O closest to surface (less common)

    Raises height if atoms are too close (min < 1.2 Å).
    """
    result = slab.copy()
    co = make_co()

    if orientation == 'C_down':
        # CO along z, C pointing down (toward surface)
        co.rotate(90, 'y')
        # Flip so C is at bottom
        co.positions[:, 2] *= -1
        co.positions -= co.get_center_of_mass()
        co.positions += site_pos + np.array([0, 0, height])
    elif orientation == 'O_down':
        co.rotate(90, 'y')
        co.positions -= co.get_center_of_mass()
        co.positions += site_pos + np.array([0, 0, height])
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    result += co

    # Safety: raise CO if too close to slab atoms
    # adsorbate_sizes=[2] excludes intra-CO bond (C-O = 1.13 Å) from check
    for attempt in range(5):
        ok, min_d = check_min_distance(result, adsorbate_sizes=[2])
        if ok:
            break
        print(f"    WARNING: min dist {min_d:.2f} Å < 1.2 Å, raising CO by 0.3 Å (attempt {attempt+1})", flush=True)
        # Raise only the CO atoms (last 2)
        result.positions[-2:, 2] += 0.3
    else:
        ok, min_d = check_min_distance(result, adsorbate_sizes=[2])
        if not ok:
            print(f"    ERROR: still too close ({min_d:.2f} Å) after 5 attempts!", flush=True)

    return result


def generate_co_on_pentlandite():
    """CO on pentlandite (111) — 6 configs."""
    configs = []
    pent = build_pentlandite()
    slab = surface(pent, (1, 1, 1), layers=2, vacuum=15.0)

    syms = np.array(slab.get_chemical_symbols())
    fe_mask = syms == 'Fe'
    ni_mask = syms == 'Ni'
    s_mask = syms == 'S'

    fe_pos = slab.positions[fe_mask]
    ni_pos = slab.positions[ni_mask]
    s_pos = slab.positions[s_mask]

    if len(fe_pos) == 0:
        return configs

    top_fe = fe_pos[np.argmax(fe_pos[:, 2])]
    top_ni = ni_pos[np.argmax(ni_pos[:, 2])] if len(ni_pos) > 0 else top_fe

    sites = {
        'top_Fe': top_fe,
        'top_Ni': top_ni,
        'bridge_FeNi': (top_fe + top_ni) / 2,
    }

    for site_name, site_pos in sites.items():
        for orient in ['C_down', 'O_down']:
            s = place_co_on_site(slab, site_pos, orientation=orient, height=2.2)
            configs.append((s, f"pent_111_CO_{site_name}_{orient}"))

    return configs


def generate_co_on_greigite():
    """CO on greigite (111) — 4 configs."""
    configs = []
    greig = build_greigite_conventional()
    slab = surface(greig, (1, 1, 1), layers=2, vacuum=15.0)

    syms = np.array(slab.get_chemical_symbols())
    fe_mask = syms == 'Fe'
    s_mask = syms == 'S'

    fe_pos = slab.positions[fe_mask]
    s_pos = slab.positions[s_mask]

    if len(fe_pos) == 0:
        return configs

    top_fe = fe_pos[np.argmax(fe_pos[:, 2])]
    top_s = s_pos[np.argmax(s_pos[:, 2])]

    sites = {
        'top_Fe': top_fe,
        'bridge_FeS': (top_fe + top_s) / 2,
    }

    for site_name, site_pos in sites.items():
        for orient in ['C_down', 'O_down']:
            s = place_co_on_site(slab, site_pos, orientation=orient, height=2.2)
            configs.append((s, f"greigite_111_CO_{site_name}_{orient}"))

    return configs


def generate_all_configs():
    configs = []

    print("=" * 60, flush=True)
    print("Generating CO adsorption configs (Tier 2E)", flush=True)
    print("=" * 60, flush=True)

    print("\n--- Pentlandite (111) + CO ---", flush=True)
    cfgs = generate_co_on_pentlandite()
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs", flush=True)

    print("--- Greigite (111) + CO ---", flush=True)
    cfgs = generate_co_on_greigite()
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL CO ADSORPTION CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


def run_gpaw_single_point(atoms, config_label):
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)
    kpts = (1, 1, 1) if n_atoms > 60 else (2, 2, 1)

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
    parser = argparse.ArgumentParser(description="Generate CO adsorption DFT data (Tier 2E)")
    parser.add_argument('--output', type=str, default='/workspace/results/co_ads_train.xyz')
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

    log_path = output_path.parent / 'co_ads_log.txt'

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
