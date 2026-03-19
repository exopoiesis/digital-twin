#!/usr/bin/env python3
"""
Generate DFT training data for S-vacancy surface configs (Tier 1D).

S-vacancies are real catalytic active sites on sulfide surfaces.
Removing one surface S atom creates under-coordinated Fe/Ni — the actual
centers for CO2 reduction and H2 evolution.

Config breakdown (15 configs):
  Mackinawite (001) − 1S:   eq + 4 rattles =  5
  Pyrite (100) − 1S:        eq + 4 rattles =  5
  Pentlandite (111) − 1S:   eq + 4 rattles =  5
  TOTAL:                                      15

Usage:
    python -u generate_s_vacancy_surface_configs.py --output /workspace/results/s_vac_surface.xyz
    python -u generate_s_vacancy_surface_configs.py --output /workspace/results/s_vac_surface.xyz --resume
    python -u generate_s_vacancy_surface_configs.py --dry-run
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from ase.build import surface
from ase.io import write, read
from ase.spacegroup import crystal


# ===========================================================================
#  Mineral builders (same as v2 datagen)
# ===========================================================================

def build_mackinawite():
    """Mackinawite FeS (P4/nmm, #129). Layered structure."""
    return crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
        primitive_cell=True,
    )


def build_pyrite():
    """Pyrite FeS2 (Pa-3, #205)."""
    return crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.385, 0.385, 0.385)],
        spacegroup=205,
        cellpar=[5.418, 5.418, 5.418, 90, 90, 90],
        primitive_cell=True,
    )


def build_pentlandite():
    """Pentlandite (Fe,Ni)9S8 (Fm-3m, #225). Conventional cell."""
    return crystal(
        symbols=['Fe', 'Ni', 'S'],
        basis=[
            (0.0, 0.0, 0.0),       # 4a: Fe
            (0.625, 0.625, 0.625),  # 8c: Ni
            (0.25, 0.25, 0.25),     # 8c: S
        ],
        spacegroup=225,
        cellpar=[10.07, 10.07, 10.07, 90, 90, 90],
        primitive_cell=False,
    )


# ===========================================================================
#  S-vacancy surface builder
# ===========================================================================

def check_min_distance(atoms, min_dist=1.2):
    """Check if any two atoms are closer than min_dist (Å).

    Returns (ok, min_d).
    """
    if len(atoms) < 2:
        return True, float('inf')
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    return min_d >= min_dist, min_d


def make_s_vacancy_slab(bulk, miller, label_prefix, layers=2, vacuum=12.0,
                        repeat=None, n_rattles=4, rattle_stdev=0.05):
    """Build a slab, remove the topmost S atom, and generate rattled variants.

    Returns list of (atoms, label, True) tuples.
    Validates min distance after vacancy creation and rattling.
    """
    configs = []

    slab = surface(bulk, miller, layers=layers, vacuum=vacuum)
    if repeat is not None:
        slab = slab.repeat(repeat)

    syms = np.array(slab.get_chemical_symbols())
    s_indices = np.where(syms == 'S')[0]

    if len(s_indices) == 0:
        print(f"  WARNING: {label_prefix} slab has no S atoms!", flush=True)
        return configs

    # Find the topmost S atom
    s_z = slab.positions[s_indices, 2]
    top_s_idx = s_indices[np.argmax(s_z)]

    # Create vacancy slab
    vac_slab = slab.copy()
    del vac_slab[top_s_idx]

    ok, min_d = check_min_distance(vac_slab)
    if not ok:
        print(f"  WARNING: {label_prefix}_Svac_eq min dist {min_d:.2f} Å < 1.2 Å!", flush=True)

    configs.append((vac_slab.copy(), f"{label_prefix}_Svac_eq", True))

    # Rattled variants
    for i in range(n_rattles):
        rattled = vac_slab.copy()
        rattled.rattle(stdev=rattle_stdev, seed=1000 + hash(label_prefix) % 10000 + i)
        ok, min_d = check_min_distance(rattled)
        if not ok:
            print(f"  WARNING: {label_prefix}_Svac_rattle_{i:02d} min dist {min_d:.2f} Å — re-rattling", flush=True)
            # Retry with smaller stdev
            rattled = vac_slab.copy()
            rattled.rattle(stdev=rattle_stdev * 0.5, seed=2000 + hash(label_prefix) % 10000 + i)
        configs.append((rattled, f"{label_prefix}_Svac_rattle_{i:02d}", True))

    return configs


def generate_all_configs():
    """Generate all S-vacancy surface configurations."""
    configs = []

    print("=" * 60, flush=True)
    print("Generating S-vacancy surface configs (Tier 1D)", flush=True)
    print("=" * 60, flush=True)

    # Mackinawite (001) − 1S
    print("\n--- Mackinawite (001) ---", flush=True)
    mack = build_mackinawite()
    cfgs = make_s_vacancy_slab(mack, (0, 0, 1), "mack_001",
                               layers=2, vacuum=12.0, repeat=(2, 2, 1))
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs ({len(cfgs[0][0]) if cfgs else '?'} atoms each)", flush=True)

    # Pyrite (100) − 1S
    print("\n--- Pyrite (100) ---", flush=True)
    pyr = build_pyrite()
    cfgs = make_s_vacancy_slab(pyr, (1, 0, 0), "pyrite_100",
                               layers=2, vacuum=12.0, repeat=(2, 2, 1))
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs ({len(cfgs[0][0]) if cfgs else '?'} atoms each)", flush=True)

    # Pentlandite (111) − 1S
    print("\n--- Pentlandite (111) ---", flush=True)
    pent = build_pentlandite()
    cfgs = make_s_vacancy_slab(pent, (1, 1, 1), "pent_111",
                               layers=2, vacuum=12.0)
    configs.extend(cfgs)
    print(f"  {len(cfgs)} configs ({len(cfgs[0][0]) if cfgs else '?'} atoms each)", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL S-VACANCY SURFACE CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


# ===========================================================================
#  DFT + I/O (same as other Tier 1 scripts)
# ===========================================================================

def run_gpaw_single_point(atoms, config_label):
    """Run GPAW single-point (slab settings)."""
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)

    if n_atoms > 60:
        kpts = (1, 1, 1)
    elif n_atoms > 30:
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
    """Save to extended XYZ (append mode)."""
    atoms_copy = atoms.copy()
    atoms_copy.info['energy'] = results['energy']
    atoms_copy.info['config_type'] = results['config_type']
    atoms_copy.arrays['forces'] = results['forces']
    if results['stress'] is not None:
        atoms_copy.info['stress'] = results['stress']
    write(output_path, atoms_copy, format='extxyz', append=True)


def load_existing_labels(output_path):
    """Load already computed config labels."""
    if not Path(output_path).exists():
        return set()
    try:
        all_atoms = read(output_path, index=':', format='extxyz')
        return {a.info.get('config_type', '') for a in all_atoms}
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(description="Generate S-vacancy surface DFT data (Tier 1D)")
    parser.add_argument('--output', type=str, default='/workspace/results/s_vac_surface.xyz')
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

    log_path = output_path.parent / 's_vac_surface_log.txt'

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
