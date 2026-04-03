#!/usr/bin/env python3
"""
Generate DFT training data from MD snapshots at elevated temperatures (Tier 3A).

For hydrothermal conditions we need training data at 350K and 500K.
Strategy: run short MD with MACE-MP-0 (fast), extract snapshots,
then GPAW single-point on each snapshot.

If MACE-MP-0 is not available, falls back to enhanced thermal rattling
calibrated to the target temperature.

Config breakdown (~180 configs):
  Mackinawite bulk 2x2x2 (32 at) × 2T × 20 snaps:     40
  Greigite bulk primitive (14 at) × 2T × 20 snaps:     40
  Pyrite bulk 2x2x1 (24 at) × 2T × 20 snaps:          40
  Pentlandite bulk primitive (17 at) × 2T × 20 snaps:  40
  Pyrrhotite bulk (30 at) × 350K × 20 snaps:           20
  TOTAL:                                               180

Usage:
    python -u generate_aimd_snapshots_configs.py --output /workspace/results/aimd_snapshots.xyz
    python -u generate_aimd_snapshots_configs.py --output /workspace/results/aimd_snapshots.xyz --resume
    python -u generate_aimd_snapshots_configs.py --dry-run
    python -u generate_aimd_snapshots_configs.py --md-only  # Only generate snapshots, no DFT
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from ase import units
from ase.io import write, read
from ase.spacegroup import crystal


# ===========================================================================
#  Mineral builders
# ===========================================================================

def build_mackinawite_supercell():
    """Mackinawite 2x2x2 supercell (32 atoms)."""
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
        primitive_cell=True,
    )
    return atoms.repeat((2, 2, 2))


def build_greigite_primitive():
    """Greigite primitive cell (14 atoms)."""
    return crystal(
        symbols=['Fe', 'Fe', 'S'],
        basis=[
            (0.125, 0.125, 0.125),
            (0.5, 0.5, 0.5),
            (0.254, 0.254, 0.254),
        ],
        spacegroup=227,
        cellpar=[9.876, 9.876, 9.876, 90, 90, 90],
        primitive_cell=True,
    )


def build_pyrite_supercell():
    """Pyrite 2x2x1 supercell (~24 atoms)."""
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.385, 0.385, 0.385)],
        spacegroup=205,
        cellpar=[5.418, 5.418, 5.418, 90, 90, 90],
        primitive_cell=True,
    )
    return atoms.repeat((2, 2, 1))


def build_pentlandite_primitive():
    """Pentlandite primitive cell (~17 atoms)."""
    return crystal(
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


def build_pyrrhotite():
    """Pyrrhotite Fe7S8 (30 atoms)."""
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
#  MD snapshot generator
# ===========================================================================

def generate_md_snapshots(atoms, temperature, n_snapshots=20, label_prefix="",
                          md_steps=2000, snapshot_interval=None, seed=42):
    """Generate thermally displaced snapshots using Langevin MD.

    Tries MACE-MP-0 first. If unavailable, uses temperature-calibrated rattling.

    Args:
        atoms: initial structure
        temperature: target temperature in K
        n_snapshots: number of snapshots to extract
        label_prefix: for config labels
        md_steps: total MD steps (dt=1 fs)
        snapshot_interval: steps between snapshots (auto if None)
        seed: random seed
    """
    configs = []

    if snapshot_interval is None:
        snapshot_interval = max(1, md_steps // n_snapshots)

    try:
        # Try MACE-MP-0 for fast MD
        from mace.calculators import mace_mp
        calc = mace_mp(model="small", device="cpu", default_dtype="float32")

        from ase.md.langevin import Langevin

        md_atoms = atoms.copy()
        md_atoms.calc = calc

        # Initialize velocities
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        MaxwellBoltzmannDistribution(md_atoms, temperature_K=temperature, rng=np.random.RandomState(seed))

        dyn = Langevin(md_atoms, timestep=1.0 * units.fs,
                       temperature_K=temperature, friction=0.01,
                       rng=np.random.RandomState(seed + 1))

        # Equilibrate for 500 steps
        dyn.run(500)

        # Collect snapshots
        snapshot_count = 0
        for step in range(md_steps):
            dyn.run(1)
            if (step + 1) % snapshot_interval == 0 and snapshot_count < n_snapshots:
                snap = md_atoms.copy()
                snap.calc = None
                configs.append((snap, f"{label_prefix}_{temperature}K_md_{snapshot_count:03d}"))
                snapshot_count += 1

        print(f"    MACE MD: {len(configs)} snapshots at {temperature}K", flush=True)

    except (ImportError, Exception) as e:
        # Fallback: temperature-calibrated thermal rattling
        print(f"    MACE unavailable ({type(e).__name__}), using thermal rattling", flush=True)

        # Thermal displacement: sigma = sqrt(kB*T / k_eff)
        # For typical sulfide phonon frequencies (~5-15 THz):
        # At 350K: sigma ~ 0.06-0.10 Å
        # At 500K: sigma ~ 0.08-0.12 Å
        sigma_base = 0.04  # Å per sqrt(100K)
        sigma = sigma_base * np.sqrt(temperature / 100.0)

        rng = np.random.RandomState(seed)
        for i in range(n_snapshots):
            rattled = atoms.copy()
            # Variable rattle magnitude for diversity
            sigma_i = sigma * rng.uniform(0.7, 1.3)
            rattled.rattle(stdev=sigma_i, seed=seed + i * 7)
            configs.append((rattled, f"{label_prefix}_{temperature}K_rattle_{i:03d}"))

        print(f"    Thermal rattle: {len(configs)} configs at {temperature}K "
              f"(sigma={sigma:.3f} Å)", flush=True)

    return configs


def generate_all_configs():
    configs = []

    print("=" * 60, flush=True)
    print("Generating AIMD/thermal snapshot configs (Tier 3A)", flush=True)
    print("=" * 60, flush=True)

    minerals = [
        ("mackinawite", build_mackinawite_supercell, [350, 500]),
        ("greigite", build_greigite_primitive, [350, 500]),
        ("pyrite", build_pyrite_supercell, [350, 500]),
        ("pentlandite", build_pentlandite_primitive, [350, 500]),
        ("pyrrhotite", build_pyrrhotite, [350]),  # only 350K per plan
    ]

    for name, builder, temperatures in minerals:
        print(f"\n--- {name} ---", flush=True)
        atoms = builder()
        print(f"  Cell: {len(atoms)} atoms", flush=True)

        for temp in temperatures:
            snaps = generate_md_snapshots(
                atoms, temp, n_snapshots=20,
                label_prefix=f"{name}_bulk",
                seed=42 + hash(name) % 10000 + temp
            )
            configs.extend([(a, l, False) for a, l in snaps])

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL AIMD/THERMAL CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


# ===========================================================================
#  DFT + I/O
# ===========================================================================

def run_gpaw_single_point(atoms, config_label):
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)

    if n_atoms > 40:
        kpts = (2, 2, 2)
    elif n_atoms > 20:
        kpts = (3, 3, 3)
    else:
        kpts = (4, 4, 4)

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
    parser = argparse.ArgumentParser(description="Generate AIMD snapshot DFT data (Tier 3A)")
    parser.add_argument('--output', type=str, default='/workspace/results/aimd_snapshots.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Just count configs, no DFT')
    parser.add_argument('--md-only', action='store_true', help='Only generate snapshots, skip DFT')
    args = parser.parse_args()

    configs = generate_all_configs()

    if args.dry_run:
        print("\nDRY RUN — config list:")
        for atoms, label, is_slab in configs:
            print(f"  {label}: {len(atoms)} atoms")
        print(f"\nTotal: {len(configs)} configs")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.md_only:
        snap_path = output_path.with_suffix('.snapshots.xyz')
        for atoms, label, _ in configs:
            atoms.info['config_type'] = label
            write(snap_path, atoms, format='extxyz', append=True)
        print(f"Saved {len(configs)} snapshots to {snap_path}")
        return

    existing = set()
    if args.resume:
        existing = load_existing_labels(output_path)
        print(f"Resuming: found {len(existing)} existing configs", flush=True)

    remaining = [(a, l, s) for a, l, s in configs if l not in existing]
    print(f"Remaining: {len(remaining)} configs to compute\n", flush=True)

    log_path = output_path.parent / 'aimd_snapshots_log.txt'

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
