#!/usr/bin/env python3
"""
Generate DFT training data for pyrrhotite Fe7S8 (Tier 2A).

Pyrrhotite is the most abundant iron sulfide in hydrothermal systems.
It is a Fe-deficient derivative of troilite (NiAs-type FeS) with ordered
Fe vacancies. Structure: monoclinic 4C superstructure, but for DFT training
we use a simplified model -- NiAs 2x2x2 supercell with 2 Fe removed.

Fe16S16 → remove 2 Fe → Fe14S16 ≡ Fe7S8

Vacancy ordering: alternating (001) Fe layers, mimicking 4C pyrrhotite.

Config breakdown (~44 configs):
  Bulk eq:                           1
  Bulk rattles (σ=0.03-0.20):       15
  Bulk strains (±1-5%):             10
  Bulk shears:                        5
  2x2x1 supercell rattles:          10  (from 2x2x2 base → 2x2x1 of that = too big, skip)
  (001) slab + rattles:               6
  H adsorption (3 sites):            3
  TOTAL:                            ~50  (adjusted from plan estimate of 44)

Note: pyrrhotite is ferrimagnetic. Initial magnetic moments set to ±3.5 μB
on Fe (alternating layers) to help SCF convergence.

Usage:
    python -u generate_pyrrhotite_configs.py --output /workspace/results/pyrrhotite_train.xyz
    python -u generate_pyrrhotite_configs.py --output /workspace/results/pyrrhotite_train.xyz --resume
    python -u generate_pyrrhotite_configs.py --dry-run
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


# ===========================================================================
#  Pyrrhotite builders
# ===========================================================================

def build_troilite_unit():
    """Build troilite FeS unit cell (NiAs-type, P6_3/mmc, #194).

    a = 3.446 Å, c = 5.877 Å (experimental for FeS)
    Fe at 2a: (0, 0, 0)
    S  at 2c: (1/3, 2/3, 1/4)
    """
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (1/3, 2/3, 0.25)],
        spacegroup=194,
        cellpar=[3.446, 3.446, 5.877, 90, 90, 120],
        primitive_cell=True,
    )
    return atoms


def build_pyrrhotite(seed=42):
    """Build pyrrhotite Fe7S8 from 2x2x2 troilite supercell with 2 Fe vacancies.

    Vacancy ordering: remove 1 Fe from layer z≈0 and 1 from layer z≈c/2,
    on different (x,y) positions to approximate the 4C ordering.
    Total: 14 Fe + 16 S = 30 atoms.
    """
    troilite = build_troilite_unit()
    supercell = troilite.repeat((2, 2, 2))  # 16 Fe + 16 S = 32 atoms

    syms = np.array(supercell.get_chemical_symbols())
    fe_indices = np.where(syms == 'Fe')[0]
    fe_z = supercell.positions[fe_indices, 2]

    # Split Fe into two layers by z coordinate
    z_median = np.median(fe_z)
    layer_lo = fe_indices[fe_z < z_median]  # bottom layer
    layer_hi = fe_indices[fe_z >= z_median]  # top layer

    # Remove 1 Fe from each layer (at different x,y to break symmetry)
    rng = np.random.RandomState(seed)
    remove_lo = rng.choice(layer_lo)

    # For high layer, pick one far from the low vacancy
    lo_pos = supercell.positions[remove_lo, :2]
    hi_dists = np.linalg.norm(supercell.positions[layer_hi, :2] - lo_pos, axis=1)
    remove_hi = layer_hi[np.argmax(hi_dists)]

    # Delete in reverse order to preserve indices
    to_remove = sorted([remove_lo, remove_hi], reverse=True)
    pyrrhotite = supercell.copy()
    for idx in to_remove:
        del pyrrhotite[idx]

    # Set initial magnetic moments (ferrimagnetic: ±3.5 on Fe)
    new_syms = np.array(pyrrhotite.get_chemical_symbols())
    new_fe = np.where(new_syms == 'Fe')[0]
    new_fe_z = pyrrhotite.positions[new_fe, 2]
    z_med = np.median(new_fe_z)

    magmoms = np.zeros(len(pyrrhotite))
    for i in new_fe:
        if pyrrhotite.positions[i, 2] < z_med:
            magmoms[i] = 3.5
        else:
            magmoms[i] = -3.5
    pyrrhotite.set_initial_magnetic_moments(magmoms)

    return pyrrhotite


def set_pyrrhotite_magmoms(atoms):
    """Set ferrimagnetic initial magnetic moments on any Fe-S structure."""
    syms = np.array(atoms.get_chemical_symbols())
    fe_mask = syms == 'Fe'
    if not np.any(fe_mask):
        return

    fe_z = atoms.positions[fe_mask, 2]
    z_med = np.median(fe_z)

    magmoms = np.zeros(len(atoms))
    fe_indices = np.where(fe_mask)[0]
    for i in fe_indices:
        if atoms.positions[i, 2] < z_med:
            magmoms[i] = 3.5
        else:
            magmoms[i] = -3.5
    atoms.set_initial_magnetic_moments(magmoms)


# ===========================================================================
#  Config generators
# ===========================================================================

def rattle_atoms(atoms, stdev, label_prefix, count=5):
    """Generate rattled structures with preserved magnetic moments."""
    configs = []
    for i in range(count):
        rattled = atoms.copy()
        rattled.rattle(stdev=stdev, seed=42 + int(stdev * 100) + i)
        set_pyrrhotite_magmoms(rattled)
        configs.append((rattled, f"{label_prefix}_rattle_{stdev:.2f}_{i:02d}"))
    return configs


def strain_atoms(atoms, label_prefix):
    """Generate volumetrically strained structures."""
    configs = []
    for strain_pct in [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]:
        strained = atoms.copy()
        factor = 1.0 + strain_pct / 100.0
        strained.set_cell(strained.cell * factor, scale_atoms=True)
        set_pyrrhotite_magmoms(strained)
        configs.append((strained, f"{label_prefix}_strain_{strain_pct:+d}pct"))
    return configs


def shear_atoms(atoms, label_prefix):
    """Generate shear-strained structures."""
    configs = []
    for i, (c1, c2) in enumerate([(0, 1), (0, 2), (1, 2)]):
        sheared = atoms.copy()
        cell = sheared.cell.copy()
        cell[c1, c2] += 0.02 * np.linalg.norm(cell[c1])
        sheared.set_cell(cell, scale_atoms=True)
        set_pyrrhotite_magmoms(sheared)
        configs.append((sheared, f"{label_prefix}_shear_{i:02d}"))
    for i, (c1, c2) in enumerate([(0, 1), (1, 2)]):
        sheared = atoms.copy()
        cell = sheared.cell.copy()
        cell[c1, c2] += 0.04 * np.linalg.norm(cell[c1])
        sheared.set_cell(cell, scale_atoms=True)
        set_pyrrhotite_magmoms(sheared)
        configs.append((sheared, f"{label_prefix}_shear_large_{i:02d}"))
    return configs


def build_pyrrhotite_surface_001():
    """Build pyrrhotite (001) slab + rattled variants.

    Since pyrrhotite is already 30 atoms, 2-layer (001) slab will be ~60 atoms.
    """
    pyrrh = build_pyrrhotite()
    slab = surface(pyrrh, (0, 0, 1), layers=2, vacuum=12.0)
    set_pyrrhotite_magmoms(slab)

    configs = [(slab.copy(), "pyrrhotite_001_slab")]
    for i in range(5):
        rattled = slab.copy()
        rattled.rattle(stdev=0.05, seed=500 + i)
        set_pyrrhotite_magmoms(rattled)
        configs.append((rattled, f"pyrrhotite_001_slab_rattle_{i:02d}"))

    return configs


def build_pyrrhotite_H_adsorption():
    """Build pyrrhotite (001) surface with H at different sites."""
    pyrrh = build_pyrrhotite()
    slab = surface(pyrrh, (0, 0, 1), layers=2, vacuum=12.0)
    set_pyrrhotite_magmoms(slab)

    configs = []
    syms = np.array(slab.get_chemical_symbols())
    fe_mask = syms == 'Fe'
    s_mask = syms == 'S'
    fe_pos = slab.positions[fe_mask]
    s_pos = slab.positions[s_mask]

    if len(fe_pos) == 0 or len(s_pos) == 0:
        return configs

    top_fe = fe_pos[np.argmax(fe_pos[:, 2])]
    top_s = s_pos[np.argmax(s_pos[:, 2])]

    # Site 1: H on-top Fe (1.6 Å above)
    s1 = slab.copy()
    s1 += Atoms('H', positions=[top_fe + [0, 0, 1.6]])
    set_pyrrhotite_magmoms(s1)
    configs.append((s1, "pyrrhotite_001_H_ontop_Fe"))

    # Site 2: H bridge Fe-S (1.8 Å above midpoint)
    h_bridge = (top_fe + top_s) / 2 + [0, 0, 1.8]
    # Safety: check min distance
    dists = np.linalg.norm(slab.positions - h_bridge, axis=1)
    if np.min(dists) < 1.0:
        h_bridge[2] += 0.5
    s2 = slab.copy()
    s2 += Atoms('H', positions=[h_bridge])
    set_pyrrhotite_magmoms(s2)
    configs.append((s2, "pyrrhotite_001_H_bridge_FeS"))

    # Site 3: H hollow (Fe-Fe-S)
    fe_dists = np.linalg.norm(fe_pos - top_fe, axis=1)
    fe_dists[np.argmax(fe_pos[:, 2])] = np.inf
    second_fe = fe_pos[np.argmin(fe_dists)]
    h_hollow = (top_fe + top_s + second_fe) / 3 + [0, 0, 1.5]
    dists = np.linalg.norm(slab.positions - h_hollow, axis=1)
    if np.min(dists) < 1.0:
        h_hollow[2] += 0.5
    s3 = slab.copy()
    s3 += Atoms('H', positions=[h_hollow])
    set_pyrrhotite_magmoms(s3)
    configs.append((s3, "pyrrhotite_001_H_hollow"))

    return configs


def generate_all_configs():
    """Generate all pyrrhotite configurations."""
    configs = []  # List of (atoms, label, is_slab)

    print("=" * 60, flush=True)
    print("Generating pyrrhotite Fe7S8 configs (Tier 2A)", flush=True)
    print("=" * 60, flush=True)

    pyrrh = build_pyrrhotite()
    print(f"  Pyrrhotite cell: {len(pyrrh)} atoms "
          f"({sum(1 for s in pyrrh.get_chemical_symbols() if s == 'Fe')} Fe, "
          f"{sum(1 for s in pyrrh.get_chemical_symbols() if s == 'S')} S)", flush=True)

    # Bulk equilibrium
    configs.append((pyrrh.copy(), "pyrrhotite_bulk_eq", False))

    # Rattles on bulk cell (30 atoms)
    for stdev in [0.03, 0.05, 0.08, 0.10, 0.20]:
        n = 3
        configs.extend([(a, l, False) for a, l in rattle_atoms(pyrrh, stdev, "pyrrhotite_bulk", n)])

    # Strains
    configs.extend([(a, l, False) for a, l in strain_atoms(pyrrh, "pyrrhotite_bulk")])

    # Shears
    configs.extend([(a, l, False) for a, l in shear_atoms(pyrrh, "pyrrhotite_bulk")])

    n_bulk = len(configs)
    print(f"  Bulk configs: {n_bulk}", flush=True)

    # (001) surface + rattles
    print("\n  Building (001) surface...", flush=True)
    slab_configs = build_pyrrhotite_surface_001()
    configs.extend([(a, l, True) for a, l in slab_configs])
    print(f"  Slab configs: {len(slab_configs)} "
          f"({len(slab_configs[0][0]) if slab_configs else '?'} atoms)", flush=True)

    # H adsorption
    print("\n  Building H adsorption configs...", flush=True)
    h_configs = build_pyrrhotite_H_adsorption()
    configs.extend([(a, l, True) for a, l in h_configs])
    print(f"  H adsorption configs: {len(h_configs)}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL PYRRHOTITE CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


# ===========================================================================
#  DFT + I/O (same as other datagen scripts)
# ===========================================================================

def run_gpaw_single_point(atoms, config_label, is_slab=False):
    """Run GPAW single-point with spin polarization."""
    from gpaw import GPAW, PW, FermiDirac, MixerDif, Mixer

    n_atoms = len(atoms)

    mode = PW(400)

    if is_slab:
        if n_atoms > 60:
            kpts = (1, 1, 1)
        else:
            kpts = (2, 2, 1)
    else:
        if n_atoms > 30:
            kpts = (2, 2, 2)
        else:
            kpts = (3, 3, 3)

    # MixerDif for AFM/ferrimagnetic slabs (charge sloshing prevention)
    mixer = MixerDif(0.02, 5) if is_slab else Mixer(0.05, 5)

    calc = GPAW(
        mode=mode,
        xc='PBE',
        kpts=kpts,
        occupations=FermiDirac(0.1),
        convergence={'energy': 1e-5},
        mixer=mixer,
        maxiter=500,
        parallel={'augment_grids': True},
        txt=f'/workspace/results/{config_label}.txt',
    )

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = None
    if all(atoms.pbc):
        stress = atoms.get_stress(voigt=True)

    return {'energy': energy, 'forces': forces, 'stress': stress, 'config_type': config_label}


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
    parser = argparse.ArgumentParser(description="Generate pyrrhotite Fe7S8 DFT training data (Tier 2A)")
    parser.add_argument('--output', type=str, default='/workspace/results/pyrrhotite_train.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Just count configs, no DFT')
    args = parser.parse_args()

    configs = generate_all_configs()

    if args.dry_run:
        print("\nDRY RUN -- config list:")
        for atoms, label, is_slab in configs:
            slab_tag = " [SLAB]" if is_slab else ""
            mag = atoms.get_initial_magnetic_moments()
            mag_str = f", |m|={np.abs(mag).sum():.1f}" if np.any(mag != 0) else ""
            print(f"  {label}: {len(atoms)} atoms{slab_tag}{mag_str}")
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

    log_path = output_path.parent / 'pyrrhotite_log.txt'
    n_success = 0
    n_failed = 0

    for i, (atoms, label, is_slab) in enumerate(remaining):
        t0 = time.time()
        try:
            results = run_gpaw_single_point(atoms, label, is_slab)
            save_to_extxyz(atoms, results, output_path)
            # Clean up GPAW log on success
            gpaw_log = Path(f'/workspace/results/{label}.txt')
            if gpaw_log.exists():
                gpaw_log.unlink()
            dt = time.time() - t0
            msg = (f"[{i+1}/{len(remaining)}] {label}: "
                   f"E={results['energy']:.4f} eV, "
                   f"max|F|={np.max(np.linalg.norm(results['forces'], axis=1)):.4f} eV/A "
                   f"({dt:.1f}s)")
            print(msg, flush=True)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')
            n_success += 1
        except Exception as e:
            msg = f"[{i+1}/{len(remaining)}] {label}: FAILED -- {e}"
            print(msg, flush=True)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')
            traceback.print_exc()
            n_failed += 1

    print(f"\n{'=' * 60}", flush=True)
    print(f"Done. Output: {output_path}", flush=True)
    print(f"  Success: {n_success}/{len(remaining)}", flush=True)
    print(f"  Failed:  {n_failed}/{len(remaining)}", flush=True)
    if output_path.exists():
        final = read(output_path, index=':', format='extxyz')
        print(f"Total configs in file: {len(final)}", flush=True)

    # DONE marker for monitoring
    done_path = output_path.parent / 'DONE'
    done_path.write_text(
        f"pyrrhotite completed {n_success}/{len(remaining)} at "
        f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
    )


if __name__ == '__main__':
    main()
