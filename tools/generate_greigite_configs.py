#!/usr/bin/env python3
"""
Generate DFT training data for greigite Fe3S4 (Tier 1 expansion).

Greigite is an inverse thiospinel (Fd-3m, #227) — the key mineral for
CO2 reduction in origin-of-life research (Roldan 2015).

MACE-MP-0 is known to fail on spinels, making this data critical.

Config breakdown (~74 configs):
  Bulk eq + rattles (0.03-0.20):    25
  Bulk strains (±1-5%):             10
  Bulk shears:                       5
  2x2x2 supercell rattles:          10
  (001) slab + rattles:              6
  (111) slab + rattles:              6
  H adsorption (3 sites × 2 surf):  6
  S-vacancy bulk:                    3
  S-vacancy surface:                 3
  TOTAL:                            ~74

Usage:
    python -u generate_greigite_configs.py --output /workspace/results/greigite_train.xyz
    python -u generate_greigite_configs.py --output /workspace/results/greigite_train.xyz --resume
    python -u generate_greigite_configs.py --dry-run  # Just count configs, no DFT
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from ase import Atoms
from ase.build import surface, add_adsorbate
from ase.io import write, read
from ase.spacegroup import crystal
from gpaw_checkpoint import register_sigterm_handler, is_shutdown_requested


def build_greigite() -> Atoms:
    """Build greigite Fe3S4 unit cell (Fd-3m, #227, inverse thiospinel).

    Inverse spinel structure:
      8a  (1/8, 1/8, 1/8):   Fe3+ (tetrahedral A-site)
      16d (1/2, 1/2, 1/2):   Fe2.5+ (octahedral B-site, mixed valence)
      32e (u, u, u), u≈0.380 (setting 1): S2-

    Unit cell: 8 Fe3S4 = 56 atoms (8×Fe_A + 16×Fe_B + 32×S)
    For training data we use the primitive cell (14 atoms).
    """
    # Primitive cell of Fd-3m has 2 formula units = 14 atoms
    # Use conventional cell parameters: a = 9.876 Å
    atoms = crystal(
        symbols=['Fe', 'Fe', 'S'],
        basis=[
            (0.125, 0.125, 0.125),   # 8a tetrahedral Fe
            (0.5, 0.5, 0.5),         # 16d octahedral Fe
            (0.380, 0.380, 0.380),   # 32e sulfur (setting 1, ASE default)
        ],
        spacegroup=227,
        cellpar=[9.876, 9.876, 9.876, 90, 90, 90],
        primitive_cell=True,
    )
    return atoms


def build_greigite_conventional() -> Atoms:
    """Build full conventional cell (56 atoms) for supercell calculations."""
    atoms = crystal(
        symbols=['Fe', 'Fe', 'S'],
        basis=[
            (0.125, 0.125, 0.125),
            (0.5, 0.5, 0.5),
            (0.380, 0.380, 0.380),
        ],
        spacegroup=227,
        cellpar=[9.876, 9.876, 9.876, 90, 90, 90],
        primitive_cell=False,
    )
    return atoms


def rattle_atoms(atoms, stdev, label_prefix, count=10):
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


def shear_atoms(atoms, label_prefix):
    """Generate shear-strained structures."""
    configs = []
    for i, (c1, c2) in enumerate([(0, 1), (0, 2), (1, 2)]):
        sheared = atoms.copy()
        cell = sheared.cell.copy()
        cell[c1, c2] += 0.02 * cell[c1, c1]
        sheared.set_cell(cell, scale_atoms=True)
        configs.append((sheared, f"{label_prefix}_shear_{i:02d}"))
    # Extra shear magnitudes
    for i, (c1, c2) in enumerate([(0, 1), (1, 2)]):
        sheared = atoms.copy()
        cell = sheared.cell.copy()
        cell[c1, c2] += 0.04 * cell[c1, c1]
        sheared.set_cell(cell, scale_atoms=True)
        configs.append((sheared, f"{label_prefix}_shear_large_{i:02d}"))
    return configs


def build_greigite_surface_001():
    """Build greigite (001) slab + rattled variants."""
    greig = build_greigite_conventional()
    slab = surface(greig, (0, 0, 1), layers=2, vacuum=12.0)
    # Greigite conventional cell is large, so 1x1 slab is already ~56 atoms/layer

    configs = [(slab.copy(), "greigite_001_slab")]
    for i in range(5):
        rattled = slab.copy()
        rattled.rattle(stdev=0.05, seed=500 + i)
        configs.append((rattled, f"greigite_001_slab_rattle_{i:02d}"))
    return configs


def build_greigite_surface_111():
    """Build greigite (111) slab — octahedral termination, relevant for CO2RR."""
    greig = build_greigite_conventional()
    slab = surface(greig, (1, 1, 1), layers=2, vacuum=12.0)

    configs = [(slab.copy(), "greigite_111_slab")]
    for i in range(5):
        rattled = slab.copy()
        rattled.rattle(stdev=0.05, seed=600 + i)
        configs.append((rattled, f"greigite_111_slab_rattle_{i:02d}"))
    return configs


def build_greigite_H_adsorption():
    """Build greigite surfaces with H at different adsorption sites."""
    greig = build_greigite_conventional()
    configs = []

    for miller, miller_str, seed_base in [((0,0,1), '001', 700), ((1,1,1), '111', 800)]:
        slab = surface(greig, miller, layers=2, vacuum=12.0)

        syms = np.array(slab.get_chemical_symbols())
        fe_mask = syms == 'Fe'
        s_mask = syms == 'S'

        fe_pos = slab.positions[fe_mask]
        s_pos = slab.positions[s_mask]

        if len(fe_pos) == 0 or len(s_pos) == 0:
            continue

        top_fe = fe_pos[np.argmax(fe_pos[:, 2])]
        top_s = s_pos[np.argmax(s_pos[:, 2])]

        # Site 1: H on-top Fe
        s1 = slab.copy()
        s1 += Atoms('H', positions=[top_fe + [0, 0, 1.5]])
        configs.append((s1, f"greigite_{miller_str}_H_ontop_Fe"))

        # Site 2: H bridge Fe-S
        s2 = slab.copy()
        s2 += Atoms('H', positions=[(top_fe + top_s) / 2 + [0, 0, 1.2]])
        configs.append((s2, f"greigite_{miller_str}_H_bridge_FeS"))

        # Site 3: H hollow (3-fold)
        s3 = slab.copy()
        # Find 2nd nearest Fe
        fe_dists = np.linalg.norm(fe_pos - top_fe, axis=1)
        fe_dists[np.argmax(fe_pos[:, 2])] = np.inf
        second_fe = fe_pos[np.argmin(fe_dists)]
        hollow = (top_fe + top_s + second_fe) / 3 + [0, 0, 1.0]
        s3 += Atoms('H', positions=[hollow])
        configs.append((s3, f"greigite_{miller_str}_H_hollow"))

    return configs


def build_greigite_S_vacancy():
    """Build greigite with S-vacancy (bulk and surface)."""
    configs = []

    # Bulk S-vacancy
    greig = build_greigite()
    syms = np.array(greig.get_chemical_symbols())
    s_indices = np.where(syms == 'S')[0]

    for i, s_idx in enumerate(s_indices[:3]):
        vac = greig.copy()
        del vac[s_idx]
        configs.append((vac, f"greigite_bulk_Svac_{i:02d}"))

    # Surface S-vacancy
    greig_conv = build_greigite_conventional()
    slab = surface(greig_conv, (0, 0, 1), layers=2, vacuum=12.0)
    syms_slab = np.array(slab.get_chemical_symbols())
    s_indices_slab = np.where(syms_slab == 'S')[0]

    # Remove topmost S atoms
    s_positions = slab.positions[s_indices_slab]
    top_s_indices = s_indices_slab[np.argsort(s_positions[:, 2])[-3:]]
    for i, s_idx in enumerate(top_s_indices):
        vac = slab.copy()
        del vac[s_idx]
        configs.append((vac, f"greigite_001_Svac_{i:02d}"))

    return configs


def generate_all_configs():
    """Generate all greigite configurations."""
    configs = []  # List of (atoms, label, is_slab)

    print("Generating greigite configs...", flush=True)
    greig = build_greigite()
    greig_conv = build_greigite_conventional()

    # Bulk equilibrium
    configs.append((greig.copy(), "greigite_bulk_eq", False))
    print(f"  Primitive cell: {len(greig)} atoms", flush=True)
    print(f"  Conventional cell: {len(greig_conv)} atoms", flush=True)

    # Rattles on primitive cell
    for stdev in [0.03, 0.05, 0.08, 0.10, 0.20]:
        n = 5 if stdev >= 0.10 else 5
        configs.extend([(a, l, False) for a, l in rattle_atoms(greig, stdev, "greigite_bulk", n)])

    # Strains
    configs.extend([(a, l, False) for a, l in strain_atoms(greig, "greigite_bulk")])

    # Shears
    configs.extend([(a, l, False) for a, l in shear_atoms(greig, "greigite_bulk")])

    print(f"  Bulk configs: {len(configs)}", flush=True)

    # Supercell rattles (using conventional cell)
    for i in range(5):
        rattled = greig_conv.copy()
        rattled.rattle(stdev=0.05, seed=900 + i)
        configs.append((rattled, f"greigite_conv_rattle_0.05_{i:02d}", False))
    for i in range(5):
        rattled = greig_conv.copy()
        rattled.rattle(stdev=0.10, seed=950 + i)
        configs.append((rattled, f"greigite_conv_rattle_0.10_{i:02d}", False))

    # (001) surface
    configs.extend([(a, l, True) for a, l in build_greigite_surface_001()])

    # (111) surface
    configs.extend([(a, l, True) for a, l in build_greigite_surface_111()])

    # H adsorption
    configs.extend([(a, l, True) for a, l in build_greigite_H_adsorption()])

    # S-vacancy
    s_vac = build_greigite_S_vacancy()
    for a, l in s_vac:
        is_slab = 'slab' in l or '001_Svac' in l
        configs.append((a, l, is_slab))

    print(f"\n{'='*60}", flush=True)
    print(f"TOTAL GREIGITE CONFIGS: {len(configs)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return configs


def set_magnetic_moments(atoms):
    """Set initial magnetic moments for greigite (ferrimagnetic).

    Greigite: Fe_tet=3.5, Fe_oct=-3.1 (Chang 2008).
    Simplified: all Fe=3.5 (sign doesn't matter for initial guess,
    DFT will find the antiparallel arrangement).
    """
    magmoms = []
    for sym in atoms.get_chemical_symbols():
        if sym == 'Fe':
            magmoms.append(3.5)
        else:
            magmoms.append(0.0)
    atoms.set_initial_magnetic_moments(magmoms)


def run_gpaw_single_point(atoms, config_label, is_slab=False):
    """Run GPAW single-point (same settings as v2 datagen)."""
    from gpaw import GPAW, PW, FermiDirac

    set_magnetic_moments(atoms)

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
    parser = argparse.ArgumentParser(description="Generate greigite DFT training data (Tier 1)")
    parser.add_argument('--output', type=str, default='/workspace/results/greigite_train.xyz')
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

    log_path = output_path.parent / 'greigite_log.txt'

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

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"Done. Output: {output_path}", flush=True)
    if output_path.exists():
        final = read(output_path, index=':', format='extxyz')
        print(f"Total configs in file: {len(final)}", flush=True)


if __name__ == '__main__':
    main()
