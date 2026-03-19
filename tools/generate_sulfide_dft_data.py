#!/usr/bin/env python3
"""
Generate DFT training data for MACE fine-tuning on iron sulfide minerals.

Target: 250-300 DFT single-point calculations (energy, forces, stress)
Output: Extended XYZ format for MACE training
Runtime: ~1-2 hours on RTX 4070

Usage:
    python -u generate_sulfide_dft_data.py --output /workspace/results/sulfide_train.xyz --resume
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from ase import Atoms
from ase.build import surface
from ase.io import write, read
from ase.spacegroup import crystal
from gpaw import GPAW, PW, FermiDirac


def build_mackinawite() -> Atoms:
    """Build mackinawite FeS unit cell (P4/nmm, #129)."""
    return crystal(
        symbols=['Fe', 'S'],
        basis=[(0, 0, 0), (0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
    )


def build_pentlandite() -> Atoms:
    """Build pentlandite (Fe,Ni)9S8 unit cell (Fm-3m, #225).

    Wyckoff positions:
      4a (0,0,0): Fe (tetrahedral metal site)
      8c (1/4,1/4,1/4): Ni (octahedral metal site)
      8c (0.385,0.385,0.385): S

    Gives Fe4Ni8S8 (20 atoms). Real pentlandite is Fe4.5Ni4.5S8
    with Fe/Ni disorder on 8c — ordered approximation is fine for
    training data. We swap 3 Ni→Fe to get Fe7Ni5S8 (closer to real).
    """
    atoms = crystal(
        symbols=['Fe', 'Ni', 'S'],
        basis=[(0, 0, 0),              # 4a → 4 Fe
               (0.25, 0.25, 0.25),     # 8c → 8 Ni
               (0.385, 0.385, 0.385)], # 8c → 8 S
        spacegroup=225,
        cellpar=[10.07, 10.07, 10.07, 90, 90, 90],
    )
    # Swap 3 Ni → Fe on 8c sites to approximate Fe4.5Ni4.5S8
    syms = atoms.get_chemical_symbols()
    ni_indices = [i for i, s in enumerate(syms) if s == 'Ni']
    for i in ni_indices[:3]:
        syms[i] = 'Fe'
    atoms.set_chemical_symbols(syms)
    return atoms


def build_pyrite() -> Atoms:
    """Build pyrite FeS2 unit cell (Pa-3, #205).

    Wyckoff: Fe at 4a(0,0,0), S at 8c(u,u,u) with u≈0.385.
    Gives 4 Fe + 8 S = 12 atoms (correct FeS2 stoichiometry).
    """
    return crystal(
        symbols=['Fe', 'S'],
        basis=[(0, 0, 0), (0.385, 0.385, 0.385)],
        spacegroup=205,
        cellpar=[5.418, 5.418, 5.418, 90, 90, 90],
    )


def rattle_atoms(atoms: Atoms, stdev: float, label_suffix: str) -> List[Tuple[Atoms, str]]:
    """Generate rattled structures."""
    configs = []
    n = 15 if stdev < 0.1 else 10

    for i in range(n):
        rattled = atoms.copy()
        rattled.rattle(stdev=stdev, seed=42 + i)
        label = f"{label_suffix}_rattle_{stdev:.2f}_{i:02d}"
        configs.append((rattled, label))

    return configs


def strain_atoms(atoms: Atoms, label_suffix: str) -> List[Tuple[Atoms, str]]:
    """Generate volumetrically strained structures (±1%, ±2%, ±3%)."""
    configs = []

    for strain_pct in [1, 2, 3, -1, -2, -3]:
        strained = atoms.copy()
        factor = 1.0 + strain_pct / 100.0
        strained.set_cell(strained.cell * factor, scale_atoms=True)
        label = f"{label_suffix}_strain_{strain_pct:+d}pct"
        configs.append((strained, label))

    return configs


def shear_strain_atoms(atoms: Atoms, label_suffix: str) -> List[Tuple[Atoms, str]]:
    """Generate shear-strained structures (off-diagonal perturbations)."""
    configs = []

    for i, (comp1, comp2) in enumerate([(0, 1), (0, 2), (1, 2)]):
        sheared = atoms.copy()
        cell = sheared.cell.copy()
        cell[comp1, comp2] += 0.02 * cell[comp1, comp1]  # 2% shear
        sheared.set_cell(cell, scale_atoms=True)
        label = f"{label_suffix}_shear_{i:02d}"
        configs.append((sheared, label))

    return configs


def build_mackinawite_surface() -> List[Tuple[Atoms, str]]:
    """Build mackinawite (001) surface slab and rattled variants."""
    mack = build_mackinawite()

    # Build 2x2x1 slab with 2 layers (~4-5 Å thick) + 10 Å vacuum
    slab = surface(mack, (0, 0, 1), layers=2, vacuum=10.0)
    slab = slab.repeat((2, 2, 1))  # 2x2 supercell

    configs = [(slab.copy(), "mackinawite_001_slab")]

    # 5 rattled variants
    for i in range(5):
        rattled = slab.copy()
        rattled.rattle(stdev=0.05, seed=100 + i)
        configs.append((rattled, f"mackinawite_001_slab_rattle_{i:02d}"))

    return configs


def build_mackinawite_surface_with_H() -> List[Tuple[Atoms, str]]:
    """Build mackinawite surface + H at 3 positions (interstitial, surface, bridge)."""
    mack = build_mackinawite()
    slab = surface(mack, (0, 0, 1), layers=2, vacuum=10.0)
    slab = slab.repeat((2, 2, 1))

    configs = []

    # Get surface plane z-coordinate
    z_max = slab.positions[:, 2].max()

    # Position 1: H on top of surface Fe
    slab1 = slab.copy()
    # Find topmost Fe
    fe_mask = np.array(slab1.get_chemical_symbols()) == 'Fe'
    fe_positions = slab1.positions[fe_mask]
    top_fe = fe_positions[np.argmax(fe_positions[:, 2])]
    h_pos_1 = top_fe + [0, 0, 1.5]  # 1.5 Å above Fe
    slab1 += Atoms('H', positions=[h_pos_1])
    configs.append((slab1, "mackinawite_001_H_ontop"))

    # Position 2: H at bridge site (between Fe and S)
    slab2 = slab.copy()
    s_mask = np.array(slab2.get_chemical_symbols()) == 'S'
    s_positions = slab2.positions[s_mask]
    top_s = s_positions[np.argmax(s_positions[:, 2])]
    h_pos_2 = (top_fe + top_s) / 2 + [0, 0, 1.0]
    slab2 += Atoms('H', positions=[h_pos_2])
    configs.append((slab2, "mackinawite_001_H_bridge"))

    # Position 3: H in interstitial (between layers)
    slab3 = slab.copy()
    # Find interlayer position
    z_sorted = np.sort(slab3.positions[:, 2])
    # Find largest gap in z
    gaps = np.diff(z_sorted)
    gap_idx = np.argmax(gaps)
    z_interstitial = (z_sorted[gap_idx] + z_sorted[gap_idx + 1]) / 2
    # Place H at center xy, interstitial z
    center_xy = slab3.cell[:2, :2].sum(axis=0) / 4  # Quarter cell
    h_pos_3 = [center_xy[0], center_xy[1], z_interstitial]
    slab3 += Atoms('H', positions=[h_pos_3])
    configs.append((slab3, "mackinawite_001_H_interstitial"))

    return configs


def build_pentlandite_H_path() -> List[Tuple[Atoms, str]]:
    """Build pentlandite H vacancy diffusion path snapshots (5 configs)."""
    pent = build_pentlandite()

    configs = []

    # Remove one S to create vacancy, add H nearby
    atoms = pent.copy()
    del atoms[-1]  # Remove last S atom

    # Position 1: H at vacancy site
    pos_1 = atoms.positions[-1].copy()  # Position of removed S
    atoms_1 = atoms.copy()
    atoms_1 += Atoms('H', positions=[pos_1])
    configs.append((atoms_1, "pentlandite_H_vacancy_start"))

    # Position 5: H at neighboring vacancy site (approximate)
    # Move H to adjacent octahedral site
    pos_5 = pos_1 + [pent.cell[0, 0] * 0.25, 0, 0]  # Quarter cell shift
    atoms_5 = atoms.copy()
    atoms_5 += Atoms('H', positions=[pos_5])
    configs.append((atoms_5, "pentlandite_H_vacancy_end"))

    # Interpolate 3 intermediate positions
    for i, alpha in enumerate([0.25, 0.5, 0.75]):
        pos_i = pos_1 * (1 - alpha) + pos_5 * alpha
        atoms_i = atoms.copy()
        atoms_i += Atoms('H', positions=[pos_i])
        # Add small rattle
        atoms_i.rattle(stdev=0.03, seed=200 + i)
        configs.append((atoms_i, f"pentlandite_H_vacancy_path_{i+1}"))

    return configs


def run_gpaw_single_point(atoms: Atoms, config_label: str, is_slab: bool = False) -> Dict:
    """Run GPAW single-point calculation (energy, forces, stress)."""
    # GPAW settings matching q075-dft
    mode = PW(400) if is_slab else PW(500)

    # K-points
    if 'pentlandite' in config_label:
        kpts = (2, 2, 2)  # Large cell
    elif is_slab:
        kpts = (2, 2, 1)  # Slab
    else:
        kpts = (4, 4, 4)  # Small bulk cells

    calc = GPAW(
        mode=mode,
        xc='PBE',
        kpts=kpts,
        occupations=FermiDirac(0.1),
        convergence={'energy': 1e-5},
        parallel={'augment_grids': True},
        txt=None,  # Suppress output
    )

    atoms.calc = calc

    # Calculate
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # Stress only for bulk (PBC in all directions)
    stress = None
    if all(atoms.pbc):
        stress = atoms.get_stress(voigt=True)  # 6-component Voigt

    return {
        'energy': energy,
        'forces': forces,
        'stress': stress,
        'config_type': config_label,
    }


def save_to_extxyz(atoms: Atoms, results: Dict, output_path: Path):
    """Save single config to extended XYZ (append mode)."""
    atoms_copy = atoms.copy()

    # Store in atoms.info and atoms.arrays
    atoms_copy.info['energy'] = results['energy']
    atoms_copy.info['config_type'] = results['config_type']
    atoms_copy.arrays['forces'] = results['forces']

    if results['stress'] is not None:
        atoms_copy.info['stress'] = results['stress']

    # Append to file
    write(output_path, atoms_copy, format='extxyz', append=True)


def load_existing_configs(output_path: Path) -> set:
    """Load already computed config_type labels from existing XYZ."""
    if not output_path.exists():
        return set()

    existing = set()
    try:
        all_atoms = read(output_path, index=':', format='extxyz')
        for atoms in all_atoms:
            if 'config_type' in atoms.info:
                existing.add(atoms.info['config_type'])
    except Exception as e:
        print(f"Warning: Could not load existing configs: {e}", flush=True)

    return existing


def generate_all_configs() -> List[Tuple[Atoms, str, bool]]:
    """Generate all structure configurations (atoms, label, is_slab)."""
    configs = []

    # === MACKINAWITE ===
    print("Generating mackinawite configs...", flush=True)
    mack = build_mackinawite()

    # Equilibrium
    configs.append((mack.copy(), "mackinawite_bulk_eq", False))

    # Rattled
    configs.extend([(a, l, False) for a, l in rattle_atoms(mack, 0.05, "mackinawite_bulk")])
    configs.extend([(a, l, False) for a, l in rattle_atoms(mack, 0.15, "mackinawite_bulk")])

    # Strains
    configs.extend([(a, l, False) for a, l in strain_atoms(mack, "mackinawite_bulk")])
    configs.extend([(a, l, False) for a, l in shear_strain_atoms(mack, "mackinawite_bulk")])

    # Surfaces
    configs.extend([(a, l, True) for a, l in build_mackinawite_surface()])
    configs.extend([(a, l, True) for a, l in build_mackinawite_surface_with_H()])

    print(f"  Mackinawite: {len([c for c in configs if 'mackinawite' in c[1]])} configs", flush=True)

    # === PENTLANDITE ===
    print("Generating pentlandite configs...", flush=True)
    pent = build_pentlandite()

    # Equilibrium
    configs.append((pent.copy(), "pentlandite_bulk_eq", False))

    # Rattled
    configs.extend([(a, l, False) for a, l in rattle_atoms(pent, 0.05, "pentlandite_bulk")])
    configs.extend([(a, l, False) for a, l in rattle_atoms(pent, 0.15, "pentlandite_bulk")])

    # Strains
    configs.extend([(a, l, False) for a, l in strain_atoms(pent, "pentlandite_bulk")])
    configs.extend([(a, l, False) for a, l in shear_strain_atoms(pent, "pentlandite_bulk")])

    # H diffusion path
    configs.extend([(a, l, False) for a, l in build_pentlandite_H_path()])

    print(f"  Pentlandite: {len([c for c in configs if 'pentlandite' in c[1]])} configs", flush=True)

    # === PYRITE ===
    print("Generating pyrite configs...", flush=True)
    pyrite = build_pyrite()

    # Equilibrium
    configs.append((pyrite.copy(), "pyrite_bulk_eq", False))

    # Rattled
    configs.extend([(a, l, False) for a, l in rattle_atoms(pyrite, 0.05, "pyrite_bulk")])
    configs.extend([(a, l, False) for a, l in rattle_atoms(pyrite, 0.15, "pyrite_bulk")])

    # Strains
    configs.extend([(a, l, False) for a, l in strain_atoms(pyrite, "pyrite_bulk")])
    configs.extend([(a, l, False) for a, l in shear_strain_atoms(pyrite, "pyrite_bulk")])

    print(f"  Pyrite: {len([c for c in configs if 'pyrite' in c[1]])} configs", flush=True)

    print(f"\nTotal configs: {len(configs)}", flush=True)

    return configs


def main():
    parser = argparse.ArgumentParser(description="Generate sulfide DFT training data for MACE")
    parser.add_argument('--output', type=str, default='/workspace/results/sulfide_train.xyz',
                        help='Output extended XYZ file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output (skip already computed configs)')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing configs if resuming
    existing = set()
    if args.resume:
        existing = load_existing_configs(output_path)
        print(f"Resuming: found {len(existing)} existing configs", flush=True)

    # Generate all configs
    all_configs = generate_all_configs()

    # Filter out existing
    if existing:
        all_configs = [(a, l, s) for a, l, s in all_configs if l not in existing]
        print(f"Remaining after filter: {len(all_configs)} configs\n", flush=True)

    # Run DFT calculations
    results_summary = []
    n_total = len(all_configs)
    n_success = 0
    n_failed = 0

    print(f"Starting DFT calculations ({n_total} configs)...\n", flush=True)

    for idx, (atoms, label, is_slab) in enumerate(all_configs, start=1):
        t0 = time.time()

        try:
            # Run GPAW
            results = run_gpaw_single_point(atoms, label, is_slab)

            # Save to XYZ
            save_to_extxyz(atoms, results, output_path)

            # Log
            max_force = np.max(np.linalg.norm(results['forces'], axis=1))
            elapsed = time.time() - t0

            print(f"[{idx}/{n_total}] {label}: "
                  f"E={results['energy']:.4f} eV, "
                  f"max|F|={max_force:.4f} eV/Å "
                  f"({elapsed:.1f}s)",
                  flush=True)

            n_success += 1

            # Summary stats
            results_summary.append({
                'config_type': label,
                'energy': results['energy'],
                'max_force': float(max_force),
                'n_atoms': len(atoms),
                'time_s': elapsed,
            })

        except Exception as e:
            print(f"[{idx}/{n_total}] {label}: FAILED", flush=True)
            print(f"  Error: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            n_failed += 1

    # Save summary JSON
    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    summary_data = {
        'n_total': n_total,
        'n_success': n_success,
        'n_failed': n_failed,
        'output_file': str(output_path),
        'results': results_summary,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    # Final report
    print(f"\n{'='*60}", flush=True)
    print(f"DFT data generation complete!", flush=True)
    print(f"  Success: {n_success}/{n_total}", flush=True)
    print(f"  Failed:  {n_failed}/{n_total}", flush=True)
    print(f"  Output:  {output_path}", flush=True)
    print(f"  Summary: {summary_path}", flush=True)
    print(f"{'='*60}", flush=True)

    # DONE marker for /go monitoring (prevent idle instance waste)
    done_path = output_path.parent / 'DONE'
    done_path.write_text(
        f"completed {n_success}/{n_total} at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
    )


if __name__ == '__main__':
    main()
