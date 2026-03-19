#!/usr/bin/env python3
"""
Generate DFT training configurations for smythite (Fe9S11).

Smythite: rare intermediate between mackinawite and greigite.
Rhombohedral/hexagonal structure, a≈3.47 Å, c≈34.5 Å (too large for DFT).

APPROXIMATION: Build from NiAs-type FeS (hexagonal P6_3/mmc),
create 2×2×2 supercell (32 atoms), remove 3 Fe atoms to match
Fe:S ratio ≈ 9:11.

Result: Fe13S16 ≈ Fe9.75S12 (close enough for ML training).
Labeled as "smythite_approx".

Configurations (~30):
- Bulk equilibrium (1)
- Rattled 4×3 (12)
- Strained (10)
- Sheared (3)
- (001) slab + rattles (6)

GPAW settings: PBE, PW400, FermiDirac(0.1), convergence=1e-5, parallel augment_grids.
Output: extended XYZ with energy/forces/stress.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms
from ase.spacegroup import crystal
from ase.build import surface
from ase.calculators.calculator import Calculator
from gpaw import GPAW, PW, FermiDirac


def setup_logging(log_file: Path) -> None:
    """Configure logging to file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def build_smythite_approx_bulk() -> Atoms:
    """
    Build approximate smythite (Fe9S11) structure.

    Based on NiAs-type FeS (P6_3/mmc), 2×2×2 supercell,
    with 3 Fe atoms removed to approximate Fe:S = 9:11.

    Returns Fe13S16 structure labeled as smythite_approx.
    """
    # Start with NiAs-type FeS
    a = 3.446
    c = 5.877

    primitive = crystal(
        symbols=['Fe', 'S'],
        basis=[(0, 0, 0), (1/3, 2/3, 1/4)],
        spacegroup=194,
        cellpar=[a, a, c, 90, 90, 120]
    )

    # Create 2×2×2 supercell
    supercell = primitive * (2, 2, 2)
    # This gives 32 atoms: 16 Fe + 16 S

    # Remove 3 Fe atoms to get Fe:S ≈ 9:11
    fe_indices = [i for i, sym in enumerate(supercell.get_chemical_symbols()) if sym == 'Fe']

    # Select 3 Fe to remove (well-separated)
    np.random.seed(42)
    to_remove = sorted(np.random.choice(fe_indices, size=3, replace=False), reverse=True)

    for idx in to_remove:
        del supercell[idx]

    # Result: 29 atoms (13 Fe + 16 S) ≈ Fe13S16 ≈ Fe0.8125S
    # Target: Fe9S11 = Fe0.818S (close enough)

    supercell.info['note'] = 'Smythite approximation: Fe13S16 from NiAs supercell'
    supercell.info['composition'] = f'Fe{len([s for s in supercell.get_chemical_symbols() if s == "Fe"])}S{len([s for s in supercell.get_chemical_symbols() if s == "S"])}'

    return supercell


def create_gpaw_calculator() -> Calculator:
    """Create GPAW calculator with standard settings."""
    calc = GPAW(
        mode=PW(400),
        xc='PBE',
        occupations=FermiDirac(0.1),
        convergence={'energy': 1e-5},
        parallel={'augment_grids': True},
        txt=None
    )
    return calc


def generate_bulk_configs() -> List[Atoms]:
    """Generate bulk configurations: eq + rattles + strains + shears."""
    configs = []

    # 1. Equilibrium
    bulk = build_smythite_approx_bulk()
    bulk.info['config_type'] = 'smythite_approx_bulk_eq'
    configs.append(bulk)
    logging.info(f"Generated smythite_approx bulk eq: {bulk.info['composition']}")

    # 2. Rattled structures (4 amplitudes × 3 seeds = 12)
    np.random.seed(42)
    for amp in [0.05, 0.10, 0.15, 0.20]:
        for seed in range(3):
            rattled = bulk.copy()
            rng = np.random.RandomState(seed + int(amp * 1000))
            rattled.positions += rng.normal(0, amp, rattled.positions.shape)
            rattled.info['config_type'] = f'smythite_approx_bulk_rattle_amp{amp:.2f}_seed{seed}'
            configs.append(rattled)
    logging.info("Generated 12 rattled configurations")

    # 3. Strained structures (10 configs)
    for strain in [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]:
        strained = bulk.copy()
        cell = strained.cell.copy()
        cell *= (1 + strain)
        strained.set_cell(cell, scale_atoms=True)
        strained.info['config_type'] = f'smythite_approx_bulk_strain_{strain:+.2f}'
        configs.append(strained)

    # Anisotropic strains (a vs c)
    for da, dc in [(0.02, -0.02), (-0.02, 0.02), (0.03, 0.0), (0.0, 0.03)]:
        strained = bulk.copy()
        cell = strained.cell.array.copy()
        # Scale a-axes (hexagonal: first two vectors)
        cell[0:2] *= (1 + da)
        # Scale c-axis
        cell[2] *= (1 + dc)
        strained.set_cell(cell, scale_atoms=True)
        strained.info['config_type'] = f'smythite_approx_bulk_strain_a{da:+.2f}_c{dc:+.2f}'
        configs.append(strained)

    logging.info("Generated 10 strained configurations")

    # 4. Sheared structures (3 configs)
    for shear in [0.05, 0.10, 0.15]:
        sheared = bulk.copy()
        cell = sheared.cell.array.copy()
        cell[0, 1] += shear * cell[1, 1]  # xy shear
        sheared.set_cell(cell, scale_atoms=True)
        sheared.info['config_type'] = f'smythite_approx_bulk_shear_{shear:.2f}'
        configs.append(sheared)

    logging.info("Generated 3 sheared configurations")

    return configs


def generate_surface_configs() -> List[Atoms]:
    """Generate (001) surface + rattles (6 configs)."""
    configs = []

    bulk = build_smythite_approx_bulk()

    # (001) surface
    try:
        slab = surface(bulk, (0, 0, 1), layers=3, vacuum=10.0)
        slab.info['config_type'] = 'smythite_approx_001_slab'
        slab.info['composition'] = bulk.info.get('composition', 'Fe13S16')
        configs.append(slab)
        logging.info("Generated smythite_approx (001) slab")

        # Rattled slabs
        np.random.seed(100)
        for i, amp in enumerate([0.05, 0.10, 0.15, 0.20, 0.25]):
            rattled = slab.copy()
            rng = np.random.RandomState(100 + i)
            rattled.positions += rng.normal(0, amp, rattled.positions.shape)
            rattled.info['config_type'] = f'smythite_approx_001_slab_rattle_{amp:.2f}'
            rattled.info['composition'] = bulk.info.get('composition', 'Fe13S16')
            configs.append(rattled)

        logging.info("Generated 5 rattled slab configurations")

    except Exception as e:
        logging.error(f"Could not generate (001) surface: {e}")

    return configs


def calculate_configs(configs: List[Atoms], dry_run: bool = False) -> List[Atoms]:
    """Calculate energy, forces, stress for all configurations."""
    if dry_run:
        logging.info(f"DRY RUN: Would calculate {len(configs)} configurations")
        return configs

    calc = create_gpaw_calculator()
    results = []

    for i, atoms in enumerate(configs):
        try:
            logging.info(f"Calculating {i+1}/{len(configs)}: {atoms.info['config_type']}")
            atoms.calc = calc

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            try:
                stress = atoms.get_stress()
                atoms.info['stress'] = stress
            except Exception:
                logging.warning(f"Could not calculate stress for {atoms.info['config_type']}")

            atoms.info['energy'] = energy
            atoms.arrays['forces'] = forces

            results.append(atoms)
            logging.info(f"  E = {energy:.4f} eV")

        except Exception as e:
            logging.error(f"Failed to calculate {atoms.info['config_type']}: {e}")
            continue

    return results


def save_configs(configs: List[Atoms], output_file: Path, append: bool = False) -> None:
    """Save configurations to extended XYZ format."""
    from ase.io import write

    mode = 'a' if append else 'w'

    for atoms in configs:
        write(output_file, atoms, format='extxyz', append=(mode == 'a'))
        mode = 'a'

    logging.info(f"Saved {len(configs)} configurations to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DFT training configurations for smythite (Fe9S11 approximation)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('smythite_configs.xyz'),
        help='Output XYZ file (default: smythite_configs.xyz)'
    )
    parser.add_argument(
        '--log',
        type=Path,
        default=Path('smythite_configs.log'),
        help='Log file (default: smythite_configs.log)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate structures without DFT calculations'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume calculation (append to existing file)'
    )

    args = parser.parse_args()

    setup_logging(args.log)
    logging.info("=" * 60)
    logging.info("Smythite (Fe9S11 approx) DFT configuration generator")
    logging.info("=" * 60)

    # Generate all configurations
    all_configs = []

    logging.info("Generating bulk configurations...")
    all_configs.extend(generate_bulk_configs())

    logging.info("Generating surface configurations...")
    all_configs.extend(generate_surface_configs())

    logging.info(f"Total configurations generated: {len(all_configs)}")

    # Calculate energies and forces
    calculated = calculate_configs(all_configs, dry_run=args.dry_run)

    # Save results
    if calculated:
        save_configs(calculated, args.output, append=args.resume)
        logging.info(f"Complete! {len(calculated)} configurations written to {args.output}")
    else:
        logging.warning("No configurations calculated")

    logging.info("=" * 60)


if __name__ == '__main__':
    main()
