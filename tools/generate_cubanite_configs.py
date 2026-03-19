#!/usr/bin/env python3
"""
Generate DFT training configurations for cubanite (CuFe2S3).

Cubanite: orthorhombic Pcmn (#62, standard setting Pnma).
Lattice: a=6.467 Å, b=11.117 Å, c=6.233 Å.
Conventional cell: 24 atoms (4 Cu + 8 Fe + 12 S).

Approximate positions (Wyckoff):
- Cu at 4c: (0.25, 0.088, 0.0)
- Fe at 8d: (0.375, 0.246, 0.25)
- S at 4c: (0.25, 0.367, 0.0) and 8d: (0.125, 0.130, 0.25)

Configurations (~40):
- Bulk equilibrium (1)
- Rattled 5×3 (15)
- Strained (10)
- Sheared (3)
- (010) slab + rattles (6)
- H adsorption (3)

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


def build_cubanite_bulk() -> Atoms:
    """
    Build cubanite (CuFe2S3) bulk structure.

    Orthorhombic Pcmn (#62, standard Pnma).
    a = 6.467 Å, b = 11.117 Å, c = 6.233 Å.

    Approximate structure with Cu, Fe, S positions.
    """
    a = 6.467
    b = 11.117
    c = 6.233

    # Simplified positions for CuFe2S3
    # Cu at 4c, Fe at 8d, S at mixed sites
    atoms = crystal(
        symbols=['Cu', 'Cu', 'Fe', 'Fe', 'Fe', 'Fe', 'S', 'S', 'S', 'S', 'S', 'S'],
        basis=[
            (0.25, 0.088, 0.0),      # Cu 4c
            (0.75, 0.912, 0.0),      # Cu 4c
            (0.375, 0.246, 0.25),    # Fe 8d
            (0.625, 0.754, 0.75),    # Fe 8d
            (0.875, 0.754, 0.25),    # Fe 8d
            (0.125, 0.246, 0.75),    # Fe 8d
            (0.25, 0.367, 0.0),      # S 4c
            (0.75, 0.633, 0.0),      # S 4c
            (0.125, 0.130, 0.25),    # S 8d
            (0.875, 0.870, 0.75),    # S 8d
            (0.375, 0.870, 0.25),    # S 8d
            (0.625, 0.130, 0.75),    # S 8d
        ],
        spacegroup=62,
        cellpar=[a, b, c, 90, 90, 90]
    )

    return atoms


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
    bulk = build_cubanite_bulk()
    bulk.info['config_type'] = 'cubanite_bulk_eq'
    configs.append(bulk)
    logging.info("Generated cubanite bulk equilibrium")

    # 2. Rattled structures (5 amplitudes × 3 seeds = 15)
    np.random.seed(42)
    for amp in [0.02, 0.05, 0.10, 0.15, 0.20]:
        for seed in range(3):
            rattled = bulk.copy()
            rng = np.random.RandomState(seed + int(amp * 1000))
            rattled.positions += rng.normal(0, amp, rattled.positions.shape)
            rattled.info['config_type'] = f'cubanite_bulk_rattle_amp{amp:.2f}_seed{seed}'
            configs.append(rattled)
    logging.info("Generated 15 rattled configurations")

    # 3. Strained structures (10 configs)
    for strain in [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]:
        strained = bulk.copy()
        cell = strained.cell.copy()
        cell *= (1 + strain)
        strained.set_cell(cell, scale_atoms=True)
        strained.info['config_type'] = f'cubanite_bulk_strain_{strain:+.2f}'
        configs.append(strained)

    # Anisotropic strains (a, b, c different)
    for da, db in [(0.02, -0.02), (-0.02, 0.02), (0.03, 0.0), (0.0, 0.03)]:
        strained = bulk.copy()
        cell = strained.cell.array.copy()
        cell[0] *= (1 + da)
        cell[1] *= (1 + db)
        strained.set_cell(cell, scale_atoms=True)
        strained.info['config_type'] = f'cubanite_bulk_strain_a{da:+.2f}_b{db:+.2f}'
        configs.append(strained)

    logging.info("Generated 10 strained configurations")

    # 4. Sheared structures (3 configs)
    for shear in [0.05, 0.10, 0.15]:
        sheared = bulk.copy()
        cell = sheared.cell.array.copy()
        cell[0, 1] += shear * cell[1, 1]  # xy shear
        sheared.set_cell(cell, scale_atoms=True)
        sheared.info['config_type'] = f'cubanite_bulk_shear_{shear:.2f}'
        configs.append(sheared)

    logging.info("Generated 3 sheared configurations")

    return configs


def generate_surface_configs() -> List[Atoms]:
    """Generate (010) surface + rattles (6 configs)."""
    configs = []

    bulk = build_cubanite_bulk()

    # (010) surface (b-axis normal)
    try:
        slab = surface(bulk, (0, 1, 0), layers=3, vacuum=10.0)
        slab.info['config_type'] = 'cubanite_010_slab'
        configs.append(slab)
        logging.info("Generated cubanite (010) slab")

        # Rattled slabs
        np.random.seed(100)
        for i, amp in enumerate([0.05, 0.10, 0.15, 0.20, 0.25]):
            rattled = slab.copy()
            rng = np.random.RandomState(100 + i)
            rattled.positions += rng.normal(0, amp, rattled.positions.shape)
            rattled.info['config_type'] = f'cubanite_010_slab_rattle_{amp:.2f}'
            configs.append(rattled)

        logging.info("Generated 5 rattled slab configurations")

    except Exception as e:
        logging.error(f"Could not generate (010) surface: {e}")

    return configs


def generate_adsorption_configs() -> List[Atoms]:
    """Generate H adsorption on (010) surface (3 sites)."""
    configs = []

    bulk = build_cubanite_bulk()

    try:
        slab = surface(bulk, (0, 1, 0), layers=3, vacuum=10.0)

        # Find surface atoms
        y_positions = slab.positions[:, 1]  # (010) means y is normal
        y_max = y_positions.max()
        surface_mask = y_positions > (y_max - 1.0)
        surface_indices = np.where(surface_mask)[0]

        if len(surface_indices) >= 3:
            ads_sites = surface_indices[:3]

            for i, site_idx in enumerate(ads_sites):
                ads_slab = slab.copy()
                pos = slab.positions[site_idx]
                # Add H 1.5 Å above surface in y-direction
                h_pos = pos.copy()
                h_pos[1] += 1.5
                ads_slab.append('H')
                ads_slab.positions[-1] = h_pos
                ads_slab.info['config_type'] = f'cubanite_010_H_ads_site{i}'
                configs.append(ads_slab)

            logging.info(f"Generated {len(ads_sites)} H adsorption configurations")
        else:
            logging.warning("Not enough surface sites for adsorption")

    except Exception as e:
        logging.error(f"Could not generate adsorption configs: {e}")

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
        description="Generate DFT training configurations for cubanite (CuFe2S3)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('cubanite_configs.xyz'),
        help='Output XYZ file (default: cubanite_configs.xyz)'
    )
    parser.add_argument(
        '--log',
        type=Path,
        default=Path('cubanite_configs.log'),
        help='Log file (default: cubanite_configs.log)'
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
    logging.info("Cubanite (CuFe2S3) DFT configuration generator")
    logging.info("=" * 60)

    # Generate all configurations
    all_configs = []

    logging.info("Generating bulk configurations...")
    all_configs.extend(generate_bulk_configs())

    logging.info("Generating surface configurations...")
    all_configs.extend(generate_surface_configs())

    logging.info("Generating adsorption configurations...")
    all_configs.extend(generate_adsorption_configs())

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
