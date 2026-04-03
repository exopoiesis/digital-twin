#!/usr/bin/env python3
"""
Generate DFT training configurations for troilite (FeS).

Troilite: hexagonal P6_3/mmc (#194, NiAs-type), stoichiometric FeS.
Differs from mackinawite (tetragonal P4/nmm). Meteoritic sulfide.
Lattice: a=3.446 Å, c=5.877 Å (from ICSD).
Primitive cell: 4 atoms (2 Fe + 2 S).

Configurations (~40):
- Bulk equilibrium (1)
- Rattled structures 5×3 (15)
- Strained ±1-5% (10)
- Sheared (5)
- (001) slab + rattles (6)
- H adsorption 3 sites (3)

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
from ase.build import surface, add_adsorbate
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


def build_troilite_bulk() -> Atoms:
    """
    Build troilite (FeS) bulk structure.

    Hexagonal P6_3/mmc (#194), NiAs-type.
    a = 3.446 Å, c = 5.877 Å.
    Fe at 2a (0, 0, 0), S at 2c (1/3, 2/3, 1/4).
    """
    a = 3.446
    c = 5.877

    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0, 0, 0), (1/3, 2/3, 1/4)],
        spacegroup=194,
        cellpar=[a, a, c, 90, 90, 120]
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
        txt=None  # Individual logs disabled, use main log
    )
    return calc


def generate_bulk_configs() -> List[Atoms]:
    """Generate bulk configurations: eq + rattles + strains + shears."""
    configs = []

    # 1. Equilibrium
    bulk = build_troilite_bulk()
    bulk.info['config_type'] = 'troilite_bulk_eq'
    configs.append(bulk)
    logging.info("Generated troilite bulk equilibrium")

    # 2. Rattled structures (5 amplitudes × 3 seeds = 15)
    np.random.seed(42)
    for amp in [0.02, 0.05, 0.10, 0.15, 0.20]:
        for seed in range(3):
            rattled = bulk.copy()
            rng = np.random.RandomState(seed + int(amp * 1000))
            rattled.positions += rng.normal(0, amp, rattled.positions.shape)
            rattled.info['config_type'] = f'troilite_bulk_rattle_amp{amp:.2f}_seed{seed}'
            configs.append(rattled)
    logging.info(f"Generated 15 rattled configurations")

    # 3. Strained structures (±1-5%, 10 configs)
    for strain in [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]:
        strained = bulk.copy()
        cell = strained.cell.copy()
        cell *= (1 + strain)
        strained.set_cell(cell, scale_atoms=True)
        strained.info['config_type'] = f'troilite_bulk_strain_{strain:+.2f}'
        configs.append(strained)

    # Anisotropic strains (a vs c)
    for da, dc in [(0.02, -0.02), (-0.02, 0.02), (0.03, 0.0), (0.0, 0.03)]:
        strained = bulk.copy()
        cell = strained.cell.array.copy()
        # Scale a-axis
        cell[0:2] *= (1 + da)
        # Scale c-axis
        cell[2] *= (1 + dc)
        strained.set_cell(cell, scale_atoms=True)
        strained.info['config_type'] = f'troilite_bulk_strain_a{da:+.2f}_c{dc:+.2f}'
        configs.append(strained)

    logging.info(f"Generated 10 strained configurations")

    # 4. Sheared structures (5 configs)
    shears = [0.05, 0.10, 0.15, 0.20, 0.25]
    for shear in shears:
        sheared = bulk.copy()
        cell = sheared.cell.array.copy()
        cell[0, 1] += shear * cell[1, 1]  # xy shear
        sheared.set_cell(cell, scale_atoms=True)
        sheared.info['config_type'] = f'troilite_bulk_shear_{shear:.2f}'
        configs.append(sheared)

    logging.info(f"Generated 5 sheared configurations")

    return configs


def generate_surface_configs() -> List[Atoms]:
    """Generate (001) surface + rattles (6 configs)."""
    configs = []

    bulk = build_troilite_bulk()

    # (001) surface, 3 layers, vacuum 10 Å
    slab = surface(bulk, (0, 0, 1), layers=3, vacuum=10.0)
    slab.info['config_type'] = 'troilite_001_slab'
    configs.append(slab)
    logging.info("Generated troilite (001) slab")

    # Rattled slabs (5 configs)
    np.random.seed(100)
    for i, amp in enumerate([0.05, 0.10, 0.15, 0.20, 0.25]):
        rattled = slab.copy()
        rng = np.random.RandomState(100 + i)
        rattled.positions += rng.normal(0, amp, rattled.positions.shape)
        rattled.info['config_type'] = f'troilite_001_slab_rattle_{amp:.2f}'
        configs.append(rattled)

    logging.info("Generated 5 rattled slab configurations")

    return configs


def generate_adsorption_configs() -> List[Atoms]:
    """Generate H adsorption on (001) surface (3 sites)."""
    configs = []

    bulk = build_troilite_bulk()
    slab = surface(bulk, (0, 0, 1), layers=3, vacuum=10.0)

    # Find surface atoms (top layer)
    z_positions = slab.positions[:, 2]
    z_max = z_positions.max()
    surface_mask = z_positions > (z_max - 1.0)

    surface_indices = np.where(surface_mask)[0]

    # Select 3 adsorption sites
    if len(surface_indices) >= 3:
        ads_sites = surface_indices[:3]
    else:
        ads_sites = surface_indices

    for i, site_idx in enumerate(ads_sites):
        ads_slab = slab.copy()
        pos = slab.positions[site_idx] + np.array([0, 0, 1.5])  # 1.5 Å above surface
        add_adsorbate(ads_slab, 'H', position=pos[:2], height=1.5)
        ads_slab.info['config_type'] = f'troilite_001_H_ads_site{i}'
        configs.append(ads_slab)

    logging.info(f"Generated {len(ads_sites)} H adsorption configurations")

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
        mode = 'a'  # After first write, always append

    logging.info(f"Saved {len(configs)} configurations to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DFT training configurations for troilite (FeS)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('troilite_configs.xyz'),
        help='Output XYZ file (default: troilite_configs.xyz)'
    )
    parser.add_argument(
        '--log',
        type=Path,
        default=Path('troilite_configs.log'),
        help='Log file (default: troilite_configs.log)'
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
    logging.info("Troilite (FeS) DFT configuration generator")
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
