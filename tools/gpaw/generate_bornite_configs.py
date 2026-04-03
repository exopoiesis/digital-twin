#!/usr/bin/env python3
"""
Generate DFT training configurations for bornite (Cu5FeS4).

Bornite: cubic Fm-3m (#225, high-T phase).
Lattice: a=10.94 Å.
Simplified model: Cu5FeS4 stoichiometry.
Conventional cell: ~40 atoms (20 Cu + 4 Fe + 16 S).

Note: Actual bornite structure is complex with multiple Cu sites.
Using simplified cubic model with approximate positions.

Configurations (~50):
- Bulk equilibrium (1)
- Rattled 5×3 (15)
- Strained (10)
- Sheared (5)
- (001) slab + rattles (6)
- (112) slab + rattles (6)
- H/H2O/CO2 adsorption (6)
- S vacancies (3)

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
from ase.build import bulk, surface, add_adsorbate, molecule
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


def build_bornite_bulk() -> Atoms:
    """
    Build bornite (Cu5FeS4) bulk structure.

    Simplified cubic model based on Fm-3m (#225).
    a = 10.94 Å.

    Approximation: Create FCC-based structure with Cu/Fe/S
    arranged to match Cu5FeS4 stoichiometry.
    """
    a = 10.94

    # Build FCC base structure (Fm-3m)
    # Use Cu as base, then substitute atoms
    atoms = bulk('Cu', 'fcc', a=a, cubic=True)

    # Expand to ensure we have enough atoms
    atoms = atoms * (2, 2, 2)

    # Calculate target composition for Cu5FeS4
    total = len(atoms)
    n_S = int(total * 4/10)
    n_Fe = int(total * 1/10)
    n_Cu = total - n_S - n_Fe

    # Replace atoms to match stoichiometry
    symbols = ['Cu'] * n_Cu + ['Fe'] * n_Fe + ['S'] * n_S
    np.random.seed(42)
    np.random.shuffle(symbols)
    atoms.set_chemical_symbols(symbols[:total])

    atoms.info['note'] = 'Simplified bornite model, FCC-based'

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
    bulk_atoms = build_bornite_bulk()
    bulk_atoms.info['config_type'] = 'bornite_bulk_eq'
    configs.append(bulk_atoms)
    logging.info("Generated bornite bulk equilibrium")

    # 2. Rattled structures (5 amplitudes × 3 seeds = 15)
    np.random.seed(42)
    for amp in [0.02, 0.05, 0.10, 0.15, 0.20]:
        for seed in range(3):
            rattled = bulk_atoms.copy()
            rng = np.random.RandomState(seed + int(amp * 1000))
            rattled.positions += rng.normal(0, amp, rattled.positions.shape)
            rattled.info['config_type'] = f'bornite_bulk_rattle_amp{amp:.2f}_seed{seed}'
            configs.append(rattled)
    logging.info("Generated 15 rattled configurations")

    # 3. Strained structures (10 configs)
    for strain in [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]:
        strained = bulk_atoms.copy()
        cell = strained.cell.copy()
        cell *= (1 + strain)
        strained.set_cell(cell, scale_atoms=True)
        strained.info['config_type'] = f'bornite_bulk_strain_{strain:+.2f}'
        configs.append(strained)

    # Anisotropic strains
    for dx, dy in [(0.02, -0.02), (-0.02, 0.02), (0.03, 0.0), (0.0, 0.03)]:
        strained = bulk_atoms.copy()
        cell = strained.cell.array.copy()
        cell[0] *= (1 + dx)
        cell[1] *= (1 + dy)
        strained.set_cell(cell, scale_atoms=True)
        strained.info['config_type'] = f'bornite_bulk_strain_x{dx:+.2f}_y{dy:+.2f}'
        configs.append(strained)

    logging.info("Generated 10 strained configurations")

    # 4. Sheared structures (5 configs)
    for shear in [0.05, 0.10, 0.15, 0.20, 0.25]:
        sheared = bulk_atoms.copy()
        cell = sheared.cell.array.copy()
        cell[0, 1] += shear * cell[1, 1]
        sheared.set_cell(cell, scale_atoms=True)
        sheared.info['config_type'] = f'bornite_bulk_shear_{shear:.2f}'
        configs.append(sheared)

    logging.info("Generated 5 sheared configurations")

    return configs


def generate_surface_configs() -> List[Atoms]:
    """Generate (001) and (112) surfaces + rattles (12 configs total)."""
    configs = []

    bulk_atoms = build_bornite_bulk()

    # (001) surface
    try:
        slab_001 = surface(bulk_atoms, (0, 0, 1), layers=3, vacuum=10.0)
        slab_001.info['config_type'] = 'bornite_001_slab'
        configs.append(slab_001)
        logging.info("Generated bornite (001) slab")

        # Rattled (001) slabs
        np.random.seed(100)
        for i, amp in enumerate([0.05, 0.10, 0.15, 0.20, 0.25]):
            rattled = slab_001.copy()
            rng = np.random.RandomState(100 + i)
            rattled.positions += rng.normal(0, amp, rattled.positions.shape)
            rattled.info['config_type'] = f'bornite_001_slab_rattle_{amp:.2f}'
            configs.append(rattled)
        logging.info("Generated 5 rattled (001) slab configurations")

    except Exception as e:
        logging.error(f"Could not generate (001) surface: {e}")

    # (112) surface
    try:
        slab_112 = surface(bulk_atoms, (1, 1, 2), layers=3, vacuum=10.0)
        slab_112.info['config_type'] = 'bornite_112_slab'
        configs.append(slab_112)
        logging.info("Generated bornite (112) slab")

        # Rattled (112) slabs
        np.random.seed(200)
        for i, amp in enumerate([0.05, 0.10, 0.15, 0.20, 0.25]):
            rattled = slab_112.copy()
            rng = np.random.RandomState(200 + i)
            rattled.positions += rng.normal(0, amp, rattled.positions.shape)
            rattled.info['config_type'] = f'bornite_112_slab_rattle_{amp:.2f}'
            configs.append(rattled)
        logging.info("Generated 5 rattled (112) slab configurations")

    except Exception as e:
        logging.error(f"Could not generate (112) surface: {e}")

    return configs


def generate_adsorption_configs() -> List[Atoms]:
    """Generate H/H2O/CO2 adsorption on (001) surface (6 configs)."""
    configs = []

    bulk_atoms = build_bornite_bulk()

    try:
        slab = surface(bulk_atoms, (0, 0, 1), layers=3, vacuum=10.0)

        # Find surface positions
        z_positions = slab.positions[:, 2]
        z_max = z_positions.max()
        surface_mask = z_positions > (z_max - 1.5)
        surface_indices = np.where(surface_mask)[0]

        if len(surface_indices) >= 2:
            site1, site2 = surface_indices[:2]

            # H adsorption (2 sites)
            for i, site_idx in enumerate([site1, site2]):
                ads_slab = slab.copy()
                pos = slab.positions[site_idx]
                add_adsorbate(ads_slab, 'H', position=pos[:2], height=1.5)
                ads_slab.info['config_type'] = f'bornite_001_H_ads_site{i}'
                configs.append(ads_slab)

            # H2O adsorption (2 sites)
            for i, site_idx in enumerate([site1, site2]):
                ads_slab = slab.copy()
                pos = slab.positions[site_idx]
                h2o = molecule('H2O')
                h2o.translate(np.array([pos[0], pos[1], z_max + 2.0]))
                ads_slab += h2o
                ads_slab.info['config_type'] = f'bornite_001_H2O_ads_site{i}'
                configs.append(ads_slab)

            # CO2 adsorption (2 sites)
            for i, site_idx in enumerate([site1, site2]):
                ads_slab = slab.copy()
                pos = slab.positions[site_idx]
                co2 = molecule('CO2')
                co2.translate(np.array([pos[0], pos[1], z_max + 2.5]))
                ads_slab += co2
                ads_slab.info['config_type'] = f'bornite_001_CO2_ads_site{i}'
                configs.append(ads_slab)

            logging.info(f"Generated {len(configs)} adsorption configurations")

        else:
            logging.warning("Not enough surface sites for adsorption")

    except Exception as e:
        logging.error(f"Could not generate adsorption configs: {e}")

    return configs


def generate_vacancy_configs() -> List[Atoms]:
    """Generate S vacancy configurations (3 configs)."""
    configs = []

    bulk_atoms = build_bornite_bulk()

    # Find S atoms
    s_indices = [i for i, sym in enumerate(bulk_atoms.get_chemical_symbols()) if sym == 'S']

    if len(s_indices) >= 3:
        np.random.seed(300)
        selected = np.random.choice(s_indices, size=3, replace=False)

        for i, vac_idx in enumerate(selected):
            vac_atoms = bulk_atoms.copy()
            del vac_atoms[vac_idx]
            vac_atoms.info['config_type'] = f'bornite_bulk_S_vac_{i}'
            configs.append(vac_atoms)

        logging.info("Generated 3 S vacancy configurations")
    else:
        logging.warning("Not enough S atoms for vacancy generation")

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
        description="Generate DFT training configurations for bornite (Cu5FeS4)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('bornite_configs.xyz'),
        help='Output XYZ file (default: bornite_configs.xyz)'
    )
    parser.add_argument(
        '--log',
        type=Path,
        default=Path('bornite_configs.log'),
        help='Log file (default: bornite_configs.log)'
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
    logging.info("Bornite (Cu5FeS4) DFT configuration generator")
    logging.info("=" * 60)

    # Generate all configurations
    all_configs = []

    logging.info("Generating bulk configurations...")
    all_configs.extend(generate_bulk_configs())

    logging.info("Generating surface configurations...")
    all_configs.extend(generate_surface_configs())

    logging.info("Generating adsorption configurations...")
    all_configs.extend(generate_adsorption_configs())

    logging.info("Generating vacancy configurations...")
    all_configs.extend(generate_vacancy_configs())

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
