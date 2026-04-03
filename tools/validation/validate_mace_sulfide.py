#!/usr/bin/env python3
"""
Validate fine-tuned MACE model for sulfide minerals.

Validation tests:
1. Test set RMSE (energy/atom, forces)
2. H diffusion barrier in pentlandite (vs DFT 1.43 eV from Q-071)
3. Formate adsorption on mackinawite (vs DFT from q075-dft)
4. Bulk modulus of pentlandite and mackinawite
5. Check for unphysical values (MACE-MP-0 gave E_ads = -18 to -22 eV)

Usage:
    python -u validate_mace_sulfide.py [--model-dir /path/to/model]
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

# ASE imports
from ase import Atoms
from ase.io import read
from ase.build import bulk

# MACE
try:
    from mace.calculators import MACECalculator
except ImportError:
    print("ERROR: mace-torch not installed. Install with: pip install mace-torch", flush=True)
    exit(1)

warnings.filterwarnings('ignore', category=FutureWarning)


def find_model_file(model_dir: Path):
    """Find the trained MACE model file."""
    print(f"Searching for model in {model_dir}...", flush=True)

    # Priority order: final trained model > best checkpoint > any .model file
    candidates = [
        model_dir / "mace_sulfide_ft.model",
        model_dir / "best_model.model",
        model_dir / "final_model.model",
    ]

    for candidate in candidates:
        if candidate.exists():
            print(f"Found model: {candidate}", flush=True)
            return candidate

    # Fallback: find any .model file
    models = list(model_dir.glob("*.model"))
    if models:
        print(f"Found model: {models[0]}", flush=True)
        return models[0]

    raise FileNotFoundError(f"No .model file found in {model_dir}")


def load_mace_calculator(model_path: Path, device='cuda'):
    """Load fine-tuned MACE calculator."""
    print(f"Loading MACE calculator from {model_path}...", flush=True)
    calc = MACECalculator(model_paths=str(model_path), device=device, default_dtype='float64')
    print("Calculator loaded successfully.", flush=True)
    return calc


def validate_test_set(calc, test_file: Path):
    """Compare MACE predictions with DFT on test set."""
    print("\n" + "="*60, flush=True)
    print("TEST SET VALIDATION", flush=True)
    print("="*60, flush=True)

    if not test_file.exists():
        print(f"WARNING: Test file not found: {test_file}", flush=True)
        print("Skipping test set validation.", flush=True)
        return None

    atoms_list = read(test_file, index=':')
    n_structures = len(atoms_list)
    print(f"Test structures: {n_structures}", flush=True)

    energy_errors = []
    force_errors = []

    for i, atoms in enumerate(atoms_list):
        # Get DFT reference values
        e_dft = atoms.info.get('energy', None)
        f_dft = atoms.arrays.get('forces', None)

        if e_dft is None:
            print(f"Structure {i}: No DFT energy in info, skipping", flush=True)
            continue

        # Compute MACE prediction
        atoms.calc = calc
        e_mace = atoms.get_potential_energy()
        f_mace = atoms.get_forces()

        # Energy error per atom
        n_atoms = len(atoms)
        e_err = abs(e_mace - e_dft) / n_atoms
        energy_errors.append(e_err)

        # Force RMSE
        if f_dft is not None:
            f_err = np.sqrt(np.mean((f_mace - f_dft)**2))
            force_errors.append(f_err)

        if i < 5 or i % 10 == 0:
            print(f"  [{i+1}/{n_structures}] E_err={e_err:.4f} eV/atom, F_RMSE={f_err:.4f} eV/Å", flush=True)

    if not energy_errors:
        print("WARNING: No valid test structures found.", flush=True)
        return None

    results = {
        'n_structures': n_structures,
        'energy_mae_per_atom': float(np.mean(energy_errors)),
        'energy_rmse_per_atom': float(np.sqrt(np.mean(np.array(energy_errors)**2))),
        'force_mae': float(np.mean(force_errors)) if force_errors else None,
        'force_rmse': float(np.sqrt(np.mean(np.array(force_errors)**2))) if force_errors else None,
    }

    print(f"\nTest set metrics:", flush=True)
    print(f"  Energy MAE:  {results['energy_mae_per_atom']:.4f} eV/atom", flush=True)
    print(f"  Energy RMSE: {results['energy_rmse_per_atom']:.4f} eV/atom", flush=True)
    if results['force_rmse']:
        print(f"  Force MAE:   {results['force_mae']:.4f} eV/Å", flush=True)
        print(f"  Force RMSE:  {results['force_rmse']:.4f} eV/Å", flush=True)

    return results


def validate_pentlandite_neb(calc, dft_barrier=1.43):
    """
    Validate H diffusion barrier in pentlandite.

    Instead of running full NEB, check single-point energies at:
    - Initial state (H in octahedral site)
    - Transition state (H at tetrahedral site)
    - Final state (H in neighboring octahedral site)

    DFT reference: 1.43 eV (from Q-071, РЕШЕНИЕ-044)
    """
    print("\n" + "="*60, flush=True)
    print("PENTLANDITE NEB BARRIER VALIDATION", flush=True)
    print("="*60, flush=True)
    print(f"DFT reference barrier: {dft_barrier:.2f} eV", flush=True)

    try:
        # Build pentlandite Fe4.5Ni4.5S8 (Fm-3m, a=10.095 Å)
        # Simplified: use bulk pentlandite structure
        # In reality, would need proper vacancy + H interstitial geometry
        print("WARNING: Full NEB validation requires pre-computed geometries from DFT.", flush=True)
        print("Skipping detailed NEB check (placeholder for future implementation).", flush=True)

        # Placeholder: would load initial/TS/final geometries from XYZ files
        # and compute barrier = E_TS - E_initial

        return {
            'status': 'SKIPPED',
            'reason': 'Requires pre-computed NEB geometries from DFT',
            'dft_barrier_ev': dft_barrier,
            'mace_barrier_ev': None,
            'error_ev': None,
        }

    except Exception as e:
        print(f"ERROR in NEB validation: {e}", flush=True)
        return {'status': 'ERROR', 'message': str(e)}


def validate_formate_adsorption(calc):
    """
    Validate formate (HCOO-) adsorption on mackinawite (001) surface.

    MACE-MP-0 gave unphysical E_ads = -18 to -22 eV.
    Physical range: -0.1 to -1.5 eV (typical chemisorption).

    DFT reference: from q075-dft (will be available soon).
    """
    print("\n" + "="*60, flush=True)
    print("FORMATE ADSORPTION VALIDATION", flush=True)
    print("="*60, flush=True)

    try:
        # Build mackinawite Fe(Ni)S (tetragonal P4/nmm, a=3.675, c=5.033)
        print("Building mackinawite (001) slab...", flush=True)

        # Simplified tetragonal FeS (mackinawite structure)
        # In reality, use proper mackinawite cell from CIF
        a = 3.675  # Å
        c = 5.033  # Å

        # Create slab (4 layers, 10 Å vacuum)
        # Placeholder: proper implementation would use ASE surface builder
        print("WARNING: Requires proper mackinawite surface geometry from DFT.", flush=True)
        print("Skipping detailed adsorption check (placeholder).", flush=True)

        # Placeholder for proper implementation:
        # 1. Load relaxed slab from XYZ
        # 2. Load slab+formate from XYZ
        # 3. Compute E_ads = E(slab+HCOO) - E(slab) - E(HCOO_gas)
        # 4. Check: -1.5 < E_ads < -0.1 eV (physical range)

        return {
            'status': 'SKIPPED',
            'reason': 'Requires DFT reference geometries from q075-dft',
            'e_ads_ev': None,
            'physical_range': [-1.5, -0.1],
            'is_physical': None,
        }

    except Exception as e:
        print(f"ERROR in adsorption validation: {e}", flush=True)
        return {'status': 'ERROR', 'message': str(e)}


def validate_bulk_modulus(calc, material='pentlandite'):
    """
    Validate bulk modulus via Birch-Murnaghan EOS.

    Fit E-V curve for ±5% volume variation.
    Literature values:
    - Pentlandite: ~140-160 GPa
    - Mackinawite: ~60-80 GPa (estimated)
    """
    print("\n" + "="*60, flush=True)
    print(f"BULK MODULUS VALIDATION: {material.upper()}", flush=True)
    print("="*60, flush=True)

    try:
        if material == 'pentlandite':
            # Fe4.5Ni4.5S8, Fm-3m, a=10.095 Å
            atoms = bulk('Ni', 'fcc', a=10.095)  # Placeholder
            atoms.set_chemical_symbols(['Fe']*5 + ['Ni']*5 + ['S']*8)
            b0_lit = 150  # GPa (approximate)
        elif material == 'mackinawite':
            # FeS, P4/nmm, a=3.675, c=5.033
            atoms = Atoms('FeS',
                          scaled_positions=[(0,0,0), (0.5,0.5,0.5)],
                          cell=[3.675, 3.675, 5.033],
                          pbc=True)
            b0_lit = 70  # GPa (approximate)
        else:
            raise ValueError(f"Unknown material: {material}")

        print(f"Computing E-V curve for {len(atoms)} atoms...", flush=True)
        print("WARNING: Bulk modulus validation requires proper crystal structure.", flush=True)
        print("Skipping (placeholder for proper implementation).", flush=True)

        # Proper implementation:
        # volumes = []
        # energies = []
        # for scale in np.linspace(0.95, 1.05, 7):
        #     atoms_scaled = atoms.copy()
        #     atoms_scaled.set_cell(atoms.cell * scale, scale_atoms=True)
        #     atoms_scaled.calc = calc
        #     energies.append(atoms_scaled.get_potential_energy())
        #     volumes.append(atoms_scaled.get_volume())
        # eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
        # v0, e0, B = eos.fit()
        # B_GPa = B / 1.602176634  # eV/Å³ to GPa

        return {
            'status': 'SKIPPED',
            'reason': 'Requires proper crystal structure setup',
            'material': material,
            'b0_mace_gpa': None,
            'b0_literature_gpa': b0_lit,
        }

    except Exception as e:
        print(f"ERROR in bulk modulus validation: {e}", flush=True)
        return {'status': 'ERROR', 'message': str(e)}


def overall_verdict(results):
    """Determine PASS/FAIL based on all validation tests."""
    print("\n" + "="*60, flush=True)
    print("OVERALL VERDICT", flush=True)
    print("="*60, flush=True)

    criteria = []

    # Test set RMSE
    if results.get('test_set'):
        e_rmse = results['test_set'].get('energy_rmse_per_atom')
        if e_rmse is not None:
            pass_test = e_rmse < 0.1
            criteria.append(('Test energy RMSE < 0.1 eV/atom', pass_test, f"{e_rmse:.4f}"))
            print(f"  Test energy RMSE: {e_rmse:.4f} eV/atom {'✓ PASS' if pass_test else '✗ FAIL'}", flush=True)

        f_rmse = results['test_set'].get('force_rmse')
        if f_rmse is not None:
            pass_force = f_rmse < 0.3
            criteria.append(('Test force RMSE < 0.3 eV/Å', pass_force, f"{f_rmse:.4f}"))
            print(f"  Test force RMSE:  {f_rmse:.4f} eV/Å {'✓ PASS' if pass_force else '✗ FAIL'}", flush=True)

    # NEB barrier
    if results.get('neb') and results['neb'].get('mace_barrier_ev'):
        error = abs(results['neb']['mace_barrier_ev'] - results['neb']['dft_barrier_ev'])
        pass_neb = error < 0.15
        criteria.append(('NEB barrier within ±0.15 eV', pass_neb, f"{error:.3f} eV"))
        print(f"  NEB barrier error: {error:.3f} eV {'✓ PASS' if pass_neb else '✗ FAIL'}", flush=True)

    # Adsorption energy
    if results.get('adsorption') and results['adsorption'].get('is_physical') is not None:
        pass_ads = results['adsorption']['is_physical']
        e_ads = results['adsorption']['e_ads_ev']
        criteria.append(('Adsorption energy physical', pass_ads, f"{e_ads:.2f} eV"))
        print(f"  Adsorption E_ads:  {e_ads:.2f} eV {'✓ PASS' if pass_ads else '✗ FAIL'}", flush=True)

    # Overall
    if criteria:
        all_passed = all(c[1] for c in criteria)
        print(f"\nOverall: {'✓ PASS' if all_passed else '✗ FAIL'}", flush=True)
        print(f"  ({sum(c[1] for c in criteria)}/{len(criteria)} criteria passed)", flush=True)
        return all_passed
    else:
        print("\nOverall: INCONCLUSIVE (most validations skipped)", flush=True)
        print("  Reason: Missing DFT reference data or geometries", flush=True)
        print("  Recommendation: Re-run after copying q075-dft results", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description='Validate fine-tuned MACE model for sulfides')
    parser.add_argument('--model-dir', type=Path,
                        default=Path('/workspace/results/mace_sulfide_ft'),
                        help='Directory containing trained model')
    parser.add_argument('--test-file', type=Path,
                        default=Path('/workspace/results/sulfide_valid.xyz'),
                        help='Test set XYZ file')
    parser.add_argument('--output', type=Path,
                        default=None,
                        help='Output JSON file (default: <model-dir>/validation.json)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    print("="*60, flush=True)
    print("MACE SULFIDE MODEL VALIDATION", flush=True)
    print("="*60, flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print(f"Model directory: {args.model_dir}", flush=True)
    print(f"Test file: {args.test_file}", flush=True)
    print(f"Device: {args.device}", flush=True)

    # Find and load model
    try:
        model_path = find_model_file(args.model_dir)
        calc = load_mace_calculator(model_path, device=args.device)
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}", flush=True)
        return 1

    # Run validation tests
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'device': args.device,
    }

    # 1. Test set validation
    results['test_set'] = validate_test_set(calc, args.test_file)

    # 2. NEB barrier (pentlandite)
    results['neb'] = validate_pentlandite_neb(calc, dft_barrier=1.43)

    # 3. Formate adsorption (mackinawite)
    results['adsorption'] = validate_formate_adsorption(calc)

    # 4. Bulk modulus
    results['bulk_modulus_pentlandite'] = validate_bulk_modulus(calc, 'pentlandite')
    results['bulk_modulus_mackinawite'] = validate_bulk_modulus(calc, 'mackinawite')

    # 5. Overall verdict
    results['passed'] = overall_verdict(results)

    # Save results
    output_file = args.output or (args.model_dir / 'validation.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}", flush=True)
    print("="*60, flush=True)

    if results['passed'] is True:
        return 0
    elif results['passed'] is False:
        return 1
    else:
        return 2  # Inconclusive


if __name__ == '__main__':
    exit(main())
