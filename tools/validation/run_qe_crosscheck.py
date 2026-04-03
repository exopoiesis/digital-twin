#!/usr/bin/env python3
"""
Level 2 cross-code validation: run same configs in Quantum ESPRESSO and compare with GPAW.

Selects ~20 representative configs from the dataset, runs QE single-point,
and reports energy/force discrepancies.

Usage:
    python run_qe_crosscheck.py --input results/sulfide_train_v2_final.xyz --n-configs 20
    python run_qe_crosscheck.py --input results/sulfide_train_v2_final.xyz --configs-file crosscheck_selection.json

Expected agreement: |dE| < 5 meV/atom, |dF| < 0.05 eV/A (PAW vs PAW, same functional).
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from ase.io import read
from ase.calculators.espresso import Espresso

PSEUDO_DIR = os.environ.get('ESPRESSO_PSEUDO', '/opt/pseudopotentials')

# Pseudopotential mapping (SSSP Efficiency PBE)
PSEUDOPOTENTIALS = {
    'Fe': 'Fe.upf',
    'Ni': 'Ni.upf',
    'S':  'S.upf',
    'H':  'H.upf',
    'C':  'C.upf',
    'O':  'O.upf',
}

# QE input parameters (matched to GPAW as closely as possible)
QE_INPUT = {
    'ecutwfc': 40,      # Ry (~544 eV, slightly above GPAW 400 eV for safety)
    'ecutrho': 320,     # Ry (8x ecutwfc, standard for NC PPs)
    'input_dft': 'PBE',
    'nspin': 2,         # Spin-polarized
    'occupations': 'smearing',
    'smearing': 'fd',   # Fermi-Dirac (same as GPAW)
    'degauss': 0.004,   # Ry (~0.05 eV, close to GPAW's 0.1 eV FermiDirac)
    'conv_thr': 1e-8,   # Tight convergence
    'mixing_beta': 0.3,
    'electron_maxstep': 300,
    'tprnfor': True,    # Print forces
    'tstress': True,    # Print stress
}


def select_representative_configs(atoms_list, n_configs=20):
    """Select representative configs: ~3-4 per mineral, mix of types."""
    groups = defaultdict(list)
    for i, atoms in enumerate(atoms_list):
        label = atoms.info.get('config_type', f'config_{i}')
        mineral = label.split('_')[0]

        # Classify type
        if '_bulk_eq' in label:
            config_type = 'bulk_eq'
        elif '_rattle_0.05' in label:
            config_type = 'rattle_small'
        elif '_rattle_0.20' in label:
            config_type = 'rattle_large'
        elif '_strain_+3' in label or '_strain_-3' in label:
            config_type = 'strain'
        elif '_slab' in label and 'rattle' not in label and 'H_' not in label:
            config_type = 'slab'
        elif 'H_' in label:
            config_type = 'h_ads'
        else:
            continue  # Skip less important types

        groups[(mineral, config_type)].append(i)

    # Select one from each group, up to n_configs
    selected = []
    for key in sorted(groups.keys()):
        if len(selected) >= n_configs:
            break
        idx = groups[key][0]  # Take first of each type
        selected.append(idx)

    # If we need more, add random ones
    remaining = [i for i in range(len(atoms_list)) if i not in selected]
    np.random.seed(42)
    np.random.shuffle(remaining)
    while len(selected) < n_configs and remaining:
        selected.append(remaining.pop())

    return selected


def run_qe_single_point(atoms, label, work_dir):
    """Run QE single-point calculation."""
    calc_dir = Path(work_dir) / label
    calc_dir.mkdir(parents=True, exist_ok=True)

    # Determine k-points based on system size
    n_atoms = len(atoms)
    if not all(atoms.pbc):
        kpts = (1, 1, 1)
    elif n_atoms > 40:
        kpts = (2, 2, 1) if any(atoms.cell.cellpar()[:3] > 15) else (2, 2, 2)
    elif n_atoms > 15:
        kpts = (2, 2, 2)
    else:
        kpts = (4, 4, 4)

    # Check which pseudopotentials we need
    elements = set(atoms.get_chemical_symbols())
    pseudos = {}
    for el in elements:
        if el in PSEUDOPOTENTIALS:
            pp_file = Path(PSEUDO_DIR) / PSEUDOPOTENTIALS[el]
            if pp_file.exists():
                pseudos[el] = PSEUDOPOTENTIALS[el]
            else:
                print(f"  WARNING: {pp_file} not found, QE will fail for {el}")
                pseudos[el] = PSEUDOPOTENTIALS[el]
        else:
            print(f"  ERROR: No pseudopotential for {el}")
            return None

    # Set initial magnetic moments (Fe: 4, Ni: 2, others: 0)
    init_mag = []
    for sym in atoms.get_chemical_symbols():
        if sym == 'Fe':
            init_mag.append(4.0)
        elif sym == 'Ni':
            init_mag.append(2.0)
        else:
            init_mag.append(0.0)

    input_data = QE_INPUT.copy()
    input_data['starting_magnetization(1)'] = 0.5  # Will be overridden by ASE

    calc = Espresso(
        pseudopotentials=pseudos,
        pseudo_dir=PSEUDO_DIR,
        input_data={'system': input_data, 'control': {'calculation': 'scf', 'outdir': str(calc_dir)}},
        kpts=kpts,
        directory=str(calc_dir),
    )

    atoms_copy = atoms.copy()
    atoms_copy.calc = calc

    try:
        energy = atoms_copy.get_potential_energy()
        forces = atoms_copy.get_forces()
        stress = None
        if all(atoms_copy.pbc):
            try:
                stress = atoms_copy.get_stress(voigt=True)
            except Exception:
                pass
        return {'energy': energy, 'forces': forces, 'stress': stress}
    except Exception as e:
        print(f"  QE FAILED for {label}: {e}")
        return None


def compare_results(gpaw_atoms, qe_results, label):
    """Compare GPAW and QE results."""
    n_atoms = len(gpaw_atoms)
    e_gpaw = gpaw_atoms.info.get('energy', 0)
    f_gpaw = gpaw_atoms.arrays.get('forces', np.zeros((n_atoms, 3)))

    e_qe = qe_results['energy']
    f_qe = qe_results['forces']

    # Energy difference per atom
    de_per_atom = abs(e_gpaw - e_qe) / n_atoms
    de_per_atom_meV = de_per_atom * 1000

    # Force MAE
    f_diff = f_gpaw - f_qe
    f_mae = np.mean(np.abs(f_diff))
    f_max_diff = np.max(np.abs(f_diff))

    # Force cosine similarity (direction agreement)
    f_gpaw_norms = np.linalg.norm(f_gpaw, axis=1, keepdims=True)
    f_qe_norms = np.linalg.norm(f_qe, axis=1, keepdims=True)
    mask = (f_gpaw_norms.flatten() > 0.01) & (f_qe_norms.flatten() > 0.01)
    if mask.any():
        cos_sim = np.mean([
            np.dot(f_gpaw[i], f_qe[i]) / (f_gpaw_norms[i] * f_qe_norms[i])
            for i in range(n_atoms) if mask[i]
        ])
    else:
        cos_sim = 1.0

    result = {
        'label': label,
        'n_atoms': n_atoms,
        'E_gpaw': e_gpaw,
        'E_qe': e_qe,
        'dE_per_atom_meV': de_per_atom_meV,
        'F_MAE_eV_A': f_mae,
        'F_max_diff_eV_A': f_max_diff,
        'F_cosine_similarity': float(cos_sim),
    }

    # Verdict
    e_pass = de_per_atom_meV < 5.0
    f_pass = f_mae < 0.05
    result['E_pass'] = e_pass
    result['F_pass'] = f_pass
    result['overall_pass'] = e_pass and f_pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-code validation: GPAW vs QE")
    parser.add_argument('--input', type=str, required=True, help='Input GPAW extended XYZ')
    parser.add_argument('--n-configs', type=int, default=20, help='Number of configs to check')
    parser.add_argument('--work-dir', type=str, default='/tmp/qe_crosscheck', help='QE working directory')
    parser.add_argument('--output', type=str, default='crosscheck_results.json', help='Output JSON')
    parser.add_argument('--configs-file', type=str, default=None, help='JSON with specific config indices')
    args = parser.parse_args()

    # Load dataset
    print(f"Loading {args.input}...")
    atoms_list = read(args.input, index=':', format='extxyz')
    print(f"Loaded {len(atoms_list)} configurations")

    # Select configs
    if args.configs_file:
        with open(args.configs_file) as f:
            selected = json.load(f)
        print(f"Using {len(selected)} configs from {args.configs_file}")
    else:
        selected = select_representative_configs(atoms_list, args.n_configs)
        print(f"Selected {len(selected)} representative configs")

    # Run comparisons
    results = []
    for i, idx in enumerate(selected):
        atoms = atoms_list[idx]
        label = atoms.info.get('config_type', f'config_{idx}')
        print(f"\n[{i+1}/{len(selected)}] {label} ({len(atoms)} atoms)...")

        qe_result = run_qe_single_point(atoms, label, args.work_dir)
        if qe_result is None:
            results.append({'label': label, 'status': 'FAILED'})
            continue

        comparison = compare_results(atoms, qe_result, label)
        results.append(comparison)

        status = "PASS" if comparison['overall_pass'] else "FAIL"
        print(f"  dE/atom = {comparison['dE_per_atom_meV']:.2f} meV, "
              f"F_MAE = {comparison['F_MAE_eV_A']:.4f} eV/A, "
              f"cos_sim = {comparison['F_cosine_similarity']:.4f} -> {status}")

    # Summary
    print("\n" + "="*60)
    print("CROSS-CODE VALIDATION SUMMARY")
    print("="*60)

    n_pass = sum(1 for r in results if r.get('overall_pass', False))
    n_fail = sum(1 for r in results if 'overall_pass' in r and not r['overall_pass'])
    n_error = sum(1 for r in results if r.get('status') == 'FAILED')

    print(f"  PASS:   {n_pass}")
    print(f"  FAIL:   {n_fail}")
    print(f"  ERROR:  {n_error}")

    if results:
        de_values = [r['dE_per_atom_meV'] for r in results if 'dE_per_atom_meV' in r]
        fm_values = [r['F_MAE_eV_A'] for r in results if 'F_MAE_eV_A' in r]
        if de_values:
            print(f"\n  Mean dE/atom: {np.mean(de_values):.2f} meV (target < 5)")
            print(f"  Max  dE/atom: {np.max(de_values):.2f} meV")
        if fm_values:
            print(f"  Mean F_MAE:   {np.mean(fm_values):.4f} eV/A (target < 0.05)")
            print(f"  Max  F_MAE:   {np.max(fm_values):.4f} eV/A")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
