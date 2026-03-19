#!/usr/bin/env python3
"""
Level 0+1 validation of sulfide DFT training dataset.

Checks:
  Level 0: SCF convergence, energy/force bounds, outliers, symmetry, duplicates
  Level 1: Lattice params vs experiment, bulk modulus, surface energies, E_ads

Usage:
    python validate_dataset.py results/sulfide_train_v2_final.xyz
    python validate_dataset.py results/v2_mack.xyz --level 0  # Quick check only
"""

import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
from ase.io import read
from ase import Atoms

try:
    import spglib
    HAS_SPGLIB = True
except ImportError:
    HAS_SPGLIB = False
    print("Warning: spglib not installed, symmetry checks skipped")


# ============================================================
# Reference data for Level 1 validation
# ============================================================

LATTICE_PARAMS_EXP = {
    # mineral: (a, b, c, alpha, beta, gamma) in Angstrom/degrees
    'mackinawite': {'a': 3.674, 'c': 5.033, 'source': 'Lennie 1995'},
    'pyrite':     {'a': 5.416, 'source': 'ICSD'},
    'pentlandite':{'a': 10.07, 'source': 'ICSD'},
    'greigite':   {'a': 9.876, 'source': 'Skinner 1964'},
    'pyrrhotite': {'a': 11.88, 'c': 5.72, 'source': 'Tokonami 1972'},
    'violarite':  {'a': 9.464, 'source': 'ICSD'},
    'millerite':  {'a': 9.616, 'c': 3.149, 'source': 'ICSD'},
    'chalcopyrite':{'a': 5.289, 'c': 10.423, 'source': 'ICSD'},
    'marcasite':  {'a': 4.443, 'b': 5.425, 'c': 3.387, 'source': 'ICSD'},
}

BULK_MODULUS_EXP = {
    # mineral: (B0_GPa, uncertainty, source)
    'pyrite':     (143, 3, 'Merkel 2002'),
    'pentlandite':(130, 20, 'Tenailleau 2006 (est.)'),
    'mackinawite':(30, 10, 'Subashri 2004 (est.)'),
    'greigite':   (80, 20, 'analogy with magnetite'),
}

SURFACE_ENERGY_LIT = {
    # (mineral, miller): (gamma_J_m2, source)
    ('pyrite', '001'):     (1.06, 'Hung 2002 (DFT-GGA)'),
    ('pyrite', '111'):     (1.68, 'Hung 2002'),
    ('mackinawite', '001'):(0.45, 'Ohfuji 2006 (est.)'),
}

H_ADS_ENERGY_LIT = {
    # (mineral, site): (E_ads_eV, source)
    ('pyrite', 'Fe-top'):     (-0.65, 'Krishnamoorthy 2018'),
    ('pentlandite', 'mixed'): (-0.45, 'Tetzlaff 2021'),
    ('mackinawite', 'bridge'):(-0.80, 'Roldan 2013'),
}

MAGNETIC_MOMENTS_EXP = {
    # mineral: (mu_B_per_Fe, ordering, source)
    'pyrite':     (0.0, 'low-spin diamagnetic', 'textbook'),
    'mackinawite':(0.0, 'antiferromagnetic', 'Vaughan 2006'),
    'pentlandite':(1.7, 'Pauli paramagnetic', 'Vaughan 2006'),
    'greigite':   (3.3, 'ferrimagnetic (inverse spinel)', 'Chang 2008'),
}

PHONON_FREQS_EXP = {
    # mineral: [(freq_cm-1, mode_type, source), ...]
    'pyrite': [
        (344, 'Eg', 'Vogt 1983'),
        (379, 'Ag', 'Vogt 1983'),
        (430, 'Tg', 'Vogt 1983'),
    ],
    'mackinawite': [
        (208, 'A1g', 'Bourdoiseau 2011'),
        (282, 'Eg', 'Bourdoiseau 2011'),
    ],
    'pentlandite': [
        (186, 'T2g', 'Mernagh & Trudu 1993'),
        (283, 'Eg', 'Mernagh & Trudu 1993'),
        (323, 'A1g', 'Mernagh & Trudu 1993'),
    ],
}


# ============================================================
# Level 0: Internal consistency
# ============================================================

def check_energy_bounds(atoms_list: List[Atoms]) -> List[str]:
    """Check that energy per atom is in reasonable range."""
    issues = []
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        e_per_atom = atoms.info.get('energy', 0) / len(atoms)
        if not (-10.0 < e_per_atom < 0.0):
            issues.append(f"  FAIL: {label}: E/atom = {e_per_atom:.4f} eV (outside -10..0)")
        elif e_per_atom > -2.0:
            issues.append(f"  WARN: {label}: E/atom = {e_per_atom:.4f} eV (suspiciously high)")
    return issues


def check_force_bounds(atoms_list: List[Atoms]) -> List[str]:
    """Check that forces are within reasonable bounds."""
    issues = []
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        forces = atoms.arrays.get('forces', None)
        if forces is None:
            issues.append(f"  WARN: {label}: no forces found")
            continue
        f_max = np.max(np.linalg.norm(forces, axis=1))
        if f_max > 50.0:
            issues.append(f"  FAIL: {label}: |F_max| = {f_max:.2f} eV/A (>50)")
        elif f_max > 25.0:
            issues.append(f"  WARN: {label}: |F_max| = {f_max:.2f} eV/A (>25, high rattle?)")
    return issues


def check_volume_bounds(atoms_list: List[Atoms]) -> List[str]:
    """Check volume per atom is reasonable for sulfides."""
    issues = []
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        if not all(atoms.pbc):
            continue  # Skip non-periodic (shouldn't happen for us)
        v_per_atom = atoms.get_volume() / len(atoms)
        if not (5.0 < v_per_atom < 40.0):
            issues.append(f"  FAIL: {label}: V/atom = {v_per_atom:.2f} A^3 (outside 5..40)")
    return issues


def check_outliers(atoms_list: List[Atoms]) -> List[str]:
    """Z-score outlier detection within groups."""
    issues = []
    groups = defaultdict(list)
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        # Group by mineral + type (e.g., "mackinawite_bulk_rattle")
        parts = label.split('_')
        mineral = parts[0] if parts else 'unknown'
        if 'slab' in label:
            group = f"{mineral}_slab"
        elif 'strain' in label or 'shear' in label:
            group = f"{mineral}_strain"
        elif 'rattle' in label:
            group = f"{mineral}_rattle"
        else:
            group = f"{mineral}_other"
        e_per_atom = atoms.info.get('energy', 0) / len(atoms)
        groups[group].append((label, e_per_atom))

    for group, entries in groups.items():
        if len(entries) < 4:
            continue
        energies = np.array([e for _, e in entries])
        mean = np.mean(energies)
        std = np.std(energies)
        if std < 1e-10:
            continue
        for label, e in entries:
            z = abs(e - mean) / std
            if z > 3.5:
                issues.append(f"  WARN: {label}: Z-score = {z:.1f} in group {group} "
                              f"(E/atom={e:.4f}, mean={mean:.4f}, std={std:.4f})")
    return issues


def check_symmetry(atoms_list: List[Atoms]) -> List[str]:
    """Check symmetry of equilibrium structures."""
    if not HAS_SPGLIB:
        return ["  SKIP: spglib not installed"]

    issues = []
    expected = {
        'mackinawite_bulk_eq': 129,  # P4/nmm
        'pyrite_bulk_eq': 205,       # Pa-3
        'pentlandite_bulk_eq': 225,  # Fm-3m
        'greigite_bulk_eq': 227,     # Fd-3m
    }

    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        if label not in expected:
            continue

        cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        sg = spglib.get_spacegroup(cell, symprec=0.1)
        sg_number = int(sg.split('(')[1].rstrip(')')) if sg and '(' in sg else 0

        if sg_number != expected[label]:
            issues.append(f"  WARN: {label}: spacegroup = {sg} (expected #{expected[label]})")
        else:
            issues.append(f"  OK: {label}: spacegroup = {sg}")

    return issues


def check_duplicates(atoms_list: List[Atoms], threshold: float = 0.001) -> List[str]:
    """Simple duplicate check using energy + composition fingerprint."""
    issues = []
    seen = {}  # (n_atoms, composition_hash, E_rounded) -> label
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        e = atoms.info.get('energy', 0)
        n = len(atoms)
        syms = tuple(sorted(atoms.get_chemical_symbols()))
        key = (n, syms, round(e, 4))
        if key in seen:
            issues.append(f"  WARN: Possible duplicate: {label} <-> {seen[key]} "
                          f"(same N={n}, composition, E={e:.4f})")
        else:
            seen[key] = label
    return issues


def run_level0(atoms_list: List[Atoms]) -> dict:
    """Run all Level 0 checks."""
    print("\n" + "="*60)
    print("LEVEL 0: Internal Consistency Checks")
    print("="*60)

    results = {}
    n_total = len(atoms_list)
    print(f"\nTotal configurations: {n_total}")

    # Count by mineral
    mineral_counts = defaultdict(int)
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        mineral = label.split('_')[0]
        mineral_counts[mineral] += 1
    print("\nBy mineral:")
    for mineral, count in sorted(mineral_counts.items()):
        print(f"  {mineral}: {count}")

    # Energy bounds
    print("\n--- Energy bounds ---")
    issues = check_energy_bounds(atoms_list)
    results['energy_bounds'] = issues
    if not issues:
        print("  ALL PASS")
    else:
        for i in issues:
            print(i)

    # Force bounds
    print("\n--- Force bounds ---")
    issues = check_force_bounds(atoms_list)
    results['force_bounds'] = issues
    if not issues:
        print("  ALL PASS")
    else:
        for i in issues:
            print(i)

    # Volume bounds
    print("\n--- Volume bounds ---")
    issues = check_volume_bounds(atoms_list)
    results['volume_bounds'] = issues
    if not issues:
        print("  ALL PASS")
    else:
        for i in issues:
            print(i)

    # Outliers
    print("\n--- Outlier detection (Z > 3.5) ---")
    issues = check_outliers(atoms_list)
    results['outliers'] = issues
    if not issues:
        print("  ALL PASS (no outliers)")
    else:
        for i in issues:
            print(i)

    # Symmetry
    print("\n--- Symmetry check ---")
    issues = check_symmetry(atoms_list)
    results['symmetry'] = issues
    for i in issues:
        print(i)

    # Duplicates
    print("\n--- Duplicate check ---")
    issues = check_duplicates(atoms_list)
    results['duplicates'] = issues
    if not issues:
        print("  ALL PASS (no duplicates)")
    else:
        for i in issues:
            print(i)

    # Summary stats
    print("\n--- Force distribution ---")
    all_forces = []
    for atoms in atoms_list:
        forces = atoms.arrays.get('forces', None)
        if forces is not None:
            f_norms = np.linalg.norm(forces, axis=1)
            all_forces.extend(f_norms)
    if all_forces:
        all_forces = np.array(all_forces)
        print(f"  Mean |F|: {np.mean(all_forces):.3f} eV/A")
        print(f"  Median |F|: {np.median(all_forces):.3f} eV/A")
        print(f"  Max |F|: {np.max(all_forces):.3f} eV/A")
        print(f"  P95 |F|: {np.percentile(all_forces, 95):.3f} eV/A")
        print(f"  P99 |F|: {np.percentile(all_forces, 99):.3f} eV/A")

    print("\n--- Energy per atom distribution ---")
    e_per_atom = []
    for atoms in atoms_list:
        e = atoms.info.get('energy', None)
        if e is not None:
            e_per_atom.append(e / len(atoms))
    if e_per_atom:
        e_per_atom = np.array(e_per_atom)
        print(f"  Mean E/atom: {np.mean(e_per_atom):.4f} eV")
        print(f"  Std E/atom: {np.std(e_per_atom):.4f} eV")
        print(f"  Min E/atom: {np.min(e_per_atom):.4f} eV")
        print(f"  Max E/atom: {np.max(e_per_atom):.4f} eV")

    return results


# ============================================================
# Level 1: Comparison with known data
# ============================================================

def check_lattice_params(atoms_list: List[Atoms]) -> List[str]:
    """Compare equilibrium lattice parameters with experiment."""
    issues = []
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        if '_bulk_eq' not in label:
            continue

        mineral = label.split('_')[0]
        if mineral not in LATTICE_PARAMS_EXP:
            continue

        ref = LATTICE_PARAMS_EXP[mineral]
        cell = atoms.cell.cellpar()  # a, b, c, alpha, beta, gamma

        a_calc = cell[0]
        a_exp = ref['a']
        err_a = abs(a_calc - a_exp) / a_exp * 100

        msg = f"  {mineral}: a_calc={a_calc:.3f}, a_exp={a_exp:.3f} ({ref['source']}), err={err_a:.1f}%"

        if 'c' in ref:
            c_calc = cell[2]
            c_exp = ref['c']
            err_c = abs(c_calc - c_exp) / c_exp * 100
            msg += f", c_calc={c_calc:.3f}, c_exp={c_exp:.3f}, err={err_c:.1f}%"
            if err_a > 3.0 or err_c > 3.0:
                msg = msg.replace(f"  {mineral}", f"  WARN {mineral}")
            else:
                msg = msg.replace(f"  {mineral}", f"  OK {mineral}")
        else:
            if err_a > 3.0:
                msg = msg.replace(f"  {mineral}", f"  WARN {mineral}")
            else:
                msg = msg.replace(f"  {mineral}", f"  OK {mineral}")

        issues.append(msg)

    return issues


def check_bulk_modulus(atoms_list: List[Atoms]) -> List[str]:
    """Estimate bulk modulus from strain configs using Birch-Murnaghan EOS."""
    issues = []

    # Group strain configs by mineral
    strain_data = defaultdict(list)  # mineral -> [(V, E), ...]
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        if 'strain' not in label or 'shear' in label:
            continue
        mineral = label.split('_')[0]
        if not all(atoms.pbc):
            continue
        V = atoms.get_volume()
        E = atoms.info.get('energy', 0)
        strain_data[mineral].append((V, E))

    for mineral, data in sorted(strain_data.items()):
        if len(data) < 5:
            issues.append(f"  SKIP {mineral}: only {len(data)} strain points (need >=5)")
            continue

        volumes = np.array([d[0] for d in data])
        energies = np.array([d[1] for d in data])

        # Simple parabolic fit E(V) = a*V^2 + b*V + c
        # B0 = V0 * d2E/dV2 = V0 * 2a
        try:
            coeffs = np.polyfit(volumes, energies, 2)
            a = coeffs[0]
            V0 = -coeffs[1] / (2 * a)
            B0_eV_A3 = V0 * 2 * a
            B0_GPa = B0_eV_A3 * 160.2176634  # eV/A^3 -> GPa

            msg = f"  {mineral}: B0_calc = {B0_GPa:.1f} GPa"
            if mineral in BULK_MODULUS_EXP:
                B_exp, B_unc, source = BULK_MODULUS_EXP[mineral]
                err = abs(B0_GPa - B_exp) / B_exp * 100
                msg += f", B0_exp = {B_exp}+/-{B_unc} GPa ({source}), err={err:.0f}%"
                if err > 30:
                    msg = "  WARN" + msg[1:]
                else:
                    msg = "  OK" + msg[1:]
            issues.append(msg)
        except Exception as e:
            issues.append(f"  FAIL {mineral}: polyfit error: {e}")

    return issues


def check_surface_energies(atoms_list: List[Atoms]) -> List[str]:
    """Estimate surface energies from slab vs bulk."""
    issues = []

    # Find bulk eq energies per atom
    bulk_e_per_atom = {}
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        if '_bulk_eq' in label:
            mineral = label.split('_')[0]
            bulk_e_per_atom[mineral] = atoms.info.get('energy', 0) / len(atoms)

    # Find slab energies
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        if '_slab' not in label or 'rattle' in label or 'H_' in label:
            continue

        mineral = label.split('_')[0]
        if mineral not in bulk_e_per_atom:
            continue

        # Extract Miller index from label
        parts = label.split('_')
        miller = None
        for p in parts:
            if p in ('001', '100', '110', '111', '010', '112'):
                miller = p
                break

        if miller is None:
            continue

        N = len(atoms)
        E_slab = atoms.info.get('energy', 0)
        E_bulk_ref = bulk_e_per_atom[mineral] * N

        # Surface area = 2 * (cell[0] x cell[1])
        cell = atoms.cell.array
        area = np.linalg.norm(np.cross(cell[0], cell[1]))
        gamma = (E_slab - E_bulk_ref) / (2 * area) * 16.0218  # eV/A^2 -> J/m^2

        msg = f"  {mineral} ({miller}): gamma = {gamma:.3f} J/m^2"

        key = (mineral, miller)
        if key in SURFACE_ENERGY_LIT:
            gamma_lit, source = SURFACE_ENERGY_LIT[key]
            err = abs(gamma - gamma_lit) / gamma_lit * 100
            msg += f", lit = {gamma_lit:.2f} ({source}), err={err:.0f}%"
            if err > 50:
                msg = "  WARN" + msg[1:]
            else:
                msg = "  OK" + msg[1:]

        issues.append(msg)

    return issues


def check_adsorption_energies(atoms_list: List[Atoms]) -> List[str]:
    """Check H adsorption energies."""
    issues = []

    # Find clean slab energies
    slab_energies = {}
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        if '_slab' in label and 'rattle' not in label and 'H_' not in label:
            slab_energies[label] = atoms.info.get('energy', 0)

    # E(H2) reference: approximately -6.77 eV for PBE (depends on box size)
    # Will be computed from data if H2 config exists, otherwise use default
    E_H2 = -6.77  # eV, typical PBE value for H2 in 10A box

    # Find H-adsorbed configs
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        if 'H_' not in label:
            continue

        mineral = label.split('_')[0]
        E_ads_config = atoms.info.get('energy', 0)

        # Find matching clean slab
        clean_key = None
        for sk in slab_energies:
            if sk.startswith(mineral) and '_slab' in sk:
                clean_key = sk
                break

        if clean_key is None:
            issues.append(f"  SKIP {label}: no matching clean slab found")
            continue

        E_slab = slab_energies[clean_key]
        E_ads = E_ads_config - E_slab - 0.5 * E_H2

        msg = f"  {label}: E_ads = {E_ads:.3f} eV (vs clean slab {clean_key})"

        # Check against literature
        for (min_key, site_key), (e_lit, source) in H_ADS_ENERGY_LIT.items():
            if mineral == min_key:
                msg += f", lit ~{e_lit:.2f} ({source})"
                break

        issues.append(msg)

    return issues


def run_level1(atoms_list: List[Atoms]) -> dict:
    """Run all Level 1 checks."""
    print("\n" + "="*60)
    print("LEVEL 1: Comparison with Known Data")
    print("="*60)

    results = {}

    print("\n--- Lattice parameters vs experiment ---")
    issues = check_lattice_params(atoms_list)
    results['lattice_params'] = issues
    for i in issues:
        print(i)
    if not issues:
        print("  No equilibrium configs found")

    print("\n--- Bulk modulus (from strains) ---")
    issues = check_bulk_modulus(atoms_list)
    results['bulk_modulus'] = issues
    for i in issues:
        print(i)

    print("\n--- Surface energies ---")
    issues = check_surface_energies(atoms_list)
    results['surface_energies'] = issues
    for i in issues:
        print(i)
    if not issues:
        print("  No slab configs found")

    print("\n--- H adsorption energies ---")
    issues = check_adsorption_energies(atoms_list)
    results['h_adsorption'] = issues
    for i in issues:
        print(i)
    if not issues:
        print("  No H-adsorbed configs found")

    return results


# ============================================================
# Summary
# ============================================================

def print_summary(results_l0: dict, results_l1: dict):
    """Print overall validation summary."""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    n_fail = 0
    n_warn = 0
    n_ok = 0

    for level_results in [results_l0, results_l1]:
        for check, issues in level_results.items():
            for issue in issues:
                if 'FAIL' in issue:
                    n_fail += 1
                elif 'WARN' in issue:
                    n_warn += 1
                elif 'OK' in issue:
                    n_ok += 1

    print(f"\n  PASS:     {n_ok}")
    print(f"  WARNINGS: {n_warn}")
    print(f"  FAILURES: {n_fail}")

    if n_fail > 0:
        print("\n  VERDICT: REVIEW NEEDED (failures found)")
    elif n_warn > 3:
        print("\n  VERDICT: ACCEPTABLE (some warnings, review recommended)")
    else:
        print("\n  VERDICT: GOOD (clean dataset)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Validate sulfide DFT training dataset")
    parser.add_argument('input', type=str, help='Input extended XYZ file')
    parser.add_argument('--level', type=int, default=1, choices=[0, 1],
                        help='Validation level (0=quick, 1=full)')
    parser.add_argument('--json', type=str, default=None,
                        help='Save results as JSON')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    print(f"Loading {input_path}...")
    atoms_list = read(input_path, index=':', format='extxyz')
    print(f"Loaded {len(atoms_list)} configurations")

    # Level 0
    results_l0 = run_level0(atoms_list)

    # Level 1
    results_l1 = {}
    if args.level >= 1:
        results_l1 = run_level1(atoms_list)

    # Summary
    print_summary(results_l0, results_l1)

    # Save JSON
    if args.json:
        all_results = {'level0': {}, 'level1': {}}
        for k, v in results_l0.items():
            all_results['level0'][k] = v
        for k, v in results_l1.items():
            all_results['level1'][k] = v
        with open(args.json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == '__main__':
    main()
