#!/usr/bin/env python3
"""
Level 3 physical properties validation for sulfide DFT dataset.

Extracts derived physical properties and compares with experimental data.
Complements validate_dataset.py (Level 0+1) with deeper analysis.

Properties extracted:
  3A. Elastic constants (C_ij from strain+shear configs)
  3B. Birch-Murnaghan equation of state (more rigorous than parabolic)
  3C. Magnetic moments per atom
  3D. Formation energies and convex hull distance
  3E. Force constant estimation (quasi-harmonic, no phonopy needed)
  3F. Adsorption energy summary (CO2, HCOO-, H2O if present)

Usage:
    python physical_properties_validation.py results/sulfide_train_v2_final.xyz
    python physical_properties_validation.py results/v2_mack.xyz --mineral mackinawite
    python physical_properties_validation.py results/*.xyz --json phys_props.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from ase.io import read
from ase import Atoms

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed, Birch-Murnaghan fit unavailable")


# ===========================================================================
#  Reference data
# ===========================================================================

# Elastic constants from literature (GPa)
# Format: mineral -> {C11, C12, C44, ...} or bulk modulus + shear modulus
ELASTIC_REF = {
    'pyrite': {
        'C11': 382, 'C12': 31, 'C44': 109,
        'B': 143, 'G': 133,
        'source': 'Simmons & Wang 1971, Merkel 2002',
    },
    'pentlandite': {
        'B': 130,  # Only bulk modulus estimated
        'source': 'Tenailleau 2006 (est.)',
    },
    'mackinawite': {
        'C33': 18, 'C11': 210,  # Highly anisotropic (layered)
        'B': 30,
        'source': 'Subashri 2004 (est.), Rickard 2006',
    },
}

# Formation energies (eV/atom) from Materials Project
# These are relative to elemental references (bcc Fe, ortho S, fcc Ni)
FORMATION_ENERGY_REF = {
    'pyrite':      {'Ef': -0.55, 'mp_id': 'mp-1522', 'source': 'Materials Project'},
    'mackinawite': {'Ef': -0.51, 'mp_id': 'mp-505531', 'source': 'Materials Project'},
    'pentlandite': {'Ef': -0.40, 'mp_id': 'mp-14005', 'source': 'Materials Project (est.)'},
    'greigite':    {'Ef': -0.45, 'mp_id': 'mp-510516', 'source': 'Materials Project'},
}

# Elemental reference energies (PBE, bulk)
# These must match the DFT settings used in the dataset
ELEMENT_REFS_PBE = {
    # eV/atom, approximate PBE values for standard phases
    'Fe': -8.31,  # bcc Fe
    'S':  -4.13,  # orthorhombic S8
    'Ni': -5.57,  # fcc Ni
    'H':  -3.39,  # 0.5 * E(H2)
    'C':  -9.22,  # diamond
    'O':  -4.95,  # 0.5 * E(O2)
}

# Magnetic moments (mu_B per metal atom)
MAGNETIC_REF = {
    'pyrite':      {'Fe': 0.0, 'type': 'low-spin, diamagnetic', 'source': 'textbook'},
    'mackinawite': {'Fe': 0.0, 'type': 'antiferromagnetic', 'source': 'Vaughan 2006'},
    'pentlandite': {'Fe': 1.7, 'Ni': 0.3, 'type': 'Pauli paramagnetic', 'source': 'Vaughan 2006'},
    'greigite':    {'Fe_tet': 3.5, 'Fe_oct': -3.1, 'type': 'ferrimagnetic', 'source': 'Chang 2008'},
}

# Characteristic phonon frequencies (cm-1) from Raman/IR
PHONON_REF = {
    'pyrite': [(344, 'Eg'), (379, 'Ag'), (430, 'Tg')],
    'mackinawite': [(208, 'A1g'), (282, 'Eg')],
    'pentlandite': [(186, 'T2g'), (283, 'Eg'), (323, 'A1g')],
}


# ===========================================================================
#  Utility: group configs by mineral and type
# ===========================================================================

def parse_config_label(label: str) -> Tuple[str, str, str]:
    """Parse config_type label into (mineral, category, detail).

    Examples:
        'mack_bulk_eq' -> ('mack', 'bulk', 'eq')
        'pyrite_bulk_rattle_0.05_03' -> ('pyrite', 'rattle', '0.05_03')
        'greigite_001_slab' -> ('greigite', 'slab', '001')
        'mack_001_CO2_top_Fe_vertical' -> ('mack', 'co2_ads', '001_top_Fe_vertical')
    """
    parts = label.split('_')
    mineral = parts[0]

    if 'bulk_eq' in label:
        return mineral, 'bulk_eq', ''
    elif 'strain' in label:
        return mineral, 'strain', '_'.join(parts[2:])
    elif 'shear' in label:
        return mineral, 'shear', '_'.join(parts[2:])
    elif 'rattle' in label:
        return mineral, 'rattle', '_'.join(parts[2:])
    elif 'slab' in label and 'rattle' not in label and 'H_' not in label and 'CO2' not in label and 'HCOO' not in label:
        return mineral, 'slab', '_'.join(parts[1:])
    elif 'H_' in label:
        return mineral, 'h_ads', '_'.join(parts[1:])
    elif 'CO2' in label:
        return mineral, 'co2_ads', '_'.join(parts[1:])
    elif 'HCOO' in label:
        return mineral, 'formate_ads', '_'.join(parts[1:])
    elif 'H2O' in label:
        return mineral, 'h2o_ads', '_'.join(parts[1:])
    elif 'Svac' in label:
        return mineral, 's_vacancy', '_'.join(parts[1:])
    elif 'conv' in label:
        return mineral, 'supercell', '_'.join(parts[1:])
    else:
        return mineral, 'other', '_'.join(parts[1:])


def group_configs(atoms_list: List[Atoms]) -> Dict[str, Dict[str, List]]:
    """Group configurations by mineral and category."""
    groups = defaultdict(lambda: defaultdict(list))
    for atoms in atoms_list:
        label = atoms.info.get('config_type', '?')
        mineral, category, detail = parse_config_label(label)
        groups[mineral][category].append((atoms, label, detail))
    return groups


# ===========================================================================
#  3A: Elastic constants from strain/shear data
# ===========================================================================

def extract_elastic_constants(groups: Dict, mineral: str) -> Dict:
    """Extract elastic constants from strain and shear configurations.

    Uses finite-difference approach:
        C_ij = (1/V0) * d²E/dε_i dε_j

    For cubic/tetragonal systems, the independent elastic constants are:
        Volumetric strain -> B = (C11 + 2*C12) / 3
        Shear strain xy   -> C44
    """
    result = {'mineral': mineral, 'status': 'ok'}

    # Get bulk equilibrium
    bulk_eq_list = groups[mineral].get('bulk_eq', [])
    if not bulk_eq_list:
        return {'mineral': mineral, 'status': 'no_bulk_eq'}

    atoms_eq, _, _ = bulk_eq_list[0]
    E0 = atoms_eq.info.get('energy', 0)
    V0 = atoms_eq.get_volume()
    N = len(atoms_eq)
    cell0 = atoms_eq.cell.array.copy()

    # Strain data: E(V) from volumetric strains
    strain_list = groups[mineral].get('strain', [])
    if len(strain_list) < 4:
        result['status'] = f'insufficient_strain_data ({len(strain_list)} configs)'
        return result

    # Parse strain percentages from labels and collect (strain, E)
    strain_E = [(0.0, E0)]  # Include equilibrium
    for atoms, label, detail in strain_list:
        try:
            # detail is like '+3pct' or '-2pct'
            pct_str = detail.replace('pct', '').strip()
            strain_pct = float(pct_str)
            strain = strain_pct / 100.0
            E = atoms.info.get('energy', 0)
            strain_E.append((strain, E))
        except (ValueError, IndexError):
            continue

    strain_E.sort(key=lambda x: x[0])
    strains = np.array([s for s, e in strain_E])
    energies = np.array([e for s, e in strain_E])

    # Fit parabola: E(ε) = E0 + (B*V0/2) * ε² (to leading order for volumetric)
    # Actually for volumetric: E(V) with V = V0*(1+ε)³ ≈ V0*(1+3ε)
    try:
        coeffs = np.polyfit(strains, energies, 2)
        a2 = coeffs[0]  # d²E/dε² / 2
        # For isotropic volumetric strain ε_v: E = E0 + (9*B*V0/2)*ε_v²
        # Our strain is linear (a→a*(1+ε)), so V = V0*(1+ε)³ and ε_v = 3ε + 3ε² + ε³
        # To leading order: d²E/d(ε_linear)² = 9*B*V0
        B_calc = 2 * a2 / (9 * V0) * 160.2177  # eV/A³ → GPa

        result['B_GPa'] = B_calc
        result['V0_A3'] = V0
        result['E0_eV'] = E0
        result['n_strain_points'] = len(strain_E)
    except Exception as e:
        result['status'] = f'polyfit_error: {e}'
        return result

    # Shear data: C44 from off-diagonal strains
    shear_list = groups[mineral].get('shear', [])
    if shear_list:
        shear_E = []
        for atoms, label, detail in shear_list:
            E = atoms.info.get('energy', 0)
            # Shear strain magnitude: embedded in the cell off-diagonal
            cell = atoms.cell.array
            # Estimate shear magnitude from off-diagonal elements
            shear_mag = 0.0
            for i in range(3):
                for j in range(3):
                    if i != j:
                        shear_mag = max(shear_mag, abs(cell[i, j] / cell[i, i]) if cell[i, i] != 0 else 0)
            shear_E.append((shear_mag, E))

        if len(shear_E) >= 2:
            shear_E.sort(key=lambda x: x[0])
            gammas = np.array([s for s, e in shear_E])
            Es = np.array([e for s, e in shear_E])

            # E(γ) = E0 + (C44 * V0 / 2) * γ²
            try:
                # Include equilibrium point
                gammas_full = np.concatenate([[0.0], gammas])
                Es_full = np.concatenate([[E0], Es])
                coeffs_s = np.polyfit(gammas_full, Es_full, 2)
                C44_calc = 2 * coeffs_s[0] / V0 * 160.2177  # GPa
                result['C44_GPa'] = C44_calc
                result['n_shear_points'] = len(shear_E)
            except Exception:
                pass

    # Compare with reference
    if mineral in ELASTIC_REF:
        ref = ELASTIC_REF[mineral]
        result['B_ref_GPa'] = ref.get('B')
        result['ref_source'] = ref['source']
        if ref.get('B'):
            result['B_error_pct'] = abs(B_calc - ref['B']) / ref['B'] * 100

    return result


# ===========================================================================
#  3B: Birch-Murnaghan EOS
# ===========================================================================

def birch_murnaghan_energy(V, E0, V0, B0, Bp):
    """Third-order Birch-Murnaghan equation of state."""
    eta = (V0 / V) ** (2.0 / 3.0)
    E = E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1) ** 3 * Bp + (eta - 1) ** 2 * (6 - 4 * eta)
    )
    return E


def fit_birch_murnaghan(groups: Dict, mineral: str) -> Dict:
    """Fit Birch-Murnaghan EOS to strain data."""
    if not HAS_SCIPY:
        return {'mineral': mineral, 'status': 'scipy_not_installed'}

    result = {'mineral': mineral, 'status': 'ok'}

    # Collect V, E from equilibrium + strain configs
    bulk_eq_list = groups[mineral].get('bulk_eq', [])
    if not bulk_eq_list:
        return {'mineral': mineral, 'status': 'no_bulk_eq'}

    atoms_eq, _, _ = bulk_eq_list[0]
    V0_init = atoms_eq.get_volume()
    E0_init = atoms_eq.info.get('energy', 0)
    N = len(atoms_eq)

    data = [(V0_init, E0_init)]
    strain_list = groups[mineral].get('strain', [])
    for atoms, label, detail in strain_list:
        if all(atoms.pbc):
            data.append((atoms.get_volume(), atoms.info.get('energy', 0)))

    if len(data) < 5:
        return {'mineral': mineral, 'status': f'too_few_points ({len(data)})'}

    volumes = np.array([d[0] for d in data])
    energies = np.array([d[1] for d in data])

    # Sort by volume
    idx = np.argsort(volumes)
    volumes = volumes[idx]
    energies = energies[idx]

    try:
        # Initial guesses
        B0_guess = 100 / 160.2177  # 100 GPa in eV/A³
        popt, pcov = curve_fit(
            birch_murnaghan_energy, volumes, energies,
            p0=[E0_init, V0_init, B0_guess, 4.0],
            maxfev=10000,
        )
        E0_fit, V0_fit, B0_fit, Bp_fit = popt
        perr = np.sqrt(np.diag(pcov))

        B0_GPa = B0_fit * 160.2177
        B0_err_GPa = perr[2] * 160.2177

        # Residual
        E_pred = birch_murnaghan_energy(volumes, *popt)
        rmse = np.sqrt(np.mean((energies - E_pred) ** 2))
        rmse_per_atom = rmse / N * 1000  # meV/atom

        result.update({
            'E0_eV': E0_fit,
            'V0_A3': V0_fit,
            'V0_per_atom_A3': V0_fit / N,
            'B0_GPa': B0_GPa,
            'B0_err_GPa': B0_err_GPa,
            'Bp': Bp_fit,
            'RMSE_meV_per_atom': rmse_per_atom,
            'n_points': len(data),
        })

        if mineral in ELASTIC_REF and ELASTIC_REF[mineral].get('B'):
            B_ref = ELASTIC_REF[mineral]['B']
            result['B_ref_GPa'] = B_ref
            result['B_error_pct'] = abs(B0_GPa - B_ref) / B_ref * 100
            result['ref_source'] = ELASTIC_REF[mineral]['source']

    except Exception as e:
        result['status'] = f'fit_failed: {e}'

    return result


# ===========================================================================
#  3C: Magnetic moments
# ===========================================================================

def analyze_magnetic_moments(groups: Dict, mineral: str) -> Dict:
    """Analyze magnetic moments from DFT data.

    Note: GPAW extended XYZ may not store per-atom magnetic moments.
    We estimate from total magnetic moment if available, or from
    initial_magnetic_moments in the structure.
    """
    result = {'mineral': mineral, 'status': 'ok'}

    # Check equilibrium config
    bulk_eq_list = groups[mineral].get('bulk_eq', [])
    if not bulk_eq_list:
        return {'mineral': mineral, 'status': 'no_bulk_eq'}

    atoms, _, _ = bulk_eq_list[0]
    N = len(atoms)
    syms = atoms.get_chemical_symbols()

    # Count metals
    n_fe = syms.count('Fe')
    n_ni = syms.count('Ni')
    n_metal = n_fe + n_ni

    result['n_atoms'] = N
    result['n_Fe'] = n_fe
    result['n_Ni'] = n_ni

    # Check if magnetic moments are stored
    if 'magmoms' in atoms.arrays:
        magmoms = atoms.arrays['magmoms']
        result['total_moment'] = np.sum(magmoms)
        result['abs_total_moment'] = np.sum(np.abs(magmoms))

        # Per-element moments
        fe_mask = np.array(syms) == 'Fe'
        ni_mask = np.array(syms) == 'Ni'

        if n_fe > 0:
            fe_moms = magmoms[fe_mask]
            result['mean_Fe_moment'] = float(np.mean(np.abs(fe_moms)))
            result['Fe_moments_list'] = fe_moms.tolist()
        if n_ni > 0:
            ni_moms = magmoms[ni_mask]
            result['mean_Ni_moment'] = float(np.mean(np.abs(ni_moms)))
    elif 'magmom' in atoms.info:
        result['total_moment'] = atoms.info['magmom']
        if n_metal > 0:
            result['moment_per_metal'] = atoms.info['magmom'] / n_metal
    else:
        result['status'] = 'no_magnetic_data'
        result['note'] = 'GPAW extxyz may not store magmoms. Re-run with magmom output or use GPAW .gpw files.'

    # Compare with reference
    if mineral in MAGNETIC_REF:
        ref = MAGNETIC_REF[mineral]
        result['ref_type'] = ref['type']
        result['ref_source'] = ref['source']
        if 'Fe' in ref:
            result['ref_Fe_moment'] = ref['Fe']

    return result


# ===========================================================================
#  3D: Formation energies
# ===========================================================================

def compute_formation_energy(groups: Dict, mineral: str) -> Dict:
    """Compute formation energy relative to elemental references.

    E_f = E(compound) / N - sum_i (x_i * E_ref_i)

    Note: elemental references should ideally come from the SAME DFT settings.
    We use approximate PBE values; for publication, compute E_ref explicitly.
    """
    result = {'mineral': mineral, 'status': 'ok'}

    bulk_eq_list = groups[mineral].get('bulk_eq', [])
    if not bulk_eq_list:
        return {'mineral': mineral, 'status': 'no_bulk_eq'}

    atoms, _, _ = bulk_eq_list[0]
    E_total = atoms.info.get('energy', 0)
    N = len(atoms)
    syms = atoms.get_chemical_symbols()

    # Count composition
    composition = defaultdict(int)
    for s in syms:
        composition[s] += 1

    # Compute formation energy
    E_ref_sum = 0
    for element, count in composition.items():
        if element in ELEMENT_REFS_PBE:
            E_ref_sum += count * ELEMENT_REFS_PBE[element]
        else:
            result['status'] = f'missing_ref_for_{element}'
            return result

    E_f_total = E_total - E_ref_sum  # eV
    E_f_per_atom = E_f_total / N  # eV/atom

    result.update({
        'E_total_eV': E_total,
        'E_per_atom_eV': E_total / N,
        'E_f_total_eV': E_f_total,
        'E_f_per_atom_eV': E_f_per_atom,
        'composition': dict(composition),
        'note': 'E_ref values are approximate. For publication, compute explicitly.',
    })

    if mineral in FORMATION_ENERGY_REF:
        ref = FORMATION_ENERGY_REF[mineral]
        result['E_f_ref_eV'] = ref['Ef']
        result['E_f_error_eV'] = abs(E_f_per_atom - ref['Ef'])
        result['ref_source'] = ref['source']
        result['ref_mp_id'] = ref['mp_id']

    return result


# ===========================================================================
#  3E: Force constant / phonon estimate (from rattle data)
# ===========================================================================

def estimate_force_constants(groups: Dict, mineral: str) -> Dict:
    """Rough phonon frequency estimate from rattled configs.

    For small displacements Δx, F ≈ -k*Δx, so k ≈ |F|/|Δx|.
    Characteristic frequency: ω = sqrt(k/m), ν = ω/(2π).

    This is a crude Einstein model estimate, not proper phonons.
    For proper phonons, use phonopy.
    """
    result = {'mineral': mineral, 'status': 'ok'}

    bulk_eq_list = groups[mineral].get('bulk_eq', [])
    rattle_list = groups[mineral].get('rattle', [])

    if not bulk_eq_list or not rattle_list:
        return {'mineral': mineral, 'status': 'no_rattle_data'}

    atoms_eq, _, _ = bulk_eq_list[0]
    pos_eq = atoms_eq.positions.copy()

    # Collect force constants from small rattles (σ ≤ 0.05)
    k_values = []  # force constants in eV/A²

    for atoms, label, detail in rattle_list:
        if '0.03' in detail or '0.05' in detail:  # Small displacements only
            forces = atoms.arrays.get('forces', None)
            if forces is None:
                continue

            # Displacement from equilibrium
            if len(atoms) != len(atoms_eq):
                continue
            disp = atoms.positions - pos_eq
            disp_norms = np.linalg.norm(disp, axis=1)
            force_norms = np.linalg.norm(forces, axis=1)

            # Only use atoms with significant displacement
            mask = disp_norms > 0.01  # A
            if mask.sum() == 0:
                continue

            k_atom = force_norms[mask] / disp_norms[mask]  # eV/A²
            k_values.extend(k_atom)

    if not k_values:
        return {'mineral': mineral, 'status': 'no_small_rattle_data'}

    k_mean = np.mean(k_values)  # eV/A²
    k_std = np.std(k_values)

    # Convert to frequency
    # ω = sqrt(k/m), where k in eV/A² and m in amu
    # 1 eV/A² = 1.602e-19 J / (1e-10 m)² = 1.602e+1 N/m = 16.02 N/m
    # ω = sqrt(16.02 * k / (m_amu * 1.66054e-27)) rad/s
    # ν = ω / (2π) Hz
    # ν_cm = ν / (c * 100) cm⁻¹  where c = 2.998e8 m/s

    syms = atoms_eq.get_chemical_symbols()
    masses = atoms_eq.get_masses()

    # Average over all atom types
    freq_estimates = {}
    for element in set(syms):
        mask_el = np.array(syms) == element
        m_amu = masses[mask_el][0]
        m_kg = m_amu * 1.66054e-27
        k_SI = k_mean * 16.02  # N/m

        omega = np.sqrt(k_SI / m_kg)  # rad/s
        freq_hz = omega / (2 * np.pi)
        freq_cm = freq_hz / (2.998e10)  # cm⁻¹

        freq_estimates[element] = {
            'freq_cm-1': round(freq_cm, 1),
            'mass_amu': round(m_amu, 2),
        }

    result.update({
        'k_mean_eV_A2': round(k_mean, 3),
        'k_std_eV_A2': round(k_std, 3),
        'n_samples': len(k_values),
        'freq_estimates': freq_estimates,
        'note': 'Einstein model estimate. For proper phonons, use phonopy.',
    })

    # Compare with reference
    if mineral in PHONON_REF:
        result['ref_phonons_cm-1'] = [(f, m) for f, m in PHONON_REF[mineral]]

    return result


# ===========================================================================
#  3F: Adsorption energy summary
# ===========================================================================

def summarize_adsorption(groups: Dict, mineral: str) -> Dict:
    """Summarize adsorption energies for all adsorbates."""
    result = {'mineral': mineral, 'adsorbates': {}}

    # Get clean slab energy
    slab_configs = groups[mineral].get('slab', [])
    if not slab_configs:
        return result

    # Use first clean slab as reference
    slab_atoms, slab_label, _ = slab_configs[0]
    E_slab = slab_atoms.info.get('energy', 0)
    result['clean_slab_label'] = slab_label
    result['clean_slab_E'] = E_slab

    # Gas-phase reference energies (approximate PBE, single molecule in box)
    gas_refs = {
        'H': -3.39,       # 0.5 * E(H2)
        'CO2': -22.96,    # PBE, approximate
        'HCOO': -19.55,   # PBE, approximate (formate anion)
        'H2O': -14.22,    # PBE, approximate
        'CO': -14.79,     # PBE, approximate
    }

    # Process each adsorbate type
    for ads_type in ['h_ads', 'co2_ads', 'formate_ads', 'h2o_ads']:
        ads_list = groups[mineral].get(ads_type, [])
        if not ads_list:
            continue

        # Determine gas-phase molecule
        if ads_type == 'h_ads':
            E_gas = gas_refs.get('H', 0)
            adsorbate_name = 'H'
        elif ads_type == 'co2_ads':
            E_gas = gas_refs.get('CO2', 0)
            adsorbate_name = 'CO2'
        elif ads_type == 'formate_ads':
            E_gas = gas_refs.get('HCOO', 0)
            adsorbate_name = 'HCOO-'
        elif ads_type == 'h2o_ads':
            E_gas = gas_refs.get('H2O', 0)
            adsorbate_name = 'H2O'
        else:
            continue

        ads_results = []
        for atoms, label, detail in ads_list:
            E_ads_sys = atoms.info.get('energy', 0)
            E_ads = E_ads_sys - E_slab - E_gas
            ads_results.append({
                'label': label,
                'E_ads_eV': round(E_ads, 4),
                'n_atoms': len(atoms),
            })

        result['adsorbates'][adsorbate_name] = {
            'n_configs': len(ads_results),
            'E_gas_ref_eV': E_gas,
            'configs': ads_results,
            'note': 'E_gas values are approximate. Compute explicitly for publication.',
        }

    return result


# ===========================================================================
#  Main runner
# ===========================================================================

def run_all_properties(atoms_list: List[Atoms], mineral_filter: str = None) -> Dict:
    """Run all Level 3 physical property analyses."""
    groups = group_configs(atoms_list)

    minerals = sorted(groups.keys())
    if mineral_filter:
        minerals = [m for m in minerals if mineral_filter in m]

    all_results = {}

    for mineral in minerals:
        print(f"\n{'='*60}")
        print(f"Mineral: {mineral}")
        print(f"{'='*60}")

        categories = groups[mineral]
        print(f"  Config categories: {', '.join(sorted(categories.keys()))}")
        for cat, cfgs in sorted(categories.items()):
            print(f"    {cat}: {len(cfgs)} configs")

        mineral_results = {}

        # 3A: Elastic constants
        print(f"\n--- 3A: Elastic constants ---")
        ec = extract_elastic_constants(groups, mineral)
        mineral_results['elastic'] = ec
        if ec['status'] == 'ok':
            print(f"  B0 = {ec.get('B_GPa', '?'):.1f} GPa")
            if 'C44_GPa' in ec:
                print(f"  C44 = {ec['C44_GPa']:.1f} GPa")
            if 'B_ref_GPa' in ec:
                print(f"  B0_ref = {ec['B_ref_GPa']} GPa ({ec.get('ref_source', '')})")
                print(f"  Error = {ec.get('B_error_pct', '?'):.0f}%")
        else:
            print(f"  Status: {ec['status']}")

        # 3B: Birch-Murnaghan EOS
        print(f"\n--- 3B: Birch-Murnaghan EOS ---")
        bm = fit_birch_murnaghan(groups, mineral)
        mineral_results['birch_murnaghan'] = bm
        if bm['status'] == 'ok':
            print(f"  V0 = {bm['V0_A3']:.2f} A^3 ({bm['V0_per_atom_A3']:.2f} A^3/atom)")
            print(f"  B0 = {bm['B0_GPa']:.1f} +/- {bm['B0_err_GPa']:.1f} GPa")
            print(f"  B' = {bm['Bp']:.2f}")
            print(f"  RMSE = {bm['RMSE_meV_per_atom']:.2f} meV/atom")
            if 'B_ref_GPa' in bm:
                print(f"  B0_ref = {bm['B_ref_GPa']} GPa, error = {bm['B_error_pct']:.0f}%")
        else:
            print(f"  Status: {bm['status']}")

        # 3C: Magnetic moments
        print(f"\n--- 3C: Magnetic moments ---")
        mag = analyze_magnetic_moments(groups, mineral)
        mineral_results['magnetic'] = mag
        if mag['status'] == 'ok':
            if 'total_moment' in mag:
                print(f"  Total moment: {mag['total_moment']:.2f} mu_B")
            if 'mean_Fe_moment' in mag:
                print(f"  Mean |Fe| moment: {mag['mean_Fe_moment']:.2f} mu_B")
            if 'mean_Ni_moment' in mag:
                print(f"  Mean |Ni| moment: {mag['mean_Ni_moment']:.2f} mu_B")
            if 'ref_type' in mag:
                print(f"  Expected: {mag['ref_type']} ({mag.get('ref_source', '')})")
        else:
            print(f"  Status: {mag['status']}")

        # 3D: Formation energy
        print(f"\n--- 3D: Formation energy ---")
        fe = compute_formation_energy(groups, mineral)
        mineral_results['formation_energy'] = fe
        if fe['status'] == 'ok':
            print(f"  E/atom = {fe['E_per_atom_eV']:.4f} eV")
            print(f"  E_f = {fe['E_f_per_atom_eV']:.3f} eV/atom")
            print(f"  Composition: {fe['composition']}")
            if 'E_f_ref_eV' in fe:
                print(f"  E_f_ref = {fe['E_f_ref_eV']:.3f} eV/atom ({fe['ref_source']}, {fe.get('ref_mp_id', '')})")
                print(f"  Error = {fe['E_f_error_eV']:.3f} eV/atom")
        else:
            print(f"  Status: {fe['status']}")

        # 3E: Force constants / phonon estimate
        print(f"\n--- 3E: Force constant / phonon estimate ---")
        fc = estimate_force_constants(groups, mineral)
        mineral_results['force_constants'] = fc
        if fc['status'] == 'ok':
            print(f"  k_mean = {fc['k_mean_eV_A2']:.3f} eV/A^2 (from {fc['n_samples']} samples)")
            for el, data in fc['freq_estimates'].items():
                print(f"  {el}: ~{data['freq_cm-1']:.0f} cm^-1 (Einstein, m={data['mass_amu']} amu)")
            if 'ref_phonons_cm-1' in fc:
                print(f"  Reference Raman modes: {fc['ref_phonons_cm-1']}")
        else:
            print(f"  Status: {fc['status']}")

        # 3F: Adsorption summary
        print(f"\n--- 3F: Adsorption energies ---")
        ads = summarize_adsorption(groups, mineral)
        mineral_results['adsorption'] = ads
        if ads.get('adsorbates'):
            for adsorbate_name, adata in ads['adsorbates'].items():
                print(f"  {adsorbate_name}: {adata['n_configs']} configs")
                for cfg in adata['configs']:
                    print(f"    {cfg['label']}: E_ads = {cfg['E_ads_eV']:.3f} eV")
        else:
            print(f"  No adsorption configs found")

        all_results[mineral] = mineral_results

    return all_results


def print_summary_table(results: Dict):
    """Print a summary comparison table."""
    print(f"\n{'='*80}")
    print("PHYSICAL PROPERTIES SUMMARY TABLE")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Mineral':<15} {'B0(GPa)':<12} {'B0_ref':<12} {'err%':<8} {'E_f(eV/at)':<12} {'μ(μB)':<10} {'ν_est(cm⁻¹)':<15}")
    print("-" * 84)

    for mineral, data in sorted(results.items()):
        bm = data.get('birch_murnaghan', {})
        fe = data.get('formation_energy', {})
        mag = data.get('magnetic', {})
        fc = data.get('force_constants', {})

        B0 = f"{bm['B0_GPa']:.1f}" if bm.get('status') == 'ok' and 'B0_GPa' in bm else '—'
        B_ref = f"{bm.get('B_ref_GPa', '—')}" if bm.get('B_ref_GPa') else '—'
        B_err = f"{bm.get('B_error_pct', 0):.0f}" if bm.get('B_error_pct') is not None else '—'
        Ef = f"{fe.get('E_f_per_atom_eV', 0):.3f}" if fe.get('status') == 'ok' else '—'
        mu = '—'
        if mag.get('status') == 'ok':
            if 'total_moment' in mag:
                mu = f"{mag['total_moment']:.1f}"
            elif 'moment_per_metal' in mag:
                mu = f"{mag['moment_per_metal']:.1f}"

        nu = '—'
        if fc.get('status') == 'ok' and fc.get('freq_estimates'):
            # Take max frequency estimate
            max_freq = max(d['freq_cm-1'] for d in fc['freq_estimates'].values())
            nu = f"~{max_freq:.0f}"

        print(f"{mineral:<15} {B0:<12} {B_ref:<12} {B_err:<8} {Ef:<12} {mu:<10} {nu:<15}")


def main():
    parser = argparse.ArgumentParser(description="Level 3 physical properties validation")
    parser.add_argument('input', nargs='+', type=str, help='Input extended XYZ file(s)')
    parser.add_argument('--mineral', type=str, default=None, help='Filter by mineral name')
    parser.add_argument('--json', type=str, default=None, help='Save results as JSON')
    args = parser.parse_args()

    # Load all input files
    all_atoms = []
    for inp in args.input:
        inp_path = Path(inp)
        if not inp_path.exists():
            print(f"Warning: {inp_path} not found, skipping")
            continue
        print(f"Loading {inp_path}...")
        atoms = read(inp_path, index=':', format='extxyz')
        print(f"  Loaded {len(atoms)} configurations")
        all_atoms.extend(atoms)

    if not all_atoms:
        print("Error: no configurations loaded")
        sys.exit(1)

    print(f"\nTotal: {len(all_atoms)} configurations")

    # Run analysis
    results = run_all_properties(all_atoms, mineral_filter=args.mineral)

    # Summary table
    print_summary_table(results)

    # Save JSON
    if args.json:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"\nResults saved to {args.json}")


if __name__ == '__main__':
    main()
