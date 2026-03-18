#!/usr/bin/env python3
"""
Generate DFT training data for CO2 and HCOO- adsorption on sulfide surfaces (Tier 1B+1C).

CO2 is THE key adsorbate for origin-of-life research on iron sulfides.
HCOO- (formate) is the primary CO2RR product on mackinawite/greigite.

Config breakdown (~42 configs):
  Tier 1B — CO2 adsorption:
    Mackinawite (001) + CO2 (3 sites × 2 orient):   6
    Greigite (001) + CO2 (3 sites × 2 orient):      6
    Greigite (111) + CO2 (3 sites × 2 orient):      6
    Pyrite (100) + CO2 (3 sites × 2 orient):         6
    Pentlandite (111) + CO2 (3 sites × 2 orient):    6
    SUBTOTAL CO2:                                    30

  Tier 1C — HCOO- adsorption:
    Mackinawite (001) + HCOO- (2 modes × 2 sites):  4
    Greigite (001) + HCOO- (2 modes × 2 sites):     4
    Greigite (111) + HCOO- (2 modes × 2 sites):     4
    SUBTOTAL HCOO-:                                  12

  TOTAL:                                             42

Usage:
    python -u generate_co2_adsorption_configs.py --output /workspace/results/co2_ads_train.xyz
    python -u generate_co2_adsorption_configs.py --output /workspace/results/co2_ads_train.xyz --resume
    python -u generate_co2_adsorption_configs.py --dry-run
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from ase import Atoms
from ase.build import surface, molecule
from ase.io import write, read
from ase.spacegroup import crystal


# ===========================================================================
#  Mineral builders (same as v2 datagen)
# ===========================================================================

def build_mackinawite():
    """Mackinawite FeS (P4/nmm, #129). Layered structure."""
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
        primitive_cell=True,
    )
    return atoms


def build_pyrite():
    """Pyrite FeS2 (Pa-3, #205)."""
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.385, 0.385, 0.385)],
        spacegroup=205,
        cellpar=[5.418, 5.418, 5.418, 90, 90, 90],
        primitive_cell=True,
    )
    return atoms


def build_pentlandite():
    """Pentlandite (Fe,Ni)9S8 (Fm-3m, #225)."""
    atoms = crystal(
        symbols=['Fe', 'Ni', 'S'],
        basis=[
            (0.0, 0.0, 0.0),       # 4a: Fe
            (0.625, 0.625, 0.625),  # 8c: Ni (in our model)
            (0.25, 0.25, 0.25),     # 8c: S
        ],
        spacegroup=225,
        cellpar=[10.07, 10.07, 10.07, 90, 90, 90],
        primitive_cell=False,
    )
    return atoms


def build_greigite_conventional():
    """Greigite Fe3S4 (Fd-3m, #227). Full conventional cell (56 atoms)."""
    atoms = crystal(
        symbols=['Fe', 'Fe', 'S'],
        basis=[
            (0.125, 0.125, 0.125),   # 8a tetrahedral Fe
            (0.5, 0.5, 0.5),         # 16d octahedral Fe
            (0.380, 0.380, 0.380),   # 32e sulfur (setting 1, ASE default)
        ],
        spacegroup=227,
        cellpar=[9.876, 9.876, 9.876, 90, 90, 90],
        primitive_cell=False,
    )
    return atoms


# ===========================================================================
#  Slab builders
# ===========================================================================

def build_slab(bulk, miller, layers=2, vacuum=15.0, repeat=None):
    """Build a surface slab with adequate vacuum for adsorbate calculations."""
    slab = surface(bulk, miller, layers=layers, vacuum=vacuum)
    if repeat is not None:
        slab = slab.repeat(repeat)
    return slab


def get_top_metal_and_sulfur(slab):
    """Find topmost Fe/Ni and S atoms on the slab surface."""
    syms = np.array(slab.get_chemical_symbols())
    pos = slab.positions

    fe_mask = syms == 'Fe'
    ni_mask = syms == 'Ni'
    s_mask = syms == 'S'
    metal_mask = fe_mask | ni_mask

    metal_pos = pos[metal_mask]
    s_pos = pos[s_mask]

    if len(metal_pos) == 0 or len(s_pos) == 0:
        return None, None, None, None

    top_metal = metal_pos[np.argmax(metal_pos[:, 2])]
    top_s = s_pos[np.argmax(s_pos[:, 2])]

    # Find a second metal for hollow site
    metal_dists = np.linalg.norm(metal_pos - top_metal, axis=1)
    metal_dists[np.argmax(metal_pos[:, 2])] = np.inf
    second_metal = metal_pos[np.argmin(metal_dists)] if len(metal_pos) > 1 else None

    # Find top Ni specifically (for pentlandite)
    ni_pos = pos[ni_mask]
    top_ni = ni_pos[np.argmax(ni_pos[:, 2])] if len(ni_pos) > 0 else None

    return top_metal, top_s, second_metal, top_ni


# ===========================================================================
#  CO2 molecule orientations
# ===========================================================================

def make_co2():
    """Build a CO2 molecule (linear O=C=O, bond length 1.16 A)."""
    # ASE molecule database
    co2 = molecule('CO2')
    # Center at origin
    co2.positions -= co2.get_center_of_mass()
    return co2


def check_min_distance(atoms, min_dist=1.2, adsorbate_sizes=None):
    """Check if any two atoms are closer than min_dist (Å).

    adsorbate_sizes: list of ints — sizes of adsorbate molecules appended
                     at the end of atoms. Intra-molecular distances within
                     each adsorbate are excluded from the check.
                     Example: [2] for CO, [3] for CO2, [1, 3] for H + CO2.

    Returns (ok, min_d).
    """
    if len(atoms) < 2:
        return True, float('inf')
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)

    if adsorbate_sizes:
        n = len(atoms)
        idx = n
        for mol_size in reversed(adsorbate_sizes):
            start = idx - mol_size
            dists[start:idx, start:idx] = np.inf
            idx = start

    min_d = np.min(dists)
    return min_d >= min_dist, min_d


def _raise_adsorbate_if_too_close(result, n_ads_atoms, label="adsorbate"):
    """Raise the last n_ads_atoms upward if too close to slab."""
    ads_sizes = [n_ads_atoms]
    for attempt in range(5):
        ok, min_d = check_min_distance(result, adsorbate_sizes=ads_sizes)
        if ok:
            break
        print(f"    WARNING: min dist {min_d:.2f} Å < 1.2 Å, raising {label} by 0.3 Å (attempt {attempt+1})", flush=True)
        result.positions[-n_ads_atoms:, 2] += 0.3
    else:
        ok, min_d = check_min_distance(result, adsorbate_sizes=ads_sizes)
        if not ok:
            print(f"    ERROR: still too close ({min_d:.2f} Å) after 5 attempts!", flush=True)


def place_co2_on_site(slab, site_pos, orientation='vertical', height=2.5):
    """Place CO2 molecule on a surface site.

    Orientations:
        'vertical':  C-down, O=C=O perpendicular to surface (eta-1 C-binding)
        'horizontal': O=C=O parallel to surface (eta-2 bridging)

    Height: distance from site to C atom (Angstrom).
    Raises height if atoms are too close (min < 1.2 Å).
    """
    result = slab.copy()
    co2 = make_co2()

    if orientation == 'vertical':
        # CO2 vertical: C closest to surface, O pointing up
        # CO2 is along x-axis by default, rotate to z
        co2.rotate(90, 'y')
        co2.positions -= co2.get_center_of_mass()
        co2.positions += site_pos + np.array([0, 0, height])
    elif orientation == 'horizontal':
        # CO2 horizontal: O=C=O parallel to surface
        co2.positions -= co2.get_center_of_mass()
        co2.positions += site_pos + np.array([0, 0, height])
    elif orientation == 'tilted':
        # CO2 tilted 45 degrees (common initial geometry for CO2RR)
        co2.rotate(45, 'y')
        co2.positions -= co2.get_center_of_mass()
        co2.positions += site_pos + np.array([0, 0, height])
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    result += co2
    _raise_adsorbate_if_too_close(result, 3, "CO2")
    return result


# ===========================================================================
#  HCOO- (formate) orientations
# ===========================================================================

def make_formate():
    """Build formate HCOO- molecule.

    Structure: H-C(=O)-O^-
    Geometry: C in center, two O at ~125 deg, H opposite.
    Bond lengths: C-O ~ 1.25 A, C-H ~ 1.10 A
    """
    # Formate ion geometry
    d_co = 1.25  # C-O bond length
    d_ch = 1.10  # C-H bond length
    angle_oco = 125.0  # O-C-O angle in degrees

    half_angle = np.radians(angle_oco / 2)

    positions = np.array([
        [0.0, 0.0, 0.0],                          # C (center)
        [-d_co * np.sin(half_angle), 0.0, d_co * np.cos(half_angle)],  # O1
        [d_co * np.sin(half_angle), 0.0, d_co * np.cos(half_angle)],   # O2
        [0.0, 0.0, -d_ch],                        # H
    ])

    formate = Atoms('COOH', positions=positions)
    formate.positions -= formate.get_center_of_mass()
    return formate


def place_formate_on_site(slab, site_pos, mode='bidentate', height=2.2):
    """Place formate on a surface site.

    Modes:
        'bidentate':  Both O atoms facing surface (common on Fe sites)
        'monodentate': One O toward surface, tilted (common on S sites)
    """
    result = slab.copy()
    formate = make_formate()

    if mode == 'bidentate':
        # O-C-O plane perpendicular to surface, both O facing down
        formate.rotate(180, 'x')  # Flip so O's point down
        formate.positions -= formate.get_center_of_mass()
        formate.positions += site_pos + np.array([0, 0, height])
    elif mode == 'monodentate':
        # One O closer to surface, molecule tilted
        formate.rotate(180, 'x')  # O's down
        formate.rotate(30, 'y')   # Tilt
        formate.positions -= formate.get_center_of_mass()
        formate.positions += site_pos + np.array([0, 0, height])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    result += formate
    _raise_adsorbate_if_too_close(result, 4, "HCOO")
    return result


# ===========================================================================
#  Config generators
# ===========================================================================

def generate_co2_on_mackinawite():
    """CO2 adsorption on mackinawite (001) — 6 configs."""
    configs = []
    mack = build_mackinawite()
    slab = build_slab(mack, (0, 0, 1), layers=2, vacuum=15.0, repeat=(2, 2, 1))

    top_metal, top_s, second_metal, _ = get_top_metal_and_sulfur(slab)
    if top_metal is None:
        return configs

    sites = {
        'top_Fe': top_metal,
        'bridge_FeS': (top_metal + top_s) / 2,
        'hollow': (top_metal + top_s + second_metal) / 3 if second_metal is not None else (top_metal + top_s) / 2,
    }

    for site_name, site_pos in sites.items():
        for orient in ['vertical', 'horizontal']:
            s = place_co2_on_site(slab, site_pos, orientation=orient, height=2.5)
            label = f"mack_001_CO2_{site_name}_{orient}"
            configs.append((s, label))

    return configs


def generate_co2_on_greigite_001():
    """CO2 adsorption on greigite (001) — 6 configs."""
    configs = []
    greig = build_greigite_conventional()
    slab = build_slab(greig, (0, 0, 1), layers=2, vacuum=15.0)

    top_metal, top_s, second_metal, _ = get_top_metal_and_sulfur(slab)
    if top_metal is None:
        return configs

    sites = {
        'top_Fe': top_metal,
        'top_S': top_s + np.array([0, 0, 0.5]),  # slightly above S
        'bridge': (top_metal + top_s) / 2,
    }

    for site_name, site_pos in sites.items():
        for orient in ['vertical', 'horizontal']:
            s = place_co2_on_site(slab, site_pos, orientation=orient, height=2.5)
            label = f"greigite_001_CO2_{site_name}_{orient}"
            configs.append((s, label))

    return configs


def generate_co2_on_greigite_111():
    """CO2 adsorption on greigite (111) — 6 configs."""
    configs = []
    greig = build_greigite_conventional()
    slab = build_slab(greig, (1, 1, 1), layers=2, vacuum=15.0)

    top_metal, top_s, second_metal, _ = get_top_metal_and_sulfur(slab)
    if top_metal is None:
        return configs

    sites = {
        'top_Fe': top_metal,
        'bridge': (top_metal + top_s) / 2,
        'hollow': (top_metal + top_s + second_metal) / 3 if second_metal is not None else (top_metal + top_s) / 2,
    }

    for site_name, site_pos in sites.items():
        for orient in ['vertical', 'horizontal']:
            s = place_co2_on_site(slab, site_pos, orientation=orient, height=2.5)
            label = f"greigite_111_CO2_{site_name}_{orient}"
            configs.append((s, label))

    return configs


def generate_co2_on_pyrite():
    """CO2 adsorption on pyrite (100) — 6 configs."""
    configs = []
    pyr = build_pyrite()
    slab = build_slab(pyr, (1, 0, 0), layers=2, vacuum=15.0, repeat=(2, 2, 1))

    top_metal, top_s, second_metal, _ = get_top_metal_and_sulfur(slab)
    if top_metal is None:
        return configs

    sites = {
        'top_Fe': top_metal,
        'top_S': top_s + np.array([0, 0, 0.5]),
        'bridge_FeS': (top_metal + top_s) / 2,
    }

    for site_name, site_pos in sites.items():
        for orient in ['vertical', 'horizontal']:
            s = place_co2_on_site(slab, site_pos, orientation=orient, height=2.5)
            label = f"pyrite_100_CO2_{site_name}_{orient}"
            configs.append((s, label))

    return configs


def generate_co2_on_pentlandite():
    """CO2 adsorption on pentlandite (111) — 6 configs."""
    configs = []
    pent = build_pentlandite()
    slab = build_slab(pent, (1, 1, 1), layers=2, vacuum=15.0)

    top_metal, top_s, second_metal, top_ni = get_top_metal_and_sulfur(slab)
    if top_metal is None:
        return configs

    sites = {
        'top_Fe': top_metal,
        'top_Ni': top_ni if top_ni is not None else top_metal,
        'bridge': (top_metal + (top_ni if top_ni is not None else top_s)) / 2,
    }

    for site_name, site_pos in sites.items():
        for orient in ['vertical', 'horizontal']:
            s = place_co2_on_site(slab, site_pos, orientation=orient, height=2.5)
            label = f"pent_111_CO2_{site_name}_{orient}"
            configs.append((s, label))

    return configs


def generate_formate_on_mackinawite():
    """HCOO- adsorption on mackinawite (001) — 4 configs."""
    configs = []
    mack = build_mackinawite()
    slab = build_slab(mack, (0, 0, 1), layers=2, vacuum=15.0, repeat=(2, 2, 1))

    top_metal, top_s, _, _ = get_top_metal_and_sulfur(slab)
    if top_metal is None:
        return configs

    for site_name, site_pos in [('Fe', top_metal), ('FeS_bridge', (top_metal + top_s) / 2)]:
        for mode in ['bidentate', 'monodentate']:
            s = place_formate_on_site(slab, site_pos, mode=mode, height=2.2)
            label = f"mack_001_HCOO_{site_name}_{mode}"
            configs.append((s, label))

    return configs


def generate_formate_on_greigite_001():
    """HCOO- adsorption on greigite (001) — 4 configs."""
    configs = []
    greig = build_greigite_conventional()
    slab = build_slab(greig, (0, 0, 1), layers=2, vacuum=15.0)

    top_metal, top_s, _, _ = get_top_metal_and_sulfur(slab)
    if top_metal is None:
        return configs

    for site_name, site_pos in [('Fe', top_metal), ('bridge', (top_metal + top_s) / 2)]:
        for mode in ['bidentate', 'monodentate']:
            s = place_formate_on_site(slab, site_pos, mode=mode, height=2.2)
            label = f"greigite_001_HCOO_{site_name}_{mode}"
            configs.append((s, label))

    return configs


def generate_formate_on_greigite_111():
    """HCOO- adsorption on greigite (111) — 4 configs."""
    configs = []
    greig = build_greigite_conventional()
    slab = build_slab(greig, (1, 1, 1), layers=2, vacuum=15.0)

    top_metal, top_s, _, _ = get_top_metal_and_sulfur(slab)
    if top_metal is None:
        return configs

    for site_name, site_pos in [('Fe', top_metal), ('bridge', (top_metal + top_s) / 2)]:
        for mode in ['bidentate', 'monodentate']:
            s = place_formate_on_site(slab, site_pos, mode=mode, height=2.2)
            label = f"greigite_111_HCOO_{site_name}_{mode}"
            configs.append((s, label))

    return configs


# ===========================================================================
#  Main pipeline
# ===========================================================================

def generate_all_configs():
    """Generate all CO2 + HCOO- adsorption configurations."""
    configs = []  # List of (atoms, label)

    print("=" * 60, flush=True)
    print("Generating CO2/HCOO- adsorption configs", flush=True)
    print("=" * 60, flush=True)

    # Tier 1B: CO2 adsorption
    print("\n--- Tier 1B: CO2 adsorption ---", flush=True)

    generators_co2 = [
        ("Mackinawite (001)", generate_co2_on_mackinawite),
        ("Greigite (001)", generate_co2_on_greigite_001),
        ("Greigite (111)", generate_co2_on_greigite_111),
        ("Pyrite (100)", generate_co2_on_pyrite),
        ("Pentlandite (111)", generate_co2_on_pentlandite),
    ]

    for name, gen_func in generators_co2:
        cfgs = gen_func()
        configs.extend(cfgs)
        print(f"  {name}: {len(cfgs)} configs", flush=True)

    n_co2 = len(configs)
    print(f"  CO2 subtotal: {n_co2}", flush=True)

    # Tier 1C: HCOO- adsorption
    print("\n--- Tier 1C: HCOO- adsorption ---", flush=True)

    generators_formate = [
        ("Mackinawite (001)", generate_formate_on_mackinawite),
        ("Greigite (001)", generate_formate_on_greigite_001),
        ("Greigite (111)", generate_formate_on_greigite_111),
    ]

    for name, gen_func in generators_formate:
        cfgs = gen_func()
        configs.extend(cfgs)
        print(f"  {name}: {len(cfgs)} configs", flush=True)

    n_formate = len(configs) - n_co2
    print(f"  HCOO- subtotal: {n_formate}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL CO2/HCOO- CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


def set_magnetic_moments(atoms):
    """Set initial magnetic moments (Vaughan 2006)."""
    magmoms = []
    for sym in atoms.get_chemical_symbols():
        if sym == 'Fe':
            magmoms.append(1.7)
        elif sym == 'Ni':
            magmoms.append(0.3)
        else:
            magmoms.append(0.0)
    atoms.set_initial_magnetic_moments(magmoms)


def run_gpaw_single_point(atoms, config_label):
    """Run GPAW single-point (adsorption configs are always slabs)."""
    from gpaw import GPAW, PW, FermiDirac

    set_magnetic_moments(atoms)

    n_atoms = len(atoms)
    mode = PW(400)

    if n_atoms > 60:
        kpts = (1, 1, 1)
    else:
        kpts = (2, 2, 1)

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
    stress = None  # No stress for slabs with adsorbates

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
    parser = argparse.ArgumentParser(description="Generate CO2/HCOO- adsorption DFT training data (Tier 1B+1C)")
    parser.add_argument('--output', type=str, default='/workspace/results/co2_ads_train.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Just count configs, no DFT')
    args = parser.parse_args()

    configs = generate_all_configs()

    if args.dry_run:
        print("\nDRY RUN — config list:")
        for atoms, label in configs:
            elements = set(atoms.get_chemical_symbols())
            print(f"  {label}: {len(atoms)} atoms, elements={elements}")
        print(f"\nTotal: {len(configs)} configs")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if args.resume:
        existing = load_existing_labels(output_path)
        print(f"Resuming: found {len(existing)} existing configs", flush=True)

    remaining = [(a, l) for a, l in configs if l not in existing]
    print(f"Remaining: {len(remaining)} configs to compute\n", flush=True)

    log_path = output_path.parent / 'co2_ads_log.txt'

    for i, (atoms, label) in enumerate(remaining):
        t0 = time.time()
        try:
            results = run_gpaw_single_point(atoms, label)
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
    print(f"\n{'=' * 60}", flush=True)
    print(f"Done. Output: {output_path}", flush=True)
    if output_path.exists():
        final = read(output_path, index=':', format='extxyz')
        print(f"Total configs in file: {len(final)}", flush=True)


if __name__ == '__main__':
    main()
