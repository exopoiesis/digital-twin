#!/usr/bin/env python3
"""
Generate DFT training data for chalcopyrite CuFeS2 (Tier 3C).

Chalcopyrite is the most abundant copper ore mineral and has been studied
for CO2RR and HER catalysis. Tetragonal structure (I-42d, #122).

Structure: I-42d tetragonal
  a = 5.289 Å, c = 10.423 Å
  Cu at 4a: (0, 0, 0)
  Fe at 4b: (0, 0, 0.5)
  S  at 8d: (0.2574, 0.25, 0.125)

Primitive cell: 8 atoms (2 Cu + 2 Fe + 4 S)
Conventional cell: 16 atoms (4 Cu + 4 Fe + 8 S)

Config breakdown (~63 configs):
  Bulk eq + rattles (σ=0.03-0.20):   16
  Bulk strains (±1-5%):              10
  Bulk shears:                        5
  (001) slab + 5 rattles:             6
  (112) slab + 5 rattles:             6
  H adsorption (3 sites × 2 surf):    6
  H2O adsorption (2 surf):            4
  CO2 adsorption (2 surf):            4
  S-vacancy bulk:                     3
  S-vacancy surface:                  3
  TOTAL:                             ~63

Usage:
    python -u generate_chalcopyrite_configs.py --output /workspace/results/chalcopyrite_train.xyz
    python -u generate_chalcopyrite_configs.py --output /workspace/results/chalcopyrite_train.xyz --resume
    python -u generate_chalcopyrite_configs.py --dry-run
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import surface, molecule
from ase.io import write, read
from ase.spacegroup import crystal


# ===========================================================================
#  Chalcopyrite builders
# ===========================================================================

def build_chalcopyrite():
    """Build chalcopyrite CuFeS2 primitive cell (I-42d, #122).

    a = 5.289 Å, c = 10.423 Å (experimental)
    Cu at 4a: (0, 0, 0)
    Fe at 4b: (0, 0, 0.5)
    S  at 8d: (0.2574, 0.25, 0.125)
    """
    atoms = crystal(
        symbols=['Cu', 'Fe', 'S'],
        basis=[
            (0.0, 0.0, 0.0),       # 4a Cu
            (0.0, 0.0, 0.5),       # 4b Fe
            (0.2574, 0.25, 0.125), # 8d S
        ],
        spacegroup=122,
        cellpar=[5.289, 5.289, 10.423, 90, 90, 90],
        primitive_cell=True,
    )
    return atoms


def build_chalcopyrite_conventional():
    """Build conventional cell (16 atoms: 4 Cu + 4 Fe + 8 S)."""
    atoms = crystal(
        symbols=['Cu', 'Fe', 'S'],
        basis=[
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5),
            (0.2574, 0.25, 0.125),
        ],
        spacegroup=122,
        cellpar=[5.289, 5.289, 10.423, 90, 90, 90],
        primitive_cell=False,
    )
    return atoms


# ===========================================================================
#  Config generators
# ===========================================================================

def rattle_atoms(atoms, stdev, label_prefix, count=5):
    configs = []
    for i in range(count):
        rattled = atoms.copy()
        rattled.rattle(stdev=stdev, seed=42 + int(stdev * 100) + i)
        configs.append((rattled, f"{label_prefix}_rattle_{stdev:.2f}_{i:02d}"))
    return configs


def strain_atoms(atoms, label_prefix):
    configs = []
    for strain_pct in [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]:
        strained = atoms.copy()
        factor = 1.0 + strain_pct / 100.0
        strained.set_cell(strained.cell * factor, scale_atoms=True)
        configs.append((strained, f"{label_prefix}_strain_{strain_pct:+d}pct"))
    return configs


def shear_atoms(atoms, label_prefix):
    """Generate shear-strained structures."""
    configs = []
    for i, (c1, c2) in enumerate([(0, 1), (0, 2), (1, 2)]):
        sheared = atoms.copy()
        cell = sheared.cell.copy()
        cell[c1, c2] += 0.02 * np.linalg.norm(cell[c1])
        sheared.set_cell(cell, scale_atoms=True)
        configs.append((sheared, f"{label_prefix}_shear_{i:02d}"))
    for i, (c1, c2) in enumerate([(0, 1), (1, 2)]):
        sheared = atoms.copy()
        cell = sheared.cell.copy()
        cell[c1, c2] += 0.04 * np.linalg.norm(cell[c1])
        sheared.set_cell(cell, scale_atoms=True)
        configs.append((sheared, f"{label_prefix}_shear_large_{i:02d}"))
    return configs


def safe_place_h(slab, h_position, label):
    """Place H atom, checking minimum distance > 1.0 Å.

    Raises ValueError if placement is too close to any existing atom.
    """
    dists = np.linalg.norm(slab.positions - h_position, axis=1)
    min_dist = np.min(dists)
    if min_dist < 1.0:
        raise ValueError(f"{label}: H too close to existing atom (min_dist={min_dist:.2f} Å)")
    return slab + Atoms('H', positions=[h_position])


def build_chalcopyrite_surface(miller, label_prefix):
    """Build chalcopyrite slab for given Miller indices + rattled variants."""
    chalc_conv = build_chalcopyrite_conventional()
    slab = surface(chalc_conv, miller, layers=2, vacuum=12.0)

    configs = [(slab.copy(), f"{label_prefix}_slab")]
    for i in range(5):
        rattled = slab.copy()
        rattled.rattle(stdev=0.05, seed=500 + hash(label_prefix) + i)
        configs.append((rattled, f"{label_prefix}_slab_rattle_{i:02d}"))

    return configs


def build_h_adsorption_configs(miller, label_prefix):
    """Build H adsorption configs on given surface (3 sites: Cu-top, Fe-top, bridge)."""
    chalc_conv = build_chalcopyrite_conventional()
    slab = surface(chalc_conv, miller, layers=2, vacuum=12.0)

    configs = []
    syms = np.array(slab.get_chemical_symbols())
    cu_mask = syms == 'Cu'
    fe_mask = syms == 'Fe'
    s_mask = syms == 'S'

    cu_pos = slab.positions[cu_mask]
    fe_pos = slab.positions[fe_mask]
    s_pos = slab.positions[s_mask]

    if len(cu_pos) == 0 or len(fe_pos) == 0 or len(s_pos) == 0:
        return configs

    top_cu = cu_pos[np.argmax(cu_pos[:, 2])]
    top_fe = fe_pos[np.argmax(fe_pos[:, 2])]
    top_s = s_pos[np.argmax(s_pos[:, 2])]

    # Site 1: H on-top Cu (1.6 Å above)
    try:
        s1 = safe_place_h(slab, top_cu + [0, 0, 1.6], f"{label_prefix}_H_ontop_Cu")
        configs.append((s1, f"{label_prefix}_H_ontop_Cu"))
    except ValueError as e:
        print(f"  Warning: {e}", flush=True)

    # Site 2: H on-top Fe (1.6 Å above)
    try:
        s2 = safe_place_h(slab, top_fe + [0, 0, 1.6], f"{label_prefix}_H_ontop_Fe")
        configs.append((s2, f"{label_prefix}_H_ontop_Fe"))
    except ValueError as e:
        print(f"  Warning: {e}", flush=True)

    # Site 3: H bridge metal-S (1.8 Å above midpoint between top metal and top S)
    top_metal = top_cu if top_cu[2] > top_fe[2] else top_fe
    h_bridge = (top_metal + top_s) / 2 + [0, 0, 1.8]
    dists = np.linalg.norm(slab.positions - h_bridge, axis=1)
    if np.min(dists) < 1.0:
        h_bridge[2] += 0.5
    try:
        s3 = safe_place_h(slab, h_bridge, f"{label_prefix}_H_bridge")
        configs.append((s3, f"{label_prefix}_H_bridge"))
    except ValueError as e:
        print(f"  Warning: {e}", flush=True)

    return configs


def build_h2o_adsorption_configs(miller, label_prefix):
    """Build H2O adsorption configs on given surface."""
    chalc_conv = build_chalcopyrite_conventional()
    slab = surface(chalc_conv, miller, layers=2, vacuum=12.0)

    configs = []
    syms = np.array(slab.get_chemical_symbols())
    cu_mask = syms == 'Cu'
    fe_mask = syms == 'Fe'

    cu_pos = slab.positions[cu_mask]
    fe_pos = slab.positions[fe_mask]

    if len(cu_pos) == 0 or len(fe_pos) == 0:
        return configs

    top_cu = cu_pos[np.argmax(cu_pos[:, 2])]
    top_fe = fe_pos[np.argmax(fe_pos[:, 2])]

    h2o = molecule('H2O')

    # H2O on Cu (O at 2.2 Å above Cu)
    s1 = slab.copy()
    h2o_cu = h2o.copy()
    h2o_cu.translate(top_cu + [0, 0, 2.2] - h2o_cu.positions[0])
    s1 += h2o_cu
    configs.append((s1, f"{label_prefix}_H2O_on_Cu"))

    # H2O on Fe (O at 2.2 Å above Fe)
    s2 = slab.copy()
    h2o_fe = h2o.copy()
    h2o_fe.translate(top_fe + [0, 0, 2.2] - h2o_fe.positions[0])
    s2 += h2o_fe
    configs.append((s2, f"{label_prefix}_H2O_on_Fe"))

    return configs


def build_co2_adsorption_configs(miller, label_prefix):
    """Build CO2 adsorption configs on given surface."""
    chalc_conv = build_chalcopyrite_conventional()
    slab = surface(chalc_conv, miller, layers=2, vacuum=12.0)

    configs = []
    syms = np.array(slab.get_chemical_symbols())
    cu_mask = syms == 'Cu'
    fe_mask = syms == 'Fe'

    cu_pos = slab.positions[cu_mask]
    fe_pos = slab.positions[fe_mask]

    if len(cu_pos) == 0 or len(fe_pos) == 0:
        return configs

    top_cu = cu_pos[np.argmax(cu_pos[:, 2])]
    top_fe = fe_pos[np.argmax(fe_pos[:, 2])]

    co2 = molecule('CO2')

    # CO2 on Cu (C at 2.0 Å above Cu, lying flat)
    s1 = slab.copy()
    co2_cu = co2.copy()
    co2_cu.rotate(90, 'y')  # Make CO2 horizontal
    co2_cu.translate(top_cu + [0, 0, 2.0] - co2_cu.positions[0])
    s1 += co2_cu
    configs.append((s1, f"{label_prefix}_CO2_on_Cu"))

    # CO2 on Fe (C at 2.0 Å above Fe, lying flat)
    s2 = slab.copy()
    co2_fe = co2.copy()
    co2_fe.rotate(90, 'y')
    co2_fe.translate(top_fe + [0, 0, 2.0] - co2_fe.positions[0])
    s2 += co2_fe
    configs.append((s2, f"{label_prefix}_CO2_on_Fe"))

    return configs


def build_s_vacancy_bulk():
    """Build S-vacancy defects in bulk chalcopyrite (3 configs)."""
    configs = []
    chalc = build_chalcopyrite()

    syms = np.array(chalc.get_chemical_symbols())
    s_indices = np.where(syms == 'S')[0]

    # Remove different S atoms (3 variants)
    for i in range(min(3, len(s_indices))):
        defect = chalc.copy()
        del defect[s_indices[i]]
        configs.append((defect, f"chalcopyrite_bulk_S_vacancy_{i:02d}"))

    return configs


def build_s_vacancy_surface():
    """Build S-vacancy defects on (001) surface (3 configs)."""
    configs = []
    chalc_conv = build_chalcopyrite_conventional()
    slab = surface(chalc_conv, (0, 0, 1), layers=2, vacuum=12.0)

    syms = np.array(slab.get_chemical_symbols())
    s_mask = syms == 'S'
    s_pos = slab.positions[s_mask]
    s_indices = np.where(s_mask)[0]

    if len(s_pos) == 0:
        return configs

    # Find top-most S atoms
    s_z = s_pos[:, 2]
    top_s_local = np.argsort(s_z)[-3:]  # Top 3 S atoms

    for i, idx in enumerate(top_s_local):
        defect = slab.copy()
        del defect[s_indices[idx]]
        configs.append((defect, f"chalcopyrite_001_S_vacancy_{i:02d}"))

    return configs


def generate_all_configs():
    """Generate all chalcopyrite configurations."""
    configs = []  # List of (atoms, label, is_slab)

    print("=" * 60, flush=True)
    print("Generating chalcopyrite CuFeS2 configs (Tier 3C)", flush=True)
    print("=" * 60, flush=True)

    chalc = build_chalcopyrite()
    chalc_conv = build_chalcopyrite_conventional()
    print(f"  Primitive cell: {len(chalc)} atoms", flush=True)
    print(f"  Conventional cell: {len(chalc_conv)} atoms", flush=True)

    # Bulk equilibrium
    configs.append((chalc.copy(), "chalcopyrite_bulk_eq", False))

    # Rattles on primitive cell
    for stdev in [0.03, 0.05, 0.08, 0.10, 0.20]:
        n = 3
        configs.extend([(a, l, False) for a, l in rattle_atoms(chalc, stdev, "chalcopyrite_bulk", n)])

    # Strains
    configs.extend([(a, l, False) for a, l in strain_atoms(chalc, "chalcopyrite_bulk")])

    # Shears
    configs.extend([(a, l, False) for a, l in shear_atoms(chalc, "chalcopyrite_bulk")])

    n_bulk = len(configs)
    print(f"  Bulk configs: {n_bulk}", flush=True)

    # (001) surface + rattles
    print("\n  Building (001) surface...", flush=True)
    slab_001_configs = build_chalcopyrite_surface((0, 0, 1), "chalcopyrite_001")
    configs.extend([(a, l, True) for a, l in slab_001_configs])
    print(f"  (001) slab: {len(slab_001_configs[0][0]) if slab_001_configs else '?'} atoms", flush=True)

    # (112) surface + rattles
    print("  Building (112) surface...", flush=True)
    slab_112_configs = build_chalcopyrite_surface((1, 1, 2), "chalcopyrite_112")
    configs.extend([(a, l, True) for a, l in slab_112_configs])
    print(f"  (112) slab: {len(slab_112_configs[0][0]) if slab_112_configs else '?'} atoms", flush=True)

    # H adsorption on both surfaces
    print("\n  Building H adsorption configs...", flush=True)
    for miller, label in [((0, 0, 1), 'chalcopyrite_001'), ((1, 1, 2), 'chalcopyrite_112')]:
        h_configs = build_h_adsorption_configs(miller, label)
        configs.extend([(a, l, True) for a, l in h_configs])
        print(f"  {label} H adsorption: {len(h_configs)} configs", flush=True)

    # H2O adsorption on both surfaces
    print("\n  Building H2O adsorption configs...", flush=True)
    for miller, label in [((0, 0, 1), 'chalcopyrite_001'), ((1, 1, 2), 'chalcopyrite_112')]:
        h2o_configs = build_h2o_adsorption_configs(miller, label)
        configs.extend([(a, l, True) for a, l in h2o_configs])
        print(f"  {label} H2O adsorption: {len(h2o_configs)} configs", flush=True)

    # CO2 adsorption on both surfaces
    print("\n  Building CO2 adsorption configs...", flush=True)
    for miller, label in [((0, 0, 1), 'chalcopyrite_001'), ((1, 1, 2), 'chalcopyrite_112')]:
        co2_configs = build_co2_adsorption_configs(miller, label)
        configs.extend([(a, l, True) for a, l in co2_configs])
        print(f"  {label} CO2 adsorption: {len(co2_configs)} configs", flush=True)

    # S-vacancy bulk
    print("\n  Building S-vacancy bulk configs...", flush=True)
    s_vac_bulk = build_s_vacancy_bulk()
    configs.extend([(a, l, False) for a, l in s_vac_bulk])
    print(f"  S-vacancy bulk: {len(s_vac_bulk)} configs", flush=True)

    # S-vacancy surface
    print("  Building S-vacancy surface configs...", flush=True)
    s_vac_surf = build_s_vacancy_surface()
    configs.extend([(a, l, True) for a, l in s_vac_surf])
    print(f"  S-vacancy surface: {len(s_vac_surf)} configs", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL CHALCOPYRITE CONFIGS: {len(configs)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return configs


# ===========================================================================
#  DFT + I/O (same as other datagen scripts)
# ===========================================================================

def run_gpaw_single_point(atoms, config_label, is_slab=False):
    """Run GPAW single-point calculation."""
    from gpaw import GPAW, PW, FermiDirac

    n_atoms = len(atoms)
    mode = PW(400) if is_slab or n_atoms > 30 else PW(500)

    if is_slab:
        kpts = (1, 1, 1) if n_atoms > 60 else (2, 2, 1)
    elif n_atoms > 30:
        kpts = (2, 2, 2)
    else:
        kpts = (4, 4, 4)

    calc = GPAW(
        mode=mode,
        xc='PBE',
        kpts=kpts,
        occupations=FermiDirac(0.1),
        convergence={'energy': 1e-5},
        parallel={'augment_grids': True},
        txt=None,
    )

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = None
    if all(atoms.pbc):
        stress = atoms.get_stress(voigt=True)

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
    parser = argparse.ArgumentParser(description="Generate chalcopyrite CuFeS2 DFT training data (Tier 3C)")
    parser.add_argument('--output', type=str, default='/workspace/results/chalcopyrite_train.xyz')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Just count configs, no DFT')
    args = parser.parse_args()

    configs = generate_all_configs()

    if args.dry_run:
        print("\nDRY RUN — config list:")
        for atoms, label, is_slab in configs:
            slab_tag = " [SLAB]" if is_slab else ""
            print(f"  {label}: {len(atoms)} atoms{slab_tag}")
        print(f"\nTotal: {len(configs)} configs")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if args.resume:
        existing = load_existing_labels(output_path)
        print(f"Resuming: found {len(existing)} existing configs", flush=True)

    remaining = [(a, l, s) for a, l, s in configs if l not in existing]
    print(f"Remaining: {len(remaining)} configs to compute\n", flush=True)

    log_path = output_path.parent / 'chalcopyrite_log.txt'

    for i, (atoms, label, is_slab) in enumerate(remaining):
        t0 = time.time()
        try:
            results = run_gpaw_single_point(atoms, label, is_slab)
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

    print(f"\n{'=' * 60}", flush=True)
    print(f"Done. Output: {output_path}", flush=True)
    if output_path.exists():
        final = read(output_path, index=':', format='extxyz')
        print(f"Total configs in file: {len(final)}", flush=True)


if __name__ == '__main__':
    main()
