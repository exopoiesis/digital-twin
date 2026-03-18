#!/usr/bin/env python3
"""
Fix anomalous H-bridge/hollow adsorption configs with corrected geometry.

Fixes 4 anomalies found in v2 dataset:
  ANO-001: pyrite_100_H_bridge_FeS  — max|F|=210 eV/A (H at 1.2A, too close)
  ANO-002: pyrite_100_H_hollow      — max|F|=68 eV/A  (H at 1.0A, too close)
  ANO-003/004: pentlandite NEB artifacts (just delete, not recompute)
  + predicted risk: pentlandite_001_H_bridge (Fe 4a vs Ni 8c at different z)

Generates 7 corrected configs:
  pyrite_100_H_bridge_FeS_v2       — bridge at 1.8A + safety check
  pyrite_100_H_bridge_tilt_v2      — tilted bridge (70% Fe, 30% S)
  pyrite_100_H_hollow_v2           — hollow at 1.5A + safety check
  mack_001_H_bridge_v2             — preventive recompute
  pent_001_H_bridge_v2             — bridge Fe-Ni at 1.8A + safety check
  pent_001_H_bridge_Ni_tilt_v2     — tilted toward Ni (catalytically relevant)
  pent_001_H_hollow_v2             — hollow Fe-Ni-S at 1.5A + safety check

Usage:
    python -u fix_h_bridge_configs.py --output /workspace/results/h_bridge_fix.xyz
    python -u fix_h_bridge_configs.py --dry-run
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import surface
from ase.io import write, read
from ase.spacegroup import crystal


MIN_H_DISTANCE = 1.0  # Angstrom — minimum allowed H-to-any-atom distance


def build_pyrite():
    """Pyrite FeS2 (Pa-3, #205)."""
    return crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.385, 0.385, 0.385)],
        spacegroup=205,
        cellpar=[5.418, 5.418, 5.418, 90, 90, 90],
        primitive_cell=True,
    )


def build_mackinawite():
    """Mackinawite FeS (P4/nmm, #129)."""
    return crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
        primitive_cell=True,
    )


def build_pentlandite():
    """Pentlandite (Fe,Ni)9S8 (Fm-3m, #225)."""
    return crystal(
        symbols=['Fe', 'Ni', 'S'],
        basis=[
            (0.0, 0.0, 0.0),       # 4a: Fe
            (0.625, 0.625, 0.625),  # 8c: Ni
            (0.25, 0.25, 0.25),     # 8c: S
        ],
        spacegroup=225,
        cellpar=[10.07, 10.07, 10.07, 90, 90, 90],
        primitive_cell=False,
    )


def safe_place_h(slab, h_pos, label, min_dist=MIN_H_DISTANCE):
    """Place H on slab with minimum distance safety check.

    If H is too close to any atom, raise it incrementally until safe.
    Returns (atoms_with_H, final_label, was_adjusted).
    """
    result = slab.copy()
    pos = h_pos.copy()

    adjusted = False
    for attempt in range(20):
        # Check minimum distance to all slab atoms
        dists = np.linalg.norm(slab.positions - pos, axis=1)
        min_d = np.min(dists)

        if min_d >= min_dist:
            break

        # Too close — raise by 0.3 A
        pos[2] += 0.3
        adjusted = True

    result += Atoms('H', positions=[pos])
    final_label = label + ("_raised" if adjusted else "")

    return result, final_label, adjusted


def generate_fixed_configs():
    """Generate corrected H-bridge configs for all minerals."""
    configs = []

    # === Pyrite (100) H bridge — ANO-001 fix ===
    pyr = build_pyrite()
    slab_pyr = surface(pyr, (1, 0, 0), layers=2, vacuum=10.0)
    slab_pyr = slab_pyr.repeat((2, 2, 1))

    syms = np.array(slab_pyr.get_chemical_symbols())
    fe_mask = syms == 'Fe'
    s_mask = syms == 'S'
    fe_pos = slab_pyr.positions[fe_mask]
    s_pos = slab_pyr.positions[s_mask]

    top_fe = fe_pos[np.argmax(fe_pos[:, 2])]
    top_s = s_pos[np.argmax(s_pos[:, 2])]

    # Fixed bridge: 1.8 A above midpoint (was 1.2)
    h_bridge = (top_fe + top_s) / 2 + [0, 0, 1.8]
    s1, l1, adj1 = safe_place_h(slab_pyr, h_bridge, "pyrite_100_H_bridge_FeS_v2")
    configs.append((s1, l1))
    print(f"  {l1}: H at z={h_bridge[2]:.2f}, adjusted={adj1}")

    # Extra: tilted bridge (H shifted toward Fe)
    h_tilt = top_fe * 0.7 + top_s * 0.3 + [0, 0, 1.5]
    s2, l2, adj2 = safe_place_h(slab_pyr, h_tilt, "pyrite_100_H_bridge_tilt_v2")
    configs.append((s2, l2))
    print(f"  {l2}: H at z={h_tilt[2]:.2f}, adjusted={adj2}")

    # Hollow site also with safety check
    fe_dists = np.linalg.norm(fe_pos - top_fe, axis=1)
    fe_dists[np.argmax(fe_pos[:, 2])] = np.inf
    second_fe = fe_pos[np.argmin(fe_dists)]
    h_hollow = (top_fe + top_s + second_fe) / 3 + [0, 0, 1.5]
    s3, l3, adj3 = safe_place_h(slab_pyr, h_hollow, "pyrite_100_H_hollow_v2")
    configs.append((s3, l3))
    print(f"  {l3}: H at z={h_hollow[2]:.2f}, adjusted={adj3}")

    # === Mackinawite (001) H bridge — preventive check ===
    mack = build_mackinawite()
    slab_mack = surface(mack, (0, 0, 1), layers=2, vacuum=10.0)
    slab_mack = slab_mack.repeat((2, 2, 1))

    syms_m = np.array(slab_mack.get_chemical_symbols())
    fe_pos_m = slab_mack.positions[syms_m == 'Fe']
    s_pos_m = slab_mack.positions[syms_m == 'S']
    top_fe_m = fe_pos_m[np.argmax(fe_pos_m[:, 2])]
    top_s_m = s_pos_m[np.argmax(s_pos_m[:, 2])]

    h_bridge_m = (top_fe_m + top_s_m) / 2 + [0, 0, 1.8]
    s4, l4, adj4 = safe_place_h(slab_mack, h_bridge_m, "mack_001_H_bridge_v2")
    configs.append((s4, l4))
    print(f"  {l4}: H at z={h_bridge_m[2]:.2f}, adjusted={adj4}")

    # === Pentlandite (001) H bridge — predicted risk (Fe 4a vs Ni 8c different z) ===
    pent = build_pentlandite()
    slab_pent = surface(pent, (0, 0, 1), layers=2, vacuum=10.0)

    syms_p = np.array(slab_pent.get_chemical_symbols())
    fe_pos_p = slab_pent.positions[syms_p == 'Fe']
    ni_pos_p = slab_pent.positions[syms_p == 'Ni']

    if len(fe_pos_p) > 0 and len(ni_pos_p) > 0:
        top_fe_p = fe_pos_p[np.argmax(fe_pos_p[:, 2])]
        top_ni_p = ni_pos_p[np.argmax(ni_pos_p[:, 2])]

        # Fixed bridge Fe-Ni: 1.8 A (was 1.2)
        h_bridge_p = (top_fe_p + top_ni_p) / 2 + [0, 0, 1.8]
        s5, l5, adj5 = safe_place_h(slab_pent, h_bridge_p, "pent_001_H_bridge_v2")
        configs.append((s5, l5))
        print(f"  {l5}: H at z={h_bridge_p[2]:.2f}, adjusted={adj5}, "
              f"Fe_z={top_fe_p[2]:.2f}, Ni_z={top_ni_p[2]:.2f}, delta_z={abs(top_fe_p[2]-top_ni_p[2]):.2f}")

        # Extra: H between Fe-Ni tilted toward Ni (Ni-rich = better catalyst per Tetzlaff)
        h_tilt_p = top_fe_p * 0.3 + top_ni_p * 0.7 + [0, 0, 1.5]
        s6, l6, adj6 = safe_place_h(slab_pent, h_tilt_p, "pent_001_H_bridge_Ni_tilt_v2")
        configs.append((s6, l6))
        print(f"  {l6}: H at z={h_tilt_p[2]:.2f}, adjusted={adj6}")

        # Hollow site: Fe + Ni + S
        s_pos_p = slab_pent.positions[syms_p == 'S']
        top_s_p = s_pos_p[np.argmax(s_pos_p[:, 2])]
        h_hollow_p = (top_fe_p + top_ni_p + top_s_p) / 3 + [0, 0, 1.5]
        s7, l7, adj7 = safe_place_h(slab_pent, h_hollow_p, "pent_001_H_hollow_v2")
        configs.append((s7, l7))
        print(f"  {l7}: H at z={h_hollow_p[2]:.2f}, adjusted={adj7}")
    else:
        print("  WARNING: Pentlandite slab has no Fe or Ni — check structure")

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
    """Run GPAW single-point (slab settings)."""
    from gpaw import GPAW, PW, FermiDirac

    set_magnetic_moments(atoms)

    calc = GPAW(
        mode=PW(400),
        xc='PBE',
        kpts=(2, 2, 1),
        occupations=FermiDirac(0.1),
        convergence={'energy': 1e-5},
        maxiter=500,
        parallel={'augment_grids': True},
        txt=f'/workspace/results/{config_label}.txt',
    )

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    return {'energy': energy, 'forces': forces, 'stress': None, 'config_type': config_label}


def main():
    parser = argparse.ArgumentParser(description="Fix anomalous H-bridge configs (ANO-001)")
    parser.add_argument('--output', type=str, default='/workspace/results/h_bridge_fix.xyz')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print("Generating fixed H-bridge configs...", flush=True)
    configs = generate_fixed_configs()
    print(f"\nTotal: {len(configs)} configs\n", flush=True)

    if args.dry_run:
        print("DRY RUN — no DFT calculations")
        for atoms, label in configs:
            dists = atoms.get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            print(f"  {label}: {len(atoms)} atoms, min_dist={min_d:.3f} A")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, (atoms, label) in enumerate(configs):
        t0 = time.time()
        try:
            results = run_gpaw_single_point(atoms, label)

            atoms_copy = atoms.copy()
            atoms_copy.info['energy'] = results['energy']
            atoms_copy.info['config_type'] = results['config_type']
            atoms_copy.arrays['forces'] = results['forces']
            write(output_path, atoms_copy, format='extxyz', append=True)

            dt = time.time() - t0
            fmax = np.max(np.linalg.norm(results['forces'], axis=1))
            print(f"[{i+1}/{len(configs)}] {label}: E={results['energy']:.4f} eV, "
                  f"max|F|={fmax:.4f} eV/A ({dt:.1f}s)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(configs)}] {label}: FAILED — {e}", flush=True)
            traceback.print_exc()

    print(f"\nDone. Output: {args.output}", flush=True)


if __name__ == '__main__':
    main()
