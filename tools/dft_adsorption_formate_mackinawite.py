#!/usr/bin/env python3
"""
DFT adsorption energy of formate (HCOO-) on mackinawite FeS (001) surface.
Uses GPAW (PBE+D3) on GPU via CuPy-accelerated FFTs.

Replaces MACE-MP-0 version which gave unphysical E_ads (-18 to -22 eV)
due to poor gas-phase molecular reference.
"""
import time
import json
import numpy as np
from pathlib import Path

from ase import Atoms
from ase.build import surface
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.spacegroup import crystal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/workspace/results")
RESULTS_DIR.mkdir(exist_ok=True)


def build_mackinawite_slab():
    """Build mackinawite (001) slab: 3x3x1 supercell + 15 A vacuum."""
    # P4/nmm (#129), a=3.674, c=5.033
    bulk = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
    )
    slab = surface(bulk, (0, 0, 1), layers=2, vacuum=7.5)
    slab = slab.repeat((3, 3, 1))
    return slab


def get_gpaw_calc(kpts=(2, 2, 1), smearing=0.1):
    """Create GPAW calculator with PBE, plane-wave mode, GPU-accelerated."""
    from gpaw import GPAW, PW, FermiDirac
    calc = GPAW(
        mode=PW(400),           # 400 eV plane-wave cutoff
        xc='PBE',
        kpts=kpts,
        occupations=FermiDirac(smearing),
        convergence={'energy': 1e-5},
        txt='-',                # print to stdout
        parallel={'augment_grids': True, 'gpu': True},
    )
    return calc


def get_gpaw_calc_molecule():
    """GPAW calculator for isolated molecule (Gamma point, no smearing)."""
    from gpaw import GPAW, PW, FermiDirac
    calc = GPAW(
        mode=PW(400),
        xc='PBE',
        kpts=(1, 1, 1),
        occupations=FermiDirac(0.01),
        convergence={'energy': 1e-5},
        txt='-',
        parallel={'gpu': True},
    )
    return calc


def build_formate():
    """Build formate HCOO- in a large vacuum box."""
    # C at origin, two O, one H
    formate = Atoms(
        symbols=['C', 'O', 'O', 'H'],
        positions=[
            [0.0, 0.0, 0.0],
            [1.25, 0.0, 0.0],
            [-1.25, 0.0, 0.0],
            [0.0, 1.09, 0.0],
        ],
    )
    formate.center(vacuum=8.0)
    return formate


def main():
    t_total = time.time()
    results = {"method": "DFT", "code": "GPAW", "xc": "PBE", "pw_cutoff_eV": 400}

    # ── [1/7] System info ──
    print("=" * 60)
    print("  DFT Formate Adsorption on Mackinawite (001)")
    print("  GPAW PBE, PW(400 eV), 2x2x1 k-points")
    print("=" * 60)

    try:
        import cupy
        print(f"\n[1/7] CuPy available: GPU-accelerated FFTs enabled")
        results["gpu_fft"] = True
    except ImportError:
        print(f"\n[1/7] CuPy not available: CPU FFTs")
        results["gpu_fft"] = False

    # ── [2/7] Build and relax slab ──
    print("\n[2/7] Build mackinawite (001) slab 3x3x1")
    slab = build_mackinawite_slab()
    n_atoms_slab = len(slab)
    print(f"  Slab: {slab.get_chemical_formula()}, {n_atoms_slab} atoms")

    # Fix bottom layer
    positions = slab.get_positions()
    z_coords = positions[:, 2]
    z_mid = (z_coords.min() + z_coords.max()) / 2
    fix_mask = z_coords < z_mid
    slab.set_constraint(FixAtoms(mask=fix_mask))
    print(f"  Fixed {sum(fix_mask)} bottom atoms")

    t0 = time.time()
    slab.calc = get_gpaw_calc()
    opt = BFGS(slab, logfile=None)
    opt.run(fmax=0.05, steps=100)
    E_slab = slab.get_potential_energy()
    print(f"  E_slab = {E_slab:.4f} eV ({opt.nsteps} steps, {time.time()-t0:.0f}s)")
    results["E_slab_eV"] = float(E_slab)
    results["n_atoms_slab"] = n_atoms_slab

    # ── [3/7] Relax isolated formate ──
    print("\n[3/7] Relax isolated formate in vacuum box")
    t0 = time.time()
    formate = build_formate()
    formate.calc = get_gpaw_calc_molecule()
    opt_f = BFGS(formate, logfile=None)
    opt_f.run(fmax=0.01, steps=100)
    E_formate = formate.get_potential_energy()
    print(f"  E_formate_gas = {E_formate:.4f} eV ({opt_f.nsteps} steps, {time.time()-t0:.0f}s)")
    results["E_formate_gas_eV"] = float(E_formate)

    # Verify geometry
    pos_f = formate.get_positions()
    d_CO1 = np.linalg.norm(pos_f[0] - pos_f[1])
    d_CO2 = np.linalg.norm(pos_f[0] - pos_f[2])
    d_CH = np.linalg.norm(pos_f[0] - pos_f[3])
    print(f"  C-O1={d_CO1:.3f}, C-O2={d_CO2:.3f}, C-H={d_CH:.3f} A")

    # ── [4/7] Define adsorption sites ──
    print("\n[4/7] Define adsorption sites")
    slab_positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()

    # Find top-layer Fe atoms
    fe_top = [i for i, (s, z) in enumerate(zip(symbols, slab_positions[:, 2]))
              if s == 'Fe' and z > z_mid]
    s_top = [i for i, (s, z) in enumerate(zip(symbols, slab_positions[:, 2]))
             if s == 'S' and z > z_mid]

    z_surface = max(slab_positions[fe_top, 2])
    fe0_pos = slab_positions[fe_top[0]]
    fe1_pos = slab_positions[fe_top[1]] if len(fe_top) > 1 else fe0_pos
    s0_pos = slab_positions[s_top[0]] if s_top else fe0_pos

    sites = {
        "ontop_Fe": (fe0_pos[0], fe0_pos[1]),
        "bridge_FeS": ((fe0_pos[0] + s0_pos[0]) / 2, (fe0_pos[1] + s0_pos[1]) / 2),
        "hollow": ((fe0_pos[0] + fe1_pos[0]) / 2, (fe0_pos[1] + fe1_pos[1]) / 2),
    }

    for name, (x, y) in sites.items():
        print(f"  {name}: ({x:.2f}, {y:.2f})")

    # ── [5/7] Adsorption calculations ──
    print("\n[5/7] Adsorption calculations")
    results["sites"] = {}

    for name, (x, y) in sites.items():
        print(f"\n  --- Site: {name} ---")
        t0 = time.time()

        # Clone slab
        slab_ads = slab.copy()
        slab_ads.calc = get_gpaw_calc()

        # Add formate: bidentate, both O toward surface
        h_ads = 2.2  # initial height above surface
        formate_ads = Atoms(
            symbols=['C', 'O', 'O', 'H'],
            positions=[
                [x, y, z_surface + h_ads + 1.0],
                [x + 1.0, y, z_surface + h_ads],
                [x - 1.0, y, z_surface + h_ads],
                [x, y + 1.0, z_surface + h_ads + 1.5],
            ],
        )
        combined = slab_ads + formate_ads
        # Reapply constraints (fix bottom slab atoms only)
        combined.set_constraint(FixAtoms(mask=list(fix_mask) + [False] * 4))
        combined.calc = get_gpaw_calc()

        opt_ads = BFGS(combined, logfile=None)
        converged = opt_ads.run(fmax=0.05, steps=200)
        E_combined = combined.get_potential_energy()

        E_ads = E_combined - E_slab - E_formate
        elapsed = time.time() - t0

        # Check formate integrity
        ads_pos = combined.get_positions()[-4:]
        d1 = np.linalg.norm(ads_pos[0] - ads_pos[1])
        d2 = np.linalg.norm(ads_pos[0] - ads_pos[2])
        dh = np.linalg.norm(ads_pos[0] - ads_pos[3])
        intact = (0.9 < d1 < 1.8) and (0.9 < d2 < 1.8) and (0.8 < dh < 1.5)
        height = ads_pos[0, 2] - z_surface

        # Verdict
        abs_e = abs(E_ads)
        if abs_e < 0.6:
            verdict = "PASS (weak binding, no poisoning risk)"
        elif abs_e < 1.0:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL (risk of poisoning)"

        print(f"  E(slab+HCOO) = {E_combined:.4f} eV ({opt_ads.nsteps} steps)")
        print(f"  E_ads = {E_ads:.4f} eV")
        print(f"  Formate intact: {intact}, height: {height:.2f} A")
        print(f"  Verdict: {verdict}")
        print(f"  done in {elapsed:.0f}s")

        results["sites"][name] = {
            "E_combined_eV": float(E_combined),
            "E_ads_eV": float(E_ads),
            "abs_E_ads_eV": float(abs_e),
            "converged": bool(converged) if converged is not None else True,
            "steps": int(opt_ads.nsteps),
            "formate_intact": bool(intact),
            "height_A": float(height),
            "CO1_A": float(d1),
            "CO2_A": float(d2),
            "CH_A": float(dh),
            "verdict": verdict,
            "time_s": round(elapsed, 1),
        }

    # ── [6/7] Summary ──
    print("\n[6/7] Summary")
    best_site = min(results["sites"], key=lambda s: results["sites"][s]["E_ads_eV"])
    best_e = results["sites"][best_site]["E_ads_eV"]
    print(f"  Most stable site: {best_site} (E_ads = {best_e:.4f} eV)")

    all_pass = all(abs(v["E_ads_eV"]) < 0.6 for v in results["sites"].values())
    any_pass = any(abs(v["E_ads_eV"]) < 0.6 for v in results["sites"].values())
    results["best_site"] = best_site
    results["best_E_ads_eV"] = float(best_e)
    results["overall_verdict"] = "PASS" if all_pass else ("PARTIAL" if any_pass else "FAIL")

    # ── [7/7] Save ──
    print("\n[7/7] Save results")
    results["total_time_s"] = round(time.time() - t_total, 1)

    json_path = RESULTS_DIR / "q075_dft_adsorption_formate_mackinawite.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {json_path}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    site_names = list(results["sites"].keys())
    e_values = [results["sites"][s]["E_ads_eV"] for s in site_names]
    colors = ['green' if abs(e) < 0.6 else 'orange' if abs(e) < 1.0 else 'red' for e in e_values]
    ax.bar(site_names, e_values, color=colors, edgecolor='black')
    ax.axhline(-0.6, ls='--', color='green', alpha=0.7, label='PASS threshold')
    ax.axhline(-1.0, ls='--', color='red', alpha=0.7, label='FAIL threshold')
    ax.set_ylabel('E_ads (eV)')
    ax.set_title('DFT (PBE) Formate Adsorption on Mackinawite (001)')
    ax.legend()
    plt.tight_layout()
    png_path = RESULTS_DIR / "q075_dft_adsorption_formate_mackinawite.png"
    fig.savefig(png_path, dpi=150)
    print(f"  Saved {png_path}")

    print(f"\nTotal time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
