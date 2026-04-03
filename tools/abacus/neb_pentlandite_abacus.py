#!/usr/bin/env python3
"""
DFT NEB: H diffusion in pentlandite (Fe,Ni)9S8, primitive cell (16 atoms).
ABACUS LCAO + ASE NEB. Vacancy-mediated mechanism.
Cross-verify of GPAW result: E_a = 1.115 eV.

v3: Fixed SCF convergence (sigma=0.05, scf_nmax=500, mixing_beta=0.2).
    Loads pre-relaxed endpoints from XYZ.
    v1/v2 had smearing_sigma=0.01 + scf_nmax=200 -> SCF non-convergence
    on intermediate images -> 10-30 eV energy jumps -> NEB oscillation.
"""
import sys
sys.path.insert(0, "/opt/abacus-develop-3.9.0.26/interfaces/ASE_interface")

import json
import time
import numpy as np
from pathlib import Path

from ase.io import read
from ase.mep import NEB
from ase.optimize import FIRE
from ase.constraints import FixAtoms

from abacuslite import Abacus, AbacusProfile
print("abacuslite imported OK")

RESULTS = Path("/workspace/results")
RESULTS.mkdir(parents=True, exist_ok=True)

PP_DIR = "/workspace/sg15_pp"
ORB_DIR = "/workspace/sg15_orb"

N_IMAGES = 5
FMAX_NEB = 0.05
MAX_STEPS_NEB = 300

# References
E_A_GPAW = 1.115
E_A_MACE_PRIM = 0.96
E_A_MACE_SUPER = 1.43

profile = AbacusProfile(
    command="mpirun --allow-run-as-root --bind-to none -np 1 "
            "/opt/abacus-develop-3.9.0.26/build/abacus_2p",
    omp_num_threads=16,
    pseudo_dir=PP_DIR,
    orbital_dir=ORB_DIR,
)


def make_calc(label="abacus_neb"):
    return Abacus(
        profile=profile,
        directory=f"/workspace/neb_work_v3/{label}",
        pseudopotentials={
            "Fe": "Fe_ONCV_PBE-1.2.upf",
            "Ni": "Ni_ONCV_PBE-1.2.upf",
            "S": "S_ONCV_PBE-1.2.upf",
            "H": "H_ONCV_PBE-1.2.upf",
        },
        basissets={
            "Fe": "Fe_gga_8au_100Ry_4s2p2d1f.orb",
            "Ni": "Ni_gga_8au_100Ry_4s2p2d1f.orb",
            "S": "S_gga_7au_100Ry_2s2p1d.orb",
            "H": "H_gga_6au_100Ry_2s1p.orb",
        },
        kpts={'nk': [1, 1, 1], 'kshift': [0, 0, 0],
              'gamma-centered': True, 'mode': 'mp-sampling'},
        inp={
            'basis_type': 'lcao',
            'calculation': 'scf',
            'nspin': 1,  # pentlandite = Pauli paramagnetic at 25C
            'ecutwfc': 80,
            'smearing_method': 'gaussian',
            'smearing_sigma': 0.05,    # v1/v2: 0.01 -> SCF oscillation
            'scf_thr': 1e-6,
            'scf_nmax': 500,           # v1/v2: 200 -> hit ceiling
            'mixing_type': 'broyden',
            'mixing_beta': 0.2,        # v1/v2: default 0.8 -> too aggressive
            'mixing_ndim': 12,         # more history for stability
            'cal_force': 1,
            'cal_stress': 0,
            'symmetry': 0,
        },
    )


def main():
    t_total = time.time()
    print("=" * 70)
    print("  DFT (ABACUS LCAO) NEB: pentlandite H diffusion (v3)")
    print("  Fixed SCF: sigma=0.05, scf_nmax=500, mixing_beta=0.2")
    print("  Loading pre-relaxed endpoints from XYZ")
    print("=" * 70)

    # Load pre-relaxed endpoints
    print("\n[1/4] Load endpoints")
    start = read("/workspace/pent_neb_start.xyz")
    end = read("/workspace/pent_neb_end.xyz")
    print(f"  Start: {len(start)} atoms, {start.get_chemical_formula()}")
    print(f"  End:   {len(end)} atoms, {end.get_chemical_formula()}")

    # Single-point energies for endpoints
    print("\n[2/4] Single-point energies for endpoints")
    start.calc = make_calc("endpoint_start")
    heavy = [i for i in range(len(start)) if start[i].symbol != 'H']
    start.set_constraint(FixAtoms(indices=heavy))
    e_start = start.get_potential_energy()
    fmax_start = np.max(np.abs(start.get_forces()))
    print(f"  Start: E = {e_start:.4f} eV, fmax = {fmax_start:.4f}")

    end.calc = make_calc("endpoint_end")
    end.set_constraint(FixAtoms(indices=heavy))
    e_end = end.get_potential_energy()
    fmax_end = np.max(np.abs(end.get_forces()))
    print(f"  End:   E = {e_end:.4f} eV, fmax = {fmax_end:.4f}")
    print(f"  |dE| = {abs(e_start - e_end):.4f} eV")

    # Create NEB images
    print(f"\n[3/4] CI-NEB with {N_IMAGES} images (IDPP)")
    images = [start]
    for i in range(N_IMAGES):
        img = start.copy()
        img.calc = make_calc(f"image_{i}")
        img.set_constraint(FixAtoms(indices=heavy))
        images.append(img)
    images.append(end)

    neb = NEB(images, climb=True, k=0.1, method="improvedtangent")
    neb.interpolate("idpp")

    # Find H atom index (NOT last atom — XYZ reorders by species)
    h_idx = [i for i, s in enumerate(images[0].get_chemical_symbols()) if s == 'H'][0]
    print(f"  H atom index: {h_idx}")
    for k, img in enumerate(images):
        h_pos = img.positions[h_idx]
        print(f"  image {k}: H at ({h_pos[0]:.3f}, {h_pos[1]:.3f}, {h_pos[2]:.3f})")

    print(f"\n  Running FIRE NEB (fmax={FMAX_NEB}, max={MAX_STEPS_NEB})...")
    opt = FIRE(neb, logfile="/workspace/neb_work_v3/neb.log",
               trajectory="/workspace/neb_work_v3/neb.traj")
    t0 = time.time()
    converged = opt.run(fmax=FMAX_NEB, steps=MAX_STEPS_NEB)
    dt_neb = time.time() - t0

    # Extract barrier
    energies = [img.get_potential_energy() for img in images]
    e_ref = energies[0]
    rel_e = [e - e_ref for e in energies]
    barrier = max(rel_e)

    dt_total = time.time() - t_total

    print(f"\n[4/4] Results")
    print(f"  Barrier (ABACUS v3): {barrier:.4f} eV")
    print(f"  Barrier (GPAW):      {E_A_GPAW:.4f} eV")
    print(f"  dE(ABACUS-GPAW):     {barrier - E_A_GPAW:+.4f} eV")
    print(f"  Energies: {[f'{e:.4f}' for e in rel_e]}")
    print(f"  NEB steps: {opt.nsteps}, converged: {converged}")
    print(f"  NEB time: {dt_neb:.0f}s, Total: {dt_total:.0f}s")

    # Save
    result = {
        "system": "pentlandite_vacancy_hop",
        "method": "DFT_ABACUS_PBE_LCAO_v3",
        "code": "ABACUS v3.9.0.26",
        "cross_verify_of": "GPAW PBE PW350",
        "n_atoms": len(start),
        "cell_A": start.cell.lengths().tolist(),
        "n_images": N_IMAGES,
        "fmax_neb": FMAX_NEB,
        "scf_params": {
            "smearing_sigma": 0.05,
            "scf_nmax": 500,
            "mixing_beta": 0.2,
            "mixing_ndim": 12,
        },
        "E_a_eV": float(barrier),
        "E_a_GPAW_eV": E_A_GPAW,
        "dE_ABACUS_GPAW_eV": float(barrier - E_A_GPAW),
        "energies_eV": [float(e) for e in rel_e],
        "E_start_eV": float(e_start),
        "E_end_eV": float(e_end),
        "neb_steps": int(opt.nsteps),
        "converged": bool(converged),
        "time_neb_s": round(dt_neb, 1),
        "time_total_s": round(dt_total, 1),
    }
    with open(RESULTS / "q071_dft_neb_pentlandite_abacus_v3.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {RESULTS / 'q071_dft_neb_pentlandite_abacus_v3.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
