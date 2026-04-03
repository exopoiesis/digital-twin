#!/usr/bin/env python3
"""
DFT NEB: H diffusion in pyrite FeS2, 1x1x1 cell (11 atoms).
ABACUS LCAO + ASE NEB. Vacancy-mediated mechanism.

Apple-to-apple cross-verify with GPAW 1x1x1 (0.181 eV) and QE 1x1x1 (0.190 eV).
Previous ABACUS 2x2x2 gave 0.075 eV -- this tests whether the difference is
supercell size or basis set (LCAO vs PW).

Runs on AX102 alongside JDFTx with nice -19, omp=2.
"""
import sys
sys.path.insert(0, "/opt/abacus-develop-3.9.0.26/interfaces/ASE_interface")

import json
import time
import numpy as np
from pathlib import Path

from ase import Atom
from ase.io import read, write
from ase.spacegroup import crystal
from ase.geometry import get_distances
from ase.mep import NEB
from ase.optimize import FIRE
from ase.constraints import FixAtoms

from abacuslite import Abacus, AbacusProfile
print("abacuslite imported OK")

RESULTS = Path("/workspace/results")
RESULTS.mkdir(parents=True, exist_ok=True)
WORK_DIR = Path("/workspace/neb_pyrite_1x1x1")
WORK_DIR.mkdir(parents=True, exist_ok=True)

PP_DIR = "/workspace/sg15_pp"
ORB_DIR = "/workspace/sg15_orb"

N_IMAGES = 5
FMAX_RELAX = 0.05
FMAX_NEB = 0.05
MAX_STEPS_RELAX = 60
MAX_STEPS_NEB = 300

OMP = 2  # limited to not compete with JDFTx

profile = AbacusProfile(
    command="mpirun --allow-run-as-root -np 1 /opt/abacus-develop-3.9.0.26/build/abacus_2p",
    omp_num_threads=OMP,
    pseudo_dir=PP_DIR,
    orbital_dir=ORB_DIR,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def make_calc(label="abacus_neb"):
    return Abacus(
        profile=profile,
        directory=str(WORK_DIR / label),
        pseudopotentials={
            "Fe": "Fe_ONCV_PBE-1.2.upf",
            "S": "S_ONCV_PBE-1.2.upf",
            "H": "H_ONCV_PBE-1.2.upf",
        },
        basissets={
            "Fe": "Fe_gga_8au_100Ry_4s2p2d1f.orb",
            "S": "S_gga_7au_100Ry_2s2p1d.orb",
            "H": "H_gga_6au_100Ry_2s1p.orb",
        },
        # 2x2x2 kpts for 1x1x1 cell -- matches k-space sampling of
        # QE 1x1x1 (kpts 2x2x2) and GPAW 1x1x1 (kpts 2x2x2)
        kpts={'nk': [2, 2, 2], 'kshift': [0, 0, 0], 'gamma-centered': True, 'mode': 'mp-sampling'},
        inp={
            'basis_type': 'lcao',
            'calculation': 'scf',
            'nspin': 1,
            'ecutwfc': 80,
            'smearing_method': 'gaussian',
            'smearing_sigma': 0.05,
            'scf_thr': 1e-5,
            'scf_nmax': 400,
            'mixing_type': 'broyden',
            'mixing_beta': 0.1,
            'mixing_ndim': 8,
            'cal_force': 1,
            'cal_stress': 0,
            'symmetry': 0,
        },
    )


def build_pyrite():
    """Pyrite Pa-3 (#205), a=5.416 A, conventional cell = 12 atoms."""
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0), (0.385, 0.385, 0.385)],
        spacegroup=205,
        cellpar=[5.416, 5.416, 5.416, 90, 90, 90],
        primitive_cell=False,
    )
    print(f"  Pyrite: {atoms.get_chemical_formula()}, {len(atoms)} atoms")
    print(f"  Cell: {[f'{x:.3f}' for x in atoms.cell.lengths()]} A")
    return atoms


def find_s_pair(atoms):
    """Find nearest inter-dimer S-S pair (d > 2.5 A, skip S2 dimers)."""
    s_idx = [i for i, s in enumerate(atoms.symbols) if s == 'S']
    s_pos = atoms.positions[s_idx]
    _, d_matrix = get_distances(s_pos, cell=atoms.cell, pbc=True)

    best = None
    for i in range(len(s_idx)):
        for j in range(i + 1, len(s_idx)):
            d = d_matrix[i, j]
            if 2.5 < d < 4.5:
                if best is None or d < best[2]:
                    best = (s_idx[i], s_idx[j], d)
    if best is None:
        raise RuntimeError("No inter-dimer S-S pair found!")
    print(f"  S pair: atoms {best[0]} & {best[1]}, dist = {best[2]:.3f} A")
    return (best[0], best[1]), best[2]


def make_endpoints(sc, s_pair):
    """Create NEB endpoints: remove both S from pair, place H at each site."""
    si, sj = s_pair
    pos_i, pos_j = sc.positions[si].copy(), sc.positions[sj].copy()
    del_order = sorted([si, sj], reverse=True)

    endA = sc.copy()
    for idx in del_order:
        del endA[idx]
    endA.append(Atom('H', position=pos_i))

    endB = sc.copy()
    for idx in del_order:
        del endB[idx]
    endB.append(Atom('H', position=pos_j))

    for lbl, ep in [("endA", endA), ("endB", endB)]:
        h = len(ep) - 1
        md = min(ep.get_distance(h, j, mic=True) for j in range(len(ep)) if j != h)
        print(f"  {lbl}: {len(ep)} at, formula={ep.get_chemical_formula()}, min_H_dist={md:.3f} A")
    return endA, endB


def relax_endpoint(atoms, label):
    """Relax H position only (heavy atoms fixed)."""
    atoms = atoms.copy()
    heavy = [i for i in range(len(atoms)) if atoms[i].symbol != 'H']
    atoms.set_constraint(FixAtoms(indices=heavy))
    atoms.calc = make_calc(f"relax_{label}")

    opt = FIRE(atoms, logfile=str(WORK_DIR / f"relax_{label}.log"))
    t0 = time.time()
    converged = opt.run(fmax=FMAX_RELAX, steps=MAX_STEPS_RELAX)
    dt = time.time() - t0

    e = atoms.get_potential_energy()
    fmax = np.max(np.abs(atoms.get_forces()))
    print(f"  {label}: E={e:.4f} eV, fmax={fmax:.4f}, steps={opt.nsteps}, "
          f"time={dt:.0f}s, converged={converged}")
    write(str(WORK_DIR / f"relaxed_{label}.xyz"), atoms)
    return atoms, e


def run_neb(endA, endB):
    t_total = time.time()

    print(f"\n=== CI-NEB with {N_IMAGES} images ===")
    images = [endA]
    for i in range(N_IMAGES):
        img = endA.copy()
        img.calc = make_calc(f"image_{i:02d}")
        heavy = [j for j in range(len(img)) if img[j].symbol != 'H']
        img.set_constraint(FixAtoms(indices=heavy))
        images.append(img)
    images.append(endB)

    neb = NEB(images, climb=True, k=0.1, method="improvedtangent")
    neb.interpolate("idpp")

    # Print H positions along path
    h_idx = len(images[0]) - 1
    for k, img in enumerate(images):
        p = img.positions[h_idx]
        print(f"  image {k}: H ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    print(f"  Running FIRE NEB (fmax={FMAX_NEB})...")
    opt = FIRE(neb, logfile=str(WORK_DIR / "neb.log"),
               trajectory=str(WORK_DIR / "neb.traj"))
    converged = opt.run(fmax=FMAX_NEB, steps=MAX_STEPS_NEB)

    energies = [img.get_potential_energy() for img in images]
    e_ref = energies[0]
    rel_e = [e - e_ref for e in energies]
    barrier = max(rel_e)

    dt = time.time() - t_total
    print(f"\n  E_a = {barrier:.4f} eV")
    print(f"  Energies: {[f'{e:.4f}' for e in rel_e]}")
    print(f"  NEB steps: {opt.nsteps}, converged: {converged}, time: {dt:.0f}s")

    for k, img in enumerate(images):
        write(str(WORK_DIR / f"final_{k:02d}.xyz"), img)

    return barrier, rel_e, int(opt.nsteps), bool(converged), dt


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ABACUS NEB: pyrite FeS2 (Pa-3) 1x1x1, ~11 atoms")
    print(f"  LCAO DZP, kpts=(2,2,2), OMP={OMP}")
    print("  Cross-verify: GPAW 0.181, QE 0.190, ABACUS 2x2x2 0.075")
    print("=" * 60)

    print("\n[1] Build pyrite 1x1x1")
    pyr = build_pyrite()

    print("\n[2] Find S-S pair")
    sp, sd = find_s_pair(pyr)

    print("\n[3] Build endpoints")
    endA, endB = make_endpoints(pyr, sp)

    print("\n[4] Relax endA")
    endA_r, e_A = relax_endpoint(endA, "endA")

    print("\n[5] Relax endB")
    endB_r, e_B = relax_endpoint(endB, "endB")
    print(f"  |dE| = {abs(e_A - e_B):.6f} eV")

    print("\n[6] CI-NEB")
    barrier, energies, neb_steps, converged, dt_neb = run_neb(endA_r, endB_r)

    dt_total = time.time() - t0

    result = {
        "system": "pyrite_1x1x1",
        "method": "DFT_ABACUS_PBE_LCAO",
        "code": "ABACUS v3.9.0.26",
        "basis": "LCAO DZP (Fe_4s2p2d1f, S_2s2p1d, H_2s1p)",
        "pseudopotentials": "ONCV PBE 1.2",
        "ecutwfc_Ry": 80,
        "kpts": [2, 2, 2],
        "smearing": "gaussian 0.05 Ry",
        "n_atoms": len(endA),
        "formula": endA.get_chemical_formula(),
        "cell_A": pyr.cell.lengths().tolist(),
        "S_pair_indices": list(sp),
        "S_pair_dist_A": float(sd),
        "E_endA_eV": float(e_A),
        "E_endB_eV": float(e_B),
        "dE_endpoints_eV": float(abs(e_A - e_B)),
        "E_a_eV": float(barrier),
        "E_rxn_eV": float(energies[-1]),
        "energies_eV": [float(e) for e in energies],
        "n_images": N_IMAGES,
        "neb_steps": neb_steps,
        "converged": converged,
        "time_neb_s": float(dt_neb),
        "time_total_s": float(dt_total),
        # Cross-references
        "E_a_GPAW_1x1x1_eV": 0.181,
        "E_a_QE_1x1x1_eV": 0.190,
        "E_a_ABACUS_2x2x2_eV": 0.075,
        "E_a_MACE_2x2x2_eV": 0.79,
        "NOTE": "Apple-to-apple 1x1x1 cross-verify. ABACUS 2x2x2 used Gamma-only kpts.",
        "harvested": "",
        "source": "AX102 Hetzner"
    }

    rf = RESULTS / "neb_pyrite_abacus_1x1x1_result.json"
    with open(rf, 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'=' * 60}")
    print(f"  E_a(ABACUS 1x1x1) = {barrier:.4f} eV")
    print(f"  vs GPAW 1x1x1 = 0.181 eV")
    print(f"  vs QE 1x1x1   = 0.190 eV")
    print(f"  vs ABACUS 2x2x2 = 0.075 eV")
    print(f"  Saved: {rf}")
    print(f"  Total: {dt_total:.0f}s ({dt_total/60:.1f} min)")
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
