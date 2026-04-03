#!/usr/bin/env python3
"""
QE NEB: H diffusion in mackinawite FeS (P4/nmm)
Cross-verification of GPAW result (E_a = 0.738 eV)

Mackinawite: T_N ~ 65 K << RT → NO AFM, NO DFT+U
Layered structure → kpts dense in-plane, sparse out-of-plane
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

from ase import Atoms
from ase.build import make_supercell
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.io import write, read

try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB

# ASE 3.28+ Espresso API
try:
    from ase.calculators.espresso import Espresso, EspressoProfile
    USE_NEW_API = True
except ImportError:
    from ase.calculators.espresso import Espresso
    USE_NEW_API = False

# === CONFIGURATION ===
ECUTWFC = 60       # Ry
ECUTRHO = 480      # Ry (8x)
KPTS = (3, 3, 2)   # dense in-plane for layered structure
DEGAUSS = 0.02     # Ry (metallic)
MIXING_BETA = 0.2
CONV_THR = 1.0e-8
FMAX_RELAX = 0.05  # eV/A
FMAX_NEB = 0.05    # eV/A
MAX_STEPS_RELAX = 80
MAX_STEPS_NEB = 300
N_IMAGES = 5
SUPERCELL = (2, 2, 1)

WORK_DIR = Path("/workspace/neb_mackinawite_qe")
SCRATCH_DIR = Path("/workspace/qe_scratch_mack")
RESULTS_DIR = Path("/workspace/results")
RESUME_FILE = WORK_DIR / "resume.json"

PP_DIR = "/opt/pseudopotentials"
PSEUDOPOTENTIALS = {
    'Fe': 'Fe.pbe-spn-rrkjus_psl.1.0.0.UPF',
    'S':  'S.pbe-n-rrkjus_psl.1.0.0.UPF',
    'H':  'H.pbe-rrkjus_psl.1.0.0.UPF',
}

QE_CMD = "mpirun --allow-run-as-root --bind-to none -np 1 pw.x"
OMP = os.environ.get('OMP_NUM_THREADS', str(os.cpu_count()))
os.environ['OMP_NUM_THREADS'] = OMP


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_resume(data):
    with open(RESUME_FILE, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_resume():
    if RESUME_FILE.exists():
        with open(RESUME_FILE) as f:
            return json.load(f)
    return {}


def make_espresso_calc(label, restart=False):
    """Create QE calculator for mackinawite (non-magnetic, no DFT+U)"""
    scratch = SCRATCH_DIR / label
    scratch.mkdir(parents=True, exist_ok=True)
    calc_dir = WORK_DIR / label
    calc_dir.mkdir(parents=True, exist_ok=True)

    input_data = {
        'control': {
            'calculation': 'scf',
            'restart_mode': 'restart' if restart else 'from_scratch',
            'outdir': str(scratch),
            'prefix': 'mack',
            'tprnfor': True,
            'tstress': True,
            'disk_io': 'high',
            'max_seconds': 36000,
        },
        'system': {
            'ecutwfc': ECUTWFC,
            'ecutrho': ECUTRHO,
            'occupations': 'smearing',
            'smearing': 'mv',
            'degauss': DEGAUSS,
        },
        'electrons': {
            'mixing_beta': MIXING_BETA,
            'mixing_mode': 'plain',
            'mixing_ndim': 8,
            'conv_thr': CONV_THR,
            'electron_maxstep': 200,
        },
    }

    if USE_NEW_API:
        profile = EspressoProfile(command=QE_CMD, pseudo_dir=PP_DIR)
        calc = Espresso(
            input_data=input_data,
            pseudopotentials=PSEUDOPOTENTIALS,
            kpts=KPTS,
            profile=profile,
            directory=str(calc_dir),
        )
    else:
        calc = Espresso(
            input_data=input_data,
            pseudopotentials=PSEUDOPOTENTIALS,
            kpts=KPTS,
            command=QE_CMD,
            directory=str(calc_dir),
        )
    return calc


def build_mackinawite():
    """Build mackinawite P4/nmm primitive cell manually, then supercell"""
    # Mackinawite: P4/nmm, a=3.674, c=5.033
    # 4 atoms per unit cell: 2 Fe + 2 S
    a = 3.674
    c = 5.033
    z_S = 0.260  # fractional z of S

    cell = [[a, 0, 0], [0, a, 0], [0, 0, c]]
    # Fe at Wyckoff 2a: (0,0,0) and (1/2,1/2,0)
    # S at Wyckoff 2c: (0,1/2,z) and (1/2,0,-z)
    positions = [
        [0.0,   0.0,   0.0],          # Fe1
        [a/2,   a/2,   0.0],          # Fe2
        [0.0,   a/2,   z_S * c],      # S1
        [a/2,   0.0,   (1-z_S) * c],  # S2
    ]

    mack = Atoms('Fe2S2', positions=positions, cell=cell, pbc=True)
    print(f"  Primitive: {mack.get_chemical_formula()}, {len(mack)} atoms")
    print(f"  Cell: {[f'{x:.3f}' for x in mack.cell.lengths()]} A")

    # Supercell 2x2x1
    sc_matrix = np.diag(SUPERCELL)
    sc = make_supercell(mack, sc_matrix)
    print(f"  Supercell {SUPERCELL}: {sc.get_chemical_formula()}, {len(sc)} atoms")
    print(f"  Cell: {[f'{x:.3f}' for x in sc.cell.lengths()]} A")
    print(f"  Composition: {dict(zip(*np.unique(sc.symbols, return_counts=True)))}")

    # Verify min distance
    dists = sc.get_all_distances(mic=True)
    np.fill_diagonal(dists, 999)
    min_d = dists.min()
    print(f"  Min interatomic distance: {min_d:.3f} A")

    return sc


def find_s_pair(atoms):
    """Find S-S pair suitable for vacancy hop (in-plane, 3.0-4.5 A)"""
    s_indices = [i for i, s in enumerate(atoms.symbols) if s == 'S']
    candidates = []

    for i, si in enumerate(s_indices):
        for sj in s_indices[i+1:]:
            d = atoms.get_distance(si, sj, mic=True)
            if 3.0 < d < 4.5:  # in-plane hop range
                candidates.append((si, sj, d))

    candidates.sort(key=lambda x: x[2])
    if not candidates:
        # Fallback: any S-S pair 1.5-5.0
        for i, si in enumerate(s_indices):
            for sj in s_indices[i+1:]:
                d = atoms.get_distance(si, sj, mic=True)
                if 1.5 < d < 5.0:
                    candidates.append((si, sj, d))
        candidates.sort(key=lambda x: x[2])
        print(f"  WARNING: no in-plane pair found, using closest: {candidates[0][2]:.3f} A")

    si, sj, d = candidates[0]
    print(f"  S pair: atoms {si} & {sj}, distance = {d:.3f} A")
    print(f"  S_i pos: {atoms.positions[si]}")
    print(f"  S_j pos: {atoms.positions[sj]}")
    return (si, sj), d


def make_endpoints(supercell, s_pair):
    """Create NEB endpoints: H at vacancy site A and B"""
    si, sj = s_pair
    pos_i = supercell.positions[si].copy()
    pos_j = supercell.positions[sj].copy()

    # Delete in order (higher index first)
    del_order = sorted([si, sj], reverse=True)

    endA = supercell.copy()
    for idx in del_order:
        del endA[idx]
    endA.append('H')
    endA.positions[-1] = pos_i

    endB = supercell.copy()
    for idx in del_order:
        del endB[idx]
    endB.append('H')
    endB.positions[-1] = pos_j

    for label, ep in [("endA", endA), ("endB", endB)]:
        comp = dict(zip(*np.unique(ep.symbols, return_counts=True)))
        h_idx = len(ep) - 1
        dists = [ep.get_distance(h_idx, j, mic=True) for j in range(len(ep)) if j != h_idx]
        min_d = min(dists)
        print(f"  {label}: {len(ep)} atoms ({comp}), min_H_dist={min_d:.3f} A")

    return endA, endB


def relax_endpoint(atoms, label, resume_data):
    """Relax endpoint with FIRE (only H moves)"""
    key = f"relax_{label}"
    if key in resume_data and resume_data[key].get("converged"):
        print(f"  {label}: resume, E = {resume_data[key]['energy']:.4f} eV")
        xyz_file = WORK_DIR / f"relaxed_{label}.xyz"
        if xyz_file.exists():
            relaxed = read(str(xyz_file))
            # Restore constraint
            heavy = [i for i in range(len(relaxed)) if relaxed.symbols[i] != 'H']
            relaxed.set_constraint(FixAtoms(indices=heavy))
            relaxed.calc = make_espresso_calc(label, restart=True)
            return relaxed, resume_data[key]
        return atoms, resume_data[key]

    print(f"\n  Relaxing {label}...")
    heavy = [i for i in range(len(atoms)) if atoms.symbols[i] != 'H']
    atoms.set_constraint(FixAtoms(indices=heavy))

    calc = make_espresso_calc(label)
    atoms.calc = calc

    log_file = str(WORK_DIR / f"{label}.log")
    t0 = time.time()

    with FIRE(atoms, logfile=log_file) as opt:
        opt.run(fmax=FMAX_RELAX, steps=MAX_STEPS_RELAX)

    e = float(atoms.get_potential_energy())
    fmax_val = float(np.max(np.abs(atoms.get_forces())))
    steps = opt.nsteps
    conv = fmax_val < FMAX_RELAX
    dt = time.time() - t0

    write(str(WORK_DIR / f"relaxed_{label}.xyz"), atoms)

    result = {
        "energy": e, "fmax": fmax_val, "steps": int(steps),
        "converged": conv, "time_s": dt
    }
    resume_data[key] = result
    save_resume(resume_data)

    print(f"  {label}: E={e:.4f} eV, fmax={fmax_val:.4f}, steps={steps}, "
          f"conv={conv}, {dt:.0f}s")
    return atoms, result


def run_neb(end_a, end_b, resume_data):
    """Run CI-NEB with FIRE"""
    if "neb" in resume_data and resume_data["neb"].get("converged"):
        print("  NEB: resume (already converged)")
        return resume_data["neb"]

    print(f"\n[6/7] CI-NEB with {N_IMAGES} images (IDPP)")

    images = [end_a.copy()]
    for i in range(N_IMAGES):
        img = end_a.copy()
        img.calc = make_espresso_calc(f"neb_img_{i:02d}")
        images.append(img)
    images.append(end_b.copy())

    # Endpoints need calcs for NEB
    images[0].calc = make_espresso_calc("neb_endA", restart=False)
    images[-1].calc = make_espresso_calc("neb_endB", restart=False)

    # Constraints
    for img in images:
        heavy = [i for i in range(len(img)) if img.symbols[i] != 'H']
        img.set_constraint(FixAtoms(indices=heavy))

    neb = NEB(images, climb=True, method='improvedtangent',
              allow_shared_calculator=False, k=0.1)
    neb.interpolate('idpp')

    h_idx = len(images[0]) - 1
    for i, img in enumerate(images):
        pos = img.positions[h_idx]
        print(f"  image {i}: H at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    print(f"\n[7/7] Running FIRE NEB (fmax={FMAX_NEB}, max={MAX_STEPS_NEB})...")
    log_file = str(WORK_DIR / "neb.log")
    t0 = time.time()

    with FIRE(neb, logfile=log_file) as opt:
        opt.run(fmax=FMAX_NEB, steps=MAX_STEPS_NEB)

    dt = time.time() - t0

    energies = [float(img.get_potential_energy()) for img in images]
    e_ref = energies[0]
    rel_energies = [e - e_ref for e in energies]
    e_a = max(rel_energies)
    fmax_final = float(np.max(np.abs(neb.get_forces())))
    conv = fmax_final < FMAX_NEB

    for i, img in enumerate(images):
        write(str(WORK_DIR / f"final_img_{i:02d}.xyz"), img)

    result = {
        "E_a_eV": e_a, "E_rxn_eV": rel_energies[-1],
        "energies_eV": rel_energies, "neb_steps": int(opt.nsteps),
        "converged": conv, "fmax_final": fmax_final, "time_neb_s": dt,
    }
    resume_data["neb"] = result
    save_resume(resume_data)

    print(f"\n  E_a = {e_a:.4f} eV")
    print(f"  Energies: {[f'{e:.4f}' for e in rel_energies]}")
    print(f"  Steps: {opt.nsteps}, converged: {conv}")
    print(f"  Time: {dt:.0f}s")
    return result


def main():
    t_total = time.time()

    print("=" * 70)
    print("  QE NEB: H diffusion in mackinawite FeS (P4/nmm)")
    print(f"  ecutwfc={ECUTWFC} Ry, ecutrho={ECUTRHO} Ry, kpts={KPTS}")
    print(f"  Non-magnetic (T_N=65K << RT), no DFT+U")
    print(f"  N_images={N_IMAGES}, fmax_neb={FMAX_NEB} eV/A")
    print("=" * 70)
    print(f"  OMP_NUM_THREADS = {OMP}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    resume = load_resume()

    print(f"\n[1/7] Build mackinawite supercell {SUPERCELL}")
    sc = build_mackinawite()

    print(f"\n[2/7] Find S-S pair (in-plane, 3.0-4.5 A)")
    s_pair, s_dist = find_s_pair(sc)

    print(f"\n[3/7] Make NEB endpoints")
    endA, endB = make_endpoints(sc, s_pair)

    print(f"\n[4/7] Relax endpoint A")
    endA, res_a = relax_endpoint(endA, "endA", resume)

    print(f"\n[5/7] Relax endpoint B")
    endB, res_b = relax_endpoint(endB, "endB", resume)

    neb_result = run_neb(endA, endB, resume)

    dt_total = time.time() - t_total
    final = {
        "system": "mackinawite_intra_layer",
        "method": "DFT_QE_PBE_USPP",
        "code": "QE 7.4.1",
        "ecutwfc_Ry": ECUTWFC, "ecutrho_Ry": ECUTRHO,
        "kpts": list(KPTS), "degauss_Ry": DEGAUSS,
        "supercell": list(SUPERCELL),
        "n_atoms": len(endA),
        "formula": endA.get_chemical_formula(),
        "cell_A": endA.cell.lengths().tolist(),
        "S_pair_indices": list(s_pair), "S_pair_dist_A": float(s_dist),
        "E_endA_eV": res_a["energy"], "E_endB_eV": res_b["energy"],
        "dE_endpoints_eV": abs(res_a["energy"] - res_b["energy"]),
        "E_a_eV": neb_result["E_a_eV"], "E_rxn_eV": neb_result["E_rxn_eV"],
        "energies_eV": neb_result["energies_eV"],
        "converged": neb_result["converged"],
        "neb_steps": neb_result["neb_steps"],
        "time_neb_s": neb_result["time_neb_s"],
        "time_total_s": dt_total,
        "E_a_GPAW_eV": 0.738, "E_a_MACE_eV": 0.44,
    }

    result_file = RESULTS_DIR / "neb_mackinawite_qe_result.json"
    with open(result_file, 'w') as f:
        json.dump(final, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"  E_a (QE):   {neb_result['E_a_eV']:.4f} eV")
    print(f"  E_a (GPAW): 0.738 eV")
    print(f"  E_a (MACE): 0.44 eV")
    print(f"  Saved: {result_file}")
    print(f"  Total: {dt_total:.0f}s")
    print(f"{'=' * 70}")
    print("DONE")


if __name__ == '__main__':
    main()
