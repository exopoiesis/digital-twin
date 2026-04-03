#!/usr/bin/env python3
"""
QE NEB: H diffusion in pyrite FeS2 (Pa-3), 1x1x1 cell
Cross-verification of GPAW result (E_a = 0.181 eV, 11 atoms)
Small test suitable for 8-core GCP instance.
"""

import os, sys, json, time
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.spacegroup import crystal
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.io import write, read

try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB

try:
    from ase.calculators.espresso import Espresso, EspressoProfile
    USE_NEW_API = True
except ImportError:
    from ase.calculators.espresso import Espresso
    USE_NEW_API = False

ECUTWFC = 60
ECUTRHO = 480
KPTS = (2, 2, 2)
DEGAUSS = 0.01  # pyrite is semiconductor
FMAX_RELAX = 0.05
FMAX_NEB = 0.05
N_IMAGES = 5

WORK_DIR = Path("/workspace/neb_pyrite_qe")
SCRATCH_DIR = Path("/workspace/qe_scratch_pyr")
RESULTS_DIR = Path("/workspace/results")
RESUME_FILE = WORK_DIR / "resume.json"
PP_DIR = "/opt/pseudopotentials"

PSEUDOPOTENTIALS = {
    'Fe': 'Fe.pbe-spn-rrkjus_psl.1.0.0.UPF',
    'S':  'S.pbe-n-rrkjus_psl.1.0.0.UPF',
    'H':  'H.pbe-rrkjus_psl.1.0.0.UPF',
}

QE_CMD = f"mpirun --allow-run-as-root --bind-to none -np 1 pw.x"
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


def make_calc(label, restart=False):
    scratch = SCRATCH_DIR / label
    scratch.mkdir(parents=True, exist_ok=True)
    calc_dir = WORK_DIR / label
    calc_dir.mkdir(parents=True, exist_ok=True)

    input_data = {
        'control': {
            'calculation': 'scf',
            'restart_mode': 'restart' if restart else 'from_scratch',
            'outdir': str(scratch), 'prefix': 'pyr',
            'tprnfor': True, 'tstress': True,
            'disk_io': 'high', 'max_seconds': 14400,
        },
        'system': {
            'ecutwfc': ECUTWFC, 'ecutrho': ECUTRHO,
            'occupations': 'smearing', 'smearing': 'mv',
            'degauss': DEGAUSS,
        },
        'electrons': {
            'mixing_beta': 0.3, 'mixing_mode': 'plain',
            'mixing_ndim': 8, 'conv_thr': 1.0e-8,
            'electron_maxstep': 200,
        },
    }

    if USE_NEW_API:
        profile = EspressoProfile(command=QE_CMD, pseudo_dir=PP_DIR)
        return Espresso(input_data=input_data, pseudopotentials=PSEUDOPOTENTIALS,
                        kpts=KPTS, profile=profile, directory=str(calc_dir))
    else:
        return Espresso(input_data=input_data, pseudopotentials=PSEUDOPOTENTIALS,
                        kpts=KPTS, command=QE_CMD, directory=str(calc_dir))


def build_pyrite():
    """Pyrite Pa-3 (205), a=5.416"""
    pyr = crystal(
        symbols=['Fe', 'S'],
        basis=[(0, 0, 0), (0.385, 0.385, 0.385)],
        spacegroup=205,
        cellpar=[5.416, 5.416, 5.416, 90, 90, 90],
    )
    print(f"  Pyrite: {pyr.get_chemical_formula()}, {len(pyr)} atoms")
    print(f"  Cell: {[f'{x:.3f}' for x in pyr.cell.lengths()]} A")
    return pyr


def find_s_pair(atoms):
    s_idx = [i for i, s in enumerate(atoms.symbols) if s == 'S']
    best = None
    for i, si in enumerate(s_idx):
        for sj in s_idx[i+1:]:
            d = atoms.get_distance(si, sj, mic=True)
            if 2.5 < d < 4.5:
                if best is None or d < best[2]:
                    best = (si, sj, d)
    print(f"  S pair: {best[0]} & {best[1]}, dist = {best[2]:.3f} A")
    return (best[0], best[1]), best[2]


def make_endpoints(sc, s_pair):
    si, sj = s_pair
    pos_i, pos_j = sc.positions[si].copy(), sc.positions[sj].copy()
    del_order = sorted([si, sj], reverse=True)

    endA = sc.copy()
    for idx in del_order: del endA[idx]
    endA.append('H'); endA.positions[-1] = pos_i

    endB = sc.copy()
    for idx in del_order: del endB[idx]
    endB.append('H'); endB.positions[-1] = pos_j

    for lbl, ep in [("endA", endA), ("endB", endB)]:
        h = len(ep) - 1
        md = min(ep.get_distance(h, j, mic=True) for j in range(len(ep)) if j != h)
        print(f"  {lbl}: {len(ep)} at, min_H_dist={md:.3f} A")
    return endA, endB


def relax_ep(atoms, label, resume):
    key = f"relax_{label}"
    if key in resume and resume[key].get("converged"):
        print(f"  {label}: resume E={resume[key]['energy']:.4f}")
        xyz = WORK_DIR / f"relaxed_{label}.xyz"
        if xyz.exists():
            r = read(str(xyz))
            heavy = [i for i in range(len(r)) if r.symbols[i] != 'H']
            r.set_constraint(FixAtoms(indices=heavy))
            r.calc = make_calc(label, restart=True)
            return r, resume[key]
        return atoms, resume[key]

    print(f"  Relaxing {label}...")
    heavy = [i for i in range(len(atoms)) if atoms.symbols[i] != 'H']
    atoms.set_constraint(FixAtoms(indices=heavy))
    atoms.calc = make_calc(label)
    t0 = time.time()
    with FIRE(atoms, logfile=str(WORK_DIR / f"{label}.log")) as opt:
        opt.run(fmax=FMAX_RELAX, steps=60)
    e = float(atoms.get_potential_energy())
    fm = float(np.max(np.abs(atoms.get_forces())))
    dt = time.time() - t0
    conv = fm < FMAX_RELAX
    write(str(WORK_DIR / f"relaxed_{label}.xyz"), atoms)
    result = {"energy": e, "fmax": fm, "steps": int(opt.nsteps), "converged": conv, "time_s": dt}
    resume[key] = result; save_resume(resume)
    print(f"  {label}: E={e:.4f}, fmax={fm:.4f}, steps={opt.nsteps}, {dt:.0f}s")
    return atoms, result


def run_neb(ea, eb, resume):
    if "neb" in resume and resume["neb"].get("converged"):
        print("  NEB: already converged"); return resume["neb"]

    images = [ea.copy()]
    for i in range(N_IMAGES):
        img = ea.copy(); img.calc = make_calc(f"neb_{i:02d}")
        images.append(img)
    images.append(eb.copy())
    images[0].calc = make_calc("neb_endA")
    images[-1].calc = make_calc("neb_endB")

    for img in images:
        heavy = [i for i in range(len(img)) if img.symbols[i] != 'H']
        img.set_constraint(FixAtoms(indices=heavy))

    neb = NEB(images, climb=True, method='improvedtangent', allow_shared_calculator=False, k=0.1)
    neb.interpolate('idpp')

    h = len(images[0]) - 1
    for i, img in enumerate(images):
        p = img.positions[h]
        print(f"  img {i}: H ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    print(f"  Running FIRE NEB (fmax={FMAX_NEB})...")
    t0 = time.time()
    with FIRE(neb, logfile=str(WORK_DIR / "neb.log")) as opt:
        opt.run(fmax=FMAX_NEB, steps=200)
    dt = time.time() - t0

    energies = [float(img.get_potential_energy()) for img in images]
    rel = [e - energies[0] for e in energies]
    e_a = max(rel)
    conv = float(np.max(np.abs(neb.get_forces()))) < FMAX_NEB

    for i, img in enumerate(images):
        write(str(WORK_DIR / f"final_{i:02d}.xyz"), img)

    result = {"E_a_eV": e_a, "E_rxn_eV": rel[-1], "energies_eV": rel,
              "neb_steps": int(opt.nsteps), "converged": conv, "time_s": dt}
    resume["neb"] = result; save_resume(resume)
    print(f"  E_a = {e_a:.4f} eV, steps={opt.nsteps}, conv={conv}, {dt:.0f}s")
    print(f"  Energies: {[f'{e:.4f}' for e in rel]}")
    return result


def main():
    t0 = time.time()
    print("=" * 60)
    print("  QE NEB: pyrite FeS2 (Pa-3) 1x1x1, 11 atoms")
    print(f"  ecutwfc={ECUTWFC}, kpts={KPTS}, OMP={OMP}")
    print("=" * 60)

    for d in [WORK_DIR, SCRATCH_DIR, RESULTS_DIR]: d.mkdir(parents=True, exist_ok=True)
    resume = load_resume()

    print("\n[1] Build pyrite")
    pyr = build_pyrite()
    print("\n[2] Find S-S pair")
    sp, sd = find_s_pair(pyr)
    print("\n[3] Endpoints")
    ea, eb = make_endpoints(pyr, sp)
    print("\n[4] Relax endA")
    ea, ra = relax_ep(ea, "endA", resume)
    print("\n[5] Relax endB")
    eb, rb = relax_ep(eb, "endB", resume)
    print("\n[6] NEB")
    nr = run_neb(ea, eb, resume)

    dt = time.time() - t0
    final = {
        "system": "pyrite_1x1x1", "method": "DFT_QE_PBE_USPP",
        "n_atoms": len(ea), "formula": ea.get_chemical_formula(),
        "E_a_eV": nr["E_a_eV"], "energies_eV": nr["energies_eV"],
        "converged": nr["converged"], "neb_steps": nr["neb_steps"],
        "time_total_s": dt,
        "E_a_GPAW_1x1x1_eV": 0.181, "E_a_ABACUS_2x2x2_eV": 0.075,
        "E_a_MACE_eV": 0.79,
    }
    rf = RESULTS_DIR / "neb_pyrite_qe_result.json"
    with open(rf, 'w') as f: json.dump(final, f, indent=2, cls=NumpyEncoder)
    print(f"\n  E_a(QE)={nr['E_a_eV']:.4f} vs GPAW={0.181} vs ABACUS={0.075}")
    print(f"  Saved: {rf}, Total: {dt:.0f}s")
    print("DONE")

if __name__ == '__main__':
    main()
