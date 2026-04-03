#!/usr/bin/env python3
"""
QE NEB: H diffusion in pyrite FeS2 (Pa-3), 2x2x2 supercell (~95 atoms).
GPU-accelerated QE 7.5 (OpenACC/CUDA).

Size convergence test:
  QE 1x1x1 = 0.190 eV, GPAW 1x1x1 = 0.181 eV, ABACUS 2x2x2 = 0.075 eV.
  This test: if QE 2x2x2 ~ 0.19 -> size converged, ABACUS LCAO underestimates.
             if QE 2x2x2 ~ 0.08 -> 1x1x1 has finite-size error.

GPU: RTX 3090 (sm_86), QE 7.5 OpenACC, npool=1, OMP=1.
"""

import os, sys, json, time
import numpy as np
from pathlib import Path
from ase import Atom, Atoms
from ase.spacegroup import crystal
from ase.geometry import get_distances
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

# --- Parameters ---
ECUTWFC = 60       # Ry (same as 1x1x1)
ECUTRHO = 480      # Ry
KPTS = (1, 1, 1)   # Gamma for 2x2x2 = equiv to (2,2,2) in 1x1x1
DEGAUSS = 0.01     # Ry — pyrite is semiconductor
FMAX_RELAX = 0.05  # eV/A
FMAX_NEB = 0.05    # eV/A
N_IMAGES = 5
MAX_STEPS_NEB = 200

WORK_DIR = Path("/workspace/neb_pyrite_qe_2x2x2")
SCRATCH_DIR = Path("/workspace/qe_scratch_pyr2")
RESULTS_DIR = Path("/workspace/results")
RESUME_FILE = WORK_DIR / "resume.json"
PP_DIR = "/opt/pp/pbe_paw"

PSEUDOPOTENTIALS = {
    'Fe': 'Fe.pbe-spn-kjpaw_psl.0.2.1.UPF',
    'S':  'S.pbe-n-kjpaw_psl.1.0.0.UPF',
    'H':  'H.pbe-kjpaw_psl.1.0.0.UPF',
}

# GPU mode: npool=1 (GPU handles parallelism), np=1
QE_CMD = "mpirun --allow-run-as-root --bind-to none -np 1 /opt/qe-7.5-gpu/bin/pw.x"
OMP = os.environ.get('OMP_NUM_THREADS', '1')
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
            'tprnfor': True, 'tstress': False,
            'disk_io': 'high', 'max_seconds': 86400,
        },
        'system': {
            'ecutwfc': ECUTWFC, 'ecutrho': ECUTRHO,
            'occupations': 'smearing', 'smearing': 'mv',
            'degauss': DEGAUSS,
        },
        'electrons': {
            'mixing_beta': 0.3, 'mixing_mode': 'plain',
            'mixing_ndim': 8, 'conv_thr': 1.0e-7,
            'electron_maxstep': 300,
        },
    }

    if USE_NEW_API:
        profile = EspressoProfile(command=QE_CMD, pseudo_dir=PP_DIR)
        return Espresso(input_data=input_data, pseudopotentials=PSEUDOPOTENTIALS,
                        kpts=KPTS, profile=profile, directory=str(calc_dir))
    else:
        return Espresso(input_data=input_data, pseudopotentials=PSEUDOPOTENTIALS,
                        kpts=KPTS, command=QE_CMD, directory=str(calc_dir))


def build_pyrite_2x2x2():
    """Pyrite Pa-3 (205), a=5.416 A, 2x2x2 supercell."""
    unit = crystal(
        symbols=['Fe', 'S'],
        basis=[(0, 0, 0), (0.385, 0.385, 0.385)],
        spacegroup=205,
        cellpar=[5.416, 5.416, 5.416, 90, 90, 90],
    )
    atoms = unit.repeat((2, 2, 2))
    print(f"  Pyrite 2x2x2: {atoms.get_chemical_formula()}, {len(atoms)} atoms")
    print(f"  Cell: {[f'{x:.3f}' for x in atoms.cell.lengths()]} A")
    return atoms


def find_s_pair(atoms):
    """Find nearest inter-dimer S-S pair (d > 2.5 A, skip S2 dimers)."""
    s_idx = [i for i, s in enumerate(atoms.symbols) if s == 'S']
    best = None
    for i, si in enumerate(s_idx):
        for sj in s_idx[i+1:]:
            d = atoms.get_distance(si, sj, mic=True)
            if 2.5 < d < 4.5:
                if best is None or d < best[2]:
                    best = (si, sj, d)
    if best is None:
        raise RuntimeError("No inter-dimer S-S pair found!")
    print(f"  S pair: {best[0]} & {best[1]}, dist = {best[2]:.3f} A")
    return (best[0], best[1]), best[2]


def make_endpoints(sc, s_pair):
    si, sj = s_pair
    pos_i, pos_j = sc.positions[si].copy(), sc.positions[sj].copy()
    del_order = sorted([si, sj], reverse=True)

    endA = sc.copy()
    for idx in del_order: del endA[idx]
    endA.append(Atom('H', position=pos_i))

    endB = sc.copy()
    for idx in del_order: del endB[idx]
    endB.append(Atom('H', position=pos_j))

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
        opt.run(fmax=FMAX_RELAX, steps=100)
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
        opt.run(fmax=FMAX_NEB, steps=MAX_STEPS_NEB)
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
    print("  QE NEB: pyrite FeS2 (Pa-3) 2x2x2, ~95 atoms, GPU")
    print(f"  ecutwfc={ECUTWFC}, kpts={KPTS} (Gamma), OMP={OMP}")
    print(f"  Cross-verify: QE 1x1x1=0.190, GPAW 1x1x1=0.181")
    print(f"  Size ref: ABACUS 2x2x2=0.075")
    print("=" * 60)

    for d in [WORK_DIR, SCRATCH_DIR, RESULTS_DIR]: d.mkdir(parents=True, exist_ok=True)

    # Check GPU
    try:
        import subprocess
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True)
        print(f"  GPU: {r.stdout.strip()}")
    except Exception:
        print("  GPU: not detected (CPU fallback)")

    resume = load_resume()

    print("\n[1] Build pyrite 2x2x2")
    pyr = build_pyrite_2x2x2()
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
        "system": "pyrite_2x2x2", "method": "DFT_QE_PBE_USPP_GPU",
        "code": "QE 7.5 (OpenACC GPU)",
        "ecutwfc_Ry": ECUTWFC, "ecutrho_Ry": ECUTRHO,
        "kpts": list(KPTS), "kpts_note": "Gamma for 2x2x2 = equiv (2,2,2) in 1x1x1",
        "degauss_Ry": DEGAUSS,
        "supercell": [2, 2, 2],
        "n_atoms": len(ea), "formula": ea.get_chemical_formula(),
        "cell_A": pyr.cell.lengths().tolist(),
        "S_pair_indices": list(sp), "S_pair_dist_A": float(sd),
        "E_a_eV": nr["E_a_eV"], "energies_eV": nr["energies_eV"],
        "converged": nr["converged"], "neb_steps": nr["neb_steps"],
        "time_total_s": dt,
        "E_a_QE_1x1x1_eV": 0.190, "E_a_GPAW_1x1x1_eV": 0.181,
        "E_a_ABACUS_2x2x2_eV": 0.075, "E_a_MACE_2x2x2_eV": 0.79,
        "NOTE": "Size convergence test. Same QE PW/USPP as 1x1x1, larger cell + GPU.",
        "harvested": "", "source": "Vast.ai QE GPU"
    }
    rf = RESULTS_DIR / "neb_pyrite_qe_2x2x2_result.json"
    with open(rf, 'w') as f: json.dump(final, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'=' * 60}")
    print(f"  E_a(QE 2x2x2) = {nr['E_a_eV']:.4f} eV")
    print(f"  vs QE 1x1x1    = 0.190 eV  (size convergence?)")
    print(f"  vs ABACUS 2x2x2 = 0.075 eV  (basis set effect?)")
    print(f"  vs GPAW 1x1x1  = 0.181 eV")
    print(f"  Saved: {rf}")
    print(f"  Total: {dt:.0f}s ({dt/3600:.1f} h)")
    print("=" * 60)
    print("DONE")


if __name__ == '__main__':
    main()
