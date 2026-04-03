#!/usr/bin/env python3
"""
QE NEB: H diffusion in pentlandite (Fe,Ni)9S8, primitive cell (17 atoms).
GPU-accelerated QE 7.5 (OpenACC/CUDA).

Cross-verify with GPAW (1.115 eV) and ABACUS (0.900 eV).
MACE-MP-0 reference: 1.43 eV (2x2x2), 0.96 eV (primitive).

Crystal: Fm-3m (#225), a = 10.044 A (Tsukimura 1992, CIF 0007705).
Primitive cell: Fe5Ni4S8, 17 atoms.
Mechanism: vacancy-mediated H hop between nearest S-S sites.

GPU: RTX 3090, QE 7.5 OpenACC, npool=1, OMP=1.
"""

import os, sys, json, time
import numpy as np
from pathlib import Path
from ase import Atom
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
ECUTWFC = 60       # Ry
ECUTRHO = 480      # Ry
KPTS = (2, 2, 2)   # prim cell ~7.1 A; Gamma causes ASE parse bug with nspin=2
DEGAUSS = 0.02     # Ry -- pentlandite is metallic, needs more than pyrite
FMAX_RELAX = 0.05  # eV/A
FMAX_NEB = 0.05    # eV/A
N_IMAGES = 5
MAX_STEPS_NEB = 300

WORK_DIR = Path("/workspace/neb_pentlandite_qe")
SCRATCH_DIR = Path("/workspace/qe_scratch_pent")
RESULTS_DIR = Path("/workspace/results")
RESUME_FILE = WORK_DIR / "resume.json"
PP_DIR = "/opt/pp/pbe_paw"

PSEUDOPOTENTIALS = {
    'Fe': 'Fe.pbe-spn-kjpaw_psl.0.2.1.UPF',
    'Ni': 'Ni.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'S':  'S.pbe-n-kjpaw_psl.1.0.0.UPF',
    'H':  'H.pbe-kjpaw_psl.1.0.0.UPF',
}

# GPU mode: np=1, npool=1
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
            'outdir': str(scratch), 'prefix': 'pent',
            'tprnfor': True, 'tstress': False,
            'disk_io': 'high', 'max_seconds': 86400,
        },
        'system': {
            'ecutwfc': ECUTWFC, 'ecutrho': ECUTRHO,
            'occupations': 'smearing', 'smearing': 'mv',
            'degauss': DEGAUSS,
            # nspin=1 (default): pentlandite is Pauli paramagnetic, no ordered magnetism
        },
        'electrons': {
            'mixing_beta': 0.2, 'mixing_mode': 'plain',
            'mixing_ndim': 8, 'conv_thr': 1.0e-8,
            'electron_maxstep': 400,
        },
    }

    if USE_NEW_API:
        profile = EspressoProfile(command=QE_CMD, pseudo_dir=PP_DIR)
        return Espresso(input_data=input_data, pseudopotentials=PSEUDOPOTENTIALS,
                        kpts=KPTS, profile=profile, directory=str(calc_dir))
    else:
        return Espresso(input_data=input_data, pseudopotentials=PSEUDOPOTENTIALS,
                        kpts=KPTS, command=QE_CMD, directory=str(calc_dir))


def build_pentlandite_primitive():
    """Pentlandite Fm-3m (#225), a=10.044 A, primitive cell.
    Fe5Ni4S8, 17 atoms. Tsukimura 1992 (CIF 0007705).
    """
    a = 10.044
    atoms = crystal(
        symbols=['Fe', 'Fe', 'S', 'S'],
        basis=[
            (0.5, 0.5, 0.5),       # 4b: Metal1 -> Fe (CIF occ=0.986)
            (0.125, 0.125, 0.125),  # 32f: Metal2 -> Fe (initially)
            (0.25, 0.25, 0.25),     # 8c: S1
            (0.25, 0.0, 0.0),       # 24e: S2
        ],
        spacegroup=225,
        cellpar=[a, a, a, 90, 90, 90],
        primitive_cell=True,
    )
    # Assign Fe/Ni on 32f: 4 Fe + 4 Ni -> Fe5Ni4S8
    syms = atoms.get_chemical_symbols()
    fe_on_32f = [i for i, s in enumerate(syms) if s == 'Fe' and i != 0]
    for i in fe_on_32f[4:]:
        syms[i] = 'Ni'
    atoms.set_chemical_symbols(syms)
    return atoms


def find_ss_pair(atoms):
    """Find nearest S-S pair for vacancy hop."""
    s_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'S']
    s_pos = atoms.positions[s_idx]
    _, d_mat = get_distances(s_pos, cell=atoms.cell, pbc=True)
    best = (None, None, np.inf)
    for a in range(len(s_idx)):
        for b in range(a+1, len(s_idx)):
            if d_mat[a, b] < best[2]:
                best = (s_idx[a], s_idx[b], d_mat[a, b])
    return best


def make_endpoints(atoms, si, sj):
    """Remove 2 S, place H at each vacancy site."""
    pos_i, pos_j = atoms.positions[si].copy(), atoms.positions[sj].copy()
    del_order = sorted([si, sj], reverse=True)

    endA = atoms.copy()
    for idx in del_order: del endA[idx]
    endA.append(Atom('H', position=pos_i))

    endB = atoms.copy()
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
        img = ea.copy()
        img.calc = make_calc(f"neb_{i:02d}")
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

    # Progress logging every NEB step
    step_log = WORK_DIR / "neb_step.log"
    def log_step():
        energies = [float(img.get_potential_energy()) for img in images]
        rel = [e - energies[0] for e in energies]
        barrier = max(rel)
        fmax = float(np.max(np.abs(neb.get_forces())))
        with open(step_log, 'a') as f:
            f.write(f"  NEB step {opt.nsteps:4d}: barrier_est={barrier:.4f} eV, fmax={fmax:.4f}\n")
        print(f"  NEB step {opt.nsteps:4d}: barrier_est={barrier:.4f} eV")
        sys.stdout.flush()

    with FIRE(neb, logfile=str(WORK_DIR / "neb.log")) as opt:
        for step in range(MAX_STEPS_NEB):
            opt.run(fmax=FMAX_NEB, steps=1)
            log_step()
            if opt.converged():
                break

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
    print("  QE NEB: pentlandite (Fe,Ni)9S8 primitive, 16 atoms, GPU")
    print(f"  ecutwfc={ECUTWFC}, kpts={KPTS}, OMP={OMP}")
    print(f"  Cross-verify: GPAW=1.115, ABACUS=0.900, MACE=1.43")
    print("=" * 60)

    for d in [WORK_DIR, SCRATCH_DIR, RESULTS_DIR]: d.mkdir(parents=True, exist_ok=True)

    # Check GPU
    try:
        import subprocess
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True)
        print(f"  GPU: {r.stdout.strip()}")
    except Exception:
        print("  GPU: not detected")

    # Check PPs exist
    for el, pp in PSEUDOPOTENTIALS.items():
        pp_path = Path(PP_DIR) / pp
        if not pp_path.exists():
            # Try alternative names
            alt = pp_path.parent.glob(f"{el}.*UPF")
            found = list(alt)
            if found:
                print(f"  WARNING: {pp} not found, available: {[f.name for f in found]}")
            else:
                print(f"  ERROR: No PP for {el} in {PP_DIR}!")
                sys.exit(1)
        else:
            print(f"  PP {el}: {pp} OK")

    resume = load_resume()

    print("\n[1] Build pentlandite primitive")
    atoms = build_pentlandite_primitive()
    syms = atoms.get_chemical_symbols()
    n_fe, n_ni, n_s = syms.count('Fe'), syms.count('Ni'), syms.count('S')
    print(f"  {atoms.get_chemical_formula()}, {len(atoms)} atoms "
          f"({n_fe} Fe + {n_ni} Ni + {n_s} S)")
    print(f"  Cell: {[f'{x:.3f}' for x in atoms.cell.lengths()]} A")
    assert len(atoms) == 17
    assert n_fe == 5 and n_ni == 4 and n_s == 8

    # Min distance
    D = atoms.get_all_distances(mic=True)
    np.fill_diagonal(D, np.inf)
    min_d = np.min(D)
    print(f"  Min distance: {min_d:.3f} A")
    assert min_d > 1.5

    print("\n[2] Find S-S pair")
    si, sj, sd = find_ss_pair(atoms)
    print(f"  S pair: {si} & {sj}, dist = {sd:.3f} A")

    print("\n[3] Endpoints")
    ea, eb = make_endpoints(atoms, si, sj)

    print("\n[4] Relax endA")
    ea, ra = relax_ep(ea, "endA", resume)
    print("\n[5] Relax endB")
    eb, rb = relax_ep(eb, "endB", resume)
    print("\n[6] NEB")
    nr = run_neb(ea, eb, resume)

    dt = time.time() - t0
    final = {
        "system": "pentlandite_primitive", "method": "DFT_QE_PBE_PAW_GPU",
        "code": "QE 7.5 (OpenACC GPU)",
        "ecutwfc_Ry": ECUTWFC, "ecutrho_Ry": ECUTRHO,
        "kpts": list(KPTS),
        "degauss_Ry": DEGAUSS, "smearing": "mv",
        "nspin": 1,
        "supercell": "primitive",
        "n_atoms": len(ea), "formula": ea.get_chemical_formula(),
        "composition": f"Fe{n_fe}Ni{n_ni}S{n_s-2}H1",
        "cell_A": atoms.cell.lengths().tolist(),
        "S_pair_indices": [int(si), int(sj)], "S_pair_dist_A": float(sd),
        "E_a_eV": nr["E_a_eV"], "energies_eV": nr["energies_eV"],
        "converged": nr["converged"], "neb_steps": nr["neb_steps"],
        "time_total_s": dt,
        "cross_verify": {
            "GPAW_prim_eV": 1.115,
            "ABACUS_prim_eV": 0.900,
            "MACE_prim_eV": 0.96,
            "MACE_2x2x2_eV": 1.43,
        },
        "structure_source": "Tsukimura 1992 (CIF 0007705)",
        "harvested": "", "source": "Vast.ai QE GPU"
    }
    rf = RESULTS_DIR / "neb_pentlandite_qe_result.json"
    with open(rf, 'w') as f: json.dump(final, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'=' * 60}")
    print(f"  E_a(QE prim) = {nr['E_a_eV']:.4f} eV")
    print(f"  vs GPAW prim   = 1.115 eV")
    print(f"  vs ABACUS prim = 0.900 eV")
    print(f"  vs MACE prim   = 0.96 eV")
    print(f"  vs MACE 2x2x2  = 1.43 eV")
    print(f"  Saved: {rf}")
    print(f"  Total: {dt:.0f}s ({dt/3600:.1f} h)")
    print("=" * 60)
    print("DONE")


if __name__ == '__main__':
    main()
