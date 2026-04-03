#!/usr/bin/env python3
"""
QE NEB: Grotthuss H+ transfer through interlayer water in mackinawite FeS.
GPU-accelerated QE 7.5 (OpenACC/CUDA).

Mechanism D (DECISION-069): H3O+ -> H2O -> H3O+ via H-bond chain in interlayer.
Ref: Marx 2013 (AIMD, ~0 eV); dry paths A=0.44-0.74, B=2.48, C=4.19 eV.

Structure: P4/nmm (#129), a=3.674 A, c=6.5 A (expanded for water).
Supercell: 2x2x1, 16 FeS atoms + 4 H2O + 1 H+ = 29 atoms total.
nspin=1 (T_N=65K << RT, paramagnetic at 25C).
vdW: DFT-D3(BJ) -- 87% adsorption energy is dispersion (Dzade 2016).

GPU: QE 7.5 OpenACC, npool=1, OMP=1.

Two-phase NEB:
  Phase 1: climb=False, fmax < 0.3 eV/A, max 200 steps
  Phase 2: climb=True,  fmax < 0.05 eV/A, max 300 steps
FixAtoms: all Fe and S (indices 0-15). Free: all H and O (water + proton).
"""

import os
import sys
import json
import time
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
ECUTWFC = 60        # Ry
ECUTRHO = 480       # Ry
KPTS = (2, 2, 1)    # 2x2 in ab, 1 along c (layered, large c)
DEGAUSS = 0.02      # Ry
FMAX_WATER = 0.03   # eV/A -- water relaxation
FMAX_EP = 0.05      # eV/A -- endpoint relaxation
FMAX_NEB_P1 = 0.30  # eV/A -- NEB phase 1 (no climbing)
FMAX_NEB_P2 = 0.05  # eV/A -- NEB phase 2 (CI-NEB)
N_IMAGES = 5
MAX_STEPS_WATER = 200
MAX_STEPS_EP = 150
MAX_STEPS_NEB_P1 = 200
MAX_STEPS_NEB_P2 = 300
K_SPRING = 0.1      # eV/A^2

# Mackinawite geometry
A_MACK = 3.674  # A
C_MACK = 6.5    # A (expanded for water, dry=5.033)
Z_S = 0.260     # Wyckoff 2c fractional z

WORK_DIR = Path("/workspace/neb_mack_grotthuss")
SCRATCH_DIR = Path("/workspace/qe_scratch_mack_wet")
RESULTS_DIR = Path("/workspace/results")
RESUME_FILE = WORK_DIR / "resume.json"
PP_DIR = "/opt/pp/pbe_paw"

PSEUDOPOTENTIALS = {
    'Fe': 'Fe.pbe-spn-kjpaw_psl.0.2.1.UPF',
    'S':  'S.pbe-n-kjpaw_psl.1.0.0.UPF',
    'H':  'H.pbe-kjpaw_psl.1.0.0.UPF',
    'O':  'O.pbe-n-kjpaw_psl.1.0.0.UPF',
}

# GPU mode: np=1, npool=1
QE_CMD = "mpirun --allow-run-as-root --bind-to none -np 1 /opt/qe-7.5-gpu/bin/pw.x"
OMP = os.environ.get('OMP_NUM_THREADS', '1')
os.environ['OMP_NUM_THREADS'] = OMP


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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
            'outdir': str(scratch),
            'prefix': 'mack_wet',
            'tprnfor': True,
            'tstress': False,
            'disk_io': 'high',
            'max_seconds': 86400,
        },
        'system': {
            'ecutwfc': ECUTWFC,
            'ecutrho': ECUTRHO,
            'occupations': 'smearing',
            'smearing': 'mv',
            'degauss': DEGAUSS,
            # nspin=1: mackinawite T_N=65K << RT, paramagnetic at 25C
            # DFT-D3(BJ): essential for layered structure (87% E_ads from dispersion)
            'vdw_corr': 'dft-d3',
            'dftd3_version': 4,   # BJ damping
        },
        'electrons': {
            'mixing_beta': 0.2,
            'mixing_mode': 'plain',
            'mixing_ndim': 8,
            'conv_thr': 1.0e-8,
            'electron_maxstep': 500,  # extra for water complexity
        },
    }

    if USE_NEW_API:
        profile = EspressoProfile(command=QE_CMD, pseudo_dir=PP_DIR)
        return Espresso(
            input_data=input_data,
            pseudopotentials=PSEUDOPOTENTIALS,
            kpts=KPTS,
            profile=profile,
            directory=str(calc_dir),
        )
    else:
        return Espresso(
            input_data=input_data,
            pseudopotentials=PSEUDOPOTENTIALS,
            kpts=KPTS,
            command=QE_CMD,
            directory=str(calc_dir),
        )


def build_mackinawite_with_water():
    """
    Build mackinawite FeS 2x2x1 supercell with 4 H2O in interlayer.

    P4/nmm (#129), a=3.674 A, c=6.5 A (expanded for water).
    Wyckoff: Fe 2a (0,0,0); S 2c (0,1/2,z_S), z_S=0.260.
    Repeat 2x2x1 -> 8 Fe + 8 S = 16 atoms (indices 0-15).
    4 H2O at z = c/2 (interlayer midplane), quadratic grid in ab.

    Returns atoms (28 atoms = 16 FeS + 4x3 H2O).
    Also returns O positions and supercell parameter a_sc for use in endpoints.
    """
    # Unit cell: FeS P4/nmm using ase.spacegroup.crystal
    # Fe 2a: (0, 0, 0) and (1/2, 1/2, 0) by symmetry
    # S 2c: (0, 1/2, z_S) and (1/2, 0, -z_S) by symmetry
    mack_unit = crystal(
        symbols=['Fe', 'S'],
        basis=[
            (0.0, 0.0, 0.0),       # Fe 2a
            (0.0, 0.5, Z_S),       # S 2c
        ],
        spacegroup=129,
        cellpar=[A_MACK, A_MACK, C_MACK, 90, 90, 90],
        primitive_cell=False,
    )

    # Verify unit cell: 2 Fe + 2 S
    syms = mack_unit.get_chemical_symbols()
    n_fe = syms.count('Fe')
    n_s = syms.count('S')
    print(f"  Unit cell: {mack_unit.get_chemical_formula()}, "
          f"{len(mack_unit)} atoms ({n_fe} Fe, {n_s} S)")
    assert n_fe == 2 and n_s == 2, (
        f"Expected 2 Fe + 2 S in unit cell, got {n_fe} Fe + {n_s} S"
    )

    # 2x2x1 supercell -> 8 Fe + 8 S
    mack_sc = mack_unit.repeat((2, 2, 1))
    syms_sc = mack_sc.get_chemical_symbols()
    assert syms_sc.count('Fe') == 8 and syms_sc.count('S') == 8, (
        f"Unexpected supercell: {mack_sc.get_chemical_formula()}"
    )
    print(f"  Supercell: {mack_sc.get_chemical_formula()}, {len(mack_sc)} atoms")
    print(f"  Cell: {[f'{x:.4f}' for x in mack_sc.cell.lengths()]} A")

    # Supercell lattice parameters
    a_sc = 2.0 * A_MACK   # 7.348 A
    c_sc = C_MACK          # 6.5 A
    z_mid = c_sc / 2.0     # 3.25 A (interlayer midplane)

    # 4 H2O positions: quadratic grid at interlayer midplane
    # O placed above Fe sites, ~2.2-2.5 A from surface (Dzade 2016)
    # Quadratic grid: (a/4, a/4), (3a/4, a/4), (a/4, 3a/4), (3a/4, 3a/4)
    o_positions = np.array([
        [a_sc / 4.0, a_sc / 4.0, z_mid],   # H2O_1: (1.837, 1.837, 3.25)
        [3.0 * a_sc / 4.0, a_sc / 4.0, z_mid],   # H2O_2: (5.511, 1.837, 3.25)
        [a_sc / 4.0, 3.0 * a_sc / 4.0, z_mid],   # H2O_3: (1.837, 5.511, 3.25)
        [3.0 * a_sc / 4.0, 3.0 * a_sc / 4.0, z_mid],  # H2O_4: (5.511, 5.511, 3.25)
    ])

    # H positions: O-H = 0.96 A, angle HOH = 104.5 deg
    # H1 and H2 in xy-plane (water pointing along ab), symmetric about y-axis
    # angle_half = 104.5/2 = 52.25 deg
    # delta: dx = 0.96 * cos(52.25 deg) = 0.96 * 0.610 = 0.586 A
    #        dy = 0.96 * sin(52.25 deg) = 0.96 * 0.792 = 0.760 A (in ab, not z)
    # Water oriented with O-H bonds in ab-plane, O towards FeS layer
    # H atoms point into interlayer (away from FeS), consistent with Dzade 2016
    oh_bond = 0.96   # A
    half_angle_rad = np.radians(104.5 / 2.0)  # 52.25 deg
    dx = oh_bond * np.cos(half_angle_rad)   # 0.586 A
    dy = oh_bond * np.sin(half_angle_rad)   # 0.760 A

    delta_H1 = np.array([dx,  dy, 0.0])   # (+0.586, +0.760, 0)
    delta_H2 = np.array([dx, -dy, 0.0])   # (+0.586, -0.760, 0)

    # Append 4 H2O to supercell
    mack_wet = mack_sc.copy()
    for o_pos in o_positions:
        mack_wet.append(Atom('O', position=o_pos))
        mack_wet.append(Atom('H', position=o_pos + delta_H1))
        mack_wet.append(Atom('H', position=o_pos + delta_H2))

    # Verify
    syms_wet = mack_wet.get_chemical_symbols()
    assert syms_wet.count('Fe') == 8
    assert syms_wet.count('S') == 8
    assert syms_wet.count('O') == 4
    assert syms_wet.count('H') == 8
    assert len(mack_wet) == 28, f"Expected 28, got {len(mack_wet)}"
    print(f"  + 4 H2O -> {mack_wet.get_chemical_formula()}, {len(mack_wet)} atoms")

    # Check min distance H2O from FeS
    D = mack_wet.get_all_distances(mic=True)
    np.fill_diagonal(D, np.inf)
    min_d = float(np.min(D))
    print(f"  Min distance in wet structure: {min_d:.3f} A")
    if min_d < 1.5:
        print(f"  WARNING: min_distance {min_d:.3f} A < 1.5 A -- check water positions!")

    return mack_wet, o_positions


def make_endpoints(relaxed_water, o_positions_initial):
    """
    Build endA and endB from relaxed water structure.
    endA: H+ added to O_1 (H3O+ on H2O_1) = index 16 (first O in mack_wet)
    endB: H+ added to O_3 (H3O+ on H2O_3) = index 22 (third O)

    O indices in 28-atom structure:
      O_1: index 16 (Fe=0-7, S=8-15, O_1=16, H_1a=17, H_1b=18,
                      O_2=19, H_2a=20, H_2b=21,
                      O_3=22, H_3a=23, H_3b=24,
                      O_4=25, H_4a=26, H_4b=27)

    H+ position: O_pos + (0, 0, 0.96) A (upward, into interlayer).
    """
    syms = relaxed_water.get_chemical_symbols()
    # Find O indices in relaxed structure
    o_indices = [i for i, s in enumerate(syms) if s == 'O']
    assert len(o_indices) == 4, f"Expected 4 O atoms, found {len(o_indices)}"
    print(f"  O indices: {o_indices}")
    o1_pos = relaxed_water.positions[o_indices[0]].copy()  # H2O_1
    o3_pos = relaxed_water.positions[o_indices[2]].copy()  # H2O_3

    # endA: H+ on O_1
    endA = relaxed_water.copy()
    h_pos_a = o1_pos + np.array([0.0, 0.0, 0.96])
    endA.append(Atom('H', position=h_pos_a))

    # endB: H+ on O_3
    endB = relaxed_water.copy()
    h_pos_b = o3_pos + np.array([0.0, 0.0, 0.96])
    endB.append(Atom('H', position=h_pos_b))

    for lbl, ep, hp in [("endA", endA, h_pos_a), ("endB", endB, h_pos_b)]:
        assert len(ep) == 29, f"Expected 29 atoms in {lbl}, got {len(ep)}"
        h_idx = len(ep) - 1
        md = min(ep.get_distance(h_idx, j, mic=True) for j in range(len(ep) - 1))
        print(f"  {lbl}: {len(ep)} atoms, H+ at ({hp[0]:.2f},{hp[1]:.2f},{hp[2]:.2f}), "
              f"min_H_dist={md:.3f} A")

    return endA, endB, o_indices


def relax_water(mack_wet, resume):
    """Step 2: relax water positions with FeS frozen."""
    key = "relax_water"
    xyz_path = WORK_DIR / "relaxed_water.xyz"

    if key in resume and resume[key].get("converged"):
        print(f"  Water relaxation: resume E={resume[key]['energy']:.4f} eV")
        if xyz_path.exists():
            return read(str(xyz_path)), resume[key]
        print("  WARNING: resume.json says converged but xyz not found, re-running")

    print(f"  Relaxing water (fmax={FMAX_WATER}, max {MAX_STEPS_WATER} steps)...")
    atoms = mack_wet.copy()
    # Fix FeS: indices 0-15
    fes_indices = list(range(16))
    atoms.set_constraint(FixAtoms(indices=fes_indices))
    atoms.calc = make_calc("relax_water")

    t0 = time.time()
    with FIRE(atoms, logfile=str(WORK_DIR / "relax_water.log")) as opt:
        opt.run(fmax=FMAX_WATER, steps=MAX_STEPS_WATER)

    e = float(atoms.get_potential_energy())
    fm = float(np.max(np.abs(atoms.get_forces())))
    dt = time.time() - t0
    conv = fm < FMAX_WATER

    write(str(xyz_path), atoms)
    result = {
        "energy": e, "fmax": fm, "steps": int(opt.nsteps),
        "converged": conv, "time_s": dt,
    }
    resume[key] = result
    save_resume(resume)
    print(f"  Water relax: E={e:.4f} eV, fmax={fm:.4f}, steps={opt.nsteps}, "
          f"conv={conv}, {dt:.0f}s")
    return atoms, result


def relax_endpoint(atoms, label, resume):
    """Relax endpoint (H3O+ + 3 H2O) with FeS frozen."""
    key = f"relax_{label}"
    xyz_path = WORK_DIR / f"relaxed_{label}.xyz"

    if key in resume and resume[key].get("converged"):
        print(f"  {label}: resume E={resume[key]['energy']:.4f} eV")
        if xyz_path.exists():
            ep = read(str(xyz_path))
            ep.set_constraint(FixAtoms(indices=list(range(16))))
            ep.calc = make_calc(label, restart=True)
            return ep, resume[key]
        print(f"  WARNING: resume says converged but {xyz_path} missing, re-running")

    print(f"  Relaxing {label} (fmax={FMAX_EP}, max {MAX_STEPS_EP} steps)...")
    ep = atoms.copy()
    ep.set_constraint(FixAtoms(indices=list(range(16))))
    ep.calc = make_calc(label)

    t0 = time.time()
    with FIRE(ep, logfile=str(WORK_DIR / f"{label}.log")) as opt:
        opt.run(fmax=FMAX_EP, steps=MAX_STEPS_EP)

    e = float(ep.get_potential_energy())
    fm = float(np.max(np.abs(ep.get_forces())))
    dt = time.time() - t0
    conv = fm < FMAX_EP

    write(str(xyz_path), ep)
    result = {
        "energy": e, "fmax": fm, "steps": int(opt.nsteps),
        "converged": conv, "time_s": dt,
    }
    resume[key] = result
    save_resume(resume)
    print(f"  {label}: E={e:.4f} eV, fmax={fm:.4f}, steps={opt.nsteps}, "
          f"conv={conv}, {dt:.0f}s")
    return ep, result


def run_neb(ea, eb, resume):
    """Two-phase NEB: Phase 1 no-climb (fmax<0.3), Phase 2 CI-NEB (fmax<0.05)."""

    def setup_images(ea, eb):
        images = [ea.copy()]
        for i in range(N_IMAGES):
            img = ea.copy()
            img.calc = make_calc(f"neb_{i:02d}")
            images.append(img)
        images.append(eb.copy())
        images[0].calc = make_calc("neb_endA")
        images[-1].calc = make_calc("neb_endB")
        for img in images:
            img.set_constraint(FixAtoms(indices=list(range(16))))
        return images

    def print_h_positions(images):
        """Print H+ positions (last atom) for each image."""
        for i, img in enumerate(images):
            p = img.positions[-1]
            print(f"    img {i}: H+ ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    def make_step_logger(neb_obj, opt_obj, step_log, phase_name):
        def log_step():
            try:
                energies = [float(img.get_potential_energy())
                            for img in neb_obj.images[1:-1]]
                e0 = float(neb_obj.images[0].get_potential_energy())
                rel = [e - e0 for e in energies]
                barrier = max(rel) if rel else 0.0
                f_arr = neb_obj.get_forces()
                fmax = float(np.max(np.abs(f_arr))) if f_arr is not None else float('nan')
                msg = (f"  [{phase_name}] step {opt_obj.nsteps:4d}: "
                       f"barrier_est={barrier:.4f} eV, fmax={fmax:.4f} eV/A")
                with open(step_log, 'a') as f:
                    f.write(msg + '\n')
                print(msg)
                sys.stdout.flush()
            except Exception as exc:
                print(f"  [log_step warning] {exc}")
        return log_step

    # --- Phase 1: no climbing ---
    key_p1 = "neb_phase1"
    images_p1_path = WORK_DIR / "neb_images_p1.traj"

    if key_p1 in resume and resume[key_p1].get("done"):
        print("  Phase 1: already done, loading images...")
        if images_p1_path.exists():
            all_imgs = read(str(images_p1_path), index=':')
            ea_p1 = all_imgs[0]
            ea_p1.calc = make_calc("neb_endA", restart=True)
            eb_p1 = all_imgs[-1]
            eb_p1.calc = make_calc("neb_endB", restart=True)
            images = [ea_p1]
            for ii, img in enumerate(all_imgs[1:-1]):
                img.calc = make_calc(f"neb_{ii:02d}", restart=True)
                images.append(img)
            images.append(eb_p1)
            for img in images:
                img.set_constraint(FixAtoms(indices=list(range(16))))
        else:
            print("  WARNING: Phase 1 traj not found, re-running from endA/endB")
            images = setup_images(ea, eb)
            neb_p1 = NEB(images, climb=False, method='improvedtangent',
                         allow_shared_calculator=False, k=K_SPRING)
            neb_p1.interpolate('idpp')
            resume.pop(key_p1, None)
    else:
        images = setup_images(ea, eb)
        neb_p1 = NEB(images, climb=False, method='improvedtangent',
                     allow_shared_calculator=False, k=K_SPRING)
        neb_p1.interpolate('idpp')

        print("  H+ positions after IDPP interpolation:")
        print_h_positions(images)

        step_log = WORK_DIR / "neb_step.log"
        print(f"\n  --- Phase 1: climb=False, fmax < {FMAX_NEB_P1} eV/A, "
              f"max {MAX_STEPS_NEB_P1} steps ---")
        t0 = time.time()

        # log_fn created inside context so opt_p1 reference is available
        with FIRE(neb_p1, logfile=str(WORK_DIR / "neb_p1.log")) as opt_p1:
            log_fn = make_step_logger(neb_p1, opt_p1, step_log, "P1")
            for _ in range(MAX_STEPS_NEB_P1):
                opt_p1.run(fmax=FMAX_NEB_P1, steps=1)
                log_fn()
                if opt_p1.converged():
                    break

        dt_p1 = time.time() - t0
        # Save phase 1 images
        write(str(images_p1_path), images)
        resume[key_p1] = {"done": True, "steps": int(opt_p1.nsteps), "time_s": dt_p1}
        save_resume(resume)
        print(f"  Phase 1 done: steps={opt_p1.nsteps}, {dt_p1:.0f}s")

    # --- Phase 2: CI-NEB ---
    key_p2 = "neb_phase2"
    if key_p2 in resume and resume[key_p2].get("converged"):
        print("  Phase 2: already converged")
        return resume[key_p2]

    print(f"\n  --- Phase 2: climb=True, fmax < {FMAX_NEB_P2} eV/A, "
          f"max {MAX_STEPS_NEB_P2} steps ---")

    # Re-attach calculators if needed (restart from phase 1)
    for ii, img in enumerate(images[1:-1]):
        if img.calc is None:
            img.calc = make_calc(f"neb_{ii:02d}", restart=True)

    neb_p2 = NEB(images, climb=True, method='improvedtangent',
                 allow_shared_calculator=False, k=K_SPRING)

    step_log = WORK_DIR / "neb_step.log"
    t0 = time.time()

    with FIRE(neb_p2, logfile=str(WORK_DIR / "neb_p2.log")) as opt_p2:
        log_fn = make_step_logger(neb_p2, opt_p2, step_log, "P2")
        for _ in range(MAX_STEPS_NEB_P2):
            opt_p2.run(fmax=FMAX_NEB_P2, steps=1)
            log_fn()
            if opt_p2.converged():
                break

    dt_p2 = time.time() - t0

    # Collect results
    energies_abs = []
    for img in images:
        try:
            energies_abs.append(float(img.get_potential_energy()))
        except Exception:
            energies_abs.append(float('nan'))

    e0 = energies_abs[0]
    rel = [e - e0 for e in energies_abs]
    e_a = float(max(rel))
    e_rxn = float(rel[-1])

    try:
        f_arr = neb_p2.get_forces()
        fmax_final = float(np.max(np.abs(f_arr)))
    except Exception:
        fmax_final = float('nan')

    conv = fmax_final < FMAX_NEB_P2

    # Save final images
    write(str(WORK_DIR / "neb_images_final.traj"), images)
    for i, img in enumerate(images):
        write(str(WORK_DIR / f"final_{i:02d}.xyz"), img)

    result = {
        "E_a_eV": e_a,
        "E_rxn_eV": e_rxn,
        "energies_eV": rel,
        "neb_steps_p2": int(opt_p2.nsteps),
        "fmax_final": fmax_final,
        "converged": conv,
        "time_s": dt_p2,
    }
    resume[key_p2] = result
    save_resume(resume)
    print(f"  Phase 2 done: E_a={e_a:.4f} eV, fmax={fmax_final:.4f}, "
          f"steps={opt_p2.nsteps}, conv={conv}, {dt_p2:.0f}s")
    print(f"  Energies (rel): {[f'{e:.4f}' for e in rel]}")
    return result


def main():
    try:
        t0 = time.time()
        print("=" * 65)
        print("  QE NEB: Grotthuss H+ in mackinawite FeS interlayer water")
        print(f"  P4/nmm, 2x2x1, 29 atoms (16 FeS + 4 H2O + 1 H+)")
        print(f"  ecutwfc={ECUTWFC} Ry, kpts={KPTS}, DFT-D3(BJ), nspin=1")
        print(f"  OMP_NUM_THREADS={OMP}")
        print(f"  Ref: Marx 2013 AIMD ~0 eV, dry paths A=0.44-0.74 B=2.48 C=4.19 eV")
        print("=" * 65)
        sys.stdout.flush()

        for d in [WORK_DIR, SCRATCH_DIR, RESULTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # Check GPU
        try:
            import subprocess
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True,
            )
            print(f"  GPU: {r.stdout.strip()}")
        except Exception:
            print("  GPU: not detected")
        sys.stdout.flush()

        # Check PPs
        missing_pp = []
        for el, pp in PSEUDOPOTENTIALS.items():
            pp_path = Path(PP_DIR) / pp
            if not pp_path.exists():
                alt = list(pp_path.parent.glob(f"{el}.*UPF"))
                if alt:
                    print(f"  WARNING: {pp} not found, available: {[f.name for f in alt]}")
                else:
                    print(f"  ERROR: No PP for {el} in {PP_DIR}!")
                    missing_pp.append(el)
            else:
                print(f"  PP {el}: {pp} OK")
        if missing_pp:
            print(f"  FATAL: missing pseudopotentials for: {missing_pp}")
            print(f"  Download from https://pseudopotentials.quantum-espresso.org/")
            sys.exit(1)
        sys.stdout.flush()

        resume = load_resume()

        # --- Step 1: Build structure ---
        print("\n[1] Build mackinawite 2x2x1 + 4 H2O")
        mack_wet, o_positions = build_mackinawite_with_water()

        # Verify composition
        syms = mack_wet.get_chemical_symbols()
        print(f"  Formula: {mack_wet.get_chemical_formula()}, "
              f"n_atoms={len(mack_wet)}")
        print(f"  Cell lengths: {[f'{x:.4f}' for x in mack_wet.cell.lengths()]} A")

        # Min distance check
        D_all = mack_wet.get_all_distances(mic=True)
        np.fill_diagonal(D_all, np.inf)
        min_d_all = float(np.min(D_all))
        print(f"  Min interatomic distance: {min_d_all:.3f} A")
        if min_d_all < 1.3:
            print(f"  FATAL: min_distance {min_d_all:.3f} A < 1.3 A -- atoms overlapping!")
            sys.exit(1)
        sys.stdout.flush()

        # --- Step 2: Relax water ---
        print("\n[2] Relax water (FeS frozen, fmax=0.03 eV/A)")
        relaxed_water, water_result = relax_water(mack_wet, resume)
        sys.stdout.flush()

        # --- Step 3: Make endpoints ---
        print("\n[3] Build endpoints (H+ added to O_1 and O_3)")
        endA, endB, o_indices = make_endpoints(relaxed_water, o_positions)
        sys.stdout.flush()

        # --- Step 4: Relax endpoints ---
        print("\n[4] Relax endA (H3O+ on H2O_1)")
        endA_r, ra = relax_endpoint(endA, "endA", resume)
        sys.stdout.flush()

        print("\n[5] Relax endB (H3O+ on H2O_3)")
        endB_r, rb = relax_endpoint(endB, "endB", resume)
        sys.stdout.flush()

        # --- Step 5: NEB ---
        print("\n[6] Two-phase NEB")
        print(f"  N_images={N_IMAGES}, k_spring={K_SPRING} eV/A^2")
        print(f"  Phase 1: climb=False, fmax<{FMAX_NEB_P1}, max {MAX_STEPS_NEB_P1} steps")
        print(f"  Phase 2: climb=True,  fmax<{FMAX_NEB_P2}, max {MAX_STEPS_NEB_P2} steps")
        sys.stdout.flush()
        neb_result = run_neb(endA_r, endB_r, resume)
        sys.stdout.flush()

        # --- Results ---
        dt_total = time.time() - t0
        e_a = neb_result["E_a_eV"]
        e_rxn = neb_result["E_rxn_eV"]
        conv = neb_result["converged"]

        # Cross-verify commentary
        xv_marx = 0.0    # AIMD Marx 2013
        xv_dry_A = 0.59  # average of 0.44 (MACE) and 0.738 (GPAW)
        xv_dry_B = 2.479  # QE intra-layer
        xv_dry_C = 4.19   # MACE gap

        final_result = {
            "system": "mackinawite_FeS_grotthuss_wet",
            "method": "DFT_QE_PBE_D3BJ_PAW_GPU",
            "code": "QE 7.5 (OpenACC GPU)",
            "ecutwfc_Ry": float(ECUTWFC),
            "ecutrho_Ry": float(ECUTRHO),
            "kpts": list(KPTS),
            "degauss_Ry": float(DEGAUSS),
            "smearing": "mv",
            "vdw": "DFT-D3(BJ) dftd3_version=4",
            "nspin": 1,
            "spacegroup": "P4/nmm (#129)",
            "a_A": float(A_MACK),
            "c_A": float(C_MACK),
            "supercell": "2x2x1",
            "n_atoms_per_image": 29,
            "formula": "Fe8S8(H2O)4H",
            "n_images_neb": N_IMAGES,
            "k_spring_eV_A2": float(K_SPRING),
            "mechanism": "Grotthuss H+ transfer H3O+(O_1) -> H2O(O_2) -> H3O+(O_3)",
            "path_length_hops": 2,
            "E_a_eV": float(e_a),
            "E_rxn_eV": float(e_rxn),
            "energies_eV": [float(x) for x in neb_result["energies_eV"]],
            "converged": bool(conv),
            "neb_steps_p2": int(neb_result.get("neb_steps_p2", 0)),
            "fmax_final": float(neb_result.get("fmax_final", float('nan'))),
            "time_total_s": float(dt_total),
            "cross_verify": {
                "Marx_2013_AIMD_eV": xv_marx,
                "dry_path_A_MACE_GPAW_avg_eV": float(xv_dry_A),
                "dry_path_B_QE_intra_eV": float(xv_dry_B),
                "dry_path_C_MACE_gap_eV": float(xv_dry_C),
                "note": (
                    "Grotthuss barrier expected ~0-0.3 eV per Marx 2013. "
                    "If E_a > 0.5 eV: water config suboptimal, consider MD thermalization."
                ),
            },
            "structure_refs": [
                "Mackinawite P4/nmm: Lennie 1995, a=3.674 c=5.033 A",
                "Hydrated c=6.5 A: Dzade 2016 (MkA), Swanner 2019",
                "vdW essential: Dzade 2016 (87% E_ads from dispersion)",
                "Grotthuss AIMD: Marx 2013 Nature Comm.",
            ],
            "endA_energy_eV": float(ra.get("energy", float('nan'))),
            "endB_energy_eV": float(rb.get("energy", float('nan'))),
            "water_relax_energy_eV": float(water_result.get("energy", float('nan'))),
            "harvested": "",
            "source": "Vast.ai QE GPU",
        }

        result_file = RESULTS_DIR / "neb_mackinawite_grotthuss_qe_result.json"
        with open(result_file, 'w') as f:
            json.dump(final_result, f, indent=2, cls=NumpyEncoder)

        done_file = RESULTS_DIR / "DONE_neb_mackinawite_grotthuss"
        with open(done_file, 'w') as f:
            f.write(f"E_a = {e_a:.4f} eV\nconverged = {conv}\ntime_s = {dt_total:.0f}\n")

        print(f"\n{'=' * 65}")
        print(f"  RESULT: E_a(Grotthuss wet) = {e_a:.4f} eV")
        print(f"  E_rxn = {e_rxn:.4f} eV  (should be ~0, symmetric path)")
        print(f"  Converged: {conv}")
        print(f"")
        print(f"  Cross-verify:")
        print(f"    Marx 2013 AIMD (wet)         = ~0.0 eV")
        print(f"    Dry path A (MACE/GPAW avg)   = {xv_dry_A:.3f} eV")
        print(f"    Dry path B (QE intra-layer)  = {xv_dry_B:.3f} eV")
        print(f"    Dry path C (MACE VdW gap)    = {xv_dry_C:.3f} eV")
        print(f"")
        if e_a < 0.1:
            print(f"  Interpretation: near-barrierless Grotthuss, consistent with AIMD")
        elif e_a < 0.3:
            print(f"  Interpretation: low-barrier Grotthuss, NEB captures static barrier")
        elif e_a < 0.5:
            print(f"  Interpretation: moderate barrier -- check water orientation")
        else:
            print(f"  Interpretation: E_a > 0.5 eV -- likely suboptimal water config")
            print(f"  Recommendation: use MD-thermalized water positions as input")
        print(f"")
        print(f"  Saved: {result_file}")
        print(f"  Total time: {dt_total:.0f}s ({dt_total / 3600:.1f}h)")
        print("=" * 65)
        print("DONE")
        sys.stdout.flush()

    except Exception as exc:
        import traceback
        print(f"\nFATAL ERROR: {exc}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == '__main__':
    main()
