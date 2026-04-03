#!/usr/bin/env python3
"""
DFT vacancy formation energies -- ABACUS PW GPU.

Stanford reviewer #3: vacancy formation energies + equilibrium concentrations.

E_vac = E(defect) - E(pristine) + mu_S
mu_S = E(S2_gas) / 2

Minerals:
  Pentlandite (Fe,Ni)9S8  Fm-3m  68 at  kpts 2x2x2
  Mackinawite FeS          P4/nmm 72 at  kpts 2x2x2
  Pyrite FeS2              Pa-3   96 at  kpts 2x2x2

ABACUS PW mode, PBE, ecutwfc=80 Ry, nspin=1 (all paramagnetic/diamagnetic at 25C).
SG15 ONCV PBE-1.2 pseudopotentials.

Cross-verify: GPAW PW350 running on W1 (same minerals, same kpts).
"""

import json
import os
import sys
import time
import traceback
import subprocess
import numpy as np
from pathlib import Path
from ase.spacegroup import crystal
from ase import Atoms
from ase.io import write as ase_write

# -- Configuration ----------------------------------------------------------
RESULTS = Path("/workspace/results")
RESULTS.mkdir(parents=True, exist_ok=True)
RESUME_FILE = RESULTS / "vacancy_abacus_resume.json"
SUMMARY_FILE = RESULTS / "vacancy_abacus_summary.txt"

PP_DIR = os.environ.get("ABACUS_PP_PATH", "/opt/abacus-pp")
ABACUS_BIN = "abacus"  # abacus_2g via symlink

ECUTWFC = 80        # Ry (~1088 eV, well converged for SG15 ONCV)
SMEARING = 0.01     # Ry (~0.136 eV, Gaussian)
SCF_NMAX = 500
SCF_THR = 1e-6      # Ry


# -- Resume -----------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_resume():
    if RESUME_FILE.exists():
        with open(RESUME_FILE) as f:
            return json.load(f)
    return {}


def save_resume(data):
    tmp = RESUME_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    tmp.replace(RESUME_FILE)


# -- Structure builders -----------------------------------------------------
def build_pentlandite_conv():
    """Pentlandite (Fe,Ni)9S8, Fm-3m (225), conventional 68 atoms."""
    a = 10.044
    atoms = crystal(
        symbols=['Fe', 'Fe', 'S', 'S'],
        basis=[(0.5, 0.5, 0.5),
               (0.125, 0.125, 0.125),
               (0.25, 0.25, 0.25),
               (0.25, 0.0, 0.0)],
        spacegroup=225,
        cellpar=[a, a, a, 90, 90, 90],
        primitive_cell=False,
    )
    syms = atoms.get_chemical_symbols()
    fe_indices = [i for i, s in enumerate(syms) if s == 'Fe']
    fe_32f = fe_indices[4:]
    assert len(fe_32f) == 32, f"Expected 32 atoms on 32f, got {len(fe_32f)}"
    for i in fe_32f[16:]:
        syms[i] = 'Ni'
    atoms.set_chemical_symbols(syms)
    return atoms


def build_mackinawite_supercell():
    """Mackinawite FeS, P4/nmm (129), 3x3x2 supercell, 72 atoms."""
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0),
               (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.6735, 3.6735, 5.0328, 90, 90, 90],
    )
    return atoms.repeat((3, 3, 2))


def build_pyrite_supercell():
    """Pyrite FeS2, Pa-3 (205), 2x2x2 supercell, 96 atoms."""
    a = 5.416
    atoms = crystal(
        symbols=['Fe', 'S'],
        basis=[(0.0, 0.0, 0.0),
               (0.38488, 0.38488, 0.38488)],
        spacegroup=205,
        cellpar=[a, a, a, 90, 90, 90],
    )
    return atoms.repeat((2, 2, 2))


def build_s2_molecule():
    """S2 molecule in vacuum for mu_S reference. Triplet (S=1, nspin=2)."""
    d = 1.89
    box = 12.0
    atoms = Atoms('S2',
                  positions=[[box/2, box/2, box/2 - d/2],
                             [box/2, box/2, box/2 + d/2]],
                  cell=[box, box, box],
                  pbc=True)
    return atoms


# -- Helpers ----------------------------------------------------------------
def make_vacancy(atoms, element='S'):
    """Remove last atom of given element."""
    indices = [i for i, s in enumerate(atoms.get_chemical_symbols())
               if s == element]
    if not indices:
        raise ValueError(f"No {element} atoms found!")
    vac = atoms.copy()
    del vac[indices[-1]]
    return vac, indices[-1]


def validate_structure(atoms, name):
    """Print and verify structure."""
    formula = atoms.get_chemical_formula()
    n = len(atoms)
    cell = atoms.cell.cellpar()
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, 999)
    min_dist = dists.min()
    syms = atoms.get_chemical_symbols()
    comp = {s: syms.count(s) for s in sorted(set(syms))}

    print(f"  {name}: {formula}, {n} atoms")
    print(f"  Cell: {cell[0]:.3f} x {cell[1]:.3f} x {cell[2]:.3f} A")
    print(f"  Composition: {comp}")
    print(f"  Min distance: {min_dist:.3f} A")
    sys.stdout.flush()

    if min_dist < 1.2:
        print(f"  *** ERROR: min_dist={min_dist:.3f} < 1.2 A ***")
        sys.exit(1)
    return True


def write_abacus_input(workdir, atoms, kpts, label="scf",
                       nspin=1, basis_type="pw", magmoms=None):
    """Write ABACUS INPUT, STRU, KPT files for a single-point calculation.

    Args:
        nspin: 1 (restricted) or 2 (spin-polarized, requires basis_type=lcao)
        basis_type: "pw" (GPU-accelerated) or "lcao" (needed for nspin=2)
        magmoms: dict {element: mag} for nspin=2, e.g. {'S': 1.0}
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    syms = atoms.get_chemical_symbols()
    species = list(dict.fromkeys(syms))

    pp_map = {
        'Fe': 'Fe_ONCV_PBE-1.2.upf',
        'S':  'S_ONCV_PBE-1.2.upf',
        'Ni': 'Ni_ONCV_PBE-1.2.upf',
        'H':  'H_ONCV_PBE-1.2.upf',
        'Co': 'Co_ONCV_PBE-1.2.upf',
        'Cu': 'Cu_ONCV_PBE-1.2.upf',
    }
    orb_map = {
        'Fe': 'Fe_gga_8au_100Ry_4s2p2d1f.orb',
        'S':  'S_gga_7au_100Ry_2s2p1d.orb',
        'Ni': 'Ni_gga_8au_100Ry_4s2p2d1f.orb',
        'H':  'H_gga_6au_100Ry_2s1p.orb',
    }
    masses = {'Fe': 55.845, 'S': 32.06, 'Ni': 58.693, 'H': 1.008,
              'Co': 58.933, 'Cu': 63.546}
    if magmoms is None:
        magmoms = {}

    # INPUT
    input_text = f"""INPUT_PARAMETERS
calculation      scf
basis_type       {basis_type}
ecutwfc          {ECUTWFC}
dft_functional   pbe
nspin            {nspin}
scf_nmax         {SCF_NMAX}
scf_thr          {SCF_THR}
smearing_method  gauss
smearing_sigma   {SMEARING}
mixing_type      broyden
mixing_beta      0.3
mixing_ndim      8
symmetry         0
cal_force        0
cal_stress       0
out_level        ie
"""
    (workdir / "INPUT").write_text(input_text)

    # STRU
    cell = atoms.cell
    positions = atoms.get_scaled_positions()

    stru_lines = ["ATOMIC_SPECIES"]
    for sp in species:
        stru_lines.append(f"{sp}  {masses[sp]}  {pp_map[sp]}")

    if basis_type == "lcao":
        orb_dir = os.environ.get("ABACUS_ORB_PATH", "/opt/abacus-orb")
        stru_lines.append("")
        stru_lines.append("NUMERICAL_ORBITAL")
        for sp in species:
            stru_lines.append(orb_map[sp])

    stru_lines.append("")
    stru_lines.append("LATTICE_CONSTANT")
    stru_lines.append("1.889726125")

    stru_lines.append("")
    stru_lines.append("LATTICE_VECTORS")
    for i in range(3):
        stru_lines.append(f"  {cell[i][0]:.10f}  {cell[i][1]:.10f}  {cell[i][2]:.10f}")

    stru_lines.append("")
    stru_lines.append("ATOMIC_POSITIONS")
    stru_lines.append("Direct")

    for sp in species:
        sp_indices = [i for i, s in enumerate(syms) if s == sp]
        stru_lines.append("")
        stru_lines.append(f"{sp}")
        default_mag = magmoms.get(sp, 0.0)
        stru_lines.append(f"{default_mag}")
        stru_lines.append(f"{len(sp_indices)}")
        for idx in sp_indices:
            x, y, z = positions[idx]
            stru_lines.append(f"  {x:.10f}  {y:.10f}  {z:.10f}  0 0 0")

    (workdir / "STRU").write_text("\n".join(stru_lines) + "\n")

    # Symlink orbital files for LCAO mode
    if basis_type == "lcao":
        orb_dir = Path(os.environ.get("ABACUS_ORB_PATH", "/opt/abacus-orb"))
        for sp in species:
            orb_file = orb_dir / orb_map[sp]
            dst = workdir / orb_map[sp]
            if orb_file.exists() and not dst.exists():
                os.symlink(orb_file, dst)

    # KPT
    kpt_text = f"""K_POINTS
0
Gamma
{kpts[0]} {kpts[1]} {kpts[2]} 0 0 0
"""
    (workdir / "KPT").write_text(kpt_text)


def run_abacus(workdir):
    """Run ABACUS in workdir, return total energy in eV."""
    workdir = Path(workdir)

    # Clean old output
    out_dir = workdir / "OUT.ABACUS"
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    # Symlink PP files
    for upf in Path(PP_DIR).glob("*.upf"):
        dst = workdir / upf.name
        if not dst.exists():
            os.symlink(upf, dst)

    t0 = time.time()
    result = subprocess.run(
        [ABACUS_BIN],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=7200,  # 2h max per calc
    )
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"  ABACUS FAILED (exit {result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"ABACUS failed in {workdir}")

    # Parse energy from running_scf.log
    scf_log = out_dir / "running_scf.log"
    if not scf_log.exists():
        raise RuntimeError(f"No running_scf.log in {out_dir}")

    energy_eV = None
    with open(scf_log) as f:
        for line in f:
            if "!FINAL_ETOT_IS" in line:
                # Format: !FINAL_ETOT_IS -1234.567890 eV
                energy_eV = float(line.split()[1])

    if energy_eV is None:
        # Try alternative format
        with open(scf_log) as f:
            for line in f:
                if "ETOT =" in line or "final etot is" in line.lower():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p in ("=", "is") and i + 1 < len(parts):
                            try:
                                energy_eV = float(parts[i + 1])
                            except ValueError:
                                continue

    if energy_eV is None:
        print(f"  WARNING: Could not parse energy from {scf_log}")
        print(f"  Last 10 lines:")
        with open(scf_log) as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"    {line.rstrip()}")
        raise RuntimeError(f"No energy in {scf_log}")

    return energy_eV, dt


def calc_single_point(atoms, label, kpts, nspin=1, basis_type="pw",
                      magmoms=None):
    """Set up ABACUS input, run, return energy in eV."""
    workdir = RESULTS / f"abacus_{label}"
    write_abacus_input(workdir, atoms, kpts, label,
                       nspin=nspin, basis_type=basis_type, magmoms=magmoms)

    mode_str = f"ABACUS {basis_type.upper()}"
    if nspin == 2:
        mode_str += " nspin=2"
    print(f"  Running {mode_str}: {label} ({len(atoms)} atoms, kpts={kpts})...")
    sys.stdout.flush()

    e, dt = run_abacus(workdir)
    print(f"  {label}: E = {e:.6f} eV ({dt:.0f} s, {len(atoms)} atoms)")
    sys.stdout.flush()
    return e, dt


# -- Main -------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Vacancy Formation Energies -- ABACUS PW GPU")
    print(f"  ecutwfc={ECUTWFC} Ry, nspin=1, PBE, SG15 ONCV")
    print(f"  ABACUS binary: {ABACUS_BIN}")
    print("=" * 70)
    sys.stdout.flush()

    # Check GPU
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        print(f"  GPU: {r.stdout.strip()}")
    except Exception:
        print("  GPU: not detected (CPU fallback)")
    sys.stdout.flush()

    results = load_resume()
    total_t0 = time.time()

    # -- S2 reference (LCAO nspin=2 for triplet ground state) ----------------
    # S2 is a triplet (S=1, two unpaired electrons). ABACUS PW+nspin=2 is
    # not supported in v3.9, so we use LCAO mode for this one calculation.
    # All mineral calcs remain PW (nspin=1, paramagnetic/diamagnetic).
    if "s2_eV" not in results:
        print("\n--- S2 molecule (mu_S reference, LCAO nspin=2 triplet) ---")
        s2 = build_s2_molecule()
        validate_structure(s2, "S2")
        e, dt = calc_single_point(s2, "s2_ref", kpts=(1, 1, 1),
                                  nspin=2, basis_type="lcao",
                                  magmoms={'S': 1.0})
        results["s2_eV"] = float(e)
        results["mu_S_eV"] = float(e / 2)
        results["s2_time_s"] = float(dt)
        results["s2_note"] = "LCAO nspin=2 triplet (PW+nspin=2 unsupported in v3.9)"
        save_resume(results)
    else:
        print(f"\n--- S2: resume, E = {results['s2_eV']:.6f} eV ---")

    mu_S = results["mu_S_eV"]
    print(f"  mu_S = {mu_S:.4f} eV (= E(S2)/2)")

    # -- Minerals -----------------------------------------------------------
    minerals = [
        ("pent", "PENTLANDITE (Fe,Ni)9S8, 68 at, kpts 2x2x2",
         build_pentlandite_conv, (2, 2, 2)),
        ("mack", "MACKINAWITE FeS, 72 at, kpts 2x2x2",
         build_mackinawite_supercell, (2, 2, 2)),
        ("pyrite", "PYRITE FeS2, 96 at, kpts 2x2x2",
         build_pyrite_supercell, (2, 2, 2)),
    ]

    for prefix, title, builder, kpts in minerals:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

        key_prist = f"{prefix}_pristine_eV"
        key_vac = f"{prefix}_vacancy_eV"

        if key_prist not in results:
            atoms = builder()
            validate_structure(atoms, f"{prefix} pristine")
            e, dt = calc_single_point(atoms, f"{prefix}_pristine", kpts)
            results[key_prist] = float(e)
            results[f"{prefix}_pristine_time_s"] = float(dt)
            save_resume(results)
        else:
            print(f"  Pristine: resume, E = {results[key_prist]:.6f} eV")

        if key_vac not in results:
            atoms = builder()
            vac_atoms, idx = make_vacancy(atoms, 'S')
            validate_structure(vac_atoms, f"{prefix} vacancy")
            e, dt = calc_single_point(vac_atoms, f"{prefix}_vacancy", kpts)
            results[key_vac] = float(e)
            results[f"{prefix}_vacancy_time_s"] = float(dt)
            save_resume(results)
        else:
            print(f"  Vacancy: resume, E = {results[key_vac]:.6f} eV")

        e_vac = results[key_vac] - results[key_prist] + mu_S
        results[f"E_vac_{prefix}_eV"] = float(e_vac)
        print(f"  >>> E_vac({prefix}) = {e_vac:.3f} eV")
        save_resume(results)

    # -- Summary ------------------------------------------------------------
    total_dt = time.time() - total_t0
    kB = 8.617e-5  # eV/K
    temps = [(298, "25C"), (373, "100C"), (473, "200C")]

    lines = []
    lines.append("=" * 70)
    lines.append("  VACANCY FORMATION ENERGIES -- ABACUS PW GPU")
    lines.append("=" * 70)
    lines.append(f"  mu_S = {mu_S:.4f} eV (S-rich: E(S2)/2)")
    lines.append("")

    for prefix, name in [("pent", "Pentlandite"), ("mack", "Mackinawite"),
                         ("pyrite", "Pyrite")]:
        ev = results.get(f"E_vac_{prefix}_eV", float('nan'))
        lines.append(f"  {name:14s}: E_vac = {ev:.3f} eV")

    lines.append("")
    lines.append("  Equilibrium vacancy fractions x_v = exp(-E_vac/kT):")
    for T, label in temps:
        parts = [f"T={T}K"]
        for prefix in ["pent", "mack", "pyrite"]:
            ev = results.get(f"E_vac_{prefix}_eV", 99)
            xv = float(np.exp(-max(ev, 0) / (kB * T))) if ev > 0 else 1.0
            results[f"xv_{prefix}_{label}"] = xv
            parts.append(f"{prefix}={xv:.2e}")
        lines.append("  " + ", ".join(parts))

    lines.append(f"\n  Total time: {total_dt:.0f} s ({total_dt/3600:.1f} h)")
    lines.append("  DONE")

    text = "\n".join(lines)
    print(text)

    with open(SUMMARY_FILE, 'w') as f:
        f.write(text + "\n")
    save_resume(results)

    done = RESULTS / "DONE_vacancy_abacus"
    done.write_text(
        f"completed at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
    )
    print(f"\n  JSON: {RESUME_FILE}")
    print(f"  Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n*** FATAL: {e} ***")
        traceback.print_exc()
        sys.exit(1)
