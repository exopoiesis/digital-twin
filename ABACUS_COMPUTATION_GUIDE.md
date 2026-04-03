# ABACUS DFT Computation: Lessons & Benchmarks

> A practical guide to running ABACUS DFT calculations on cloud GPU/CPU infrastructure.
> Accumulated over ~41 AI-assisted sessions of the [Third Matter](https://exopoiesis.space) project
> (sessions 51–76, through April 2026).
> Target systems: iron-sulfide minerals (8–136 atoms), LCAO and PW modes.
>
> This document is battle-tested: every bug, benchmark, and recommendation
> comes from actual production runs on Vast.ai and Hetzner AX102.
> Code version: **ABACUS v3.9.0.26** (Docker image `exopoiesis/infra-abacus-gpu`).

---

## 1. Two Modes: LCAO vs PW

### LCAO — for small cells, spin-polarized, NEB on AX102

```python
from abacuslite import Abacus, AbacusProfile

profile = AbacusProfile(
    command="mpirun --allow-run-as-root --bind-to none -np 1 /opt/abacus-develop-3.9.0.26/build/abacus_2p",
    omp_num_threads=OMP,   # physical cores, NOT hyperthreads
    pseudo_dir=PP_DIR,
    orbital_dir=ORB_DIR,
)

calc = Abacus(
    profile=profile,
    directory=str(WORK_DIR / label),
    pseudopotentials={
        "Fe": "Fe_ONCV_PBE-1.2.upf",
        "S":  "S_ONCV_PBE-1.2.upf",
        "Ni": "Ni_ONCV_PBE-1.2.upf",
        "H":  "H_ONCV_PBE-1.2.upf",
    },
    basissets={
        "Fe": "Fe_gga_8au_100Ry_4s2p2d1f.orb",
        "S":  "S_gga_7au_100Ry_2s2p1d.orb",
        "Ni": "Ni_gga_8au_100Ry_4s2p2d1f.orb",
        "H":  "H_gga_6au_100Ry_2s1p.orb",
    },
    kpts={'nk': [2, 2, 2], 'kshift': [0, 0, 0], 'gamma-centered': True, 'mode': 'mp-sampling'},
    inp={
        'basis_type': 'lcao',
        'calculation': 'scf',
        'nspin': 1,            # or 2 -- LCAO supports both
        'ecutwfc': 80,         # auxiliary PW cutoff for LCAO charge density
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
```

**When to use LCAO:**
- Small cells (< 30 atoms): pyrite 1x1x1 (12 at.), mackinawite primitive (8 at.)
- Spin-polarized calculations (`nspin=2`): S2 molecule reference, AFM minerals
- CPU-only nodes (Hetzner AX102): no CUDA requirement
- NEB where multiple single-points run in sequence (lower memory than PW)

**Key difference from PW:** `basissets` dict is required; `orbital_dir` path must be set.

### PW — for large cells, GPU acceleration, production

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"   # GPU mode: 1 OMP thread per GPU
os.environ["MKL_NUM_THREADS"] = "1"

profile = AbacusProfile(
    command="abacus",   # symlink to abacus_2g (GPU binary) in infra-abacus-gpu image
    pseudo_dir=PP_DIR,
    # No orbital_dir needed for PW
)

calc = Abacus(
    profile=profile,
    directory=str(WORK_DIR / label),
    pseudopotentials={
        "Fe": "Fe_ONCV_PBE-1.2.upf",
        "S":  "S_ONCV_PBE-1.2.upf",
        "Ni": "Ni_ONCV_PBE-1.2.upf",
        "H":  "H_ONCV_PBE-1.2.upf",
    },
    # No basissets for PW
    kpts={'nk': [2, 2, 2], 'kshift': [0, 0, 0], 'gamma-centered': True, 'mode': 'mp-sampling'},
    inp={
        'basis_type': 'pw',
        'calculation': 'scf',
        'nspin': 1,            # CRITICAL: nspin=2 + PW = UNSUPPORTED in v3.9
        'ecutwfc': 60,         # Ry (~ 816 eV). SG15 ONCV is norm-conserving: 60-80 Ry sufficient
        'smearing_method': 'gaussian',
        'smearing_sigma': 0.05,
        'scf_thr': 1e-6,
        'scf_nmax': 500,
        'mixing_type': 'broyden',
        'mixing_beta': 0.2,
        'mixing_ndim': 12,
        'cal_force': 1,
        'cal_stress': 0,
        'symmetry': 0,
    },
)
```

**When to use PW:**
- Large cells (30–100+ atoms): pentlandite conv. 68 at., mackinawite 3x3x2 72 at., pyrite 2x2x2 96 at.
- GPU instances: RTX 3060+, RTX 4070 Ti, RTX 5xxx
- Vacancy formation, NEB endpoints, datagen
- Non-spin-polarized systems only (`nspin=1`)

### Key differences

| | LCAO | PW |
|---|---|---|
| Binary | `abacus_2p` (CPU MPI) | `abacus_2g` (GPU CUDA) |
| `nspin=2` | Supported | **UNSUPPORTED in v3.9** |
| `basissets` | Required (`.orb` files) | Not used |
| `ecutwfc` | Auxiliary cutoff for charge density | Wave-function cutoff |
| GPU | No | Yes |
| Speed (small cell) | Fast | Slow (GPU overhead) |
| Speed (large cell) | Slow (matrix ops scale O(N^3)) | Fast (GPU FFT) |
| OMP | Physical cores (no HT) | 1 (GPU manages parallelism) |
| Typical time | Minutes (12 at.) | Hours (68 at.) |

---

## 2. Critical Bugs (known from production)

### CRITICAL: nspin=2 + basis_type=pw = UNSUPPORTED

ABACUS v3.9 does not support spin-polarized PW calculations. Attempting `nspin=2` with
`basis_type='pw'` produces wrong magnetic moments or crashes.

**Affected use case:** S2 molecule reference for vacancy calculations (triplet ground state, S=1).

**Fix:** Use LCAO for any `nspin=2` calculation. All mineral PW calculations can use `nspin=1`
for paramagnetic/diamagnetic systems at room temperature.

```python
# WRONG: will produce incorrect mu_S
inp={'basis_type': 'pw', 'nspin': 2, ...}  # DO NOT USE

# CORRECT: LCAO for triplet S2
inp={'basis_type': 'lcao', 'nspin': 2, 'ecutwfc': 80, ...}
```

### CRITICAL: sigma=0.01 Ry causes SCF non-convergence

Default-looking `smearing_sigma=0.01` causes charge sloshing, especially for large metallic cells.
Images in NEB (intermediate geometries) are harder to converge than endpoints.

**Symptom:** NEB images return garbage energies (±30 eV oscillation), `scf_nmax` hit on every image.

**Fix:** `smearing_sigma=0.05` + `mixing_beta=0.2` + `mixing_ndim=12` + `scf_nmax=500`.

**Lesson (session 55):** This burned 15 hours of compute on AX102 (15 NEB steps x ~1h each with
oscillating energies). Always test SCF parameters on a single endpoint before deploying NEB.

### CRITICAL: symmetry=1 (default) destroys AFM

ABACUS default `symmetry=1` enforces crystallographic symmetry, which resets AFM magnetic moment
assignments to the symmetry-equivalent (= ferromagnetic or nonmagnetic) configuration.

**Fix:** Always set `symmetry: 0` for spin-polarized or AFM calculations.

For AFM convergence also add `mixing_dmr: 1` (density matrix mixing for DFT+U).

### CRITICAL: Zombie orted processes (bare abacus_2p binary)

`abacus_2p` is an MPI binary. When called directly (without mpirun) as a bare binary inside Docker,
OpenMPI singleton initialization spawns an `orted` daemon process. With `PID 1 = sleep infinity`
(Docker default), each orted becomes an orphaned zombie after the ABACUS calculation completes.
In NEB: each FIRE step = +1 zombie. Over 300 steps = 300 zombies = memory exhaustion.

**Fix:**
```bash
# Always wrap in mpirun, even for single process:
command="mpirun --allow-run-as-root -np 1 abacus_2p"

# Or use tini as container entrypoint:
docker run --init ...
```

### CRITICAL: --bind-to none mandatory with np=1 + OMP

Without `--bind-to none`, OpenMPI binds the process to 1-2 cores. All OMP threads fight for
those 2 cores → 192% CPU instead of 2400% on a 24-core machine.

**Session 54 postmortem:** W3 (Threadripper PRO 9965WX, 48 cores) ran at ~2 cores effective
for 6.5 hours. Expected runtime: ~40 minutes.

**Fix:**
```bash
mpirun --allow-run-as-root --bind-to none -np 1 abacus_2p
```

### CRITICAL: OMP = physical cores, not hyperthreads

ABACUS LCAO matrix operations are memory-bandwidth bound. Hyperthreading adds cache contention
without bandwidth benefit.

**Fix:** `OMP_NUM_THREADS = num_physical_cores`, not `nproc`. On 24-core/48-thread CPUs: OMP=24.

### CRITICAL: timeout=7200 too short for large PW cells

Each single-point calculation for 68-72 atoms with 2x2x2 kpts on GPU takes 2.4–2.7 hours.
Default `subprocess.run(..., timeout=7200)` kills the process before it converges.

**Fix:** `timeout=28800` (8 hours) for cells > 50 atoms.

### CRITICAL: CI-NEB from step 0 = oscillation

Starting NEB with `climb=True` immediately causes wild oscillation in fmax and barrier estimates.
The climbing image tries to move uphill before the path is reasonably defined.

**Symptom:** fmax oscillates 0.41 → 9.7 → 3.7 eV over 15 steps with zero net progress.

**Fix:** Two-phase NEB protocol:
1. Phase 1: `climb=False`, run until `fmax < 0.3`
2. Phase 2: `climb=True`, run until `fmax < 0.05`

ASE NEB has no `climb_after` parameter — must restart the script between phases.

### CRITICAL: kpts must scale inverse to cell size

When comparing 1x1x1 vs 2x2x2 supercells, kpoints must scale accordingly.

**Confirmed failure:** ABACUS 2x2x2 pyrite with Gamma-only kpts → `E_a = 0.075 eV`.
Same system with 1x1x1 cells (GPAW, QE, ABACUS): `E_a = 0.181–0.190 eV`. Difference: 2.5x.

**Rule:** if 1x1x1 uses `kpts=(2,2,2)`, then 2x2x2 supercell must use at least `kpts=(1,1,1)`
with non-Gamma sampling (ideally `kpts=(2,2,2)` for Gamma-centered to match k-space density).

---

## 3. ASE NEB with ABACUS

### Standard NEB setup

```python
from ase.mep import NEB
from ase.optimize import FIRE
from ase.constraints import FixAtoms

# --- Two-phase NEB (mandatory per lesson s76) ---

# Build images (only H moves, heavy atoms frozen)
images = [endA]
for i in range(N_IMAGES):
    img = endA.copy()
    heavy = [j for j in range(len(img)) if img[j].symbol != 'H']
    img.set_constraint(FixAtoms(indices=heavy))
    img.calc = make_calc(f"image_{i:02d}")
    images.append(img)
images.append(endB)

# IDPP interpolation (mandatory for layered structures)
neb = NEB(images, climb=False, k=0.05, method="improvedtangent")
neb.interpolate("idpp")

# Phase 1: plain NEB
opt = FIRE(neb, logfile="neb_phase1.log", trajectory="neb_phase1.traj")
opt.run(fmax=0.3, steps=200)

# Phase 2: CI-NEB (restart from phase 1 geometry)
neb2 = NEB(images, climb=True, k=0.05, method="improvedtangent")
opt2 = FIRE(neb2, logfile="neb_phase2.log", trajectory="neb_phase2.traj")
opt2.run(fmax=0.05, steps=300)
```

### Why FIRE, not BFGS

FIRE optimizer is more stable for NEB than BFGS. BFGS attempts to approximate the Hessian,
which becomes poorly conditioned for NEB force projections, especially near the saddle point.

### Spring constant choice

`k=0.05 eV/A` (softer than ASE default `k=0.1`). Softer springs allow images to spread
more naturally along the minimum energy path, reducing spurious forces in metallic systems.

### IDPP interpolation is mandatory for layered structures

Linear interpolation places intermediate H atoms in void regions between layers (van der Waals gap),
where they experience near-zero forces → IDPP finds a physically reasonable initial path through
the sulfide framework.

**Verified:** mackinawite GPAW NEB with linear interpolation → H stuck in vdW gap.
IDPP → H follows S-layer correctly.

### Resume pattern (SIGKILL-resistant)

ABACUS NEB has no built-in checkpoint. Implement your own:

```python
def save_images(images, tag="phase1"):
    for k, img in enumerate(images):
        write(str(WORK_DIR / f"neb_{tag}_img{k:02d}.xyz"), img)

def load_images_if_exist(n_images, tag="phase1"):
    files = [WORK_DIR / f"neb_{tag}_img{k:02d}.xyz" for k in range(n_images + 2)]
    if all(f.exists() for f in files):
        return [read(str(f)) for f in files]
    return None

# Attach to FIRE optimizer:
class StepSaver:
    def __init__(self, neb, tag):
        self.neb = neb
        self.tag = tag
    def __call__(self):
        save_images(self.neb.images, tag=self.tag)

opt.attach(StepSaver(neb, "phase1"))
```

Vast.ai sends SIGTERM → SIGKILL after 10 seconds on spot instance preemption.
With per-step image save, maximum loss is 1 NEB step (~2.5 h for large cells).

---

## 4. Benchmarks (Real Numbers)

### Single-point energy (PW GPU)

| System | Atoms | kpts | Time/SP | GPU | Notes |
|--------|:-----:|:----:|:-------:|-----|-------|
| S2 molecule | 2 | (1,1,1) | 62 s | RTX 4070 Ti | LCAO nspin=2 |
| Pyrite 1x1x1 | 12 | (2,2,2) | ~5 min | AX102 (CPU) | LCAO |
| Pentlandite conv. | 68 | (1,1,1) Gamma | ~2.4 h | RTX 4070 Ti | PW GPU |
| Pentlandite conv. | 68 | (2,2,2) | >8 h | RTX 4070 Ti | PW GPU, timed out |
| Mackinawite 3x3x2 | 72 | (2,2,2) | ~1.9 h | RTX 4070 Ti | PW GPU |
| Pyrite 2x2x2 | 96 | (1,1,1) Gamma | ~3-4 h | RTX 4070 Ti | PW GPU, Gamma-only |

**Key benchmark:** Pentlandite 68 at., 2x2x2 kpts exceeds 8h timeout on RTX 4070 Ti.
Use Gamma-only for initial testing; 2x2x2 kpts requires stronger GPU or LCAO mode.

### Vacancy formation (session 76, RTX 4070 Ti, AMD Ryzen 9 3900X)

| Calculation | System | Time | Energy |
|-------------|--------|:----:|--------|
| S2 ref (LCAO nspin=2) | S2 in 12x12x12 box | 62 s | — |
| Pristine | Pentlandite 68 at. | 2.4 h | — |
| Vacancy | Pentlandite 67 at. | 2.7 h | — |
| **E_vac(pentlandite)** | | | **4.444 eV** |
| Pristine | Mackinawite 72 at. | 1.9 h | — |
| Vacancy | Mackinawite 71 at. | ~2 h | — |
| **E_vac(mackinawite)** | | | **5.668 eV** |

### H diffusion barriers (NEB cross-verify, all PBE, vacancy mechanism)

| Mineral | Method | Cell | E_a (eV) | Notes |
|---------|--------|------|:--------:|-------|
| Pyrite | GPAW PW | 1x1x1 | 0.181 | PW350, 2x2x2 kpts |
| Pyrite | QE PW | 1x1x1 | 0.190 | PBE, 2x2x2 kpts, ±5% vs GPAW |
| Pyrite | ABACUS LCAO | 1x1x1 | 0.187 | DZP, 2x2x2 kpts |
| Pyrite | ABACUS PW | 2x2x2 | 0.075 | **Gamma-only artifact** — do not use |
| Pentlandite | GPAW PW | primitive 17 at. | 1.115 | PW350 |
| Pentlandite | ABACUS PW | primitive 17 at. | 0.900 | GPU, Gamma kpts |
| Pentlandite | MACE FF | primitive 17 at. | 0.96 | force-field |
| Pentlandite | MACE FF | conv. 68 at. | 1.43 | **+49% vs primitive** |
| Mackinawite (intra-layer) | GPAW PW | — | 0.738 | S→S hop within FeS sheet |
| Mackinawite (cross-layer) | QE PW | 15 at. | 2.479 | S→S hop through VdW gap, not Grotthuss |
| Mackinawite | MACE FF | — | 0.44 | inter-layer |

**Cross-verify status (pyrite):** GPAW 0.181 / QE 0.190 / ABACUS LCAO 0.187 — all within 5%.
Benchmark is complete. ABACUS LCAO 1x1x1 is validated.

---

## 5. AFM Minerals: Special Considerations

For minerals with Neel or Curie temperature above room temperature (troilite T_N=315°C,
chalcopyrite, greigite T_C=330°C): DFT+U is mandatory and AFM configuration requires care.

### Required parameters for AFM (ABACUS)

```python
inp={
    'nspin': 2,
    'basis_type': 'lcao',   # nspin=2 requires LCAO in v3.9
    'symmetry': 0,           # CRITICAL: symmetry=1 destroys AFM
    'nupdown': 0,            # constrain total magnetization to 0 (AFM)
    'mixing_beta_mag': 0.2,  # separate mixing for magnetization density
    'mixing_dmr': 1,         # density matrix mixing for DFT+U convergence
    'dft_plus_u': True,
    # U values (Ueff):
    # greigite: U=1.0 eV (Devey 2009)
    # troilite, chalcopyrite, others: U=2.0 eV (Peng 2019)
}
```

### AFM collapse warning

If `|Total magnetization| > 1 Bohr mag` after 5-10 minutes of SCF → AFM has collapsed to FM.
Forces from FM state are wrong → FIRE diverges → hours lost.

**Check 5-10 minutes after start:**
```bash
grep "Total magnetism" /workspace/results/*/running_scf.log
```

**Fix:** `nupdown=0` + `mixing_beta_mag=0.2` (see `knowledge/AFM_VACANCY_DFT_LESSONS.md`).

---

## 6. Docker Image and Environment

### Image: exopoiesis/infra-abacus-gpu

Built via **Dockerfile** (not `docker commit`). `docker commit` loses the NVIDIA runtime
configuration, making GPU invisible on Vast.ai restarts. See `memory/feedback_docker_commit_gpu.md`.

Contents:
- ABACUS v3.9.0.26 with CUDA (`abacus_2g` binary, symlinked as `abacus`)
- CPU MPI binary (`abacus_2p`)
- SG15 ONCV PBE-1.2 pseudopotentials in `/opt/sg15_pp/`
- DZP orbital files in `/opt/sg15_orb/` (for LCAO)
- Python environment with `abacuslite` (ASE interface)

### Environment setup

**For LCAO / CPU runs:**
```bash
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)

# PP paths (if not set in image defaults)
export ABACUS_PP_PATH="/opt/sg15_pp"
export ABACUS_ORB_PATH="/opt/sg15_orb"
```

**For PW / GPU runs:**
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# PP paths (if not set in image defaults)
export ABACUS_PP_PATH="/opt/sg15_pp"
export ABACUS_ORB_PATH="/opt/sg15_orb"
```

### Pseudopotential and orbital file paths

```python
# Standard SG15 ONCV PBE-1.2 pseudopotentials
pp_map = {
    'Fe': 'Fe_ONCV_PBE-1.2.upf',
    'S':  'S_ONCV_PBE-1.2.upf',
    'Ni': 'Ni_ONCV_PBE-1.2.upf',
    'H':  'H_ONCV_PBE-1.2.upf',
    'Co': 'Co_ONCV_PBE-1.2.upf',
    'Cu': 'Cu_ONCV_PBE-1.2.upf',
}

# DZP orbital files for LCAO (100 Ry cutoff, DZP quality)
orb_map = {
    'Fe': 'Fe_gga_8au_100Ry_4s2p2d1f.orb',
    'S':  'S_gga_7au_100Ry_2s2p1d.orb',
    'Ni': 'Ni_gga_8au_100Ry_4s2p2d1f.orb',
    'H':  'H_gga_6au_100Ry_2s1p.orb',
}
```

### abacuslite quirks

`abacuslite` is the ASE calculator interface for ABACUS, distributed with the ABACUS source:

```python
sys.path.insert(0, "/opt/abacus-develop-3.9.0.26/interfaces/ASE_interface")
from abacuslite import Abacus, AbacusProfile
```

- Use `basissets={}` (not `basis=`!) for LCAO
- Use `inp={}` dict for INPUT parameters
- `pseudopotentials={}` (not `pp={}`)
- **Do NOT pass `omp_num_threads` in the command string** — AbacusProfile prepends it,
  causing `FileNotFoundError`. Set via `os.environ` instead.

---

## 7. NumpyEncoder (mandatory)

ABACUS returns numpy types from ASE. All JSON output must encode them explicitly.

```python
import json
import numpy as np

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

# Usage:
with open(result_file, 'w') as f:
    json.dump(result, f, indent=2, cls=NumpyEncoder)
```

This has burned us 3+ times across different sessions. The crash is silent in some Python versions.

---

## 8. Bash Wrapper Patterns

### Mandatory flags

```bash
#!/bin/bash
set -eo pipefail   # catch python crash through | tee

export OMP_NUM_THREADS=1   # PW GPU mode
python3 -u neb_script.py 2>&1 | tee /workspace/results/neb.log
```

`-u` flag is mandatory for real-time output in Docker containers. Without it, stdout is buffered
and you see no output until the buffer flushes (or never, if the process crashes first).

`set -eo pipefail` is mandatory when using `| tee`. Without `pipefail`, a Python crash returns
exit code 0 (from `tee`), and monitoring scripts think everything is fine.

### Lock file (prevent duplicate processes)

```bash
LOCK=/workspace/neb.lock
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
    echo "Already running (PID $(cat $LOCK)). Exiting."
    exit 0
fi
echo $$ > "$LOCK"
trap "rm -f $LOCK" EXIT
```

### DONE marker

```python
# At the end of every script:
done_marker = RESULTS / "DONE_neb_pentlandite"
done_marker.write_text(
    f"E_a={barrier:.4f} eV  completed={time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
)
print("DONE", flush=True)
```

The SRE watchdog checks for `DONE_*` files to detect completion.

---

## 9. Pre-deploy Checklist

### Structure verification

- [ ] `crystal()` → `len(atoms)` == expected atom count
- [ ] Stoichiometry matches mineral formula
- [ ] `min_distance > 1.5 A` (no overlaps after substitution or vacancy)
- [ ] Wyckoff positions cross-checked (pentlandite: 4b+32f+8c+24e = 68 atoms)

### ABACUS parameters

- [ ] `nspin=2` → `basis_type='lcao'` (PW + nspin=2 = unsupported)
- [ ] `symmetry=0` for any spin-polarized or NEB calculation
- [ ] `smearing_sigma >= 0.05` (never 0.01 for metallic systems)
- [ ] `scf_nmax >= 400` (500 for NEB intermediate images)
- [ ] `mixing_beta=0.2`, `mixing_ndim=12` for metallic cells
- [ ] `--bind-to none` in mpirun command (even for np=1)
- [ ] OMP = physical cores, not hyperthreads
- [ ] `timeout >= 28800` (8h) for cells > 50 atoms on GPU

### NEB-specific

- [ ] Two-phase protocol: plain NEB (climb=False) first, CI-NEB second
- [ ] `k=0.05` spring constant (not default 0.1)
- [ ] IDPP interpolation for layered sulfides
- [ ] Per-step image save for resume (SIGKILL protection)
- [ ] Test SCF params on single endpoint BEFORE running full NEB
- [ ] kpts scale correctly with cell size (2x2x2 cell → halved kpts)

### Infrastructure

- [ ] `set -eo pipefail` in bash wrapper
- [ ] `python3 -u` (unbuffered)
- [ ] Lock file to prevent duplicate processes
- [ ] DONE marker at end of script
- [ ] No em-dash `—` characters in code (crashes in ASCII Docker containers)
- [ ] `NumpyEncoder` for all `json.dump` calls

---

## 10. Bug Hall of Fame

### Tier S: many hours of compute lost

| # | Bug | Hours lost | Symptom | Fix |
|---|-----|:----------:|---------|-----|
| 1 | **sigma=0.01 Ry in NEB** | **15 h** | NEB images return ±30 eV garbage energies, hit scf_nmax every image | `sigma=0.05` + `mixing_beta=0.2` + `mixing_ndim=12` + `scf_nmax=500` |
| 2 | **--bind-to none missing** | **6.5 h** | 48-core Threadripper running at 2 cores (192% CPU instead of 4800%) | `mpirun --allow-run-as-root --bind-to none -np 1 abacus_2p` |
| 3 | **CI-NEB from step 0** | **15 h** | fmax oscillates 0.4 → 9.7 → 3.7, no convergence | Two-phase: plain NEB first (fmax<0.3), then CI |
| 4 | **timeout=7200 for large cells** | **variable** | ABACUS killed at 2h, 64 SCF iterations completed, no result | `timeout=28800` for cells > 50 atoms |
| 5 | **symmetry=1 destroys AFM** | **hours** | AFM configuration resets to FM, forces wrong, FIRE diverges | `symmetry: 0` always for magnetic calculations |

### Tier A: garbage data

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 6 | **nspin=2 + PW** | Wrong magnetic moments or crash | LCAO for any nspin=2 |
| 7 | **Gamma-only kpts in 2x2x2** | E_a 0.075 eV (pyrite) vs correct 0.187 eV | Scale kpts inversely to cell size |
| 8 | **numpy types in json.dump** | Silent crash or `TypeError` | Always use `NumpyEncoder` |
| 9 | **S2 molecule with PW nspin=1** | Wrong mu_S reference energy (singlet, not triplet) | LCAO nspin=2 for S2 |

### Tier B: operational issues

| # | Bug | Fix |
|---|-----|-----|
| 10 | Zombie orted processes | `mpirun --allow-run-as-root -np 1 abacus_2p` or `docker run --init` |
| 11 | OMP=48 on 24-core/48-thread CPU | Physical cores only: OMP=24 |
| 12 | QE NEB diverges on small metallic cells | Use ABACUS or GPAW for small metallic NEB |
| 13 | Em-dash `—` in scripts | Replace with `--` before deploy to Docker |
| 14 | `set -e` in monitor scripts | Use `set -uo pipefail` (without `-e`) for monitors |

---

## 11. Comparison: ABACUS vs GPAW vs QE

| Feature | ABACUS | GPAW | QE |
|---------|--------|------|-----|
| GPU (PW) | Yes (CUDA) | Yes (CuPy) | GPU build required |
| AFM + DFT+U | **LCAO only** | `GPAW_NEW=0` or legacy | Yes, legacy Hubbard |
| NEB stability (small metallic cell) | Good | Good | **Diverges** (seen in practice) |
| LCAO basis | Yes (DZP) | No | No |
| Implicit solvation | No | SolvationGPAW | No (in stock build) |
| ASE interface | `abacuslite` | Native | `ase.calculators.espresso` |
| Checkpoint/restart (NEB) | Script only (`--resume`) | Script only | Script only |
| Validated cross-verify | Pyrite 1x1x1 (±3% vs GPAW, ±2% vs QE) | Reference | Reference |

**When to choose ABACUS:**
- AFM minerals (GPAW_NEW=1 + AFM + DFT+U = SCF FAIL)
- GPU PW for large non-magnetic cells
- Cross-verify of GPAW results (independent code)

**When to stay with GPAW:**
- Solvation (SolvationGPAW)
- Datagen campaigns (mature infrastructure, battle-tested)
- FM systems with DFT+U on GPU (`GPAW_NEW=1` supports it)

---

## License

This guide is part of the [Third Matter](https://exopoiesis.space) project.
Released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
If you find it useful for your own ABACUS calculations, please cite.

*Last updated: 2026-04-03, session 76*
*ABACUS version: v3.9.0.26*
*Docker image: exopoiesis/infra-abacus-gpu:latest*
