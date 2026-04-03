# Quantum ESPRESSO DFT NEB: Lessons & Benchmarks

> A practical guide to running Quantum ESPRESSO (QE) NEB calculations via ASE on cloud GPU/CPU infrastructure.
> Accumulated over ~10 AI-assisted sessions of the [Third Matter](https://exopoiesis.space) project (March--April 2026).
> Target systems: iron-sulfide minerals (12--96 atoms), H-diffusion NEB, vacancy mechanism.
> **All claims verified against actual production runs on Vast.ai.**
>
> This document is battle-tested: every bug, benchmark, and recommendation
> comes from real runs. Cross-verification: pyrite E_a within 5% across QE, GPAW, and ABACUS.

---

## 1. CPU vs GPU Mode

### CPU mode (for systems < 100 atoms)

```bash
# mpirun + npool: the only way to get acceptable performance
export OMP_NUM_THREADS=2
mpirun --allow-run-as-root --bind-to none -np 8 pw.x -npool 8 -in espresso.pwi > espresso.pwo
```

**Key rule:** `np = npool = number of irreducible k-points`. QE without `-npool` computes k-points
sequentially -- with 8 k-points, 7/8 of the CPU time is wasted waiting.

**Docker image:** `exopoiesis/infra-qe:latest` (QE 7.4.1, MPI, PSLibrary USPP pseudopotentials)

### GPU mode (for systems >= 100 atoms)

```bash
# Single process, GPU handles FFT and BLAS (OpenACC/CUDA)
export OMP_NUM_THREADS=1
mpirun --allow-run-as-root --bind-to none -np 1 /opt/qe-7.5-gpu/bin/pw.x -npool 1 -in espresso.pwi
```

GPU acceleration in QE is via OpenACC (`-DQE_ENABLE_CUDA=ON -DQE_ENABLE_OPENACC=ON`).
For small systems (< 100 atoms), GPU overhead exceeds the benefit -- speedup can be < 1x.

**Docker image:** `exopoiesis/infra-qe-gpu:7.5` (QE 7.5, nvfortran, CUDA, OpenACC)

### Comparison table

| | CPU + MPI + npool | GPU (single process) |
|---|---|---|
| Optimal atom count | 12--96 | 96+ |
| Parallelism | MPI across k-points | GPU FFT + cuBLAS |
| `-npool` | = np (critical!) | 1 (single process) |
| `OMP_NUM_THREADS` | nproc / np | 1 |
| Typical time (troilite 23 at.) | 4 min/NEB step (npool=10) | slower (GPU overhead) |
| Typical time (pyrite 2x2x2, 95 at.) | hours/NEB step | hours (GPU wins here) |

---

## 2. ASE Espresso Calculator

### Basic setup (new API, ASE >= 3.23)

```python
from ase.calculators.espresso import Espresso, EspressoProfile

PSEUDOPOTENTIALS = {
    'Fe': 'Fe.pbe-spn-rrkjus_psl.1.0.0.UPF',   # USPP PBE
    'S':  'S.pbe-n-rrkjus_psl.1.0.0.UPF',
    'H':  'H.pbe-rrkjus_psl.1.0.0.UPF',
}

input_data = {
    'control': {
        'calculation': 'scf',
        'tprnfor': True,         # MANDATORY: without this QE won't print forces -> ASE crash
        'tstress': False,
        'outdir': '/workspace/qe_scratch/label',
        'prefix': 'mineral',
        'disk_io': 'high',       # writes wfc each SCF step -> checkpoint on kill
        'max_seconds': 36000,    # graceful stop at 10 h
    },
    'system': {
        'ecutwfc': 60,           # Ry -- standard for USPP PBE iron sulfides
        'ecutrho': 480,          # Ry -- 8x ecutwfc for USPP
        'occupations': 'smearing',
        'smearing': 'mv',        # Marzari-Vanderbilt cold smearing
        'degauss': 0.01,         # Ry -- small for semiconductors/insulators
    },
    'electrons': {
        'mixing_beta': 0.3,
        'mixing_mode': 'plain',
        'mixing_ndim': 8,
        'conv_thr': 1.0e-7,
        'electron_maxstep': 300,
    },
}

QE_CMD = "mpirun --allow-run-as-root --bind-to none -np 8 pw.x -npool 8"
profile = EspressoProfile(command=QE_CMD, pseudo_dir='/opt/pp/pbe_uspp')

calc = Espresso(
    input_data=input_data,
    pseudopotentials=PSEUDOPOTENTIALS,
    kpts=(2, 4, 2),
    profile=profile,
    directory='/workspace/qe_work/label',
)
```

### Old API fallback (ASE < 3.23)

```python
from ase.calculators.espresso import Espresso
calc = Espresso(input_data=input_data, pseudopotentials=PSEUDOPOTENTIALS,
                kpts=(2, 4, 2), command=QE_CMD, directory=str(work_dir))
```

Detect API version:
```python
try:
    from ase.calculators.espresso import Espresso, EspressoProfile
    USE_NEW_API = True
except ImportError:
    from ase.calculators.espresso import Espresso
    USE_NEW_API = False
```

---

## 3. `-npool` Optimization (CRITICAL)

Without `-npool`, QE processes k-points sequentially. With N k-points and 1 MPI process per k-point,
QE fully parallelizes the most expensive part (diagonalization and FFT).

**Formula:** `np = npool = N_irreducible_k`

For a (2,4,2) Monkhorst-Pack mesh, irreducible k-points depend on symmetry. A safe rule:
`npool = np`, and `np <= N_irr_k`. More MPI processes than k-points wastes ranks.

### Real benchmarks (session 70)

| System | Atoms | k-mesh | np | npool | OMP | Time/NEB step | Speedup |
|--------|:-----:|:------:|:--:|:-----:|:---:|:-------------:|:-------:|
| Mackinawite | 15 | (2,2,2) ~ 8 irr. k | 1 | 1 | 16 | 36 min | 1x (baseline) |
| Mackinawite | 15 | (2,2,2) ~ 8 irr. k | 8 | 8 | 2 | 13.5 min | **2.67x** |
| Troilite | 23 | (2,4,2) ~ 10 irr. k | 1 | 1 | 16 | ~55 min | 1x (baseline) |
| Troilite | 23 | (2,4,2) ~ 10 irr. k | 10 | 10 | 2 | **~4 min** | **~13x** |

The 13x speedup on troilite is not a typo. Without `-npool`, 90% of CPU time is wasted.

### Setting in ASE calculator

```python
QE_CMD = "mpirun --allow-run-as-root --bind-to none -np 10 pw.x -npool 10"
profile = EspressoProfile(command=QE_CMD, pseudo_dir=PP_DIR)
```

No changes needed in `input_data`. The `-npool` flag is passed directly to `pw.x`.

---

## 4. NEB with QE via ASE

### Full NEB workflow

```python
from ase.mep import NEB   # ASE >= 3.22; fallback: from ase.neb import NEB
from ase.optimize import FIRE
from ase.constraints import FixAtoms

# Build images (N_IMAGES intermediate, 2 endpoints)
N_IMAGES = 5
images = [endA.copy()]
for i in range(N_IMAGES):
    img = endA.copy()
    img.calc = make_calc(f"neb_{i:02d}")   # separate calculator per image
    images.append(img)
images.append(endB.copy())

# Endpoints also need calculators for energy evaluation
images[0].calc = make_calc("neb_endA")
images[-1].calc = make_calc("neb_endB")

# Fix heavy atoms (Fe, S); move only H
for img in images:
    heavy = [i for i in range(len(img)) if img.symbols[i] not in ('H', 'O')]
    img.set_constraint(FixAtoms(indices=heavy))

# Phase 1: plain NEB until fmax < 0.3 eV/A
neb = NEB(images, climb=False, method='improvedtangent', k=0.1)
neb.interpolate('idpp')   # IDPP is mandatory for layered structures

with FIRE(neb, logfile='neb_p1.log') as opt:
    opt.run(fmax=0.30, steps=200)

# Phase 2: CI-NEB to convergence
# ASE has no climb_after parameter -- restart the script
neb2 = NEB(images, climb=True, method='improvedtangent', k=0.1)
with FIRE(neb2, logfile='neb_p2.log') as opt:
    opt.run(fmax=0.05, steps=300)

# Extract barrier
energies = [img.get_potential_energy() for img in images]
E_a = max(e - energies[0] for e in energies)
print(f"E_a = {E_a:.4f} eV")
```

### Two-phase NEB: mandatory

**CI-NEB from step 0 = oscillation.** The climbing image algorithm destabilizes an unconverged
path. Observed failure (session 76): pentlandite NEB with `climb=True` from the start -- barrier
jumped from 0.41 to 9.7 to 3.7 eV and never converged over 15 steps.

**Correct approach:**
1. Phase 1: `climb=False`, run until `fmax < 0.3` eV/A
2. Phase 2: `climb=True`, run until `fmax < 0.05` eV/A

Since ASE NEB has no `climb_after` parameter, implement as two separate script phases with a
resume flag, or restart the script after phase 1 completes.

### IDPP vs linear interpolation

```python
# For layered structures (mackinawite, troilite): always IDPP
neb.interpolate('idpp')

# For diverging NEB on small metallic cells: try linear instead
neb.interpolate()   # default = linear
```

IDPP places the H atom on a physically reasonable path, avoiding steric clashes.
However, for very small metallic cells where periodic images of H are close (< 7 A),
IDPP can place H in bad starting positions, causing divergence.

---

## 5. SCF Parameters for Iron Sulfides

### By magnetic type

| Mineral | T_N or T_C | nspin | Starting approach | degauss (Ry) | mixing_mode |
|---------|:----------:|:-----:|:-----------------:|:------------:|:-----------:|
| Mackinawite | T_N = 65 K (param. at RT) | 1 | no magmoms | 0.02 | plain |
| Pyrite | diamagnetic | 1 | no magmoms | 0.01 | plain |
| Troilite | T_N ~ 315 C (AFM at RT) | 2 | AFM via tot_mag=0 | 0.01 | local-TF |
| Greigite | ferrimagnetic (T_C > RT) | 2 | magmoms 8a=+3.5, 16d=-3.1 | 0.05 | plain |
| Pentlandite | Pauli paramagnetic | 1 | no magmoms | 0.02 | plain |

### AFM setup (troilite example)

```python
# QE cannot do per-atom magmoms via ASE. Use per-species starting_magnetization
# with tot_magnetization=0 constraint -> QE SCF flips some Fe to AFM.
input_data = {
    ...
    'nspin': 2,
    'tot_magnetization': 0.0,             # AFM constraint
    'starting_magnetization(1)': 0.5,     # Fe: species index 1
    'starting_magnetization(2)': 0.0,     # S
    'starting_magnetization(3)': 0.0,     # H
    'mixing_mode': 'local-TF',            # key for AFM SCF convergence
    'mixing_beta': 0.1,
    'mixing_ndim': 10,
}
```

`local-TF` mixing is the Thomas-Fermi screening analog in QE -- analogous to Kerker preconditioning
in GPAW. Essential for AFM systems where charge redistribution is non-uniform.

### DFT+U for magnetic sulfides

```python
# Legacy Hubbard syntax (required -- ASE does not support new HUBBARD card)
# Works in QE 7.4.1 with deprecation warning
input_data['lda_plus_u'] = True
input_data['Hubbard_U(1)'] = 2.0   # Fe d-states, U_eff = 2.0 eV (Dudarev)
                                     # Greigite exception: U = 1.0 eV (Devey 2009)
```

DFT+U is mandatory for minerals with T_N or T_C above room temperature. Without it, the Fe
moment collapses in the SCF and the barrier is garbage.

### vdW correction for layered structures

```python
# DFT-D3 with Becke-Johnson damping: mandatory for mackinawite (87% E_ads from dispersion)
input_data['vdw_corr'] = 'dft-d3'
input_data['dftd3_version'] = 4   # BJ damping
```

---

## 6. Known Bugs and Limitations

### Bug #1: `tprnfor=True` is mandatory

Without `tprnfor=True` in `&CONTROL`, QE does not print forces in SCF mode.
ASE reads forces from QE output -- if they are absent, ASE crashes with a parse error.

```python
input_data['control']['tprnfor'] = True   # NEVER omit this
```

### Bug #2: no custom species in ASE Espresso

ASE Espresso uses the chemical symbol as the species label. You cannot define `Fe1` and `Fe2`
with different magnetic moments -- ASE will crash on the species mismatch.

**Consequence:** per-atom magnetic ordering (true sublattice AFM) is impossible via ASE.
Only per-species `starting_magnetization` with a `tot_magnetization=0` constraint is available.
This produces a good approximation but not exact sublattice AFM.

**Workaround for exact AFM:** use native `neb.x` from QE or write the PWscf input manually.

### Bug #3: `additional_cards` not supported

The new `HUBBARD` card syntax (QE >= 7.3) is not passed through ASE's `additional_cards` support.
Use the legacy `lda_plus_u = True` + `Hubbard_U(species_index)` syntax instead.
Both syntaxes work in QE 7.4.1 (legacy gives deprecation warning, not error).

### Bug #4: CI-NEB from step 0 diverges

Documented above in section 4. Always use two-phase approach.

### Bug #5: QE NEB diverges on small metallic cells

Pentlandite NEB on a 17-atom primitive cell diverged with QE: `fmax` went 0.81 -> 0.58 -> 1.13 -> 1.44.
GPAW and ABACUS converged on the same cell and same endpoints.

The failure mode is specific to QE + FIRE + IDPP on a dense metallic cell where H periodic images
are close (cell a ~ 7.1 A). Remedies to try:
1. Linear interpolation instead of IDPP
2. Reduce spring constant: `k=0.01--0.05` (default 0.1 is too stiff)
3. Reduce number of images: 3 instead of 5
4. Use a supercell (2x2x2): removes the periodic image problem entirely

### Bug #6: Gamma-only kpts in large supercells = artifact

For the 2x2x2 pyrite supercell (95 atoms), using Gamma-only k-points gives E_a = 0.075 eV.
The same system with a proper (2,2,2) k-mesh gives E_a ~ 0.19 eV (matching the 1x1x1 result
with a 4x4x4 mesh).

**Rule:** k-points scale inversely with cell size. A 2x2x2 supercell with Gamma-only for a
metallic system is not converged -- use at minimum a (2,2,2) mesh.

### Bug #7: em-dash `--` in QE scripts

Same as GPAW and ABACUS: em-dash characters (`--`, U+2014) cause `UnicodeEncodeError` in
ASCII-locale Docker containers. Replace with `--` (double hyphen) before deploying.

### Bug #8: NumpyEncoder for json.dump

QE results processed through ASE/numpy contain `np.float64`, `np.bool_`, etc. These types
are not JSON-serializable. Always use a custom encoder:

```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open('result.json', 'w') as f:
    json.dump(data, f, indent=2, cls=NumpyEncoder)
```

---

## 7. AFM Systems: Limitations Summary

| Feature | Supported via ASE? | Workaround |
|---------|:-----------------:|------------|
| Per-species starting_magnetization | Yes | `input_data['starting_magnetization(N)'] = val` |
| Per-atom magnetic moments | **No** | Can't define Fe1/Fe2 species -- ASE crash |
| tot_magnetization=0 (AFM constraint) | Yes | `input_data['tot_magnetization'] = 0.0` |
| Legacy Hubbard U (lda_plus_u) | Yes | `input_data['lda_plus_u'] = True` + `Hubbard_U(N)` |
| New HUBBARD card syntax | **No** | Use legacy syntax (works in QE 7.4.1) |
| additional_cards | **No** | Legacy dict keys in input_data |
| nspin=4 (noncollinear) | **No** | Write input manually or use neb.x directly |

### AFM via tot_magnetization=0: what actually happens

QE starts with `starting_magnetization(Fe) = +0.5` (all Fe ferromagnetic).
The `tot_magnetization=0` constraint forces the total moment to be zero.
During SCF, the QE optimizer redistributes the moment to satisfy the constraint.
For strongly AFM systems (troilite T_N ~ 315 C), QE reliably finds the AFM state.
For weakly magnetic systems, the constraint may give an incorrect magnetic structure.

---

## 8. Benchmarks

### NEB benchmarks (production runs)

| System | Supercell | Atoms | Method | npool | Time/step | Notes |
|--------|:---------:|:-----:|--------|:-----:|:---------:|-------|
| Troilite FeS | 1x1x1 conv | 23 | CPU MPI | 10 | **~4 min** | fmax 0.103, converging |
| Mackinawite FeS | 2x2x1 | 15 | CPU MPI | 8 | **~13.5 min** | E_a = 2.479 eV (cross-layer) |
| Mackinawite FeS | 2x2x1 | 15 | CPU OMP=16 | 1 | **36 min** | baseline (no npool) |
| Pyrite FeS2 | 1x1x1 | 12 | CPU | 4 | fast (min) | E_a = 0.190 eV |
| Pyrite FeS2 | 2x2x2 | 95 | GPU RTX 3090 | 1 | hours/step | size convergence test |

### Cross-verification results (three-code comparison)

| System | GPAW PW | QE USPP | ABACUS PW | Spread | Status |
|--------|:-------:|:-------:|:---------:|:------:|--------|
| Pyrite 1x1x1 H diffusion | 0.181 eV | **0.190 eV** | 0.187 eV | 5% | **Validated** |
| Troilite 1x1x1 H diffusion | -- | converging | -- | -- | In progress |
| Mackinawite dry S-hop | 0.738 eV (intra) | **2.479 eV (cross)** | -- | -- | Different mechanisms |
| Pentlandite H diffusion | 1.115 eV | diverged | 0.900 eV | -- | QE fails on this cell |

**Interpretation of mackinawite discrepancy:** GPAW 0.738 eV is the intra-layer path (S-S hop
within the FeS sheet). QE 2.479 eV is the cross-layer path (hop through the van der Waals gap
between FeS sheets). These are different physical mechanisms, not a code disagreement.

---

## 9. Checkpoint and Resume

QE writes wavefunctions to `outdir/prefix.save/` after each SCF step when `disk_io='high'`.
On crash, QE can restart from this checkpoint using `restart_mode='restart'` in `&CONTROL`.

```python
# Detect if QE checkpoint exists and restart automatically
checkpoint = Path(scratch_dir) / 'prefix.save' / 'data-file-schema.xml'
restart_mode = 'restart' if checkpoint.exists() else 'from_scratch'
input_data['control']['restart_mode'] = restart_mode
```

**Note:** QE checkpoint is per-SCF-image, not per-NEB-step. If NEB is running 5 intermediate
images and the script crashes, all 5 images lose their NEB progress (forces, path). The atomic
positions are lost unless you save them explicitly.

Recommended pattern: save the full NEB path after each FIRE step:

```python
def save_neb_snapshot(images, step, work_dir):
    from ase.io import write
    write(str(work_dir / f'neb_snapshot_{step:04d}.traj'), images)
```

For script-level resume (skip completed phases):
```python
resume = {}
if RESUME_FILE.exists():
    resume = json.load(open(RESUME_FILE))

if 'relax_endA' not in resume:
    endA, energy = relax_endpoint(endA, 'endA')
    resume['relax_endA'] = {'energy': energy, 'converged': True}
    json.dump(resume, open(RESUME_FILE, 'w'), cls=NumpyEncoder)
```

---

## 10. Pre-deploy Checklist

### Structure
- [ ] `len(atoms)` == expected atom count (verify Wyckoff positions!)
- [ ] `min_distance > 1.5 A` after adding H / building vacancy
- [ ] For layered structures: check that H is inside the layer (not in the vdW gap)

### Calculator
- [ ] `tprnfor=True` in `&CONTROL` -- forces are required for ASE
- [ ] `disk_io='high'` -- enables SCF-level checkpoint
- [ ] `max_seconds` set to instance lifetime (graceful stop before SIGKILL)
- [ ] `outdir` on fast local disk (not NFS)
- [ ] `-npool N` = N irreducible k-points in QE command
- [ ] `--bind-to none` in mpirun (prevents core pinning)
- [ ] Separate `directory` per image (otherwise ASE Espresso writes clash)

### NEB specific
- [ ] `climb=False` in phase 1 until `fmax < 0.3`
- [ ] `climb=True` only in phase 2
- [ ] `NumpyEncoder` used in all `json.dump` calls
- [ ] `images[0].calc` and `images[-1].calc` assigned (endpoints need energy for profile)
- [ ] `FixAtoms` applied to all images consistently

### AFM specific
- [ ] `nspin=2`, `tot_magnetization=0`, `starting_magnetization(N)` set
- [ ] `mixing_mode='local-TF'` for AFM systems
- [ ] `lda_plus_u=True` + `Hubbard_U(N)` for minerals with T_N/T_C > RT
- [ ] No em-dash characters in code or comments

### Bash wrapper
- [ ] `python3 -u` (unbuffered stdout for Docker)
- [ ] `set -eo pipefail` (catches Python crash through pipe)
- [ ] Log rotation before restart: `cp neb.log neb.log.prev`

---

## 11. Quick Reference: Pseudopotential Filenames

| Element | USPP PBE (PSLibrary 1.0) | PAW PBE (PSLibrary 1.0) |
|---------|--------------------------|--------------------------|
| Fe | `Fe.pbe-spn-rrkjus_psl.1.0.0.UPF` | `Fe.pbe-spn-kjpaw_psl.0.2.1.UPF` |
| S | `S.pbe-n-rrkjus_psl.1.0.0.UPF` | `S.pbe-n-kjpaw_psl.1.0.0.UPF` |
| H | `H.pbe-rrkjus_psl.1.0.0.UPF` | `H.pbe-kjpaw_psl.1.0.0.UPF` |
| Ni | `Ni.pbe-spn-rrkjus_psl.1.0.0.UPF` | `Ni.pbe-spn-kjpaw_psl.0.1.UPF` |
| O | `O.pbe-n-rrkjus_psl.1.0.0.UPF` | `O.pbe-n-kjpaw_psl.1.0.0.UPF` |

USPP (ultrasoft) is the default choice -- requires `ecutrho = 8 * ecutwfc`.
PAW is more accurate for magnetic systems but heavier; use for publication-quality results.

**Standard cutoffs for iron sulfides (PBE USPP):**
- `ecutwfc = 60 Ry`
- `ecutrho = 480 Ry`
- `degauss = 0.01--0.02 Ry` (semiconductors / metals)

---

## License

This guide is part of the [Third Matter](https://exopoiesis.space) project.
Released under CC-BY-4.0. If you find it useful for your own QE calculations, please cite.

*Last updated: 2026-04-03, session 76*
*Verified against production runs on Vast.ai (sessions 67--76)*
