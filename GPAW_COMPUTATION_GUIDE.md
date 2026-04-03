# GPAW DFT Computation: Lessons & Benchmarks

> A practical guide to running GPAW DFT calculations on cloud GPU/CPU infrastructure.
> Accumulated over ~77 AI-assisted sessions of the [Third Matter](https://exopoiesis.space) project (March–April 2026).
> Target systems: iron-sulfide minerals (8–136 atoms), PW and FD modes.
> **All claims verified against [GPAW source code](https://gitlab.com/gpaw/gpaw).**
>
> This document is battle-tested: every bug, benchmark, and recommendation
> comes from actual production runs on Vast.ai, Hetzner, and local GPU servers.
> Total compute cost: ~$350+ across GPAW, ABACUS, QE, JDFTx, and MACE.
> See also: [ABACUS guide](ABACUS_COMPUTATION_GUIDE.md), [QE guide](QE_COMPUTATION_GUIDE.md), [JDFTx guide](JDFTX_COMPUTATION_GUIDE.md).

---

## 1. GPAW: Two Modes, Two Worlds

### Plane-Wave (PW) — for single-point and datagen

```python
# GPU (RTX 3060+)
export GPAW_NEW=1 GPAW_USE_GPUS=1 OMP_NUM_THREADS=1
calc = GPAW(mode=PW(400),
            parallel={'gpu': True, 'domain': 1},
            eigensolver={'name': 'ppcg'},
            convergence={'energy': 1e-5},
            occupations=FermiDirac(0.1))

# CPU-only (Blackwell sm_120 or no GPU)
export GPAW_NEW=0 OMP_NUM_THREADS=$(nproc)
calc = GPAW(mode=PW(400),
            convergence={'energy': 1e-5},
            occupations=FermiDirac(0.1),
            parallel={'augment_grids': True})
```

**When:** bulk, slab, H-adsorption. Single-point energy + forces for ML training data.

### Finite Difference (FD) + Solvation — for electrochemistry

```python
# MPI parallelism. OMP depends on hardware:
#   Homogeneous cores (Xeon/EPYC): OMP = nproc / np  (e.g. 16 cores, np=4 → OMP=4)
#   Heterogeneous cores (Intel P+E): OMP = 1  (E-cores stall MPI sync)
mpirun --allow-run-as-root -np 4..8 python3 -u script.py

from gpaw.utilities import h2gpts
gpts = h2gpts(0.18, atoms.cell, idiv=32)  # MANDATORY: ensures 6+ multigrid levels

calc = SolvationGPAW(mode='fd', gpts=gpts,   # NOT h=0.18 — explicit gpts!
                     eigensolver={'name': 'rmmdiis'},
                     mixer={'beta': 0.05, 'nmaxold': 8},
                     convergence={'energy': 0.0005, 'density': 1e-4},
                     restart=None,            # NEVER use restart= for SolvationGPAW
                     **solvation_kwargs)

opt = BFGS(atoms, maxstep=0.1)  # NOT default 0.2 — smaller steps for cavity stability
```

**When:** BFGS geometry optimization with implicit solvation. Electrochemical potentials, adsorption energies.
**Do NOT use:** BFGSLineSearch (forces/energy inconsistency with SolvationGPAW — [GPAW docs](https://gpaw.readthedocs.io/tutorialsexercises/electrostatics/sjm/solvated_jellium_method.html)).

### Key differences

| | PW | FD + Solvation |
|---|---|---|
| GPU | Yes (CuPy FFT) | No |
| Parallelism | OMP (threads) | MPI (processes) + OMP for FFT/BLAS |
| OMP_NUM_THREADS | 1 (GPU) / nproc (CPU) | nproc/np on homogeneous, 1 on heterogeneous |
| Grid setup | Automatic | **Must use h2gpts(h, cell, idiv=32)** |
| Typical time | 35 s – 60 min / config | 45 min – 3 h / BFGS step |
| Memory bottleneck | VRAM | RAM bandwidth |
| BFGS maxstep | N/A | **0.1** (not default 0.2) |

### NEB (Nudged Elastic Band)

GPAW PW mode works well for NEB via ASE:

```python
from ase.neb import NEB
from ase.optimize import FIRE
from ase.neb import idpp_interpolate

images = [initial] + [initial.copy() for _ in range(5)] + [final]
neb = NEB(images, climb=True, k=0.1)
idpp_interpolate(neb)

for img in images[1:-1]:
    img.calc = GPAW(mode=PW(350), kpts=kpts, ...)

opt = FIRE(neb)
opt.run(fmax=0.05)
```

**Validated NEB results (GPAW PW, CI-NEB):**
- Pentlandite H⁺ hop: **1.115 eV** (primitive cell, 17 atoms)
- Mackinawite intra-layer: **0.738 eV** (2x2x1, 15 atoms)
- Pyrite S-S dimer hop: **0.181 eV** (1x1x1, 12 atoms)

Cross-verified against ABACUS and QE within 5-20% — see `results/` for full data.

> **Verdict (session 62):** SolvationGPAW FD + BFGS is **non-viable** for production.
> Use GPAW PW for NEB/single-point and JDFTx CANDLE for solvation.

---

## 2. Hardware Selection

### Rule #1: GHz > cores > VRAM

For PW mode (datagen), **CPU clock speed is the dominant factor**. GPU is only used for FFT.

| GPU | CPU GHz | Time/config (pentlandite, 20 atoms) | $/hr | Verdict |
|-----|:-------:|:-----------------------------------:|:----:|---------|
| Quadro P4000 | 3.8 | 6226 s (103 min) | $0.068 | **Best value** |
| RTX 3060 | 5.9 | ~3000 s (50 min) | $0.069 | Excellent |
| RTX 5060 Ti | 5.5 | ~2100 s (35 min) | $0.070 | Excellent |
| RTX 3080 | 6.0 | ~2400 s (40 min) | $0.117 | Good |
| RTX 5070 Ti | 8.5 | ~1800 s (30 min) | $0.117 | Good |
| RTX 4070 | **3.0** | 13002 s (**217 min**) | $0.075 | **Bad** (low GHz!) |

> **Takeaway:** a cheap instance with high GHz beats an expensive one with a powerful GPU.
> RTX 4070 at 3.0 GHz is 2x slower than Quadro P4000 at 3.8 GHz.

### Rule #2: For FD/MPI — server CPUs only, never desktop

FD mode (SolvationGPAW) is memory-bandwidth bound. The Poisson solver operates on stencils over a large 3D grid.

| Platform | CPU | DDR5 Channels | BW (GB/s) | Time/BFGS step (72 atoms) | Verdict |
|----------|-----|:-------------:|:---------:|:-------------------------:|---------|
| Vast.ai (typical) | Xeon/EPYC | 8–12 | 200–400 | **~45 min** | OK |
| Hetzner EX63 | Core Ultra 7 265 | **2** | **77** | **~10 hours** | **13x slower!** |
| Hetzner AX102-U | EPYC 9454P | **12** | ~460 | ~30–45 min (expected) | OK |

> **Takeaway:** desktop Intel (P+E cores, 2-channel DDR5) is a trap for MPI.
> E-cores slow down MPI sync, 2 DDR5 channels = stencil contention.
> Server Xeon/EPYC with 8+ channels is the only viable option for FD.

### Rule #3: RAM matters

| System | Atoms | VRAM (PW+GPU) | RAM (FD+MPI) |
|--------|:-----:|:-------------:|:------------:|
| Mackinawite bulk | 8 | ~2 GB | ~4 GB |
| Pyrite bulk | 12 | ~3 GB | ~6 GB |
| Pentlandite bulk | 20 | ~6 GB | ~8 GB |
| Pentlandite conv. cell | 68 | ~14 GB (OOM at 10!) | ~20 GB |
| Mackinawite slab (72 at.) | — | N/A (FD only) | ~16 GB |
| Pentlandite CO ads (138 at.) | 14+ GB (OOM) | — | ~40 GB |

> **Takeaway:** pentlandite PW needs GPU with >= 16 GB VRAM.
> FD solvation needs >= 64 GB RAM per instance.

### Pre-purchase checklist

- [ ] GHz >= 5.0 (for PW datagen)
- [ ] RAM > 16 GB (pentlandite), > 64 GB (FD solvation)
- [ ] For MPI FD: homogeneous cores (Xeon/EPYC), >= 8 DDR channels
- [ ] For MPI FD: 6–8 cores optimal (np=4–8); np=16 crashes
- [ ] NOT desktop Intel 12th+/Core Ultra (P+E cores = MPI disaster)
- [ ] Blackwell GPU (RTX 5xxx, sm_120): needs NVRTC >= 12.8 for CuPy

---

## 3. Benchmarks: Real Numbers

### PW datagen (single-point energy + forces)

| Mineral | SG | Atoms | Config type | Time/config | GPU | Configs | Total |
|---------|:--:|:-----:|-------------|:-----------:|-----|:-------:|:-----:|
| Mackinawite | P4/nmm | 8 | bulk rattle | 35 s | RTX 3060 Ti | 44 | ~25 min |
| Pyrite | Pa-3 | 12 | bulk rattle | 200 s | RTX 3060 | 51 | ~2.8 h |
| Pentlandite | Fm3m | 20 | bulk rattle | 30–60 min | RTX 3080 | 113 (4 workers) | ~3 days |
| Marcasite | Pnnm | 24 | bulk rattle | ~500 s | RTX 4000 Ada | 26 | ~3.6 h |
| Marcasite | Pnnm | ~48 | slab eq | **10372 s (2.9 h)** | RTX 4000 Ada | 1/5 | slab FAIL |
| Greigite | Fd-3m | ~80 | bulk+slab | ~5760 s (1.6 h) | RTX 5070 Ti | 22 | ~35 h |
| Troilite | P-62c | 24 | bulk+slab | ~600–8000 s | — | 53 planned | QA PASS |
| Cubanite | Pnma | 24 | bulk+slab | ~600–8000 s | — | 55 planned | QA PASS |
| Chalcopyrite | I-42d | 16 | bulk+slab | ~8000 s | RTX 4000 Ada | 7/61 running | — |
| Violarite | Fd-3m | 14–56 | bulk+conv+slab | 340–17300 s | RTX 4000 Ada | 33/49 running | — |

### FD + Solvation (BFGS optimization)

| System | Atoms | np | OMP | gpts | Time/step | Instance | Notes |
|--------|:-----:|:--:|:---:|:----:|:---------:|----------|-------|
| Mack (001) + HCOO⁻ ontop | 77 | 4 | **4** | 64×64×192 | **~20 min** | Vast.ai RTX 3080 | gpts fix, working |
| Mack (001) + HCOO⁻ hollow | 77 | 4 | **4** | 64×64×192 | **~20 min** | Vast.ai RTX 3080 | gpts fix, working |
| Mack (001) + HCOO⁻ bridge | 77 | 8 | 1 | old (60×60×180) | computing | Hetzner P-cores | slow MG, restarting |
| Mack (001) + HCOO⁻ ontop (old) | 77 | 4 | 4 | **60×60×180** | **~45 min** | Vast.ai | 3 MG levels, slow but worked (26 steps) |
| Mack (001) + HCOO⁻ ontop (old) | 77 | 4 | **1** | **60×60×180** | **>>2h** | Vast.ai | 3 MG levels + OMP=1 = extremely slow |
| ref test | 72 | 1 | OMP | auto | 52318 s (14.5 h) total | RTX 4070 (CPU FD) | 200 BFGS steps |

**Key insight:** the "Poisson hang" was never a hang — it was catastrophically slow convergence
with only 3 multigrid levels (60×60×180). With 6 levels (64×64×192), same calculation is 8–16x faster.
Verified from GPAW source: `FDPoissonSolver` has `maxiter=1000` and raises `PoissonConvergenceError`,
not an infinite loop (`gpaw/poisson.py`).

### Cost summary

| Campaign | Instances | $/hr total | Days | Total |
|----------|:---------:|:----------:|:----:|:-----:|
| v1 datagen (5 cheap) | 5 | $0.32 | 2 | ~$15 |
| v2 datagen (6 workers) | 6 | $0.60 | 4 | ~$58 |
| q075 solvation (3 sites) | 3 | $0.33 | 7+ | ~$55+ |
| Hetzner EX63 | 1 | ~$0.58/hr | 2 | **~$30 (cancelled — 13x slow)** |

> **Total for March 2026:** ~$130 Vast.ai + ~$30 Hetzner (wasted) = ~$160.

---

## 4. Bug Hall of Fame

### Tier S: tens of hours lost

| # | Bug | Hours lost | Symptom | Root cause | Fix |
|---|-----|:----------:|---------|------------|-----|
| 1 | **Insufficient multigrid levels** | **~130 h** | Poisson solver "hangs" (actually: catastrophically slow convergence). Log stops writing during first SCF iter of step 1. | `h=0.18` → 60×60×180 grid → only 3 MG levels. Poisson solver has `maxiter=1000` + throws `PoissonConvergenceError`, not infinite loop. With 3 levels each iteration is very slow. After BFGS move cavity fully recomputes → solver starts from scratch in new dielectric. | **`gpts=h2gpts(0.18, cell, idiv=32)`** → 64×64×192 = 6 MG levels. Also `maxstep=0.1` (smaller cavity perturbation) |
| 2 | **False diagnosis "OMP=1 always"** | **~130 h** | Killed working process (26 BFGS steps) based on wrong lesson. Restarted with OMP=1 → 4x slower → looked even more "stuck" | Pattern-matching from Hetzner (P+E cores need OMP=1) applied blindly to Vast.ai (homogeneous cores where OMP=4 is fine) | OMP=nproc/np on homogeneous (Xeon/EPYC). OMP=1 ONLY on heterogeneous (P+E) |
| 3 | **SolvationGPAW restart=** | **~42 h** | `restart=file` fails to initialize eigensolver → crash on `set_positions()` | `read()` with `reading=True` doesn't create eigensolver | Fresh calculator every time, NEVER use `restart=` |
| 4 | **MPI + BFGS .traj race** | **~50 h** | N ranks open .traj in `mode='w'` → file empty | ASE BFGS opens trajectory in `mode='w'`, N MPI ranks = race condition | Rank 0 only: `traj = path if world.rank == 0 else None` |
| 5 | **Hetzner P+E cores** | **~20 h** | 13x slower than Vast.ai. E-cores slow MPI, 2-ch DDR5 = BW bottleneck | Desktop CPU ≠ server CPU. P+E = heterogeneous = MPI disaster | Server CPUs (Xeon/EPYC) only for MPI |
| 6 | **Logs overwritten on restart** | **diagnostics** | `> log` on restart destroys crash data | No log rotation | `cp log log.prev` BEFORE restart |

### Tier A: hours lost or garbage data

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 7 | **Greigite u=0.254 vs 0.380** | S-S overlap 0.11 A, garbage energies | ASE `crystal()` setting 1 (default): u=0.380 |
| 8 | **numpy 2.x JSON bool** | `np.bool_` serialization fail, MPI rank race on JSON write | `_sanitize()` with `np.generic.item()`, rank 0 guard |
| 9 | **calc.attach() for SolvationGPAW** | Callback never fires (SCF loop doesn't call it) | Use `opt.attach()` instead of `calc.attach()` |
| 10 | **kpts: slab gets bulk kpts** | is_slab check AFTER label-based kpts → slab with (4,4,4) → OOM | Check is_slab FIRST |
| 11 | **Em-dash `—` (U+2014)** | UnicodeEncodeError in ASCII containers | Replace `—` → `--` before deploy |
| 12 | **Pyrrhotite: wrong SCF fix** | Applied marcasite pattern (MixerDif for all + FermiDirac 0.15) → magnetic moments collapsed to 0 | Read the SCF log first! density was -3.94 at threshold -4.0. Correct fix: `convergence={'density': 1e-3}`. MixerDif only for slabs. FermiDirac ≤ 0.1 |

### Tier B: annoying but manageable

| # | Bug | Fix |
|---|-----|-----|
| 13 | `pkill -f` kills SSH session | `kill PID` of specific process, not `pkill -f` |
| 14 | `grep gpaw` = 0 (GPAW runs as python3) | `ps aux --sort=-pcpu \| head -5` |
| 15 | Vast.ai onstart-cmd spawns duplicates | Lock-file + PID alive check |
| 16 | SIGTERM → SIGKILL = 10 sec on Vast.ai | Append after each config + `--resume`, max loss = 1 config |
| 17 | H adsorption wrong axis | ASE `surface()` always makes z the normal. Use `[:, 2]` and `[2] +=` |

### Meta-bug: pattern-matching fixes

**The most expensive class of bugs is applying a "lesson" from one postmortem to a different situation
without reading the actual diagnostic data.**

Examples:
- Marcasite fix (MixerDif for everything) ≠ pyrrhotite fix (relax density threshold)
- Hetzner lesson (OMP=1 on P+E cores) ≠ Vast.ai (OMP=4 on homogeneous cores is correct)
- "Poisson hangs" ≠ "Poisson converges slowly with too few multigrid levels"

**Rule: read the SCF log / diagnostic data FIRST, then diagnose. Never apply a template fix.**

---

## 5. SCF Convergence: Recipes by Magnetism

### Initial magnetic moments (mandatory!)

```python
magmoms_dict = {'Fe': 2.0, 'Ni': 0.6, 'Co': 1.5, 'Cu': 0.0, 'S': 0.0}
atoms.set_initial_magnetic_moments([magmoms_dict[s] for s in atoms.get_chemical_symbols()])
```

Without magmoms → SCF won't converge (Fe/Ni/Co) or produces garbage (greigite, troilite).

### AFM minerals (marcasite, troilite, mackinawite)

```python
# Mixer: MixerDif for SLABS (different spin channels across vacuum)
#        Mixer(0.05) for BULK (even if AFM — sufficient for most cases)
mixer = MixerDif(beta=0.02, nmaxold=5, weight=50.0) if is_slab else Mixer(0.05, 5)
maxiter = 600
occupations = FermiDirac(0.1)   # ≤ 0.1 for magnetics! 0.15+ kills magnetic order
```

**Slab rattle sigma for AFM: <= 0.01!** At sigma >= 0.03 → charge sloshing → SCF FAIL.
`retry_limit=2` mandatory for slab configs (otherwise infinite fail loop).

### Ferrimagnetic minerals (pyrrhotite, greigite)

```python
# Key insight: density convergence is the bottleneck, NOT the mixer
convergence = {'energy': 1e-5, 'density': 1e-3}  # relaxed density (default 1e-4 too tight)
mixer = Mixer(0.05, 5)  # standard mixer OK for bulk ferrimagnetics
occupations = FermiDirac(0.1)  # NEVER > 0.1 for magnetics
```

**Pyrrhotite postmortem:** bulk_eq had density at -3.94 with threshold -4.0 (ALMOST converged).
Wrong fix: MixerDif(0.02) for all + FermiDirac(0.15) → magnetic moments collapsed to 0.
Correct fix: just `convergence={'density': 1e-3}`. Read the log, don't apply templates!

### DFT+U (mandatory for T_N/T_C > RT)

```python
# GPAW Hubbard U correction — required for magnetic minerals with ordering above room temperature
setups = {'Fe': ':d,2.0'}       # U=2.0 eV on Fe 3d (troilite, chalcopyrite, violarite)
# Greigite: U=1.0 eV (weaker, ferrimagnetic)
setups = {'Fe': ':d,1.0'}       # greigite only
```

Without DFT+U: GGA collapses magnetic moments for T_N > RT. GPAW_NEW=1 + AFM + DFT+U = SCF FAIL — use legacy GPAW_NEW=0 or switch to ABACUS.

### Magnetism by mineral

| Mineral | Magnetism | Without magmoms usable? | Mixer (bulk) | Mixer (slab) | FermiDirac | density conv | DFT+U |
|---------|-----------|:-----------------------:|:------------:|:------------:|:----------:|:------------:|:-----:|
| Pentlandite | Pauli paramagnetic | YES (~5–10% error) | Mixer(0.05) | Mixer(0.05) | 0.1 | default | no |
| Mackinawite | AF, T_N=65K | YES (paramagnetic at 25C) | Mixer(0.05) | MixerDif(0.02) | 0.1 | default | no |
| Pyrite | Diamagnetic | YES | Mixer(0.05) | Mixer(0.05) | 0.1 | default | no |
| **Greigite** | **Ferrimagnetic (3.5 µB)** | **NO — garbage** | Mixer(0.05) | MixerDif(0.02) | **≤ 0.1** | **1e-3** | **U=1.0** |
| **Troilite** | **AF, T_N~315C** | **NO — garbage** | Mixer(0.05) | MixerDif(0.02) | **≤ 0.1** | default | **U=2.0** |
| **Cubanite** | **FM, T~250–300C** | **NO — garbage** | Mixer(0.05) | Mixer(0.05) | **≤ 0.1** | default | **U=2.0** |
| **Pyrrhotite** | **Ferrimagnetic** | **NO — garbage** | Mixer(0.05) | MixerDif(0.02) | **≤ 0.1** | **1e-3** | **U=2.0** |
| **Chalcopyrite** | **AF, T_N~550C** | **NO — garbage** | Mixer(0.05) | MixerDif(0.02) | **≤ 0.1** | default | **U=2.0** |
| Marcasite | AF, T_N~65K | YES (paramagnetic at 25C) | Mixer(0.05) | MixerDif(0.02) | 0.1 | default | no |

---

## 6. SolvationGPAW: Deep Dive (from GPAW source code)

### Multigrid levels — the #1 performance factor

The FDPoissonSolver uses a multigrid V-cycle. Each level coarsens the grid by 2x in each dimension.
Coarsening stops when it can't divide evenly → fewer levels = slower convergence.

```
Grid 60×60×180  (no idiv)  → 3 levels (min dim)  → VERY SLOW
Grid 64×64×184  (idiv=8)   → 3 levels (z=8×23!)  → STILL SLOW — 23 is prime!
Grid 64×64×192  (idiv=32)  → 6 levels (all dims)  → FAST
```

**WARNING:** `idiv=8` is NOT sufficient! For cell ~11×11×33 Å with h=0.18:
- z: ceil(32.75/0.18)=182, next_mult(182,8)=**184=8×23** → only 3 MG levels
- z: next_mult(182,32)=**192=3×2^6** → 6 MG levels

Empirically confirmed: step 0 SCF converges in ~90 min with 64×64×184,
but step 1 (after BFGS move changes solvation cavity) goes silent for 3+ hours.

From [GPAW docs](https://gpaw.readthedocs.io/documentation/poisson.html):
> "FDPoissonSolver will **converge inefficiently if at all, or yield wrong results**
> with unsuitable grids."

[GitLab Issue #103](https://gitlab.com/gpaw/gpaw/-/issues/103) — still open.

**Fix:** always use `h2gpts(h, cell, idiv=32)` (or idiv=16 for large systems).

### Cavity recomputes every BFGS step

Source: `gpaw/solvation/hamiltonian.py`

```
set_positions() → initialize_positions() → hamiltonian.update_atoms()
→ cavity.update(new_atoms, density)      [Boltzmann equation]
→ dielectric.update(cavity)              [full recalculation of eps_gradeps]
→ WeightedFDPoissonSolver.solve()        [restrict_op_weights() + Poisson V-cycle]
```

After BFGS move: old potential in new dielectric → Poisson needs many iterations to converge.
Larger step (maxstep) = larger cavity change = more Poisson iterations = slower.

### calc.write() is safe, restart= is not

- **`calc.write(mode='all')`** — read-only on calculator state (`gpaw/solvation/calculator.py`).
  Writes cavity/dielectric PARAMETERS, not grid state. Safe to call during BFGS.
  But the .gpw file is useless because...
- **`SolvationGPAW(restart='file.gpw')`** — BUGGY. ASE calls `read()` which calls
  `initialize(reading=True)` which does NOT create eigensolver.
  → crash on next `set_positions()`.
- **Correct checkpoint:** `atoms.write('checkpoint.xyz')` (geometry only, light, actually useful for resume).

### BFGSLineSearch — do NOT use

From [GPAW SJM documentation](https://gpaw.readthedocs.io/tutorialsexercises/electrostatics/sjm/solvated_jellium_method.html):
> "BFGSLineSearch may have trouble with SJM, since the tolerance set on the desired potential
> can lead to small inconsistencies between the forces and energy."

Use plain BFGS.

### SolvationGPAW + BFGS = untested in GPAW

No test files in `gpaw/test/solvation/` use BFGS or any optimizer.
No examples in documentation show BFGS + checkpoint with solvation.
We are in uncharted territory.

---

## 7. Operational Lessons (Vast.ai / Cloud)

### Launching on Vast.ai

```bash
# Optimal onstart one-liner
export TZ=Europe/Kyiv OMP_NUM_THREADS=$(nproc) MKL_NUM_THREADS=$(nproc) OPENBLAS_NUM_THREADS=$(nproc)
git clone --depth 1 https://github.com/exopoiesis/digital-twin.git /workspace/digital-twin
cp /workspace/digital-twin/tools/infra/*.sh /workspace/digital-twin/tools/gpaw/*.py /workspace/ 2>/dev/null
chmod +x /workspace/*.sh; mkdir -p /workspace/results
nohup bash /workspace/vast_monitor.sh >/dev/null 2>&1 & disown
```

### Monitoring

- **vast_check.sh** — quick status: process, heartbeat, DONE, crash_info
- **vast_monitor.sh** — background daemon: disk, load, GPU, XYZ count, python3 alive
- **Heartbeat** → `/workspace/results/heartbeat`. If > 5 min stale — problem
- **DONE file** → `/workspace/results/DONE*`. If exists — collect data, consider destroy

### Script patterns (mandatory)

```bash
set -eo pipefail              # Catch python crash through | tee
python3 -u script.py          # -u = unbuffered (real-time output in Docker)
nohup ... > log 2>&1 &        # Detach from SSH
disown $!                     # Don't kill on SSH disconnect
```

```python
# MPI guard for IO
from gpaw.mpi import world
traj_file = 'output.traj' if world.rank == 0 else None
log_file  = 'output.log'  if world.rank == 0 else None
```

### Diagnosing a "stuck" process

**Most likely NOT stuck — just slow Poisson convergence with insufficient multigrid levels.**

1. `ps aux --sort=-pcpu | head -5` — is python3 alive and using CPU?
2. `find /workspace/results -mmin -30` — writing anything?
3. Check .txt log: if last SCF iter was step 1 iter 0 and nothing new for hours → Poisson solver is
   computing (log only updates AFTER each SCF iter, during Poisson solve = silence)
4. `/proc/PID/fd/ | grep .txt` — if fd open but position not growing over 60s → inside Poisson solve
5. `cat /proc/PID/environ | tr '\0' '\n' | grep OMP` — verify OMP env
6. **Before killing:** check BFGS log! If steps are progressing (even slowly), the process is WORKING.
   Killing a slow-but-working process and restarting = losing all completed steps.

### Log rotation (mandatory on restart)

```bash
cp /workspace/results/site.txt /workspace/results/site.txt.prev
# Only then start new process
```

---

## 8. Crystallographic Traps

### General rule

**Verify EVERY crystal() structure before DFT:**
1. `len(atoms)` == expected atom count
2. Stoichiometry == mineral formula
3. `min_distance > 1.5 A` (no overlaps)
4. Cross-check Wyckoff positions with literature CIF

### Specific traps

| Mineral | SG | Trap | Correct |
|---------|:--:|------|---------|
| Greigite | Fd-3m (#227) | u=0.254 → overlaps 0.11 A | u=0.380 (setting 1, ASE default) |
| Troilite | P-62c (#190) | Using NiAs-type P63/mmc → wrong positions | CIF 0004158, Skala 2006: 24 atoms |
| Cubanite | Pnma (#62) | CIF in Pcmn → need coordinate transform | (x,y,z)→(z,y,1-x), cell (a,b,c)→(c,b,a) |
| Pentlandite | Fm3m (#225) | "8c → 8 S, 20 atoms" | 32f → 32 metal, 8c+24e → 32 S, **68 atoms** |
| Bornite | Fm3m | Cu/Fe disorder | **No ordered CIF → unsuitable for DFT** |

### check_min_distance (v2, adsorbate-aware)

```python
def check_min_distance(atoms, min_dist=1.2, adsorbate_sizes=None):
    """Check minimum interatomic distance. adsorbate_sizes masks intramolecular bonds."""
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    if adsorbate_sizes:
        n = len(atoms)
        idx = n
        for mol_size in reversed(adsorbate_sizes):
            start = idx - mol_size
            dists[start:idx, start:idx] = np.inf
            idx = start
    return np.min(dists) >= min_dist, np.min(dists)
```

---

## 9. Pre-deploy Checklist

### Structure
- [ ] `crystal()` → `len(atoms)` == expected
- [ ] Stoichiometry == mineral formula
- [ ] `min_distance > 1.5 A`
- [ ] Wyckoff positions cross-checked with CIF

### DFT parameters
- [ ] `set_initial_magnetic_moments()` for Fe/Ni/Co
- [ ] `check_min_distance()` after rattle/strain/adsorbate
- [ ] `txt=f'/workspace/results/{label}.txt'` (NOT None — needed for SCF diagnostics)
- [ ] kpts: slabs (x,x,1), bulk (x,x,x). is_slab check FIRST
- [ ] maxiter=500+ for adsorption
- [ ] No em-dash `—` in code
- [ ] Vacuum >= 12 A for slabs

### SCF strategy (magnetic minerals)
- [ ] MixerDif(0.02) for AFM/ferri SLABS, Mixer(0.05) for BULK
- [ ] FermiDirac ≤ 0.1 for all magnetic minerals
- [ ] `convergence={'density': 1e-3}` for ferrimagnetics (pyrrhotite, greigite)

### SolvationGPAW (FD mode)
- [ ] **`gpts=h2gpts(h, cell, idiv=32)`** — NEVER bare `h=` for FD+solvation
- [ ] **`maxstep=0.1`** in BFGS (not default 0.2)
- [ ] **BFGS** (not BFGSLineSearch)
- [ ] **`restart=None`** (never `restart=` for SolvationGPAW)
- [ ] Checkpoint: `atoms.write()` (not `calc.write()`)
- [ ] OMP: nproc/np on homogeneous cores, 1 on heterogeneous (P+E)

### Bash/infra
- [ ] `set -eo pipefail` in bash wrapper
- [ ] `python3 -u` (unbuffered)
- [ ] Log rotation before restart

---

## License

This guide is part of the [Third Matter](https://exopoiesis.space) project.
Released under CC-BY-4.0. If you find it useful for your own GPAW calculations, please cite.

*Last updated: 2026-04-04, session 77*
*Verified against [GPAW source code](https://gitlab.com/gpaw/gpaw)*
