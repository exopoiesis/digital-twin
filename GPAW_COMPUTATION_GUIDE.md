# GPAW DFT Computation: Lessons & Benchmarks

> A practical guide to running GPAW DFT calculations on cloud GPU/CPU infrastructure.
> Accumulated over ~40 AI-assisted sessions of the [Third Matter](https://exopoiesis.space) project (March 2026).
> Target systems: iron-sulfide minerals (8–136 atoms), PW and FD modes.
>
> This document is battle-tested: every bug, benchmark, and recommendation
> comes from actual production runs on Vast.ai, Hetzner, and local GPU servers.
> Total compute cost: ~$160 for ~500 DFT configs + 3 solvation BFGS optimizations.

---

## Table of Contents

1. [Two Modes, Two Worlds](#1-gpaw-two-modes-two-worlds) — PW vs FD+Solvation
2. [Hardware Selection](#2-hardware-selection) — GHz, RAM, bandwidth
3. [Benchmarks](#3-benchmarks-real-numbers) — actual timings and costs
4. [Bug Hall of Fame](#4-bug-hall-of-fame) — 15 bugs, ~220 hours lost
5. [SCF Convergence](#5-scf-convergence-recipes-by-magnetism) — magnetism recipes
6. [Operational Lessons](#6-operational-lessons-vastai--cloud) — Vast.ai, monitoring, diagnostics
7. [Crystallographic Traps](#7-crystallographic-traps) — ASE/CIF pitfalls
8. [Pre-deploy Checklist](#8-pre-deploy-checklist) — 15-point checklist

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
# ALWAYS MPI, ALWAYS OMP=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mpirun --allow-run-as-root -np 4..8 python3 -u script.py

calc = SolvationGPAW(mode={'name': 'fd', 'h': 0.18},
                     eigensolver={'name': 'rmmdiis'},
                     mixer={'beta': 0.05, 'nmaxold': 8},
                     convergence={'energy': 0.0005, 'density': 1e-4})
```

**When:** BFGS geometry optimization with implicit solvation. Electrochemical potentials, adsorption energies.

### Key differences

| | PW | FD + Solvation |
|---|---|---|
| GPU | Yes (CuPy FFT) | No |
| Parallelism | OMP (threads) | MPI (processes) |
| OMP_NUM_THREADS | 1 (GPU) / nproc (CPU) | **STRICTLY 1** |
| Typical time | 35 s – 60 min / config | 45 min – 3 h / BFGS step |
| Memory bottleneck | VRAM | RAM bandwidth |

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
| Chalcopyrite | I-42d | 16 | bulk+slab | ~8000 s | RTX 4000 Ada | 7/61 | running |
| Violarite | Fd-3m | 14–56 | bulk+conv+slab | 340–17300 s | RTX 4000 Ada | 33/49 | running |

### FD + Solvation (BFGS optimization)

| System | Atoms | np | OMP | Time/step | Instance | Notes |
|--------|:-----:|:--:|:---:|:---------:|----------|-------|
| Mackinawite (001) + HCOO⁻ ontop | 77 | 4 | 1 | ~45 min | Vast.ai RTX 3080 | OMP=1 fix, working |
| Mackinawite (001) + HCOO⁻ hollow | 77 | 4 | 1 | ~45 min | Vast.ai RTX 3080 | OMP=1 fix, working |
| Mackinawite (001) + HCOO⁻ bridge | 77 | 8 | 1 | computing | Hetzner P-cores | Restarted OMP=1 |
| Mackinawite (001) + HCOO⁻ bridge | 77 | 10 | **2** | **STUCK** | Hetzner all cores | Poisson hang |
| Mackinawite (001) + HCOO⁻ (any) | 77 | 4 | **4** | **STUCK** | Vast.ai | Poisson hang |
| Local workstation test | 72 | 1 | OMP | 52318 s (14.5 h) total | RTX 4070 (CPU FD) | 200 BFGS steps |

### Cost summary

| Campaign | Instances | $/hr total | Days | Total |
|----------|:---------:|:----------:|:----:|:-----:|
| v1 datagen (5 cheap) | 5 | $0.32 | 2 | ~$15 |
| v2 datagen (6 workers) | 6 | $0.60 | 4 | ~$58 |
| Solvation BFGS (3 sites) | 3 | $0.33 | 7+ | ~$55+ |
| Hetzner EX63 | 1 | ~$0.58/hr | 2 | **~$30 (cancelled — 13x slow)** |

> **Total for March 2026:** ~$130 Vast.ai + ~$30 Hetzner (wasted) = ~$160.

---

## 4. Bug Hall of Fame

### Tier S: tens of hours lost

| # | Bug | Hours lost | Symptom | Fix |
|---|-----|:----------:|---------|-----|
| 1 | **OMP>1 + MPI FD** | **~108 h** | Poisson solver hangs on step 1+. CPU 100%, log stops writing, fd for .txt closed. Confirmed 3/3 (OMP=4, OMP=4, OMP=2) | `OMP_NUM_THREADS=1` ALWAYS for MPI FD |
| 2 | **SolvationGPAW restart=** | **~42 h** | `restart=file` fails to initialize eigensolver → crash on `set_positions()` | Fresh calculator every time, NEVER use restart= |
| 3 | **MPI + BFGS .traj race** | **~50 h** | N ranks open .traj in mode='w' → file empty | Rank 0 only: `traj = path if world.rank == 0 else None` |
| 4 | **Hetzner P+E cores** | **~20 h** | 13x slower than Vast.ai. E-cores slow MPI, 2-ch DDR5 = BW bottleneck | Server CPUs (Xeon/EPYC) only |
| 5 | **Logs overwritten on restart** | **diagnostics** | `> log` on restart destroys crash data | `cp log log.prev` BEFORE restart |

### Tier A: hours lost or garbage data

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 6 | **Greigite u=0.254 vs 0.380** | S-S overlap 0.11 A, garbage energies | ASE crystal() setting 1 (default): u=0.380 |
| 7 | **numpy 2.x JSON bool** | `np.bool_` serialization fail, MPI rank race on JSON write | `_sanitize()` with `np.generic.item()`, rank 0 guard |
| 8 | **calc.attach() for SolvationGPAW** | Callback never fires (SCF loop doesn't call it) | Use `opt.attach()` instead of `calc.attach()` |
| 9 | **kpts: slab gets bulk kpts** | is_slab check AFTER label-based kpts → slab with (4,4,4) → OOM | Check is_slab FIRST |
| 10 | **Em-dash `—` (U+2014)** | UnicodeEncodeError in ASCII containers | Replace `—` → `--` before deploy |

### Tier B: annoying but manageable

| # | Bug | Fix |
|---|-----|-----|
| 11 | `pkill -f` kills SSH session | `kill PID` of specific process, not pkill -f |
| 12 | `grep gpaw` = 0 (GPAW runs as python3) | `ps aux --sort=-pcpu \| head -5` |
| 13 | Vast.ai onstart-cmd spawns duplicates | Lock-file + PID alive check |
| 14 | SIGTERM → SIGKILL = 10 sec on Vast.ai | Append after each config + `--resume`, max loss = 1 config |
| 15 | H adsorption wrong axis | ASE surface() always makes z the normal. Use `[:, 2]` and `[2] +=` |

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
# Mixer: must use MixerDif for different spin channels
mixer = MixerDif(beta=0.02, nmaxold=5, weight=50.0)
maxiter = 600
occupations = FermiDirac(0.2)   # Increased smearing for stability
```

**Slab rattle sigma for AFM: <= 0.01!** At sigma >= 0.03 → charge sloshing → SCF FAIL.
`retry_limit=2` mandatory for slab configs (otherwise infinite fail loop).

### Magnetism by mineral

| Mineral | Magnetism | Without magmoms usable? | Mixer |
|---------|-----------|:-----------------------:|-------|
| Pentlandite | Pauli paramagnetic | YES (~5–10% error) | Mixer(0.05) |
| Mackinawite | AF, T_N=65K | YES (paramagnetic at 25C) | MixerDif(0.02) |
| Pyrite | Diamagnetic | YES | Mixer(0.05) |
| **Greigite** | **Ferrimagnetic (3.5 µB)** | **NO — garbage** | Mixer(0.05) |
| **Troilite** | **AF, T_N~315C** | **NO — garbage** | MixerDif(0.02) |
| **Cubanite** | **FM, T~250–300C** | **NO — garbage** | Mixer(0.05) |
| **Pyrrhotite** | **Ferrimagnetic** | **NO — garbage** | Mixer(0.05) |
| Marcasite | AF, T_N~65K | YES (paramagnetic at 25C) | MixerDif(0.02) |

---

## 6. Operational Lessons (Vast.ai / Cloud)

### Launching on Vast.ai

```bash
# Optimal onstart one-liner
export TZ=Europe/Kyiv OMP_NUM_THREADS=$(nproc) MKL_NUM_THREADS=$(nproc) OPENBLAS_NUM_THREADS=$(nproc)
git clone --depth 1 https://github.com/exopoiesis/digital-twin.git /workspace/digital-twin
cp /workspace/digital-twin/tools/*.sh /workspace/digital-twin/tools/*.py /workspace/ 2>/dev/null
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

### Diagnosing a stuck process

1. `ps aux --sort=-pcpu | head -5` — is python3 alive?
2. `find /workspace/results -mmin -30` — writing anything?
3. `/proc/PID/fd/ | grep .txt` — is log fd open? If fd closed but process is R → **STUCK**
4. `cat /proc/PID/environ | tr '\0' '\n' | grep OMP` — verify OMP env

### Log rotation (mandatory on restart)

```bash
cp /workspace/results/site.txt /workspace/results/site.txt.prev
# Only then start new process
```

---

## 7. Crystallographic Traps

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

## 8. Pre-deploy Checklist

- [ ] `crystal()` → `len(atoms)` == expected
- [ ] Stoichiometry == mineral formula
- [ ] `min_distance > 1.5 A`
- [ ] Wyckoff positions cross-checked with CIF
- [ ] `set_initial_magnetic_moments()` for Fe/Ni/Co
- [ ] `check_min_distance()` after rattle/strain/adsorbate
- [ ] `txt=f'/workspace/results/{label}.txt'` (NOT None — needed for SCF diagnostics)
- [ ] kpts: slabs (x,x,1), bulk (x,x,x). is_slab check FIRST
- [ ] maxiter=500+ for adsorption
- [ ] No em-dash `—` in code
- [ ] Vacuum >= 12 A for slabs
- [ ] MixerDif(0.02) for AFM slabs
- [ ] `set -eo pipefail` in bash wrapper
- [ ] `python3 -u` (unbuffered)
- [ ] For FD/MPI: OMP=1, MKL=1, OPENBLAS=1

---

## License

This guide is part of the [Third Matter](https://exopoiesis.space) project.
Released under the same license as the parent repository.

*Last updated: 2026-03-21*
