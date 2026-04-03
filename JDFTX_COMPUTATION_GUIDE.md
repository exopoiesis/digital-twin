# JDFTx DFT Computation: Lessons & Benchmarks

> A practical guide to running JDFTx calculations on cloud CPU infrastructure.
> Accumulated over ~41 AI-assisted sessions of the [Third Matter](https://exopoiesis.space) project (March–April 2026).
> Target systems: iron-sulfide mineral slabs (72–108 atoms), CANDLE implicit solvation, CO2RR adsorption.
> **All claims verified against actual production runs on Hetzner AX102.**
>
> This document is battle-tested: every bug, benchmark, and recommendation
> comes from actual production runs. Total compute lost to undocumented traps: ~$50, ~71 hours.

---

## 1. What JDFTx Excels At

JDFTx is the best open-source code for **slab electrochemistry with implicit solvation**.

| Feature | JDFTx | GPAW SolvationGPAW | Comment |
|---------|:------:|:------------------:|---------|
| CANDLE solvation | **Native, PW solver** | Add-on FD Poisson, buggy BFGS | CANDLE = state-of-the-art PCM for electrochem |
| Slab + electrolyte | **Built-in Coulomb truncation** | Manual setup | Proper 2D periodic boundary |
| Large Fe/Ni slabs (50+ atoms) | OK with tuned mixing | Charge sloshing | JDFTx mixing more controllable |
| GPU acceleration | PW basis only (requires custom build) | Yes (CuPy FFT) | JDFTx GPU not tested in production |
| Parallelism | MPI + threads | OMP (PW) / MPI (FD) | JDFTx scales well to 16 cores |
| LCAO initialization | Built-in, fast | External (LCAO mode) | Critical for large slabs |

**Our primary use case:** CANDLE slab calculation for CO2RR adsorption energies (formate/CO) on mackinawite (001).

---

## 2. Input File Format

JDFTx uses a single `.in` text file — **not ASE Python API**. All units are **Hartree/Bohr** unless stated.

```
# ============================================================
# Minimal mackinawite slab + CANDLE solvation template
# ============================================================

# --- Cell (COLUMNS = lattice vectors, units: Bohr) ---
# 1 Angstrom = 1.889726 Bohr
lattice \
    6.94489  0.00000  0.00000 \
    0.00000  6.94489  0.00000 \
    0.00000  0.00000  50.0000

# --- Atom positions (fractional) ---
coords-type Lattice
ion Fe  0.0  0.0  0.23  1        # element x y z movable(0/1)
ion S   0.0  0.5  0.26  1
# ... (repeat for all atoms)

# --- Pseudopotentials: GBRV USPPs (built-in via $ID template) ---
ion-species GBRV/$ID_pbe_v1.2.uspp
ion-species GBRV/$ID_pbe_v1.01.uspp
ion-species GBRV/$ID_pbe_v1.uspp

# --- Basis set (Hartree, not Ry!) ---
elec-cutoff 20 100               # wfn=20 Ha(=40 Ry), density=100 Ha(=200 Ry)

# --- k-points: k_z=1 mandatory for slabs ---
kpoint-folding 4 4 1

# --- Spin: collinear z-spin ---
spintype z-spin
initial-magnetic-moments Fe 2 -2 2 -2  # per-atom, alternating for AFM

# --- Smearing: MANDATORY >= 0.03 Ha for metallic Fe slabs ---
elec-smearing Fermi 0.03         # 0.03 Ha = 0.816 eV. Default 0.01 = divergence!

# --- SCF: MUST tune mixing for >50 atoms ---
electronic-scf  nIterations 200 \
    energyDiffThreshold 1e-6 \
    mixFraction 0.1 \
    mixFractionMag 0.5 \
    qKerker 0.8 \
    history 20 \
    subspaceRotationFactor 0     # mandatory for metals

# --- LCAO initialization ---
lcao-params 100                  # nIterations (positional syntax, NOT lcao-params nIter=100)

# --- Slab Coulomb truncation (z-axis) ---
coulomb-interaction Slab 001
coulomb-truncation-embed 0 0 0
# center slab at z=0.5 fractional, vacuum >= 20 Ang each side

# --- CANDLE implicit solvation ---
fluid LinearPCM
pcm-variant CANDLE
fluid-solvent H2O
fluid-cation Na+ 0.1             # electrolyte concentration (mol/L)
fluid-anion Cl- 0.1

# --- DFT+U (if needed for magnetic minerals) ---
# add-U Fe d 0.07350             # U-J = 2 eV = 0.07350 Ha. Troilite, pyrrhotite

# --- Ionic relaxation ---
ionic-minimize nIterations 50

# --- Checkpointing: ALWAYS include both lines ---
dump-name slab_candle.$VAR
dump End State Ecomponents ElecDensity BandEigs  # wfns at end
dump Electronic State                            # wfns every 5 SCF iters (two-line form, portable)
dump-interval Electronic 5                       # NOTE: single-line "dump-interval Electronic 5 State" only works in custom build (PR #439)
initial-state slab_candle.$VAR                   # read checkpoint on restart
```

### Unit conversion table

| Parameter | JDFTx unit | Conversion from SI |
|-----------|-----------|-------------------|
| `lattice` vectors | Bohr | 1 Å = 1.889726 Bohr |
| `elec-cutoff` | Hartree | 1 Ha = 2 Ry = 27.2114 eV |
| `elec-smearing` | Hartree | 0.03 Ha = 0.816 eV |
| `add-U` value | Hartree | 2.0 eV = 0.07350 Ha |
| `coords-type Lattice` | fractional | — |
| `coords-type Cartesian` | Bohr | — |

---

## 3. SCF Mixing for Metallic Slabs (Critical)

**Default JDFTx mixing is designed for small molecules and insulators. For Fe/Ni slabs > 50 atoms,
defaults cause catastrophic SCF divergence within 2–3 iterations.**

### What happens with defaults

```
Iteration 1: F = -272.5 Ha  (normal)
Iteration 2: F = -270.1 Ha  (residual growing)
Iteration 3: F = +2228.4 Ha  (+2500 Ha jump — diverged)
```

**31 CPU-hours lost** before this was diagnosed (session 64, mackinawite 108-atom slab).

### Working parameters for metallic Fe/Ni slabs

| Parameter | Default | For >50-atom Fe/Ni slabs | Why |
|-----------|:-------:|:------------------------:|-----|
| `mixFraction` | 0.5 | **0.1** | Defaults too aggressive; 0.1 damps charge oscillations |
| `mixFractionMag` | 1.5 | **0.5** | Magnetic channel even more unstable |
| `elec-smearing` | 0.01 | **0.03** (minimum for Fe) | Smearing stabilizes Fermi surface; 0.01 Ha too sharp |
| `qKerker` | 0.8 | 0.8 (OK) | Kerker preconditioner, default fine |
| `history` | 8 | **20** | More history = better extrapolation |
| `subspaceRotationFactor` | auto | **0** | Mandatory for metals: disables subspace rotation that destabilizes metals |

**Rule:** For ANY calculation with >50 Fe/Ni atoms, ALWAYS set `mixFraction<=0.1`, `mixFractionMag<=0.5`, `smearing>=0.03`.

---

## 4. LCAO Initialization

JDFTx uses LCAO (Linear Combination of Atomic Orbitals) to generate initial wavefunctions before the plane-wave SCF loop.

### Default LCAO is insufficient for large slabs

```
lcao-params 30          # DEFAULT: 30 iterations — not enough for 68+ atom slabs
lcao-params 100         # Use this. Positional syntax! NOT "lcao-params nIterations 100"
```

**Syntax warning:** `lcao-params` takes positional arguments: `[nIter] [Ediff] [smearingWidth]`.
Writing `lcao-params nIterations 100` is a **parse error**.

### LCAO progress monitoring

```bash
grep "LCAOMinimize: Iter" output.out | tail -20
```

Output example:
```
LCAOMinimize: Iter:   1  F: -272.1234  |grad|_K: 2.31e-05  alpha: 0.99  t[s]: 2487
LCAOMinimize: Iter:   2  F: -272.1456  |grad|_K: 1.87e-05  alpha: 1.12  t[s]: 4973
```

**Key fields:**
- `F` — energy (Hartree). Should decrease monotonically.
- `|grad|_K` — gradient norm. Oscillates around 1.5–2.5e-05 for large slabs — this is **normal**, do not kill.
- `t[s]` — cumulative wall time. On AX102: ~2500 s/iteration for 108-atom slab.
- `alpha` — step size. Stays near 1.0 when converging.

### Magnetic initialization for AFM slabs

```
spintype z-spin
initial-magnetic-moments Fe 2 -2 2 -2 2 -2  # alternating signs per Fe atom
```

JDFTx `initial-magnetic-moments` takes per-atom values in order. For an AFM slab with 6 Fe atoms,
list 6 values with alternating signs. Magnetic moments are in units of µB.

---

## 5. Checkpointing (Critical Lesson)

### The trap: `dump End` without `State`

The default output configuration in most JDFTx examples:

```
dump End Ecomponents          # WRONG: saves energies but NOT wavefunctions
```

This dumps components at the end of the run, but **`State` (wavefunctions) is not included**.
Without `.wfns`, there is **no restart possible** — LCAO must run from scratch.

**Session 56:** W1 instance was migrated to AX102 mid-run. No `.wfns` file.
Result: 40 CPU-hours of LCAO + early SCF **completely lost**.

### Correct checkpoint configuration

```
dump-name prefix.$VAR                   # $VAR = placeholder for file type extension
dump End State Ecomponents ElecDensity  # ALWAYS include State
dump Electronic State                   # Checkpoint state every 5 SCF iterations (two-line portable form)
dump-interval Electronic 5
initial-state prefix.$VAR               # Read checkpoint on restart
```

**`dump-interval` syntax (v1.7.0):** The inline syntax `dump-interval Electronic 5 State` causes a **parse error**
in stock JDFTx 1.7.0. Use the two-line form above (portable, works everywhere).
The single-line form only works in our **custom build (PR #439)**.

### Custom build vs stock

| Feature | Stock `/opt/jdftx-1.7.0/` | Custom `/workspace/jdftx-custom/build/jdftx` |
|---------|:-:|:-:|
| LCAO checkpoints (`dump-interval` during LCAO) | NO | **YES** |
| Emergency dump on SIGQUIT | NO | **YES** |
| Respects `OMP_NUM_THREADS` | NO (uses physical cores) | **YES** |
| Deployment | Pre-installed on AX102 | `wget tar.gz → cmake → make -j16` |

**Rule: Never use stock JDFTx 1.7.0 for multi-hour runs without custom build.**

### Restart from checkpoint

```
jdftx -i input.in -o output_restart.out
# JDFTx reads initial-state automatically if the file exists.
# The -d flag does NOT mean dry-run! It means "skip reading initial-state".
# Do NOT use -d when you want to restart.
```

---

## 6. CANDLE Solvation

CANDLE (Charge-Asymmetric Nonlocally Determined Local-Electric) is JDFTx's flagship implicit solvation model.
It outperforms PCM, VASPsol, and SolvationGPAW for electrochemical surface calculations.

### Why CANDLE beats SolvationGPAW for our use case

SolvationGPAW FD + BFGS = **unviable** (4+ hours per BFGS step, no tested checkpoint, Poisson stall).
See `GPAW_COMPUTATION_GUIDE.md` for details. JDFTx CANDLE + PW basis:
- PW Poisson solver (not FD stencil) → no multigrid level issues
- Native slab Coulomb truncation
- Cavity recomputes fast with PW (not FD grid convolution)

### Slab configuration

```
# Required for any surface calculation:
coulomb-interaction Slab 001         # 2D periodic, truncation along z
coulomb-truncation-embed 0 0 0       # embed point for truncation

# Slab geometry rules:
# - Center slab at z = 0.5 fractional
# - Vacuum >= 20 Ang on each side (total cell z >= slab + 40 Ang)
# - coulomb-truncation-ion-margin 1   (if using older syntax)
```

### Electrolyte parameters

```
fluid LinearPCM
pcm-variant CANDLE
fluid-solvent H2O
fluid-cation Na+ 0.1    # 0.1 mol/L NaCl electrolyte
fluid-anion Cl- 0.1
```

For pure water (no salt): omit `fluid-cation`/`fluid-anion`.
For acidic solutions (CO2RR): add `fluid-cation H3O+ 0.01` for pH ≈ 2.

### Tutorial reference

http://jdftx.org/MetalSurfaces.html — formate adsorption on Pt(111), analogous to our mackinawite case.

---

## 7. Hardware & Parallelism

### JDFTx is pure CPU

JDFTx GPU support exists (`-DENABLE_GPU=ON`) but requires custom compilation and is **not tested in production**.
All our production runs use CPU-only mode.

```bash
# Optimal for AX102 (16 physical cores):
export OMP_NUM_THREADS=16
mpirun -np 1 /workspace/jdftx-custom/build/jdftx -i input.in -o output.out

# Or with MPI parallelism (k-point parallel):
mpirun -np 4 /workspace/jdftx-custom/build/jdftx -i input.in -o output.out
```

**Note:** Stock JDFTx ignores `OMP_NUM_THREADS` and auto-detects physical cores.
The custom build (PR #439) respects the environment variable.

### AX102 benchmarks (Hetzner, AMD Ryzen 9 7950X3D)

| System | Atoms | LCAO/iter | SCF iter (converged) | Notes |
|--------|:-----:|:---------:|:--------------------:|-------|
| Mackinawite slab (test) | 8 | ~5 s | ~2 min total | GBRV, AFM, passes |
| Mackinawite CANDLE slab | 108 | **~2500 s** | ~20–40 min/SCF iter | Production target |
| Mackinawite CANDLE v1 | 108 | — | **diverged** | Default mixing |
| Mackinawite CANDLE v2 | 108 | ~2500 s/iter | running | Fixed mixing |

**AX102 vs W1 (i9-14900KF, 24 cores) per-thread:** AX102 is **2.17× faster**.
Reasons: 128 MB V-Cache, homogeneous cores (no E-cores), high IPC at 5.76 GHz.

### Process identification

```bash
ps aux | grep jdftx                  # Shows as "jdftx" (easy to find, unlike GPAW's python3)
ps aux --sort=-pcpu | head -5        # Verify CPU usage
cat /proc/PID/environ | tr '\0' '\n' | grep OMP   # Verify OMP_NUM_THREADS
```

### Log monitoring

```bash
tail -50 output.out
grep "LCAOMinimize: Iter" output.out | tail -5    # LCAO progress
grep "SCFMinimize: Iter"  output.out | tail -5    # SCF progress
grep "F ="                output.out | tail -5    # Energy convergence
grep "IonicMinimize"      output.out | tail -5    # Ionic relaxation steps
```

---

## 8. DFT+U for Magnetic Minerals

```
add-U Fe d 0.07350    # U-J = 2.0 eV = 0.07350 Ha (Dudarev simplified scheme)
```

JDFTx uses Dudarev DFT+U by default. The value is `(U - J)` in Hartree.

| Mineral | T_N / T_C | Required? | U value | Notes |
|---------|:---------:|:---------:|:-------:|-------|
| Troilite | T_N = 588 K | YES | 2.0 eV = 0.07350 Ha | Without U: Fe moment collapses |
| Pyrrhotite | T_C ~580 K | YES | 2.0 eV | Ferrimagnetic |
| Greigite | T_C >400 K | YES | **1.0 eV = 0.03675 Ha** | Lower U (Devey 2009) |
| Chalcopyrite | T_N = 823 K | YES | 2.0 eV | AFM, strong ordering |
| Mackinawite | T_N = 65 K | No (25°C > T_N) | — | Paramagnetic at room temperature |
| Pyrite | Diamagnetic | No | — | Non-magnetic |
| Pentlandite | Pauli paramagnetic | No | — | Weak paramagnetism |

---

## 9. Bug Hall of Fame

### Tier S: tens of hours lost

| # | Bug | Hours lost | Root cause | Fix |
|---|-----|:----------:|------------|-----|
| 1 | **`dump End` without `State`** | **40 h** | Default `dump End Ecomponents` saves energy but not wavefunctions. Instance migration = total loss. | **`dump End State Ecomponents`** + `dump-interval Electronic 5 State` |
| 2 | **Default mixFraction=0.5 for Fe slab** | **31 h** | SCF divergence within 3 iterations. F jumped +2500 Ha. | `mixFraction 0.1`, `mixFractionMag 0.5`, `smearing 0.03` |
| 3 | **`lcao-params nIterations 100` syntax** | **hours** | Parse error — positional syntax required. LCAO used default 30 iters. | `lcao-params 100` (positional, no keyword) |
| 4 | **Orphaned CWD** | data risk | Run directory deleted while JDFTx was active. Files written to orphaned inode → gone when process exits. | `chattr +i /workspace/rundir/`. Periodic backup: `cp /proc/PID/cwd/*.wfns /safe/` |

### Tier A: misleading behavior

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 5 | **`-d` flag misread as dry-run** | `-d` = skip reading initial-state, NOT dry-run. Full calculation runs. | Don't use `-d` when restarting. Check output: "Reading initial state" |
| 6 | **LCAO `\|grad\|_K` oscillation** | Gradient ~1.5–2.5e-05 never reaches 0 → looks stuck | Normal behavior for large slabs. Don't kill. Watch `F` decreasing. |
| 7 | **`dump-interval Electronic 5 State` inline** | Parse error in stock v1.7.0 | Two lines: `dump Electronic State` + `dump-interval Electronic 5` |
| 8 | **kspacing not rescaled for supercell** | kspacing=0.15 for 95-atom cell → 4 k-pts. Same for 24-atom cell → 26 k-pts = 6× more expensive | Scale kspacing by cell size. For NEB relative energies: kspacing=0.20–0.25 sufficient |
| 9 | **OMP_NUM_THREADS ignored by stock** | Set `OMP_NUM_THREADS=32` on 16-core machine → JDFTx uses 16 anyway | Check log: "Run totals: 1 processes, N threads". Use custom build to respect env |

### Meta-lesson

LCAO stage for large slabs (100+ atoms) is **the most vulnerable phase**: hours of computation,
no native checkpointing in stock binary, no output between iterations. Always use custom build
before starting any run expected to take >30 min.

---

## 10. Operational Patterns

### Backup process (mandatory for long LCAO runs)

```bash
# Run alongside main JDFTx process:
while true; do
    PID=$(pgrep -x jdftx)
    if [ -z "$PID" ]; then break; fi
    cp /proc/$PID/cwd/slab_candle.wfns /workspace/backup/ 2>/dev/null
    cp /proc/$PID/cwd/slab_candle.fillings /workspace/backup/ 2>/dev/null
    cp /proc/$PID/fd/18 /workspace/backup/output_live.out 2>/dev/null  # output fd
    sleep 60
done
```

Script: `/workspace/jdftx_fd_backup.sh` (already on AX102).

### Protecting finished artifacts

> **WARNING:** Do NOT apply `chattr +i` to the active run directory — JDFTx writes
> checkpoint files and output there continuously. An immutable directory will cause
> write errors and crash the run. Only apply `+i` to **completed** result directories
> or specific finished files.

```bash
# WRONG — breaks active run:
# chattr +i /workspace/candle_production/

# CORRECT — protect a finished result snapshot:
cp -r /workspace/candle_production /workspace/candle_production_DONE
chattr +i /workspace/candle_production_DONE/   # Prevent accidental rm -rf of results
# To remove protection later: chattr -i /workspace/candle_production_DONE/
```

### Script pattern

```bash
#!/bin/bash
set -eo pipefail

JDFTX=/workspace/jdftx-custom/build/jdftx
INPUT=/workspace/candle_production/slab_candle.in
OUTPUT=/workspace/candle_production/slab_candle.out

export OMP_NUM_THREADS=16   # AX102: 16 physical cores

# Start backup process
nohup bash /workspace/jdftx_fd_backup.sh >/dev/null 2>&1 &

# Run (absolute output path to survive orphaned CWD)
$JDFTX -i $INPUT -o $OUTPUT
```

### Pre-deploy checklist

- [ ] Custom build at `/workspace/jdftx-custom/build/jdftx` (not stock `/opt/jdftx-1.7.0/`)
- [ ] `mixFraction 0.1` + `mixFractionMag 0.5` + `elec-smearing Fermi 0.03`
- [ ] `subspaceRotationFactor 0` in `electronic-scf`
- [ ] `lcao-params 100` (positional syntax)
- [ ] `dump End State ...` — includes `State`
- [ ] `dump-interval Electronic 5` + `dump Electronic State` (two lines)
- [ ] `initial-state prefix.$VAR` for restart
- [ ] `coulomb-interaction Slab 001` for slabs
- [ ] Slab centered at z=0.5, vacuum >= 20 Ang each side
- [ ] `chattr +i` on run directory
- [ ] Backup script running alongside JDFTx
- [ ] Output to absolute path (`-o /workspace/results/output.out`)
- [ ] k-points verified: `kpoint-folding N N 1` for slabs, N=1 along z

---

## License

This guide is part of the [Third Matter](https://exopoiesis.space) project.
Released under CC-BY-4.0. If you find it useful for your own JDFTx calculations, please cite.

*Last updated: 2026-04-03, session 76*
*Verified against production runs on Hetzner AX102 (Ryzen 9 7950X3D, 128 GB DDR5)*
*JDFTx v1.7.0 (stock) + exopoiesis/jdftx PR #439 (custom build)*
