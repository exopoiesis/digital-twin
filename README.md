# Third Matter — Digital Twin (ORACLE)

Computational digital twin of a self-maintaining mineral membrane protocell.
Uses Sobol sampling, stochastic Gillespie simulations, Fourier Neural Operators (FNO),
and Physics-Informed Neural Networks (PINN) to map the viability landscape of the
TM6v3 architecture (pentlandite + mackinawite membrane, two-chamber design).

## Repository structure

```
oracle/                 — ORACLE pipeline (Phases A → B → C → D)
  oracle/data/          — Precomputed datasets + trained models (~58 MB)
hypothesis-tester/      — CLI for rapid hypothesis testing against the trained surrogate
results/                — DFT NEB barriers and vacancy formation energies (per mineral)
tools/                  — DFT NEB scripts (GPAW, ABACUS, QE), config generators, monitoring
example_results/        — Small JSON summaries for ORACLE reference
```

**Included pretrained artifacts** (no GPU needed to get started):
- `oracle/data/oracle_full.npz` — 262K Sobol×Gillespie samples (29 MB)
- `oracle/data/oracle_membrane_50k.npz` — 50K PDE solutions (3.2 MB)
- `oracle/data/oracle_membrane_fno.pt` — trained FNO surrogate (13 MB)
- `oracle/data/oracle_membrane_pinn.pt` — PINN fine-tuned model (13 MB)
- `oracle/data/oracle_phase_b_scalers.pkl` — FNO input scalers (6 KB)

## Key results

### ORACLE (surrogate modeling)

| Metric | Value |
|--------|-------|
| Sobol × Gillespie samples | 262,144 |
| RF classifier AUC | 0.982 |
| Most critical parameter | kd_A (formiate degradation), importance 0.36 |
| Steady-state [formate] A* (PDE) | 2.29 mM |
| k_cat decomplexation range | 7 orders of magnitude (1e-8 → 0.1 s⁻¹) |
| Phase C 72h survival (4-species) | 100% |

### DFT cross-verification (H⁺ diffusion barriers)

Multi-code NEB calculations on iron sulfide minerals using GPAW, ABACUS, Quantum ESPRESSO, and MACE-MP-0.
Full data in `results/` (one JSON per mineral).

| Mineral | GPAW | ABACUS | QE | MACE | Consensus |
|---------|:----:|:------:|:--:|:----:|-----------|
| **Pentlandite** (Fe,Ni)₉S₈ | 1.115 eV | 0.900 eV | — | 0.96 eV | **0.9–1.1 eV** (strong proton barrier) |
| **Mackinawite** FeS (intra-layer) | 0.738 eV | — | 2.479 eV* | 0.44 eV | **0.7 eV** lattice / **~0 eV** Grotthuss |
| **Pyrite** FeS₂ | 0.181 eV | 0.187 eV | 0.190 eV | 0.79 eV | **0.18–0.19 eV** (three-code, ±5%) |
| **Troilite** FeS | — | — | 0.375 eV† | 0.31 eV | ~0.37 eV (running) |

\* QE mackinawite = cross-layer path (different from GPAW intra-layer). † Running, fmax=0.10.

| Mineral | E_vac (S-vacancy) | Method |
|---------|:-----------------:|--------|
| Pentlandite | 4.444 eV | ABACUS PW GPU |
| Mackinawite | 5.668 eV | ABACUS PW GPU |
| Pyrite | running | ABACUS PW GPU |

## Quick start

All pretrained models and datasets are included — no GPU needed to start exploring.

```bash
pip install -r requirements.txt

# Test a single hypothesis (uses pretrained FNO from oracle/data/)
python hypothesis-tester/oracle_hypothesis_tester.py --mode single --no-plot

# Sweep mackinawite thickness
python hypothesis-tester/oracle_hypothesis_tester.py --mode sweep \
  --param L_mack --range 5,200 --n 30

# Monte Carlo robustness check
python hypothesis-tester/oracle_hypothesis_tester.py --mode montecarlo \
  --vary L_mack --spread 0.3 --n 100
```

## Reproducing from scratch

If you want to regenerate everything from raw simulations:

```bash
# Phase A: Sobol × Gillespie sampling (~2h on 8 cores, CPU)
python oracle/oracle_phase_a.py --output oracle/data/oracle_full.npz

# Phase A analysis: train RF classifier, extract feature importances (~2 min, CPU)
# Produces oracle_rf_classifier.pkl (~1.1 GB) + summary plots
python oracle/oracle_analysis.py oracle/data/oracle_full.npz \
  --output-dir oracle/data/

# Phase B.1: generate PDE training data (~4h, GPU recommended)
python oracle/oracle_phase_b_datagen.py --output oracle/data/oracle_membrane_50k.npz --n_samples 50000

# Phase B.2: train FNO surrogate (~30 min, GPU)
python oracle/oracle_phase_b_train.py --data oracle/data/oracle_membrane_50k.npz \
  --output-dir oracle/data/

# Phase B.3: PINN fine-tuning (~20 min, GPU)
python oracle/oracle_phase_b3_pinn.py --data oracle/data/oracle_membrane_50k.npz \
  --pretrained oracle/data/oracle_membrane_fno.pt \
  --scalers oracle/data/oracle_phase_b_scalers.pkl \
  --output-dir oracle/data/

# Phase C: degradation analysis (~1h, CPU)
python oracle/oracle_phase_c_degradation.py

# Phase D: FNO-ODE geometry scans (~5 min, CPU)
python oracle/oracle_phase_d_fno_ode.py --validate --scan
```

See `oracle/README.md` and `hypothesis-tester/README.md` for details.

## DFT Computation Guides

Battle-tested guides for running DFT on iron-sulfide minerals in the cloud:

- **[GPAW](GPAW_COMPUTATION_GUIDE.md)** — PW/FD modes, GPU FFT, SolvationGPAW, Bug Hall of Fame (~220h lost, documented)
- **[ABACUS](ABACUS_COMPUTATION_GUIDE.md)** — PW GPU + LCAO, NEB, vacancy formation, nspin=2 caveats
- **[Quantum ESPRESSO](QE_COMPUTATION_GUIDE.md)** — CPU/GPU NEB, npool optimization (up to 13x speedup), AFM limitations
- **[JDFTx](JDFTX_COMPUTATION_GUIDE.md)** — CANDLE implicit solvation, metallic slab mixing, checkpointing lessons

Each covers hardware selection, real benchmarks with costs, known bugs, and SCF convergence recipes.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (GPU optional but recommended for FNO/PINN)
- See `requirements.txt`

## License

Code: MIT. Data files: CC-BY-4.0. See `LICENSE`.

## Citation

If you use this code, please cite:

> Morozov, I. (2026). Third Matter: A self-maintaining mineral membrane protocell.
> Preprint in preparation.

## Contact

Igor Morozov — igor@exopoiesis.space
