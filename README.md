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
tools/                  — Vast.ai launch/monitoring scripts, DFT config generators
example_results/        — Small JSON summaries for reference
```

**Included pretrained artifacts** (no GPU needed to get started):
- `oracle/data/oracle_full.npz` — 262K Sobol×Gillespie samples (29 MB)
- `oracle/data/oracle_membrane_50k.npz` — 50K PDE solutions (3.2 MB)
- `oracle/data/oracle_membrane_fno.pt` — trained FNO surrogate (13 MB)
- `oracle/data/oracle_membrane_pinn.pt` — PINN fine-tuned model (13 MB)
- `oracle/data/oracle_phase_b_scalers.pkl` — FNO input scalers (6 KB)

## Key results

| Metric | Value |
|--------|-------|
| Sobol × Gillespie samples | 262,144 |
| RF classifier AUC | 0.982 |
| Most critical parameter | kd_A (formiate degradation), importance 0.36 |
| Steady-state [formate] A* (PDE) | 2.29 mM |
| k_cat decomplexation range | 7 orders of magnitude (1e-8 → 0.1 s⁻¹) |
| Phase C 72h survival (4-species) | 100% |

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
python oracle/oracle_phase_b_datagen.py --output oracle/data/oracle_membrane_50k.npz

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

## GPAW DFT: Computation Guide

Running DFT on iron-sulfide minerals in the cloud? See **[GPAW_COMPUTATION_GUIDE.md](GPAW_COMPUTATION_GUIDE.md)** — a battle-tested reference covering:

- **Hardware selection** — why CPU GHz matters more than GPU for GPAW, and why desktop Intel is a trap for MPI
- **Real benchmarks** — timings for 10 minerals (8–136 atoms) across RTX 3060 to RTX 5070 Ti, with costs
- **Bug Hall of Fame** — 15 documented bugs that cost us ~220 hours, so you don't repeat them
- **SCF convergence recipes** — mixer settings, magnetic moments, and smearing by mineral type
- **Operational patterns** — monitoring, log rotation, MPI guards, and stuck-process diagnostics

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
