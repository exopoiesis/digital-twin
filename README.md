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
example_results/        — Small JSON summaries for reference
```

**Included pretrained artifacts** (no GPU needed to get started):
- `oracle/data/oracle_full.npz` — 262K Sobol×Gillespie samples (29 MB)
- `oracle/data/oracle_membrane_50k.npz` — 50K PDE solutions (3.2 MB)
- `oracle/data/oracle_membrane_fno.pt` — trained FNO surrogate (13 MB)
- `oracle/data/oracle_membrane_pinn.pt` — PINN fine-tuned model (13 MB)

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

```bash
pip install -r requirements.txt

# Phase A: generate Sobol samples + Gillespie (takes ~2h on 8 cores)
python oracle/oracle_phase_a.py

# Analyse Phase A results (RF classifier)
python oracle/oracle_analysis.py

# Hypothesis testing (requires trained FNO from Phase B)
python hypothesis-tester/oracle_hypothesis_tester.py --mode sweep \
  --param L_mack --min 5 --max 200 --n 30
```

See `oracle/README.md` and `hypothesis-tester/README.md` for details.

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
