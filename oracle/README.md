# ORACLE Pipeline

**O**ptimal **R**ange **A**nalysis via **C**lassifier-**L**ed **E**xploration —
a four-phase pipeline to map the viability landscape of the TM6v3 protocell.

## Architecture

```
Phase A (Sobol × Gillespie) ──→ RF classifier (feature importance, thresholds)
         │
         └─→ Phase B (PDE data → FNO → PINN) ──→ Phase D (FNO-ODE scans)
                                                        │
                                                        └─→ hypothesis-tester/
         Phase C (degradation analysis, standalone)
```

The hypothesis-tester uses **Phase B/D artifacts only** (FNO model + scalers).
The RF classifier from Phase A is an analytical tool — it identifies which
parameters matter most, but is not required for running predictions.

## Included in `data/`

| File | Size | Description |
|------|------|-------------|
| `oracle_full.npz` | 29 MB | 262K Sobol×Gillespie samples (Phase A) |
| `oracle_membrane_50k.npz` | 3.2 MB | 50K PDE solutions (Phase B.1) |
| `oracle_membrane_fno.pt` | 13 MB | Trained FNO surrogate (Phase B.2) |
| `oracle_membrane_pinn.pt` | 13 MB | PINN fine-tuned model (Phase B.3) |
| `oracle_phase_b_scalers.pkl` | 6 KB | FNO input scalers (Phase B.2) |
| `oracle_c_3sp.npz` | 77 KB | Phase C results (3-species) |
| `oracle_c_4sp.npz` | 113 KB | Phase C results (4-species) |

## Phases

### Phase A — Sobol × Gillespie (`oracle_phase_a.py`)
- 262,144 parameter combinations (Sobol quasi-random, 18 dimensions)
- Each sample: deterministic ODE + stochastic Gillespie (N=1000 molecules)
- Binary outcome: ALIVE (A* > R_c) vs DEAD
- Output: `data/oracle_full.npz` (~29 MB, included)

### Phase A — Analysis (`oracle_analysis.py`)
- Random Forest classifier (AUC = 0.982)
- Feature importance ranking: kd_A (0.36) >> k1 (0.13) > kd_m (0.10)
- Critical thresholds at 25°C: kd_A < 2.7e-4, k1 > 5.2e-5, kd_m < 1.15e-5
- Output: `oracle_rf_classifier.pkl` (~1.1 GB, **not included** — regenerate in ~2 min)

### Phase B — Surrogate model
1. **B.1 Data generation** (`oracle_phase_b_datagen.py`): PDE solutions on a membrane geometry grid
2. **B.2 FNO training** (`oracle_phase_b_train.py`): Fourier Neural Operator as fast PDE surrogate
3. **B.3 PINN fine-tuning** (`oracle_phase_b3_pinn.py`): Physics-Informed NN for boundary refinement

### Phase C — Degradation (`oracle_phase_c_degradation.py`)
- Time-dependent parameter degradation over 72h horizon
- Runner: `oracle_c_runner.py`, post-processing: `oracle_c_analyze.py`
- Result: 4-species model survives 100% at 72h (half-life 7.2h per degradation mode)

### Phase D — FNO-ODE geometry scans (`oracle_phase_d_fno_ode.py`)
- Couples FNO surrogate (membrane PDE) with ODE (reaction network)
- Scans geometry parameters (L_pent, L_mack, L_chamber, delta_pH)
- Result: A* = 2.29 mM (PDE) vs 4.43 mM (ODE well-mixed)

## Reproducing Phase A analysis

```bash
# Uses included data/oracle_full.npz, generates RF classifier + plots (~2 min, CPU)
python oracle_analysis.py data/oracle_full.npz --output-dir data/
```

## Not included (reproducible)

| File | Size | Reproduced by | Time |
|------|------|---------------|------|
| `oracle_rf_classifier.pkl` | ~1.1 GB | `oracle_analysis.py` | ~2 min, CPU |
