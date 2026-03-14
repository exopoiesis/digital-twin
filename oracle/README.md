# ORACLE Pipeline

**O**ptimal **R**ange **A**nalysis via **C**lassifier-**L**ed **E**xploration —
a four-phase pipeline to map the viability landscape of the TM6v3 protocell.

## Phases

### Phase A — Sobol × Gillespie (`oracle_phase_a.py`)
- 262,144 parameter combinations (Sobol quasi-random, 18 dimensions)
- Each sample: deterministic ODE + stochastic Gillespie (N=1000 molecules)
- Binary outcome: ALIVE (A* > R_c) vs DEAD
- Output: `oracle_sobol_results.npz` (~29 MB)

### Phase A — Analysis (`oracle_analysis.py`)
- Random Forest classifier (AUC = 0.982)
- Feature importance ranking: kd_A (0.36) >> k1 (0.13) > kd_m (0.10)
- Critical thresholds at 25°C: kd_A < 2.7e-4, k1 > 5.2e-5, kd_m < 1.15e-5
- Output: `oracle_rf_classifier.pkl` (~1.1 GB, reproducible in ~2 min)

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

## Usage

```bash
# Full pipeline (sequential)
python oracle_phase_a.py            # ~2h, 8 cores
python oracle_analysis.py           # ~2 min
python oracle_phase_b_datagen.py    # ~4h, GPU recommended
python oracle_phase_b_train.py      # ~30 min, GPU
python oracle_phase_b3_pinn.py      # ~20 min, GPU
python oracle_phase_d_fno_ode.py    # ~5 min
python oracle_phase_c_degradation.py  # ~1h
```

## Large files (not in repo, reproducible)

| File | Size | Reproduced by |
|------|------|---------------|
| `oracle_rf_classifier.pkl` | ~1.1 GB | `oracle_analysis.py` |
| `oracle_sobol_results.npz` | ~29 MB | `oracle_phase_a.py` |
| `fno_checkpoint.pt` | ~13 MB | `oracle_phase_b_train.py` |
