# ORACLE Hypothesis Tester

CLI tool for rapid hypothesis testing against the ORACLE digital twin.
Requires a trained FNO model from Phase B (see `oracle/README.md`).

## Modes

| Mode | Description |
|------|-------------|
| `single` | One geometry/param set → A*, alive, stability |
| `sweep` | Sweep 1 parameter → A*(param) curve + CSV |
| `threshold` | Binary search for min param value yielding A* > target |
| `grid2d` | 2D sweep → heatmap (any 2 parameters) |
| `montecarlo` | N random points around nominal (±spread) → survival rate |

## Usage

```bash
# Single query
python oracle_hypothesis_tester.py --mode single \
  --param L_mack --value 50

# Parameter sweep (e.g., mackinawite thickness 5–200 nm)
python oracle_hypothesis_tester.py --mode sweep \
  --param L_mack --min 5 --max 200 --n 30

# Find minimum L_mack for A* > 1 mM
python oracle_hypothesis_tester.py --mode threshold \
  --param L_mack --min 1 --max 200 --target 1.0

# 2D heatmap (L_mack × delta_pH)
python oracle_hypothesis_tester.py --mode grid2d \
  --param L_mack --min 5 --max 200 \
  --param2 delta_pH --min2 2 --max2 8 --n 15

# Monte Carlo robustness (100 random samples, ±30%)
python oracle_hypothesis_tester.py --mode montecarlo --n 100 --spread 0.3
```

## Parameters

| Parameter | Unit | Nominal | Scale |
|-----------|------|---------|-------|
| `L_pent` | nm | 350 | linear |
| `L_mack` | nm | 35 | linear |
| `L_chamber` | µm | 10 | linear |
| `delta_pH` | pH | 4.5 | linear |
| `D_H_pent` | m²/s | 5e-27 | log |
| `D_H_mack_intra` | m²/s | 1e-10 | log |
| `k_cat` | s⁻¹ | 1e-4 | log |

## Key findings (via this tool)

- **L_mack is the only lever**: 20 nm → 0.9 mM, 50 nm → 3.2 mM, 100 nm → 6.5 mM
- **k_cat fully decomplexed**: 30/30 ALIVE across 7 orders of magnitude
- **D_H_pent decomplexed**: no effect on A*
- **Monte Carlo ±30% L_mack**: 100% survival

## Architecture

```
oracle_hypothesis_tester.py   — CLI entry point (argparse)
oracle_hypothesis/
├── __init__.py
├── common.py                 — shared utils, imports from oracle/
├── sweep.py                  — sweep mode
├── threshold.py              — threshold mode
├── grid2d.py                 — grid2d mode
├── montecarlo.py             — montecarlo mode
└── plotting.py               — visualization helpers
```
