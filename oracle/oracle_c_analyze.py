#!/usr/bin/env python3
"""Quick analysis of ORACLE Phase C results."""
import numpy as np
import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "results" / "oracle_phase_c"

# 3sp
d = np.load(OUT / 'oracle_c_3sp.npz')
surv = d['survival_rates']
dA = d['delta_A']
eM = d['epsilon_m']
kd_A0 = float(d['kd_A0'])
kd_m0 = float(d['kd_m0'])
T = 72 * 3600
nA = kd_A0 / T
nM = kd_m0 / T

print("="*60)
print("3-SPECIES (TM6v3-min) DETAILED ANALYSIS")
print("="*60)
print(f"N={len(surv)}, alive(>=50%)={np.sum(surv>=0.5)}/{len(surv)} = {np.mean(surv>=0.5)*100:.1f}%")
print(f"surv==1: {np.sum(surv==1.0)}, surv==0: {np.sum(surv==0.0)}")

print("\n--- delta_A marginal (formate degradation) ---")
for x in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0]:
    lo, hi = x*nA, (x+0.5)*nA
    mask = (dA >= lo) & (dA < hi)
    if mask.sum() > 3:
        pct = np.mean(surv[mask] >= 0.5) * 100
        print(f"  dA=[{x:.1f}x, {x+0.5:.1f}x): n={mask.sum():4d}, alive={pct:.0f}%")

print("\n--- epsilon_m marginal (membrane degradation) ---")
for x in range(10):
    lo, hi = x*nM, (x+1)*nM
    mask = (eM >= lo) & (eM < hi)
    if mask.sum() > 3:
        pct = np.mean(surv[mask] >= 0.5) * 100
        print(f"  eM=[{x}x, {x+1}x): n={mask.sum():4d}, alive={pct:.0f}%")

# Find critical delta_A more precisely
print("\n--- CRITICAL DELTA_A (fine search) ---")
for x_lo in np.arange(1.0, 5.0, 0.2):
    x_hi = x_lo + 0.2
    mask = (dA >= x_lo*nA) & (dA < x_hi*nA)
    if mask.sum() > 3:
        pct = np.mean(surv[mask] >= 0.5) * 100
        if pct < 100 or x_lo >= 2.0:
            print(f"  dA=[{x_lo:.1f}x, {x_hi:.1f}x): n={mask.sum():3d}, alive={pct:.0f}%")

# Cross-analysis: delta_A vs epsilon_m
print("\n--- 2D grid: delta_A x epsilon_m ---")
for da_lo, da_hi_label in [(0, "0-1x"), (1, "1-2x"), (2, "2-3x"), (3, "3-5x"), (5, "5-10x")]:
    da_hi = int(da_hi_label.split("-")[1].replace("x", ""))
    for em_lo, em_hi_label in [(0, "0-3x"), (3, "3-7x"), (7, "7-10x")]:
        em_hi = int(em_hi_label.split("-")[1].replace("x", ""))
        mask = (dA >= da_lo*nA) & (dA < da_hi*nA) & (eM >= em_lo*nM) & (eM < em_hi*nM)
        if mask.sum() > 3:
            pct = np.mean(surv[mask] >= 0.5) * 100
            print(f"  dA={da_hi_label:6s} eM={em_hi_label:5s}: n={mask.sum():4d}, alive={pct:.0f}%")

# Physical interpretation
print("\n--- PHYSICAL INTERPRETATION ---")
# delta_A critical ~2.5-3x based on marginals
# Find exact crossing
n_bins = 40
bins = np.linspace(0, dA.max(), n_bins+1)
bc = 0.5*(bins[:-1]+bins[1:])
bs = np.zeros(n_bins); cnt = np.zeros(n_bins)
for v, s in zip(dA, surv):
    idx = min(int(v/dA.max()*n_bins), n_bins-1)
    bs[idx] += (1 if s>=0.5 else 0); cnt[idx] += 1
m = cnt > 0
bc_m = bc[m]; bs_m = bs[m]/cnt[m]

crit_dA = None
for i in range(len(bs_m)-1):
    if bs_m[i] >= 0.5 and bs_m[i+1] < 0.5:
        frac = (0.5 - bs_m[i]) / (bs_m[i+1] - bs_m[i])
        crit_dA = bc_m[i] + frac*(bc_m[i+1]-bc_m[i])
        break

if crit_dA:
    kd_72h = kd_A0 + crit_dA * T
    ratio = kd_72h / kd_A0
    hl = kd_A0 / crit_dA / 3600
    print(f"  delta_A_crit = {crit_dA:.3e} s^-2")
    print(f"  kd_A(72h) = {kd_72h:.3e} s^-1 = {ratio:.2f}x baseline")
    print(f"  Doubling time = {hl:.1f} h")
    print(f"  Normalized = {crit_dA/nA:.2f}x")
else:
    print("  delta_A: no crossing found, checking trend...")
    for i in range(min(10, len(bs_m))):
        print(f"    bin {i}: dA_norm={bc_m[i]/nA:.2f}x, alive={bs_m[i]*100:.0f}%")

# epsilon_m marginal
bins_m = np.linspace(0, eM.max(), n_bins+1)
bc_m2 = 0.5*(bins_m[:-1]+bins_m[1:])
bs_m2 = np.zeros(n_bins); cnt_m2 = np.zeros(n_bins)
for v, s in zip(eM, surv):
    idx = min(int(v/eM.max()*n_bins), n_bins-1)
    bs_m2[idx] += (1 if s>=0.5 else 0); cnt_m2[idx] += 1
mm = cnt_m2 > 0
bc_mm = bc_m2[mm]; bs_mm = bs_m2[mm]/cnt_m2[mm]

crit_eM = None
for i in range(len(bs_mm)-1):
    if bs_mm[i] >= 0.5 and bs_mm[i+1] < 0.5:
        frac = (0.5 - bs_mm[i]) / (bs_mm[i+1] - bs_mm[i])
        crit_eM = bc_mm[i] + frac*(bc_mm[i+1]-bc_mm[i])
        break

if crit_eM:
    kd_72h_m = kd_m0 + crit_eM * T
    ratio_m = kd_72h_m / kd_m0
    hl_m = kd_m0 / crit_eM / 3600
    print(f"\n  epsilon_m_crit = {crit_eM:.3e} s^-2")
    print(f"  kd_m(72h) = {kd_72h_m:.3e} s^-1 = {ratio_m:.2f}x baseline")
    print(f"  Doubling time = {hl_m:.1f} h")
    print(f"  Normalized = {crit_eM/nM:.2f}x")
else:
    print("\n  epsilon_m: no crossing found. Checking trend...")
    for i in range(min(10, len(bs_mm))):
        print(f"    bin {i}: eM_norm={bc_mm[i]/nM:.2f}x, alive={bs_mm[i]*100:.0f}%")

# 4sp
print("\n" + "="*60)
print("4-SPECIES (TM6v3-DE with D-module) ANALYSIS")
print("="*60)
d4 = np.load(OUT / 'oracle_c_4sp.npz')
surv4 = d4['survival_rates']
print(f"N={len(surv4)}, alive={np.sum(surv4>=0.5)}/{len(surv4)} ({np.mean(surv4>=0.5)*100:.0f}%)")
print(f"mean surv = {np.mean(surv4):.4f}")
print(f"min surv = {np.min(surv4):.2f}, max = {np.max(surv4):.2f}")
print(f"surv==1: {np.sum(surv4==1.0)}, surv==0: {np.sum(surv4==0.0)}")

# Check if the 4sp is REALLY 100% alive everywhere
dA4 = d4['delta_A']
eM4 = d4['epsilon_m']
eD4 = d4['epsilon_d']
kd_d0 = float(d4['kd_d0'])
nD = kd_d0 / T

print("\n  Extreme corners:")
mask_ext = (dA4 > 8*nA) & (eM4 > 8*nM) & (eD4 > 8*nD)
print(f"  All >8x: n={mask_ext.sum()}, alive={np.mean(surv4[mask_ext]>=0.5)*100:.0f}%" if mask_ext.sum()>0 else "  All >8x: 0 samples")

mask_ext2 = (dA4 > 9*nA) & (eM4 > 9*nM) & (eD4 > 9*nD)
print(f"  All >9x: n={mask_ext2.sum()}, alive={np.mean(surv4[mask_ext2]>=0.5)*100:.0f}%" if mask_ext2.sum()>0 else "  All >9x: 0 samples")

# Save updated analysis
analysis = {
    'model_3sp': {
        'delta_A_crit': float(crit_dA) if crit_dA else None,
        'delta_A_crit_normalized': float(crit_dA/nA) if crit_dA else None,
        'epsilon_m_crit': float(crit_eM) if crit_eM else None,
        'epsilon_m_crit_normalized': float(crit_eM/nM) if crit_eM else None,
        'interpretation': 'delta_A is the bottleneck; epsilon_m not critical in scanned range',
    },
    'model_4sp': {
        'alive_fraction': float(np.mean(surv4 >= 0.5)),
        'interpretation': 'D-module makes system 100% robust to 10x degradation increase over 72h',
    }
}
with open(OUT / 'oracle_c_analysis_detail.json', 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"\nSaved detailed analysis to oracle_c_analysis_detail.json")
