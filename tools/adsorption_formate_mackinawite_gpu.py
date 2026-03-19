#!/usr/bin/env python3
"""
Adsorption energy of formate (HCOO⁻) on mackinawite FeS (001) surface.
Uses MACE-MP-0 large model on GPU.

Crystal: P4/nmm (129), a=b=3.674 A, c=5.033 A, layered structure.
Slab: 3x3x1 supercell with 15 A vacuum on top.
Adsorbate: formate ion HCOO⁻ (H-C(=O)-O⁻)

Adsorption sites tested:
  (1) on-top Fe  — formate O directly above Fe atom
  (2) bridge Fe-S — formate between Fe and S atoms
  (3) hollow      — formate above hollow site (center of 4 Fe atoms)

E_ads = E(slab+HCOO) - E(slab) - E(HCOO_gas)

Criteria:
  |E_ads| < 0.6 eV  → easy desorption (PASS for catalysis)
  |E_ads| > 1.0 eV  → risk of catalyst poisoning (FAIL)

Output: q075_adsorption_formate_mackinawite.json, .png
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import numpy as np
from pathlib import Path

import torch
from ase import Atoms
from ase.spacegroup import crystal
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from mace.calculators import mace_mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/workspace/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- VRAM preflight ---
def _check_vram(required_gb):
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - torch.cuda.memory_allocated(0) / 1e9
        print(f"[VRAM] {total:.1f} GB total, {free:.1f} GB free, {required_gb:.1f} GB required")
        if free < required_gb:
            print(f"[VRAM] WARNING: only {free:.1f} GB free, need {required_gb:.1f} GB — risk of OOM")
_check_vram(2.0)


# ---------------------------------------------------------------------------
# Build structures
# ---------------------------------------------------------------------------

def build_mackinawite_slab():
    """Build mackinawite FeS (001) slab: 3x3x1 supercell + 15 A vacuum.

    Mackinawite: tetragonal, P4/nmm (129)
    a = b = 3.674 A, c = 5.033 A
    Fe at (0, 0, 0), S at (0, 0.5, 0.2602)

    Bottom 2 layers are fixed during relaxation.
    """
    unit = crystal(
        symbols=["Fe", "S"],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
    )
    # 3x3x1 supercell (1 unit cell in z — gives 2 Fe layers + 2 S layers)
    slab = unit.repeat((3, 3, 1))

    # Add vacuum: shift atoms to bottom, extend cell in z by 15 A
    positions = slab.get_positions()
    z_min = positions[:, 2].min()
    positions[:, 2] -= z_min  # shift to z=0
    slab.set_positions(positions)

    cell = slab.get_cell().copy()
    vacuum = 15.0  # A
    cell[2, 2] += vacuum
    slab.set_cell(cell)
    slab.pbc = [True, True, True]

    return slab


def get_layer_info(slab):
    """Identify layers and top-surface atoms for adsorption site placement.

    Returns dict with top Fe positions, top S positions, surface z-height.
    """
    symbols = slab.get_chemical_symbols()
    positions = slab.get_positions()

    fe_indices = [i for i, s in enumerate(symbols) if s == "Fe"]
    s_indices = [i for i, s in enumerate(symbols) if s == "S"]

    fe_z = positions[fe_indices, 2]
    s_z = positions[s_indices, 2]

    # Top Fe layer: atoms with z close to max Fe z
    fe_z_max = fe_z.max()
    top_fe_mask = np.abs(fe_z - fe_z_max) < 0.3
    top_fe_indices = [fe_indices[i] for i in range(len(fe_indices)) if top_fe_mask[i]]

    # Top S layer: atoms with z close to max S z
    s_z_max = s_z.max()
    top_s_mask = np.abs(s_z - s_z_max) < 0.3
    top_s_indices = [s_indices[i] for i in range(len(s_indices)) if top_s_mask[i]]

    # Bottom layers: everything NOT in top layers — will be frozen
    all_top = set(top_fe_indices) | set(top_s_indices)
    bottom_indices = [i for i in range(len(slab)) if i not in all_top]

    surface_z = max(fe_z_max, s_z_max)

    return {
        "top_fe_indices": top_fe_indices,
        "top_s_indices": top_s_indices,
        "bottom_indices": bottom_indices,
        "surface_z": surface_z,
        "top_fe_positions": positions[top_fe_indices],
        "top_s_positions": positions[top_s_indices],
    }


def build_formate():
    """Build isolated formate ion HCOO⁻ in a large box.

    Geometry: planar, C2v symmetry
      C at origin
      O1 at (+1.25, 0, 0) — double-bond-like
      O2 at (-1.25, 0, 0) — carboxylate
      H  at (0, +1.10, 0) — bonded to C

    Note: both C-O bonds in formate are equivalent (~1.25 A each)
    due to resonance. H-C ~ 1.10 A.

    Large box (20 A) to avoid periodic interactions.
    """
    formate = Atoms(
        symbols=["C", "O", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0],        # C (center)
            [1.246, 0.0, 0.0],       # O1
            [-1.246, 0.0, 0.0],      # O2
            [0.0, 1.097, 0.0],       # H
        ],
    )
    formate.set_cell([20.0, 20.0, 20.0])
    formate.center()
    formate.pbc = [True, True, True]
    return formate


def place_formate_on_slab(slab, site_xy, surface_z, height=2.0, orientation="bidentate"):
    """Place formate on slab at given xy position above surface.

    Args:
        slab: relaxed slab (will be copied)
        site_xy: (x, y) position of adsorption site
        surface_z: z-coordinate of topmost surface atom
        height: distance above surface (A)
        orientation: "bidentate" — both O atoms face surface (most common for
                     formate on metal sulfides), or "monodentate" — one O down.

    Returns:
        combined Atoms object (slab + formate, 4 new atoms at end)
    """
    combined = slab.copy()
    x0, y0 = site_xy
    z0 = surface_z + height

    if orientation == "bidentate":
        # Formate lying flat, both O atoms ~equal height, pointing toward surface
        # C slightly above O's, H pointing up
        # O-C-O angle ~ 130 deg, C-O ~ 1.25 A
        # Both O's at z0, C at z0+0.5, H at z0+1.6
        angle_half = 65 * np.pi / 180  # half of O-C-O angle
        d_CO = 1.25
        d_CH = 1.10
        o1_pos = [x0 + d_CO * np.sin(angle_half), y0, z0]
        o2_pos = [x0 - d_CO * np.sin(angle_half), y0, z0]
        c_pos = [x0, y0, z0 + d_CO * np.cos(angle_half)]
        h_pos = [x0, y0, c_pos[2] + d_CH]
    else:  # monodentate
        # One O pointing down, other O and H up
        c_pos = [x0, y0, z0 + 1.25]
        o1_pos = [x0, y0, z0]  # O toward surface
        o2_pos = [x0 + 1.25 * np.sin(130 * np.pi / 180), y0,
                  c_pos[2] + 1.25 * np.cos(130 * np.pi / 180)]
        h_pos = [x0 - 1.10 * np.sin(65 * np.pi / 180), y0,
                 c_pos[2] + 1.10 * np.cos(65 * np.pi / 180)]

    from ase import Atom
    combined.append(Atom("C", position=c_pos))
    combined.append(Atom("O", position=o1_pos))
    combined.append(Atom("O", position=o2_pos))
    combined.append(Atom("H", position=h_pos))

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()
    results = {
        "mineral": "mackinawite",
        "surface": "(001)",
        "model": "MACE-MP-0 large",
        "device": "cuda",
        "structure": "P4/nmm (129), layered FeS",
        "supercell": "3x3x1 + 15 A vacuum",
        "adsorbate": "formate HCOO-",
        "question": "Q-075: formate desorption energy",
    }

    # ── [1/8] GPU info ──────────────────────────────────────────────
    print("=" * 60)
    print("Q-075: Formate adsorption on mackinawite (001)")
    print("=" * 60)
    print("\n[1/8] GPU info")
    t0 = time.time()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")
        results["gpu"] = gpu_name
        results["vram_GB"] = round(vram_gb, 1)
    else:
        print("  WARNING: CUDA not available, falling back to CPU")
        results["gpu"] = "N/A (CPU fallback)"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [2/8] Load MACE-MP-0 large ─────────────────────────────────
    print("\n[2/8] Load MACE-MP-0 large on GPU")
    t0 = time.time()
    calc = mace_mp(model="large", device=device, default_dtype="float64")
    print(f"  loaded on {device} in {time.time()-t0:.1f}s")

    # ── [3/8] Build and relax slab ──────────────────────────────────
    print("\n[3/8] Build mackinawite (001) slab 3x3x1 + vacuum")
    t0 = time.time()
    slab_raw = build_mackinawite_slab()
    n_slab = len(slab_raw)
    formula_slab = slab_raw.get_chemical_formula()
    print(f"  Slab: {formula_slab}, {n_slab} atoms")
    print(f"  Cell: {slab_raw.cell.lengths()}")

    layer_info = get_layer_info(slab_raw)
    n_top_fe = len(layer_info["top_fe_indices"])
    n_top_s = len(layer_info["top_s_indices"])
    n_bottom = len(layer_info["bottom_indices"])
    print(f"  Top Fe atoms: {n_top_fe}, Top S atoms: {n_top_s}")
    print(f"  Bottom (frozen) atoms: {n_bottom}")
    print(f"  Surface z: {layer_info['surface_z']:.3f} A")

    results["n_slab_atoms"] = n_slab
    results["slab_formula"] = formula_slab
    results["cell_A"] = slab_raw.cell.lengths().tolist()
    results["n_frozen_atoms"] = n_bottom

    # Relax slab with bottom layers fixed
    slab = slab_raw.copy()
    slab.calc = calc
    slab.set_constraint(FixAtoms(indices=layer_info["bottom_indices"]))

    print(f"  Relaxing slab (BFGS, fmax=0.05, max 300 steps)...")
    opt_slab = BFGS(slab, logfile=None)
    opt_slab.run(fmax=0.05, steps=300)
    E_slab = slab.get_potential_energy()
    print(f"  E_slab = {E_slab:.4f} eV ({opt_slab.nsteps} steps)")

    results["E_slab_eV"] = float(E_slab)
    results["slab_relax_steps"] = opt_slab.nsteps

    # Update layer info after relaxation
    layer_info = get_layer_info(slab)
    print(f"  Surface z after relax: {layer_info['surface_z']:.3f} A")
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [4/8] Relax isolated formate ────────────────────────────────
    print("\n[4/8] Relax isolated formate in vacuum")
    t0 = time.time()
    formate_gas = build_formate()
    formate_gas.calc = calc
    print(f"  Formate: {formate_gas.get_chemical_formula()}, {len(formate_gas)} atoms")

    opt_formate = BFGS(formate_gas, logfile=None)
    opt_formate.run(fmax=0.01, steps=200)
    E_formate_gas = formate_gas.get_potential_energy()
    print(f"  E_formate_gas = {E_formate_gas:.4f} eV ({opt_formate.nsteps} steps)")

    # Verify geometry
    pos_f = formate_gas.get_positions()
    d_CO1 = np.linalg.norm(pos_f[0] - pos_f[1])
    d_CO2 = np.linalg.norm(pos_f[0] - pos_f[2])
    d_CH = np.linalg.norm(pos_f[0] - pos_f[3])
    print(f"  C-O1 = {d_CO1:.3f} A, C-O2 = {d_CO2:.3f} A, C-H = {d_CH:.3f} A")

    results["E_formate_gas_eV"] = float(E_formate_gas)
    results["formate_CO1_A"] = round(float(d_CO1), 3)
    results["formate_CO2_A"] = round(float(d_CO2), 3)
    results["formate_CH_A"] = round(float(d_CH), 3)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [5/8] Define adsorption sites ───────────────────────────────
    print("\n[5/8] Define adsorption sites")
    t0 = time.time()

    top_fe_pos = layer_info["top_fe_positions"]
    top_s_pos = layer_info["top_s_positions"]

    # Site 1: on-top Fe — directly above first top Fe atom
    site_ontop_fe = top_fe_pos[0, :2].copy()
    print(f"  Site 1 (on-top Fe): xy = ({site_ontop_fe[0]:.3f}, {site_ontop_fe[1]:.3f})")

    # Site 2: bridge Fe-S — midpoint between nearest Fe and S in top layer
    # Find nearest Fe-S pair
    min_d = 1e10
    best_fe_idx = 0
    best_s_idx = 0
    for i, fe_p in enumerate(top_fe_pos):
        for j, s_p in enumerate(top_s_pos):
            d = np.linalg.norm(fe_p[:2] - s_p[:2])
            if d < min_d:
                min_d = d
                best_fe_idx = i
                best_s_idx = j
    site_bridge = 0.5 * (top_fe_pos[best_fe_idx, :2] + top_s_pos[best_s_idx, :2])
    print(f"  Site 2 (bridge Fe-S): xy = ({site_bridge[0]:.3f}, {site_bridge[1]:.3f}), "
          f"Fe-S dist = {min_d:.3f} A")

    # Site 3: hollow — center of 4 nearest Fe atoms (square arrangement in P4/nmm)
    # In mackinawite (001), Fe atoms form a square lattice
    # Pick a Fe atom and find its 4 nearest Fe neighbors in the top layer
    ref_fe = top_fe_pos[0]
    fe_dists = np.linalg.norm(top_fe_pos[:, :2] - ref_fe[:2], axis=1)
    # Sort by distance, skip self (d=0), take next 4
    sorted_idx = np.argsort(fe_dists)
    # Include ref + 3 nearest neighbors to form a 4-atom square
    nn_fe_idx = sorted_idx[1:5]  # 4 nearest neighbors
    hollow_xy = np.mean(top_fe_pos[nn_fe_idx, :2], axis=0)
    # Check: also include ref to get center if hollow is at center of 4 atoms
    # Actually the hollow of 4 Fe in a square lattice is at the center
    four_fe = top_fe_pos[nn_fe_idx]
    print(f"  Site 3 (hollow): xy = ({hollow_xy[0]:.3f}, {hollow_xy[1]:.3f})")
    print(f"    4 Fe neighbors at distances: "
          f"{', '.join(f'{fe_dists[i]:.2f}' for i in nn_fe_idx)} A")

    sites = {
        "ontop_Fe": site_ontop_fe,
        "bridge_FeS": site_bridge,
        "hollow": hollow_xy,
    }

    results["adsorption_sites"] = {
        name: {"x": float(xy[0]), "y": float(xy[1])}
        for name, xy in sites.items()
    }
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [6/8] Relax slab+formate at each site ──────────────────────
    print("\n[6/8] Relax slab+formate at each adsorption site")
    t0_all = time.time()

    adsorption_results = {}
    E_ads_values = {}

    for site_name, site_xy in sites.items():
        print(f"\n  --- Site: {site_name} ---")
        t0 = time.time()

        # Place formate on slab (bidentate orientation — most common)
        combined = place_formate_on_slab(
            slab, site_xy, layer_info["surface_z"],
            height=2.0, orientation="bidentate"
        )
        n_total = len(combined)
        print(f"  Total atoms: {n_total}")

        # Set calculator (ASE .copy() does NOT copy calculator!)
        combined.calc = calc

        # Fix bottom layers of slab (same indices as before)
        combined.set_constraint(FixAtoms(indices=layer_info["bottom_indices"]))

        # Relax
        print(f"  Relaxing (BFGS, fmax=0.05, max 500 steps)...")
        opt = BFGS(combined, logfile=None)
        converged = opt.run(fmax=0.05, steps=500)
        E_combined = combined.get_potential_energy()
        print(f"  E(slab+HCOO) = {E_combined:.4f} eV ({opt.nsteps} steps, "
              f"converged: {converged})")

        # Adsorption energy
        E_ads = E_combined - E_slab - E_formate_gas
        print(f"  E_ads = {E_ads:.4f} eV")

        # Check formate integrity: verify C-O and C-H bonds still exist
        pos_final = combined.get_positions()
        symbols = combined.get_chemical_symbols()
        # Formate atoms are last 4: C, O, O, H
        i_C = n_total - 4
        i_O1 = n_total - 3
        i_O2 = n_total - 2
        i_H = n_total - 1
        d_CO1_final = np.linalg.norm(pos_final[i_C] - pos_final[i_O1])
        d_CO2_final = np.linalg.norm(pos_final[i_C] - pos_final[i_O2])
        d_CH_final = np.linalg.norm(pos_final[i_C] - pos_final[i_H])
        z_C = pos_final[i_C, 2]
        z_O1 = pos_final[i_O1, 2]
        z_O2 = pos_final[i_O2, 2]
        z_H = pos_final[i_H, 2]
        formate_z_min = min(z_C, z_O1, z_O2, z_H)
        height_above_surface = formate_z_min - layer_info["surface_z"]

        intact = (d_CO1_final < 1.6 and d_CO2_final < 1.6 and d_CH_final < 1.4)
        print(f"  Final geometry: C-O1={d_CO1_final:.3f}, C-O2={d_CO2_final:.3f}, "
              f"C-H={d_CH_final:.3f} A")
        print(f"  Height above surface: {height_above_surface:.3f} A")
        print(f"  Formate intact: {intact}")

        # Verdict
        abs_E_ads = abs(E_ads)
        if abs_E_ads < 0.6:
            verdict = "PASS (easy desorption)"
        elif abs_E_ads < 1.0:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL (risk of poisoning)"
        print(f"  Verdict: {verdict}")

        site_result = {
            "E_combined_eV": float(E_combined),
            "E_ads_eV": float(E_ads),
            "abs_E_ads_eV": float(abs_E_ads),
            "relax_steps": opt.nsteps,
            "converged": bool(converged),
            "formate_intact": intact,
            "final_CO1_A": round(float(d_CO1_final), 3),
            "final_CO2_A": round(float(d_CO2_final), 3),
            "final_CH_A": round(float(d_CH_final), 3),
            "height_above_surface_A": round(float(height_above_surface), 3),
            "verdict": verdict,
        }
        adsorption_results[site_name] = site_result
        E_ads_values[site_name] = E_ads

        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")

    results["sites"] = adsorption_results
    print(f"\n  All sites done in {time.time()-t0_all:.1f}s")

    # ── [7/8] Summary and analysis ──────────────────────────────────
    print("\n[7/8] Summary and analysis")
    t0 = time.time()

    # Find most stable site
    most_stable_site = min(E_ads_values, key=E_ads_values.get)
    E_ads_min = E_ads_values[most_stable_site]
    most_stable_abs = abs(E_ads_min)

    print(f"  Most stable site: {most_stable_site} (E_ads = {E_ads_min:.4f} eV)")

    # Overall verdict
    if most_stable_abs < 0.6:
        overall = "PASS — formate desorbs easily from all sites"
    elif most_stable_abs < 1.0:
        overall = "MARGINAL — moderate binding, needs experimental check"
    else:
        overall = "FAIL — strong binding, risk of catalyst poisoning"
    print(f"  Overall verdict: {overall}")

    results["most_stable_site"] = most_stable_site
    results["E_ads_most_stable_eV"] = float(E_ads_min)
    results["overall_verdict"] = overall

    # Context for Third Matter project
    results["context"] = (
        "For TM6v3 mackinawite catalyzes CO2->formate (R1). "
        "If |E_ads| > 1.0 eV, formate stays on surface = catalyst poisoning. "
        "If |E_ads| < 0.6 eV, formate desorbs into solution = catalytic turnover OK."
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # ── [8/8] Save results ──────────────────────────────────────────
    print("\n[8/8] Save results")
    t0 = time.time()

    results["total_time_s"] = round(float(time.time() - t_total), 1)

    # Save JSON (convert numpy types)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = RESULTS_DIR / "q075_adsorption_formate_mackinawite.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved {json_path}")

    # Save PNG — bar chart of E_ads for each site
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of adsorption energies
    ax1 = axes[0]
    site_names = list(E_ads_values.keys())
    E_vals = [E_ads_values[s] for s in site_names]
    colors = []
    for e in E_vals:
        ae = abs(e)
        if ae < 0.6:
            colors.append("green")
        elif ae < 1.0:
            colors.append("orange")
        else:
            colors.append("red")
    bars = ax1.bar(site_names, E_vals, color=colors, edgecolor="black", alpha=0.8)
    ax1.axhline(-0.6, color="green", linestyle="--", alpha=0.7, label="|E_ads|=0.6 eV (PASS)")
    ax1.axhline(0.6, color="green", linestyle="--", alpha=0.7)
    ax1.axhline(-1.0, color="red", linestyle="--", alpha=0.7, label="|E_ads|=1.0 eV (FAIL)")
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.7)
    ax1.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax1.set_ylabel("E_ads (eV)", fontsize=12)
    ax1.set_xlabel("Adsorption site", fontsize=12)
    ax1.set_title("Formate adsorption energy on mackinawite (001)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, E_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val,
                 f"{val:.3f}", ha="center",
                 va="bottom" if val > 0 else "top", fontsize=10, fontweight="bold")

    # Right: summary table
    ax2 = axes[1]
    ax2.axis("off")

    table_data = [["Site", "E_ads (eV)", "|E_ads|", "Verdict"]]
    for site_name in site_names:
        sr = adsorption_results[site_name]
        table_data.append([
            site_name,
            f"{sr['E_ads_eV']:.3f}",
            f"{sr['abs_E_ads_eV']:.3f}",
            sr["verdict"],
        ])
    table_data.append(["", "", "", ""])
    table_data.append(["OVERALL", "", "", overall.split(" — ")[0]])

    table = ax2.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.22, 0.18, 0.15, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Color header row
    for j in range(4):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Color verdict cells
    for i in range(1, len(site_names) + 1):
        verdict_text = table_data[i][3]
        if "PASS" in verdict_text:
            table[i, 3].set_facecolor("#C6EFCE")
        elif "MARGINAL" in verdict_text:
            table[i, 3].set_facecolor("#FFEB9C")
        elif "FAIL" in verdict_text:
            table[i, 3].set_facecolor("#FFC7CE")

    ax2.set_title(
        f"MACE-MP-0 large | Slab: 3x3x1 + 15 A vac\n"
        f"E_slab = {E_slab:.2f} eV | E_formate_gas = {E_formate_gas:.2f} eV",
        fontsize=10,
    )

    fig.suptitle(
        "Q-075: Formate (HCOO$^-$) desorption from mackinawite (001)\n"
        "MACE-MP-0 large",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    png_path = RESULTS_DIR / "q075_adsorption_formate_mackinawite.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {png_path}")

    print(f"[VRAM] Peak: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")

    print(f"\n{'=' * 60}")
    print(f"Total time: {results['total_time_s']:.1f}s")
    print(f"{'=' * 60}")
    print("Results summary:")
    print(f"  E_slab = {E_slab:.4f} eV")
    print(f"  E_formate_gas = {E_formate_gas:.4f} eV")
    for site_name in site_names:
        sr = adsorption_results[site_name]
        print(f"  {site_name}: E_ads = {sr['E_ads_eV']:.4f} eV "
              f"(|E_ads| = {sr['abs_E_ads_eV']:.4f}) — {sr['verdict']}")
    print(f"  Most stable: {most_stable_site} ({E_ads_min:.4f} eV)")
    print(f"  OVERALL: {overall}")
    print("=" * 60)


if __name__ == "__main__":
    main()
