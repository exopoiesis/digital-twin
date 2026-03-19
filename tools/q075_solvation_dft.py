#!/usr/bin/env python3
"""
Q-075/Q-091: Formate adsorption on mackinawite (001) with implicit solvation.

Parallelizable: each --step runs independently, results saved to JSON.
Merge results with --step summary.

Steps and dependencies:
  ref_slab      → independent (relax slab in solvent)
  ref_formate   → independent (formate in solvent box)
  site_ontop    → needs ref_slab (loads relaxed slab)
  site_bridge   → needs ref_slab
  site_hollow   → needs ref_slab
  summary       → needs all above (reads JSONs, computes E_ads)

Example — 3 instances:
  Instance 1: python3 -u q075_solvation_dft.py --step ref_slab,ref_formate
  Instance 2: python3 -u q075_solvation_dft.py --step site_ontop,site_bridge
  Instance 3: python3 -u q075_solvation_dft.py --step site_hollow

Or single machine:
  python3 -u q075_solvation_dft.py --step all
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import surface
from ase.constraints import FixAtoms
from ase.io import write, read
from ase.optimize import BFGS
from ase.spacegroup import crystal

from gpaw import FermiDirac
from gpaw.solvation import SolvationGPAW, get_HW14_water_kwargs


# ─── Configuration ───────────────────────────────────────────────────────────

GRID_SPACING = 0.18   # Å — real-space FD grid (SolvationGPAW requires FD, not PW)
KPTS_SLAB = (2, 2, 1)
KPTS_MOL = (1, 1, 1)  # molecule in box
FMAX = 0.05           # eV/Å for BFGS
MAX_STEPS_REF = 100
MAX_STEPS_SITE = 300
VACUUM_MOL = 8.0      # Å, box padding for isolated formate


# ─── Structure builders ──────────────────────────────────────────────────────

def build_mackinawite_slab():
    """Build mackinawite (001) 3x3x1 slab (72 atoms) + 12 Å vacuum."""
    mack = crystal(
        symbols=['Fe', 'S'],
        basis=[(0, 0, 0), (0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
    )
    slab = surface(mack, (0, 0, 1), layers=2, vacuum=12.0)
    slab = slab.repeat((3, 3, 1))
    return slab


def build_formate_in_box():
    """Build formate anion HCOO⁻ in a vacuum box."""
    # Formate geometry (planar, C2v)
    formate = Atoms(
        symbols='COOH',
        positions=[
            [0.000, 0.000, 0.000],   # C
            [0.000, 1.100, 0.640],    # O1
            [0.000, -1.100, 0.640],   # O2
            [0.000, 0.000, -1.100],   # H
        ],
    )
    formate.center(vacuum=VACUUM_MOL)
    formate.pbc = True
    return formate


def place_formate_on_slab(slab, site_name):
    """Place formate on relaxed slab at specified site. Returns combined atoms."""
    combined = slab.copy()
    z_max = slab.positions[:, 2].max()
    symbols = np.array(slab.get_chemical_symbols())

    if site_name == 'ontop':
        # On top of highest Fe atom
        fe_mask = symbols == 'Fe'
        fe_pos = slab.positions[fe_mask]
        top_fe = fe_pos[np.argmax(fe_pos[:, 2])]
        anchor = top_fe.copy()
        anchor[2] = z_max + 2.0  # 2.0 Å above surface

    elif site_name == 'bridge':
        # Between top Fe and top S
        fe_mask = symbols == 'Fe'
        s_mask = symbols == 'S'
        fe_pos = slab.positions[fe_mask]
        s_pos = slab.positions[s_mask]
        top_fe = fe_pos[np.argmax(fe_pos[:, 2])]
        top_s = s_pos[np.argmax(s_pos[:, 2])]
        anchor = (top_fe + top_s) / 2
        anchor[2] = z_max + 2.2

    elif site_name == 'hollow':
        # Hollow site — center of top-3 Fe atoms (by z)
        fe_mask = symbols == 'Fe'
        fe_pos = slab.positions[fe_mask]
        top3_idx = np.argsort(fe_pos[:, 2])[-3:]
        anchor = fe_pos[top3_idx].mean(axis=0)
        anchor[2] = z_max + 2.5

    else:
        raise ValueError(f"Unknown site: {site_name}")

    # Place formate: O-down bidentate orientation
    formate_positions = [
        anchor + [0.0, 0.0, 0.5],     # C
        anchor + [0.0, 1.1, -0.2],     # O1
        anchor + [0.0, -1.1, -0.2],    # O2
        anchor + [0.0, 0.0, 1.6],      # H
    ]
    formate = Atoms('COOH', positions=formate_positions)
    combined += formate
    return combined


# ─── Calculator setup ────────────────────────────────────────────────────────

def _get_solv_kwargs():
    """Build solvation kwargs with custom radii (reusable)."""
    solv_kwargs = get_HW14_water_kwargs()
    custom_radii = {
        'H': 1.20,
        'C': 1.70,
        'O': 1.52,
        'S': 1.80,
        'Fe': 2.00,
        'Ni': 1.63,
    }
    solv_kwargs['cavity'].effective_potential.atomic_radii = custom_radii
    return solv_kwargs


def make_solvation_calc(kpts, txt=None, restart=None):
    """Create SolvationGPAW calculator with HW14 water model.

    Uses FD (finite-difference) mode -- SolvationGPAW does NOT support PW.
    Custom vdW radii for Fe, S (not in HW14 defaults).

    If restart is given, attempts to load wavefunctions from .gpw checkpoint.
    Falls back to fresh calculator on any error.
    """
    solv_kwargs = _get_solv_kwargs()

    base_kwargs = dict(
        mode='fd',
        h=GRID_SPACING,
        xc='PBE',
        kpts=kpts,
        occupations=FermiDirac(0.1),
        convergence={'energy': 1e-5},
        txt=txt,
    )

    if restart and Path(restart).exists() and Path(restart).stat().st_size > 0:
        try:
            calc = SolvationGPAW(restart=restart, **base_kwargs, **solv_kwargs)
            print(f"  [checkpoint] Loaded wavefunctions from {restart}", flush=True)
            return calc
        except Exception as e:
            print(f"  [checkpoint] Restart failed ({e}), starting fresh", flush=True)

    calc = SolvationGPAW(**base_kwargs, **solv_kwargs)
    return calc


# ─── Step functions ──────────────────────────────────────────────────────────

def step_ref_slab(output_dir):
    """Relax mackinawite slab in implicit solvent."""
    print("=== ref_slab: relaxing slab in solvent ===", flush=True)
    t0 = time.time()

    slab = build_mackinawite_slab()

    # Fix bottom layer (lowest 50% of atoms by z)
    z_mid = (slab.positions[:, 2].max() + slab.positions[:, 2].min()) / 2
    fix_mask = slab.positions[:, 2] < z_mid
    slab.set_constraint(FixAtoms(mask=fix_mask))
    print(f"  Atoms: {len(slab)}, fixed: {fix_mask.sum()}", flush=True)

    calc = make_solvation_calc(KPTS_SLAB, txt=str(output_dir / 'ref_slab.txt'))
    slab.calc = calc

    opt = BFGS(slab, trajectory=str(output_dir / 'ref_slab.traj'),
               logfile=str(output_dir / 'ref_slab_bfgs.log'))
    opt.run(fmax=FMAX, steps=MAX_STEPS_REF)

    energy = slab.get_potential_energy()
    forces = slab.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))
    elapsed = time.time() - t0

    # Save relaxed structure
    write(output_dir / 'ref_slab_relaxed.xyz', slab, format='extxyz')

    result = {
        'step': 'ref_slab',
        'energy_eV': energy,
        'max_force_eV_A': float(max_force),
        'converged': max_force < FMAX,
        'n_steps': opt.nsteps,
        'n_atoms': len(slab),
        'time_s': elapsed,
    }
    save_json(output_dir / 'ref_slab.json', result)
    print(f"  E_slab_solv = {energy:.4f} eV, max|F| = {max_force:.4f}, "
          f"steps = {opt.nsteps}, time = {elapsed:.0f}s", flush=True)
    return result


def step_ref_formate(output_dir):
    """Optimize formate in implicit solvent box."""
    print("=== ref_formate: formate in solvent ===", flush=True)
    t0 = time.time()

    formate = build_formate_in_box()
    calc = make_solvation_calc(KPTS_MOL, txt=str(output_dir / 'ref_formate.txt'))
    formate.calc = calc

    opt = BFGS(formate, trajectory=str(output_dir / 'ref_formate.traj'),
               logfile=str(output_dir / 'ref_formate_bfgs.log'))
    opt.run(fmax=FMAX, steps=MAX_STEPS_REF)

    energy = formate.get_potential_energy()
    elapsed = time.time() - t0

    result = {
        'step': 'ref_formate',
        'energy_eV': energy,
        'converged': True,
        'n_steps': opt.nsteps,
        'time_s': elapsed,
    }
    save_json(output_dir / 'ref_formate.json', result)
    print(f"  E_formate_solv = {energy:.4f} eV, time = {elapsed:.0f}s", flush=True)
    return result


def step_site(site_name, output_dir):
    """Relax slab+formate at a given site in implicit solvent."""
    print(f"=== site_{site_name}: slab + formate in solvent ===", flush=True)
    t0 = time.time()

    # Load relaxed slab
    slab_xyz = output_dir / 'ref_slab_relaxed.xyz'
    if not slab_xyz.exists():
        print(f"  ERROR: {slab_xyz} not found. Run ref_slab first.", flush=True)
        return None

    slab = read(slab_xyz)
    combined = place_formate_on_slab(slab, site_name)

    # Fix bottom layer of slab (same criterion)
    z_mid = (slab.positions[:, 2].max() + slab.positions[:, 2].min()) / 2
    fix_mask_slab = slab.positions[:, 2] < z_mid
    # Extend mask for formate atoms (not fixed)
    n_formate = 4
    fix_mask = np.concatenate([fix_mask_slab, np.zeros(n_formate, dtype=bool)])
    combined.set_constraint(FixAtoms(mask=fix_mask))

    prefix = f'site_{site_name}'

    # Resume BFGS positions from trajectory (MPI-safe: rank 0 reads, broadcast)
    traj_path = output_dir / f'{prefix}.traj'
    from gpaw.mpi import world
    n_frames = np.array([0], dtype=int)
    last_pos = None
    if world.rank == 0 and traj_path.exists():
        try:
            from ase.io import Trajectory as Traj
            traj = Traj(str(traj_path))
            n_frames[0] = len(traj)
            if n_frames[0] > 0:
                last_pos = traj[-1].positions.copy()
            traj.close()
        except Exception:
            n_frames[0] = 0
    world.broadcast(n_frames, 0)
    if n_frames[0] > 0:
        if world.rank != 0:
            last_pos = np.zeros_like(combined.positions)
        world.broadcast(last_pos, 0)
        combined.set_positions(last_pos)
        if world.rank == 0:
            print(f"  Resumed positions from {traj_path} ({n_frames[0]} frames)", flush=True)

    # Try to resume SCF from checkpoint (wavefunctions), fall back to fresh
    ckpt_path = output_dir / f'{prefix}_checkpoint.gpw'
    calc = make_solvation_calc(
        KPTS_SLAB,
        txt=str(output_dir / f'{prefix}.txt'),
        restart=str(ckpt_path),
    )
    combined.calc = calc

    # Attach periodic checkpoint saving (every 5 SCF iters, min 3 min apart)
    try:
        import sys
        if '/workspace' not in sys.path:
            sys.path.insert(0, '/workspace')
        from gpaw_checkpoint import CheckpointManager
        ckpt_mgr = CheckpointManager(combined, ckpt_path, interval=5,
                                     min_save_interval=180)
        ckpt_mgr.attach_to_calc(calc)
    except Exception as e:
        if world.rank == 0:
            print(f"  [checkpoint] Attach failed ({e}), continuing without", flush=True)

    opt = BFGS(combined, trajectory=str(output_dir / f'{prefix}.traj'),
               logfile=str(output_dir / f'{prefix}_bfgs.log'))
    opt.run(fmax=FMAX, steps=MAX_STEPS_SITE)

    energy = combined.get_potential_energy()
    forces = combined.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))
    elapsed = time.time() - t0

    # Check formate integrity
    formate_atoms = combined[-n_formate:]
    pos = formate_atoms.positions
    co1 = np.linalg.norm(pos[1] - pos[0])  # C-O1
    co2 = np.linalg.norm(pos[2] - pos[0])  # C-O2
    ch = np.linalg.norm(pos[3] - pos[0])   # C-H
    z_surface = slab.positions[:, 2].max()
    height = pos[0, 2] - z_surface  # C height above surface

    intact = (1.1 < co1 < 1.5) and (1.1 < co2 < 1.5) and (0.9 < ch < 1.3)

    write(output_dir / f'{prefix}_relaxed.xyz', combined, format='extxyz')

    result = {
        'step': prefix,
        'site': site_name,
        'energy_eV': energy,
        'max_force_eV_A': float(max_force),
        'converged': max_force < FMAX,
        'n_steps': opt.nsteps,
        'n_atoms': len(combined),
        'time_s': elapsed,
        'formate_intact': intact,
        'height_A': float(height),
        'CO1_A': float(co1),
        'CO2_A': float(co2),
        'CH_A': float(ch),
    }
    save_json(output_dir / f'{prefix}.json', result)
    print(f"  E = {energy:.4f} eV, max|F| = {max_force:.4f}, intact = {intact}, "
          f"height = {height:.2f} Å, time = {elapsed:.0f}s", flush=True)
    return result


def step_summary(output_dir):
    """Read all JSONs and compute E_ads for each site."""
    print("=== summary ===", flush=True)

    slab_data = load_json(output_dir / 'ref_slab.json')
    formate_data = load_json(output_dir / 'ref_formate.json')

    if not slab_data or not formate_data:
        print("  ERROR: missing ref_slab.json or ref_formate.json", flush=True)
        return None

    E_slab = slab_data['energy_eV']
    E_form = formate_data['energy_eV']
    E_ref = E_slab + E_form

    print(f"  E_slab_solv  = {E_slab:.4f} eV", flush=True)
    print(f"  E_formate_solv = {E_form:.4f} eV", flush=True)
    print(f"  E_ref (sum)  = {E_ref:.4f} eV\n", flush=True)

    sites = {}
    for site_name in ['ontop', 'bridge', 'hollow']:
        data = load_json(output_dir / f'site_{site_name}.json')
        if not data:
            print(f"  site_{site_name}: NOT FOUND", flush=True)
            continue

        E_ads = data['energy_eV'] - E_ref
        data['E_ads_eV'] = E_ads

        # Verdict
        abs_E = abs(E_ads)
        if abs_E < 0.6:
            verdict = 'PASS (weak adsorption, easy desorption)'
        elif abs_E < 1.5:
            verdict = 'MARGINAL (moderate adsorption)'
        else:
            verdict = 'FAIL (strong adsorption, poisoning risk)'

        data['verdict'] = verdict
        sites[site_name] = data

        print(f"  {site_name}: E_ads = {E_ads:.4f} eV → {verdict}", flush=True)
        print(f"    intact={data['formate_intact']}, height={data['height_A']:.2f} Å, "
              f"time={data['time_s']:.0f}s", flush=True)

    # Overall
    if sites:
        best_site = min(sites, key=lambda s: sites[s]['E_ads_eV'])
        best_E_ads = sites[best_site]['E_ads_eV']
        print(f"\n  Most stable: {best_site} (E_ads = {best_E_ads:.4f} eV)", flush=True)
    else:
        best_site = None
        best_E_ads = None

    summary = {
        'method': 'DFT + implicit solvation (HW14 water)',
        'code': 'GPAW SolvationGPAW',
        'xc': 'PBE',
        'mode': 'fd',
        'grid_spacing_A': GRID_SPACING,
        'solvent': 'water (eps=78.36, HW14)',
        'E_slab_solv_eV': E_slab,
        'E_formate_solv_eV': E_form,
        'sites': sites,
        'best_site': best_site,
        'best_E_ads_eV': best_E_ads,
    }
    save_json(output_dir / 'q075_solvation_summary.json', summary)
    print(f"\n  Saved: {output_dir / 'q075_solvation_summary.json'}", flush=True)
    return summary


# ─── Utilities ───────────────────────────────────────────────────────────────

def _sanitize(obj):
    """Recursively convert numpy types to native Python for JSON serialization.
    Works with numpy 2.x where np.bool_.__name__ == 'bool'."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(path, data):
    """Save JSON with numpy-safe serialization. MPI-safe: only rank 0 writes."""
    try:
        from gpaw.mpi import world
        if world.rank != 0:
            return
    except ImportError:
        pass
    with open(path, 'w') as f:
        json.dump(_sanitize(data), f, indent=2)


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── Main ────────────────────────────────────────────────────────────────────

STEP_MAP = {
    'ref_slab': lambda d: step_ref_slab(d),
    'ref_formate': lambda d: step_ref_formate(d),
    'site_ontop': lambda d: step_site('ontop', d),
    'site_bridge': lambda d: step_site('bridge', d),
    'site_hollow': lambda d: step_site('hollow', d),
    'summary': lambda d: step_summary(d),
}

ALL_STEPS = ['ref_slab', 'ref_formate', 'site_ontop', 'site_bridge', 'site_hollow', 'summary']


def main():
    parser = argparse.ArgumentParser(
        description='Q-075: Formate adsorption on mackinawite with implicit solvation')
    parser.add_argument('--step', type=str, default='all',
                        help='Comma-separated steps: ref_slab, ref_formate, '
                             'site_ontop, site_bridge, site_hollow, summary, all')
    parser.add_argument('--output-dir', type=str, default='/workspace/results/q075_solv',
                        help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.step == 'all':
        steps = ALL_STEPS
    else:
        steps = [s.strip() for s in args.step.split(',')]

    print(f"Q-075 solvation DFT", flush=True)
    print(f"  Steps: {steps}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Mode: FD (h={GRID_SPACING} Å)", flush=True)
    print(f"  Solvation: HW14 water (eps=78.36)\n", flush=True)

    for step_name in steps:
        if step_name not in STEP_MAP:
            print(f"  Unknown step: {step_name}", flush=True)
            continue

        # Skip if already done
        json_path = output_dir / f'{step_name}.json'
        if json_path.exists():
            print(f"  {step_name}: SKIPPED (already done)", flush=True)
            continue

        try:
            STEP_MAP[step_name](output_dir)
        except Exception as e:
            print(f"  {step_name}: FAILED -- {e}", flush=True)
            traceback.print_exc()
            # Save error
            save_json(output_dir / f'{step_name}_error.json', {
                'step': step_name, 'error': str(e),
            })

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
