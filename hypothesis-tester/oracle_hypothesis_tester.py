#!/usr/bin/env python3
"""
ORACLE Hypothesis Tester — CLI
================================
Unified CLI for rapid hypothesis testing against the ORACLE digital twin.

Modes:
  single      — one geometry/param set → A*, alive, stability
  sweep       — sweep 1 parameter → A*(param) curve + CSV
  threshold   — binary search for minimum param value yielding A* > target
  grid2d      — 2D sweep → heatmap (any 2 parameters)
  montecarlo  — N random points around nominal (±spread) → survival rate

Examples:
  python oracle_hypothesis_tester.py --mode single --k_cat 1e-3
  python oracle_hypothesis_tester.py --mode sweep --param k_cat --range 1e-6,1e-1 --n 30
  python oracle_hypothesis_tester.py --mode threshold --param k_cat --target 1.0
  python oracle_hypothesis_tester.py --mode grid2d --param1 L_mack --range1 5,100 --param2 k_cat --range2 1e-6,1e-1 --n 20
  python oracle_hypothesis_tester.py --mode montecarlo --vary L_mack --spread 0.3 --n 1000

Author: Third Matter Research Project
Date: 2026-03-13
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure this directory is on path for package import
sys.path.insert(0, str(Path(__file__).parent))

from oracle_hypothesis.common import (
    PARAM_META, NOMINAL_MEMBRANE_PARAMS, make_sweep_values, query_single,
    NumpyEncoder, load_fno, print,
)
from oracle_hypothesis.sweep import run_sweep
from oracle_hypothesis.threshold import run_threshold
from oracle_hypothesis.grid2d import run_grid2d
from oracle_hypothesis.montecarlo import run_montecarlo
from oracle_hypothesis.plotting import plot_sweep, plot_grid2d, plot_montecarlo


def parse_range(s: str) -> tuple:
    """Parse 'min,max' string into (float, float)."""
    parts = s.split(',')
    return float(parts[0]), float(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description='ORACLE Hypothesis Tester — rapid digital twin queries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query with modified k_cat:
  python oracle_hypothesis_tester.py --mode single --k_cat 1e-3

  # Sweep k_cat over 30 points:
  python oracle_hypothesis_tester.py --mode sweep --param k_cat --range 1e-6,1e-1 --n 30

  # Find minimum k_cat for A* >= 1 mM:
  python oracle_hypothesis_tester.py --mode threshold --param k_cat --target 1.0

  # 2D grid of L_mack × k_cat:
  python oracle_hypothesis_tester.py --mode grid2d --param1 L_mack --range1 5,100 --param2 k_cat --range2 1e-6,1e-1 --n 20

  # Monte Carlo robustness (±30% L_mack):
  python oracle_hypothesis_tester.py --mode montecarlo --vary L_mack --spread 0.3 --n 1000
        """
    )

    parser.add_argument('--mode', required=True,
                        choices=['single', 'sweep', 'threshold', 'grid2d', 'montecarlo'],
                        help='Query mode')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: digital-twin/hypothesis_results/)')
    parser.add_argument('--tag', default=None,
                        help='Tag for output filenames')
    parser.add_argument('--device', default='cpu', help='Torch device')
    parser.add_argument('--model-path', default=None, help='Path to FNO .pt checkpoint')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')

    # Parameter overrides (for single mode and base params)
    for p in PARAM_META:
        parser.add_argument(f'--{p}', type=float, default=None,
                            help=f'Override {p} ({PARAM_META[p]["unit"]})')

    # Sweep / threshold
    parser.add_argument('--param', type=str, help='Parameter to sweep/threshold')
    parser.add_argument('--range', type=str, help='min,max for sweep/threshold')
    parser.add_argument('--n', type=int, default=20, help='Number of points')
    parser.add_argument('--target', type=float, help='Target A* (mM) for threshold mode')

    # Grid2D
    parser.add_argument('--param1', type=str, help='First parameter for grid2d')
    parser.add_argument('--range1', type=str, help='min,max for param1')
    parser.add_argument('--param2', type=str, help='Second parameter for grid2d')
    parser.add_argument('--range2', type=str, help='min,max for param2')

    # Monte Carlo
    parser.add_argument('--vary', type=str, help='Comma-separated params to vary')
    parser.add_argument('--spread', type=str, default='0.3',
                        help='Relative spread (single value or comma-separated per param)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent / 'hypothesis_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build base params with overrides
    base = dict(NOMINAL_MEMBRANE_PARAMS)
    for p in PARAM_META:
        val = getattr(args, p, None)
        if val is not None:
            base[p] = val

    # Load FNO once
    print("Loading FNO model...")
    model_path = args.model_path
    if model_path is None:
        model_path = str(Path(__file__).parent / 'fno_results' / 'oracle_membrane_fno.pt')

    model, scaler_params, scaler_ph, scaler_scalars, x_grid, param_order = load_fno(
        model_path, device=args.device
    )
    fno_scalers = (scaler_params, scaler_ph, scaler_scalars, x_grid, param_order)
    print("FNO loaded.")

    # Dispatch
    tag = args.tag or args.mode
    result = None

    if args.mode == 'single':
        print(f"\nBase params: {json.dumps(base, indent=2)}")
        result = query_single(base, model, fno_scalers, args.device)
        fname = f'{tag}_result.json'

    elif args.mode == 'sweep':
        if not args.param or not args.range:
            parser.error('--param and --range required for sweep mode')
        vmin, vmax = parse_range(args.range)
        values = make_sweep_values(args.param, vmin, vmax, args.n)
        result = run_sweep(args.param, values, model, fno_scalers, base, args.device)
        fname = f'{tag}_{args.param}_sweep.json'

        if not args.no_plot:
            plot_sweep(result, str(out_dir / f'{tag}_{args.param}_sweep.png'))

    elif args.mode == 'threshold':
        if not args.param or args.target is None:
            parser.error('--param and --target required for threshold mode')
        vmin, vmax = (None, None)
        if args.range:
            vmin, vmax = parse_range(args.range)
        result = run_threshold(args.param, args.target, model, fno_scalers, base,
                              args.device, vmin, vmax)
        fname = f'{tag}_{args.param}_threshold.json'

    elif args.mode == 'grid2d':
        if not all([args.param1, args.range1, args.param2, args.range2]):
            parser.error('--param1, --range1, --param2, --range2 required for grid2d')
        v1min, v1max = parse_range(args.range1)
        v2min, v2max = parse_range(args.range2)
        values1 = make_sweep_values(args.param1, v1min, v1max, args.n)
        values2 = make_sweep_values(args.param2, v2min, v2max, args.n)
        result = run_grid2d(args.param1, values1, args.param2, values2,
                           model, fno_scalers, base, args.device)
        fname = f'{tag}_{args.param1}_{args.param2}_grid2d.json'

        if not args.no_plot:
            plot_grid2d(result, str(out_dir / f'{tag}_{args.param1}_{args.param2}_grid2d.png'))

    elif args.mode == 'montecarlo':
        if not args.vary:
            parser.error('--vary required for montecarlo mode')
        vary_params = [p.strip() for p in args.vary.split(',')]
        spread_parts = [float(s.strip()) for s in args.spread.split(',')]
        if len(spread_parts) == 1:
            spreads = spread_parts * len(vary_params)
        else:
            spreads = spread_parts
        result = run_montecarlo(vary_params, spreads, args.n, model, fno_scalers,
                               base, args.device, args.seed)
        fname = f'{tag}_montecarlo.json'

        if not args.no_plot:
            plot_montecarlo(result, str(out_dir / f'{tag}_montecarlo.png'))

    # Save JSON
    if result is not None:
        out_path = out_dir / fname
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
        print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
