"""ORACLE mode: single-parameter sweep."""

from .common import query_single, PARAM_META, NOMINAL_MEMBRANE_PARAMS, print
import numpy as np


def run_sweep(param: str, values: np.ndarray, fno_model, fno_scalers,
              base_params: dict = None, device='cpu') -> dict:
    """Sweep one parameter, return results."""
    if base_params is None:
        base_params = dict(NOMINAL_MEMBRANE_PARAMS)

    results = []
    print(f"\n{'='*60}")
    print(f"SWEEP: {param} over {len(values)} values [{values[0]:.4e} .. {values[-1]:.4e}]")
    print(f"{'='*60}")

    for i, val in enumerate(values):
        mp = dict(base_params)
        mp[param] = float(val)
        print(f"\n  [{i+1}/{len(values)}] {param} = {val:.4e}", end='')

        r = query_single(mp, fno_model, fno_scalers, device, verbose=False)
        results.append(r)
        status = 'ALIVE' if r['alive'] else 'DEAD'
        print(f"  →  A* = {r['A_mM']:.4f} mM  [{status}]")

    # Summary
    A_vals = [r['A_mM'] for r in results]
    alive_count = sum(1 for r in results if r['alive'])
    print(f"\n  Summary: {alive_count}/{len(results)} alive, "
          f"A* range [{min(A_vals):.4f}, {max(A_vals):.4f}] mM")

    return {
        'mode': 'sweep',
        'param': param,
        'values': values.tolist(),
        'A_mM': A_vals,
        'alive': [r['alive'] for r in results],
        'stable': [r['stable'] for r in results],
        'fno_A_steady_mM': [r['fno_A_steady_mM'] for r in results],
        'base_params': base_params,
        'results': results,
    }
