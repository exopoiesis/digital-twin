"""ORACLE mode: 2D parameter grid sweep."""

import time
from .common import query_single, PARAM_META, NOMINAL_MEMBRANE_PARAMS, print
import numpy as np


def run_grid2d(param1: str, values1: np.ndarray, param2: str, values2: np.ndarray,
               fno_model, fno_scalers, base_params: dict = None, device='cpu') -> dict:
    """2D parameter sweep → heatmap data."""
    if base_params is None:
        base_params = dict(NOMINAL_MEMBRANE_PARAMS)

    n1, n2 = len(values1), len(values2)
    print(f"\n{'='*60}")
    print(f"GRID2D: {param1} [{values1[0]:.4e}..{values1[-1]:.4e}] × "
          f"{param2} [{values2[0]:.4e}..{values2[-1]:.4e}]")
    print(f"  Total: {n1} × {n2} = {n1*n2} evaluations")
    print(f"{'='*60}")

    A_grid = np.zeros((n1, n2))
    alive_grid = np.zeros((n1, n2), dtype=bool)

    total = n1 * n2
    t0 = time.time()

    for i, v1 in enumerate(values1):
        for j, v2 in enumerate(values2):
            mp = dict(base_params)
            mp[param1] = float(v1)
            mp[param2] = float(v2)

            r = query_single(mp, fno_model, fno_scalers, device, verbose=False)
            A_grid[i, j] = r['A_mM']
            alive_grid[i, j] = r['alive']

            done = i * n2 + j + 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}] {param1}={v1:.3e}, {param2}={v2:.3e} "
                      f"→ A*={r['A_mM']:.3f} mM  (ETA {eta:.0f}s)")

    alive_pct = alive_grid.sum() / alive_grid.size * 100
    print(f"\n  Alive: {alive_grid.sum()}/{alive_grid.size} ({alive_pct:.1f}%)")
    print(f"  A* range: [{A_grid.min():.4f}, {A_grid.max():.4f}] mM")

    return {
        'mode': 'grid2d',
        'param1': param1, 'values1': values1.tolist(),
        'param2': param2, 'values2': values2.tolist(),
        'A_grid': A_grid.tolist(),
        'alive_grid': alive_grid.tolist(),
        'base_params': base_params,
        'alive_pct': alive_pct,
    }
