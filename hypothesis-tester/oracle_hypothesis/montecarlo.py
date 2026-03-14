"""ORACLE mode: Monte Carlo robustness sampling."""

import time
from .common import query_single, PARAM_META, NOMINAL_MEMBRANE_PARAMS, print
import numpy as np


def run_montecarlo(vary_params: list, spreads: list, n: int,
                   fno_model, fno_scalers, base_params: dict = None,
                   device='cpu', seed: int = 42) -> dict:
    """
    Monte Carlo sampling around nominal.

    vary_params: list of param names to vary
    spreads: relative spread for each (0.3 = ±30%)
    n: number of samples
    """
    if base_params is None:
        base_params = dict(NOMINAL_MEMBRANE_PARAMS)

    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"MONTE CARLO: {n} samples")
    for p, s in zip(vary_params, spreads):
        print(f"  {p} = {base_params[p]:.4e} ± {s*100:.0f}%")
    print(f"{'='*60}")

    results = []
    t0 = time.time()

    for i in range(n):
        mp = dict(base_params)
        for p, s in zip(vary_params, spreads):
            nominal = base_params[p]
            meta = PARAM_META.get(p, {})
            if meta.get('log', False):
                # Log-normal sampling
                log_val = np.log(nominal) + rng.normal(0, s)
                mp[p] = float(np.exp(log_val))
            else:
                # Normal sampling, clip to positive
                mp[p] = float(max(nominal * (1 + rng.normal(0, s)), 1e-10))

        r = query_single(mp, fno_model, fno_scalers, device, verbose=False)
        results.append({
            'params': {p: mp[p] for p in vary_params},
            'A_mM': r['A_mM'],
            'alive': r['alive'],
        })

        if (i + 1) % 50 == 0 or i == n - 1:
            alive_so_far = sum(1 for x in results if x['alive'])
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (n - i - 1)
            print(f"  [{i+1}/{n}] alive={alive_so_far}/{i+1} "
                  f"({alive_so_far/(i+1)*100:.1f}%)  ETA {eta:.0f}s")

    alive_count = sum(1 for r in results if r['alive'])
    A_vals = [r['A_mM'] for r in results]

    print(f"\n  RESULT: {alive_count}/{n} alive ({alive_count/n*100:.1f}%)")
    print(f"  A* mean={np.mean(A_vals):.4f}, median={np.median(A_vals):.4f}, "
          f"std={np.std(A_vals):.4f} mM")

    return {
        'mode': 'montecarlo',
        'vary_params': vary_params,
        'spreads': spreads,
        'n': n,
        'seed': seed,
        'survival_rate': alive_count / n,
        'alive_count': alive_count,
        'A_mean_mM': float(np.mean(A_vals)),
        'A_median_mM': float(np.median(A_vals)),
        'A_std_mM': float(np.std(A_vals)),
        'A_min_mM': float(np.min(A_vals)),
        'A_max_mM': float(np.max(A_vals)),
        'results': results,
        'base_params': base_params,
    }
