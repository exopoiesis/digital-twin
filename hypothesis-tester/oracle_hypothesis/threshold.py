"""ORACLE mode: binary search for parameter threshold."""

from .common import query_single, PARAM_META, NOMINAL_MEMBRANE_PARAMS, print
import numpy as np


def run_threshold(param: str, target_A_mM: float, fno_model, fno_scalers,
                  base_params: dict = None, device='cpu',
                  vmin: float = None, vmax: float = None,
                  max_iter: int = 30, tol: float = 0.01) -> dict:
    """Binary search for minimum param value yielding A* >= target_A_mM."""
    if base_params is None:
        base_params = dict(NOMINAL_MEMBRANE_PARAMS)

    meta = PARAM_META.get(param, {})
    is_log = meta.get('log', False)

    # Default search bounds
    if vmin is None:
        vmin = meta['nominal'] * 0.001 if is_log else meta['nominal'] * 0.1
    if vmax is None:
        vmax = meta['nominal'] * 1000 if is_log else meta['nominal'] * 10

    print(f"\n{'='*60}")
    print(f"THRESHOLD: {param} for A* >= {target_A_mM:.2f} mM")
    print(f"  Search range: [{vmin:.4e}, {vmax:.4e}]")
    print(f"{'='*60}")

    # Determine direction: does increasing param increase or decrease A*?
    mp_lo = dict(base_params); mp_lo[param] = vmin
    mp_hi = dict(base_params); mp_hi[param] = vmax

    r_lo = query_single(mp_lo, fno_model, fno_scalers, device, verbose=False)
    r_hi = query_single(mp_hi, fno_model, fno_scalers, device, verbose=False)

    print(f"  {param}={vmin:.4e} → A*={r_lo['A_mM']:.4f} mM")
    print(f"  {param}={vmax:.4e} → A*={r_hi['A_mM']:.4f} mM")

    increasing = r_hi['A_mM'] > r_lo['A_mM']

    if increasing:
        lo, hi = vmin, vmax
    else:
        lo, hi = vmin, vmax

    history = []
    for i in range(max_iter):
        if is_log:
            mid = np.exp((np.log(lo) + np.log(hi)) / 2)
        else:
            mid = (lo + hi) / 2

        mp = dict(base_params); mp[param] = float(mid)
        r = query_single(mp, fno_model, fno_scalers, device, verbose=False)
        A = r['A_mM']
        history.append({'value': float(mid), 'A_mM': A, 'alive': r['alive']})
        print(f"  [{i+1}] {param}={mid:.4e} → A*={A:.4f} mM")

        if increasing:
            if A < target_A_mM:
                lo = mid
            else:
                hi = mid
        else:
            if A < target_A_mM:
                hi = mid
            else:
                lo = mid

        # Check convergence
        rel_gap = abs(hi - lo) / max(abs(hi), 1e-30)
        if rel_gap < tol:
            break

    # Final value
    if is_log:
        threshold_val = np.exp((np.log(lo) + np.log(hi)) / 2)
    else:
        threshold_val = (lo + hi) / 2

    print(f"\n  THRESHOLD: {param} = {threshold_val:.4e} for A* >= {target_A_mM:.2f} mM")

    return {
        'mode': 'threshold',
        'param': param,
        'target_A_mM': target_A_mM,
        'threshold_value': float(threshold_val),
        'unit': meta.get('unit', '?'),
        'increasing': increasing,
        'history': history,
        'base_params': base_params,
    }
