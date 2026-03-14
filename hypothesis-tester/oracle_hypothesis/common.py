"""
Common utilities for ORACLE hypothesis testing.

Contains: PARAM_META, make_sweep_values, query_single, flush print.
Re-exports NOMINAL_MEMBRANE_PARAMS and NumpyEncoder from oracle_phase_d_fno_ode.
"""

import builtins
import sys
import time
from pathlib import Path

import numpy as np

# Ensure oracle/ directory is on path for oracle_phase_d_fno_ode
_oracle_dir = str(Path(__file__).resolve().parents[2] / 'oracle')
if _oracle_dir not in sys.path:
    sys.path.insert(0, _oracle_dir)

from oracle_phase_d_fno_ode import (
    run_phase_d,
    load_fno,
    NOMINAL_MEMBRANE_PARAMS,
    PhaseDParams,
    NumpyEncoder,
)

# Flush print for Docker compatibility
_builtin_print = builtins.print
def print(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ============================================================================
# PARAMETER METADATA
# ============================================================================

PARAM_META = {
    # Membrane (Phase D) — input units as expected by run_phase_d
    'L_pent':          {'unit': 'nm',    'log': False, 'nominal': 350.0},
    'L_mack':          {'unit': 'nm',    'log': False, 'nominal': 35.0},
    'L_chamber':       {'unit': 'um',    'log': False, 'nominal': 10.0},
    'delta_pH':        {'unit': 'pH',    'log': False, 'nominal': 4.5},
    'D_H_pent':        {'unit': 'm2/s',  'log': True,  'nominal': 5e-27},
    'D_H_mack_intra':  {'unit': 'm2/s',  'log': True,  'nominal': 1e-10},
    'k_cat':           {'unit': 's-1',   'log': True,  'nominal': 1e-4},
}


def make_sweep_values(param: str, vmin: float, vmax: float, n: int) -> np.ndarray:
    """Generate n values between vmin and vmax (log-spaced if param is log-scale)."""
    meta = PARAM_META.get(param, {})
    if meta.get('log', False) or (vmin > 0 and vmax / vmin > 100):
        return np.geomspace(vmin, vmax, n)
    return np.linspace(vmin, vmax, n)


# ============================================================================
# CORE: SINGLE QUERY
# ============================================================================

def query_single(membrane_params: dict, fno_model, fno_scalers, device='cpu',
                 ode_params=None, verbose=True) -> dict:
    """Run a single Phase D query. Returns result dict."""
    t0 = time.time()
    result = run_phase_d(
        membrane_params,
        ode_params=ode_params,
        fno_model=fno_model,
        fno_scalers=fno_scalers,
        device=device,
    )
    dt = time.time() - t0

    A_mM = result['ode_steady_state_mM']['A_mM']
    alive = result['alive']
    stable = result['stability']['stable']

    if verbose:
        print(f"\n  Result: A* = {A_mM:.4f} mM | Alive: {alive} | Stable: {stable} | {dt:.1f}s")

    return {
        'membrane_params': membrane_params,
        'A_mM': A_mM,
        'alive': alive,
        'stable': stable,
        'fno_A_steady_mM': result['fno_prediction']['A_steady_mM'],
        'fno_rate': result['fno_rate'],
        'ode_steady_state_mM': result['ode_steady_state_mM'],
        'stability': result['stability'],
        'elapsed_s': dt,
    }
