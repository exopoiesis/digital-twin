"""
ORACLE Hypothesis Testing Package
===================================
Modular interface to the ORACLE digital twin for hypothesis testing.

Usage:
    from oracle_hypothesis import query_single, run_sweep, run_threshold
    from oracle_hypothesis import run_grid2d, run_montecarlo
    from oracle_hypothesis import plot_sweep, plot_grid2d, plot_montecarlo
"""

from .common import PARAM_META, make_sweep_values, query_single, NOMINAL_MEMBRANE_PARAMS
from .sweep import run_sweep
from .threshold import run_threshold
from .grid2d import run_grid2d
from .montecarlo import run_montecarlo
from .plotting import plot_sweep, plot_grid2d, plot_montecarlo

__all__ = [
    'PARAM_META', 'NOMINAL_MEMBRANE_PARAMS',
    'make_sweep_values', 'query_single',
    'run_sweep', 'run_threshold', 'run_grid2d', 'run_montecarlo',
    'plot_sweep', 'plot_grid2d', 'plot_montecarlo',
]
