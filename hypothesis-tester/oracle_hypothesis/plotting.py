"""ORACLE plotting functions for hypothesis test results."""

import numpy as np
from .common import PARAM_META, print


def plot_sweep(data: dict, output_path: str):
    """Plot A*(param) curve from sweep results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    param = data['param']
    values = np.array(data['values'])
    A_vals = np.array(data['A_mM'])
    alive = np.array(data['alive'])
    meta = PARAM_META.get(param, {})

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by alive/dead
    ax.scatter(values[alive], A_vals[alive], c='green', s=40, label='Alive', zorder=5)
    ax.scatter(values[~alive], A_vals[~alive], c='red', s=40, label='Dead', zorder=5)
    ax.plot(values, A_vals, 'b-', alpha=0.5, linewidth=1)

    # Threshold line
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Death threshold (0.1 mM)')

    # Nominal marker
    nom = meta.get('nominal')
    if nom is not None and values.min() <= nom <= values.max():
        ax.axvline(x=nom, color='gray', linestyle=':', alpha=0.5, label=f'Nominal ({nom})')

    if meta.get('log', False) or (values.max() / values.min() > 100):
        ax.set_xscale('log')

    ax.set_xlabel(f'{param} ({meta.get("unit", "")})', fontsize=12)
    ax.set_ylabel('A* (mM)', fontsize=12)
    ax.set_title(f'ORACLE Sweep: A* vs {param}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_grid2d(data: dict, output_path: str):
    """Plot 2D heatmap from grid2d results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p1, p2 = data['param1'], data['param2']
    v1 = np.array(data['values1'])
    v2 = np.array(data['values2'])
    A_grid = np.array(data['A_grid'])
    alive_grid = np.array(data['alive_grid'])

    m1 = PARAM_META.get(p1, {})
    m2 = PARAM_META.get(p2, {})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap: A*
    ax = axes[0]
    im = ax.pcolormesh(v2, v1, A_grid, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label='A* (mM)')
    ax.set_xlabel(f'{p2} ({m2.get("unit", "")})')
    ax.set_ylabel(f'{p1} ({m1.get("unit", "")})')
    ax.set_title('A* (mM)')
    if m2.get('log') or (v2.max()/v2.min() > 100): ax.set_xscale('log')
    if m1.get('log') or (v1.max()/v1.min() > 100): ax.set_yscale('log')

    # Heatmap: alive/dead boundary
    ax = axes[1]
    im2 = ax.pcolormesh(v2, v1, alive_grid.astype(float), shading='auto',
                        cmap='RdYlGn', vmin=0, vmax=1)
    fig.colorbar(im2, ax=ax, label='Alive (1) / Dead (0)')
    # Contour at boundary
    try:
        ax.contour(v2, v1, A_grid, levels=[0.1], colors='red', linewidths=2)
    except Exception:
        pass
    ax.set_xlabel(f'{p2} ({m2.get("unit", "")})')
    ax.set_ylabel(f'{p1} ({m1.get("unit", "")})')
    ax.set_title('Alive/Dead boundary')
    if m2.get('log') or (v2.max()/v2.min() > 100): ax.set_xscale('log')
    if m1.get('log') or (v1.max()/v1.min() > 100): ax.set_yscale('log')

    fig.suptitle(f'ORACLE Grid2D: {p1} × {p2}', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_montecarlo(data: dict, output_path: str):
    """Plot histogram of A* from Monte Carlo."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    A_vals = [r['A_mM'] for r in data['results']]
    alive = [r['alive'] for r in data['results']]

    fig, ax = plt.subplots(figsize=(10, 6))
    A_alive = [a for a, al in zip(A_vals, alive) if al]
    A_dead = [a for a, al in zip(A_vals, alive) if not al]

    bins = np.linspace(min(A_vals), max(A_vals), 40)
    if A_alive:
        ax.hist(A_alive, bins=bins, color='green', alpha=0.7, label=f'Alive ({len(A_alive)})')
    if A_dead:
        ax.hist(A_dead, bins=bins, color='red', alpha=0.7, label=f'Dead ({len(A_dead)})')

    ax.axvline(x=0.1, color='red', linestyle='--', label='Death threshold')
    ax.set_xlabel('A* (mM)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    sr = data['survival_rate']
    ax.set_title(f'Monte Carlo: survival {sr*100:.1f}%, '
                 f'A* = {data["A_mean_mM"]:.3f} ± {data["A_std_mM"]:.3f} mM', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")
