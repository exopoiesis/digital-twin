#!/usr/bin/env python3
"""
ORACLE Phase A Analysis Script
Analyzes survival maps, fits GP/RF surrogate models, generates plots.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
import joblib

# Parameter defaults for fixing other params in 2D slices
PARAM_DEFAULTS = {
    'k1': 1e-4, 'Ka': 7e-4, 'f1': 5e-3, 'Km_f': 5e-4, 'km': 3e-2,
    'k_fe_gen': 5e-5, 'fe_supply': 5e-8, 'kd_A': 1e-4,
    'kd_m': 3e-6, 'kd_fe': 3e-4, 'N_A': 100, 'T_celsius': 25
}


def load_data(npz_path):
    """Load ORACLE Phase A results from NPZ file."""
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    print(f"  Keys: {list(data.keys())}")
    print(f"  Samples: {data['samples'].shape[0]}")
    return data


def merge_npz_files(files, output_path):
    """Merge multiple NPZ files into one."""
    print(f"Merging {len(files)} NPZ files...")
    all_data = [np.load(f) for f in files]

    # Verify consistency
    param_names = all_data[0]['param_names']
    param_ranges = all_data[0]['param_ranges']
    param_log_scale = all_data[0]['param_log_scale']

    for i, d in enumerate(all_data[1:], 1):
        if not np.array_equal(d['param_names'], param_names):
            raise ValueError(f"File {i}: param_names mismatch")
        if not np.allclose(d['param_ranges'], param_ranges):
            raise ValueError(f"File {i}: param_ranges mismatch")

    # Concatenate arrays
    merged = {
        'param_names': param_names,
        'param_ranges': param_ranges,
        'param_log_scale': param_log_scale,
        'samples': np.vstack([d['samples'] for d in all_data]),
        'survived': np.hstack([d['survived'] for d in all_data]),
        'ode_alive': np.hstack([d['ode_alive'] for d in all_data]),
        'a_star_ode': np.hstack([d['a_star_ode'] for d in all_data]),
        't_death_h': np.hstack([d['t_death_h'] for d in all_data]),
        'final_A': np.hstack([d['final_A'] for d in all_data]),
        'final_M': np.hstack([d['final_M'] for d in all_data]),
        'final_Fe': np.hstack([d['final_Fe'] for d in all_data]),
        'max_A_molecules': np.hstack([d['max_A_molecules'] for d in all_data]),
    }

    np.savez_compressed(output_path, **merged)
    print(f"Merged {merged['samples'].shape[0]} samples → {output_path}")
    return merged


def print_summary_stats(data):
    """Print summary statistics to stdout."""
    N = len(data['survived'])
    ode_alive_pct = 100 * data['ode_alive'].sum() / N
    gillespie_survived_pct = 100 * data['survived'].sum() / N

    print("\n" + "="*60)
    print("ORACLE PHASE A SUMMARY")
    print("="*60)
    print(f"Total samples: {N}")
    print(f"ODE alive: {ode_alive_pct:.2f}%")
    print(f"Gillespie survived (72h): {gillespie_survived_pct:.2f}%")

    # Breakdown by N_A and T_celsius
    param_names = [str(p) for p in data['param_names']]
    idx_NA = param_names.index('N_A')
    idx_T = param_names.index('T_celsius')

    N_A_vals = data['samples'][:, idx_NA]
    T_vals = data['samples'][:, idx_T]

    print("\nSurvival rate by N_A:")
    for NA_bin in [10, 30, 100, 300, 1000]:
        mask = np.abs(N_A_vals - NA_bin) < 1
        if mask.sum() > 0:
            surv_rate = 100 * data['survived'][mask].mean()
            print(f"  N_A={NA_bin}: {surv_rate:.1f}% ({mask.sum()} samples)")

    print("\nSurvival rate by temperature:")
    for T_bin in [25, 50, 75, 100]:
        mask = np.abs(T_vals - T_bin) < 5
        if mask.sum() > 0:
            surv_rate = 100 * data['survived'][mask].mean()
            print(f"  T={T_bin}°C: {surv_rate:.1f}% ({mask.sum()} samples)")

    return {
        'total_samples': int(N),
        'ode_alive_pct': float(ode_alive_pct),
        'gillespie_survived_pct': float(gillespie_survived_pct)
    }


def transform_samples(samples, param_log_scale):
    """Apply log transform to log-scaled parameters."""
    X = samples.copy()
    for i, is_log in enumerate(param_log_scale):
        if is_log:
            X[:, i] = np.log10(X[:, i])
    return X


def fit_random_forest(X, y, param_names, cv_subsample=50000):
    """Fit Random Forest classifier and compute feature importances."""
    print("\nFitting Random Forest...", flush=True)
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, max_depth=20)
    rf.fit(X, y)

    # Feature importances
    importances = dict(zip([str(p) for p in param_names], rf.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    print("\nTop-5 most influential parameters (RF):", flush=True)
    for i, (param, imp) in enumerate(sorted_imp[:5], 1):
        print(f"  {i}. {param}: {imp:.4f}")

    # Cross-validation on subsample to avoid OOM
    if len(X) > cv_subsample:
        print(f"\n5-fold cross-validation (RF, subsample {cv_subsample}/{len(X)})...", flush=True)
        idx = np.random.RandomState(42).choice(len(X), cv_subsample, replace=False)
        X_cv, y_cv = X[idx], y[idx]
    else:
        print("\n5-fold cross-validation (RF)...", flush=True)
        X_cv, y_cv = X, y

    cv_acc = cross_val_score(rf, X_cv, y_cv, cv=5, scoring='accuracy', n_jobs=1)
    cv_auc = cross_val_score(rf, X_cv, y_cv, cv=5, scoring='roc_auc', n_jobs=1)

    print(f"  Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}", flush=True)
    print(f"  AUC-ROC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}", flush=True)

    # Confusion matrix on subsample
    y_pred_cv = cross_val_predict(rf, X_cv, y_cv, cv=5, n_jobs=1)
    cm = confusion_matrix(y_cv, y_pred_cv)
    print(f"\nConfusion matrix:\n{cm}", flush=True)

    return rf, importances, {
        'rf_accuracy': float(cv_acc.mean()),
        'rf_accuracy_std': float(cv_acc.std()),
        'rf_auc': float(cv_auc.mean()),
        'rf_auc_std': float(cv_auc.std())
    }


def fit_gp_classifier(X, y, param_names, max_samples=10000):
    """Fit Gaussian Process classifier."""
    print("\nFitting GP Classifier...", flush=True)

    # Subsample if too large
    if len(X) > max_samples:
        print(f"  Subsampling to {max_samples} for GP fitting (O(N³) complexity)...", flush=True)
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_gp = X[idx]
        y_gp = y[idx]
    else:
        X_gp = X
        y_gp = y

    # Fit with RBF kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0]*X.shape[1], (1e-2, 1e2))
    gp = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=1, max_iter_predict=200)

    try:
        gp.fit(X_gp, y_gp)

        # Cross-validation on subsample
        print("  5-fold cross-validation (GP)...")
        cv_acc = cross_val_score(gp, X_gp, y_gp, cv=5, scoring='accuracy', n_jobs=1)
        try:
            cv_auc = cross_val_score(gp, X_gp, y_gp, cv=5, scoring='roc_auc', n_jobs=1)
            auc_mean = cv_auc.mean()
        except:
            print("    AUC-ROC failed (GP predict_proba issue), skipping")
            auc_mean = None

        print(f"  Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
        if auc_mean is not None:
            print(f"  AUC-ROC: {auc_mean:.4f}")

        # Length scales
        if hasattr(gp.kernel_, 'k2'):
            length_scales = gp.kernel_.k2.length_scale
            print(f"\n  Learned length scales (inverse = importance):")
            inv_ls = 1.0 / length_scales
            inv_ls_norm = inv_ls / inv_ls.sum()
            ls_imp = dict(zip([str(p) for p in param_names], inv_ls_norm))
            for param, imp in sorted(ls_imp.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {param}: {imp:.4f}")
        else:
            ls_imp = {}

        return gp, {
            'gp_accuracy': float(cv_acc.mean()),
            'gp_accuracy_std': float(cv_acc.std()),
            'gp_auc': float(auc_mean) if auc_mean is not None else None,
            'gp_length_scale_importances': ls_imp
        }

    except Exception as e:
        print(f"  GP fitting failed: {e}")
        print("  Falling back to RF only")
        return None, {'gp_accuracy': None, 'gp_auc': None}


def fit_gp_regressor(X, y_val, param_names, max_samples=10000):
    """Fit GP regressor for A* prediction (on alive samples only)."""
    print("\nFitting GP Regressor for A*...", flush=True)

    # Use log(A*) as target
    y_log = np.log10(y_val)

    # Subsample if needed
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_gp = X[idx]
        y_gp = y_log[idx]
    else:
        X_gp = X
        y_gp = y_log

    kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0]*X.shape[1], (1e-2, 1e2))
    gp_reg = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=2, alpha=1e-6)

    try:
        gp_reg.fit(X_gp, y_gp)
        print("  GP Regressor fitted successfully")
        return gp_reg
    except Exception as e:
        print(f"  GP Regressor fitting failed: {e}")
        return None


def plot_2d_survival_maps(gp, rf, data, output_dir, top_params, param_defaults):
    """Plot 2D survival probability maps for top parameter pairs."""
    print("\nGenerating 2D survival maps...")

    param_names = [str(p) for p in data['param_names']]
    param_log_scale = data['param_log_scale']
    param_ranges = data['param_ranges']

    X_transform = transform_samples(data['samples'], param_log_scale)

    # Top 4 parameters, 6 pairs
    top_4 = top_params[:4]
    pairs = [(top_4[i], top_4[j]) for i in range(4) for j in range(i+1, 4)]

    for param1, param2 in pairs:
        idx1 = param_names.index(param1)
        idx2 = param_names.index(param2)

        # Create grid
        if param_log_scale[idx1]:
            p1_vals = np.logspace(np.log10(param_ranges[idx1, 0]), np.log10(param_ranges[idx1, 1]), 50)
            p1_grid = np.log10(p1_vals)
        else:
            p1_vals = np.linspace(param_ranges[idx1, 0], param_ranges[idx1, 1], 50)
            p1_grid = p1_vals

        if param_log_scale[idx2]:
            p2_vals = np.logspace(np.log10(param_ranges[idx2, 0]), np.log10(param_ranges[idx2, 1]), 50)
            p2_grid = np.log10(p2_vals)
        else:
            p2_vals = np.linspace(param_ranges[idx2, 0], param_ranges[idx2, 1], 50)
            p2_grid = p2_vals

        P1, P2 = np.meshgrid(p1_grid, p2_grid)

        # Fix other parameters at defaults (transformed)
        X_grid = np.zeros((P1.size, len(param_names)))
        for i, pname in enumerate(param_names):
            default_val = param_defaults[pname]
            if param_log_scale[i]:
                X_grid[:, i] = np.log10(default_val)
            else:
                X_grid[:, i] = default_val

        X_grid[:, idx1] = P1.ravel()
        X_grid[:, idx2] = P2.ravel()

        # Predict with GP or RF
        if gp is not None:
            try:
                Z = gp.predict_proba(X_grid)[:, 1].reshape(P1.shape)
            except:
                Z = rf.predict_proba(X_grid)[:, 1].reshape(P1.shape)
        else:
            Z = rf.predict_proba(X_grid)[:, 1].reshape(P1.shape)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(p1_vals, p2_vals, Z, levels=20, cmap='RdYlGn', alpha=0.8)
        plt.colorbar(contour, label='Survival Probability', ax=ax)

        # Overlay actual samples
        mask1 = (data['samples'][:, idx1] >= param_ranges[idx1, 0]) & (data['samples'][:, idx1] <= param_ranges[idx1, 1])
        mask2 = (data['samples'][:, idx2] >= param_ranges[idx2, 0]) & (data['samples'][:, idx2] <= param_ranges[idx2, 1])
        mask = mask1 & mask2

        survived_mask = mask & (data['survived'] == 1)
        dead_mask = mask & (data['survived'] == 0)

        ax.scatter(data['samples'][survived_mask, idx1], data['samples'][survived_mask, idx2],
                   c='green', s=1, alpha=0.3, label='Survived')
        ax.scatter(data['samples'][dead_mask, idx1], data['samples'][dead_mask, idx2],
                   c='red', s=1, alpha=0.3, label='Dead')

        if param_log_scale[idx1]:
            ax.set_xscale('log')
        if param_log_scale[idx2]:
            ax.set_yscale('log')

        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f'ORACLE Survival Map: {param1} vs {param2}')
        ax.legend(markerscale=3)

        output_path = output_dir / f'oracle_survival_{param1}_vs_{param2}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {output_path.name}")


def plot_marginal_survival_curves(gp, rf, data, output_dir, param_defaults):
    """Plot 1D marginal survival curves for all parameters."""
    print("\nGenerating marginal survival curves...")

    param_names = [str(p) for p in data['param_names']]
    param_log_scale = data['param_log_scale']
    param_ranges = data['param_ranges']

    X_transform = transform_samples(data['samples'], param_log_scale)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()

    for i, pname in enumerate(param_names):
        ax = axes[i]

        # Binned observed survival rate
        if param_log_scale[i]:
            bins = np.logspace(np.log10(param_ranges[i, 0]), np.log10(param_ranges[i, 1]), 21)
            bin_centers = np.sqrt(bins[:-1] * bins[1:])  # Geometric mean
        else:
            bins = np.linspace(param_ranges[i, 0], param_ranges[i, 1], 21)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

        obs_rate, _, _ = binned_statistic(data['samples'][:, i], data['survived'], statistic='mean', bins=bins)

        ax.plot(bin_centers, obs_rate, 'o-', label='Observed', color='blue', alpha=0.7)

        # GP prediction (marginalized)
        if gp is not None or rf is not None:
            p_vals = bin_centers
            X_pred = np.zeros((len(p_vals), len(param_names)))
            for j, pname_j in enumerate(param_names):
                default_val = param_defaults[pname_j]
                if param_log_scale[j]:
                    X_pred[:, j] = np.log10(default_val)
                else:
                    X_pred[:, j] = default_val

            if param_log_scale[i]:
                X_pred[:, i] = np.log10(p_vals)
            else:
                X_pred[:, i] = p_vals

            if gp is not None:
                try:
                    pred_proba = gp.predict_proba(X_pred)[:, 1]
                    ax.plot(p_vals, pred_proba, '-', label='GP', color='orange', linewidth=2)
                except:
                    pred_proba = rf.predict_proba(X_pred)[:, 1]
                    ax.plot(p_vals, pred_proba, '-', label='RF', color='orange', linewidth=2)
            else:
                pred_proba = rf.predict_proba(X_pred)[:, 1]
                ax.plot(p_vals, pred_proba, '-', label='RF', color='orange', linewidth=2)

        if param_log_scale[i]:
            ax.set_xscale('log')
        ax.set_xlabel(pname)
        ax.set_ylabel('Survival Rate')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(pname, fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'oracle_marginal_survival.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path.name}")


def plot_astar_maps(gp_reg, data, output_dir, top_params, param_defaults):
    """Plot 2D A* heatmaps."""
    if gp_reg is None:
        print("\nSkipping A* maps (no GP regressor)")
        return

    print("\nGenerating A* heatmaps...")

    param_names = [str(p) for p in data['param_names']]
    param_log_scale = data['param_log_scale']
    param_ranges = data['param_ranges']

    top_4 = top_params[:4]
    pairs = [(top_4[i], top_4[j]) for i in range(4) for j in range(i+1, 4)]

    for param1, param2 in pairs:
        idx1 = param_names.index(param1)
        idx2 = param_names.index(param2)

        # Create grid (same as survival maps)
        if param_log_scale[idx1]:
            p1_vals = np.logspace(np.log10(param_ranges[idx1, 0]), np.log10(param_ranges[idx1, 1]), 50)
            p1_grid = np.log10(p1_vals)
        else:
            p1_vals = np.linspace(param_ranges[idx1, 0], param_ranges[idx1, 1], 50)
            p1_grid = p1_vals

        if param_log_scale[idx2]:
            p2_vals = np.logspace(np.log10(param_ranges[idx2, 0]), np.log10(param_ranges[idx2, 1]), 50)
            p2_grid = np.log10(p2_vals)
        else:
            p2_vals = np.linspace(param_ranges[idx2, 0], param_ranges[idx2, 1], 50)
            p2_grid = p2_vals

        P1, P2 = np.meshgrid(p1_grid, p2_grid)

        X_grid = np.zeros((P1.size, len(param_names)))
        for i, pname in enumerate(param_names):
            default_val = param_defaults[pname]
            if param_log_scale[i]:
                X_grid[:, i] = np.log10(default_val)
            else:
                X_grid[:, i] = default_val

        X_grid[:, idx1] = P1.ravel()
        X_grid[:, idx2] = P2.ravel()

        try:
            Z_log = gp_reg.predict(X_grid).reshape(P1.shape)
            Z = 10**Z_log  # Convert back from log10(A*)

            fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(p1_vals, p2_vals, 1000*Z, levels=20, cmap='viridis', alpha=0.8)
            plt.colorbar(contour, label='A* (mM)', ax=ax)

            if param_log_scale[idx1]:
                ax.set_xscale('log')
            if param_log_scale[idx2]:
                ax.set_yscale('log')

            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_title(f'ORACLE A* Map: {param1} vs {param2}')

            output_path = output_dir / f'oracle_astar_{param1}_vs_{param2}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {output_path.name}")
        except Exception as e:
            print(f"  Failed to plot A* map for {param1} vs {param2}: {e}")


def plot_feature_importance(rf_importances, gp_ls_importances, output_dir):
    """Plot feature importance comparison (RF vs GP length scales)."""
    print("\nGenerating feature importance plot...")

    params = list(rf_importances.keys())
    rf_imp = [rf_importances[p] for p in params]

    if gp_ls_importances:
        gp_imp = [gp_ls_importances.get(p, 0) for p in params]
    else:
        gp_imp = [0] * len(params)

    # Sort by RF importance
    sorted_idx = np.argsort(rf_imp)[::-1]
    params_sorted = [params[i] for i in sorted_idx]
    rf_sorted = [rf_imp[i] for i in sorted_idx]
    gp_sorted = [gp_imp[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(params))

    ax.barh(y_pos - 0.2, rf_sorted, 0.4, label='RF Importance', color='steelblue')
    if any(g > 0 for g in gp_sorted):
        ax.barh(y_pos + 0.2, gp_sorted, 0.4, label='GP Length Scale⁻¹ (norm)', color='darkorange')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(params_sorted)
    ax.set_xlabel('Importance')
    ax.set_title('ORACLE Feature Importance: RF vs GP')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    output_path = output_dir / 'oracle_feature_importance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path.name}")


def plot_survival_NA_vs_T(gp, rf, data, output_dir):
    """Heatmap: survival vs N_A × T_celsius."""
    print("\nGenerating N_A vs T survival heatmap...")

    param_names = [str(p) for p in data['param_names']]
    idx_NA = param_names.index('N_A')
    idx_T = param_names.index('T_celsius')

    # Binned observed survival
    NA_bins = np.array([10, 30, 100, 300, 1000, 3000])
    T_bins = np.linspace(25, 100, 11)

    survival_grid = np.zeros((len(T_bins)-1, len(NA_bins)-1))

    for i in range(len(T_bins)-1):
        for j in range(len(NA_bins)-1):
            mask = (data['samples'][:, idx_T] >= T_bins[i]) & (data['samples'][:, idx_T] < T_bins[i+1]) & \
                   (data['samples'][:, idx_NA] >= NA_bins[j]) & (data['samples'][:, idx_NA] < NA_bins[j+1])
            if mask.sum() > 0:
                survival_grid[i, j] = data['survived'][mask].mean()
            else:
                survival_grid[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    T_centers = 0.5 * (T_bins[:-1] + T_bins[1:])
    NA_centers = np.sqrt(NA_bins[:-1] * NA_bins[1:])

    im = ax.imshow(survival_grid, aspect='auto', cmap='RdYlGn', origin='lower',
                   extent=[np.log10(NA_bins[0]), np.log10(NA_bins[-1]), T_bins[0], T_bins[-1]],
                   vmin=0, vmax=1)
    plt.colorbar(im, label='Survival Rate', ax=ax)

    ax.set_xlabel('N_A (molecules)')
    ax.set_ylabel('T (°C)')
    ax.set_title('ORACLE Survival: N_A vs Temperature')

    # Set x-ticks at NA_bins
    ax.set_xticks(np.log10(NA_bins))
    ax.set_xticklabels([str(int(n)) for n in NA_bins])

    output_path = output_dir / 'oracle_survival_NA_vs_T.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path.name}")


def find_critical_values(gp, rf, data, output_dir, param_defaults):
    """Find critical parameter values where survival drops below 50%."""
    print("\nFinding critical parameter values...")

    param_names = [str(p) for p in data['param_names']]
    param_log_scale = data['param_log_scale']
    param_ranges = data['param_ranges']
    idx_T = param_names.index('T_celsius')

    T_values = [25, 50, 100]
    critical_values = {T: {} for T in T_values}

    for T in T_values:
        print(f"\n  At T={T}°C:")
        for i, pname in enumerate(param_names):
            if pname == 'T_celsius':
                continue

            # Scan parameter range
            if param_log_scale[i]:
                p_vals = np.logspace(np.log10(param_ranges[i, 0]), np.log10(param_ranges[i, 1]), 100)
                p_grid = np.log10(p_vals)
            else:
                p_vals = np.linspace(param_ranges[i, 0], param_ranges[i, 1], 100)
                p_grid = p_vals

            X_scan = np.zeros((len(p_vals), len(param_names)))
            for j, pname_j in enumerate(param_names):
                if pname_j == 'T_celsius':
                    X_scan[:, j] = T
                else:
                    default_val = param_defaults[pname_j]
                    if param_log_scale[j]:
                        X_scan[:, j] = np.log10(default_val)
                    else:
                        X_scan[:, j] = default_val

            X_scan[:, i] = p_grid

            # Predict survival
            model = gp if gp is not None else rf
            try:
                survival_prob = model.predict_proba(X_scan)[:, 1]
            except:
                survival_prob = np.ones(len(p_vals))  # Fallback

            # Find where it crosses 0.5
            idx_cross = np.where(np.diff(survival_prob > 0.5))[0]
            if len(idx_cross) > 0:
                critical_val = p_vals[idx_cross[0]]
                critical_values[T][pname] = float(critical_val)
                print(f"    {pname}: {critical_val:.3e}")
            else:
                critical_values[T][pname] = None

    # Plot critical values vs T
    fig, ax = plt.subplots(figsize=(12, 8))

    for pname in param_names:
        if pname == 'T_celsius':
            continue
        vals = [critical_values[T].get(pname) for T in T_values]
        if any(v is not None for v in vals):
            vals_clean = [v if v is not None else np.nan for v in vals]
            ax.plot(T_values, vals_clean, 'o-', label=pname)

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Critical Value (50% survival)')
    ax.set_title('ORACLE Critical Parameter Values vs Temperature')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    output_path = output_dir / 'oracle_critical_values.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path.name}")

    return critical_values


def main():
    parser = argparse.ArgumentParser(description='Analyze ORACLE Phase A results')
    parser.add_argument('input', nargs='+', help='NPZ file(s) to analyze')
    parser.add_argument('--output-dir', default='results/oracle_plots', help='Output directory for plots')
    parser.add_argument('--merge', '-m', action='store_true', help='Merge multiple NPZ files before analysis')
    parser.add_argument('-o', '--output', help='Output path for merged NPZ (if --merge)')
    parser.add_argument('--skip-gp', action='store_true', help='Skip GP fitting (saves RAM)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or merge data
    if args.merge and len(args.input) > 1:
        if not args.output:
            print("Error: --merge requires -o OUTPUT")
            sys.exit(1)
        data = merge_npz_files(args.input, args.output)
    else:
        if len(args.input) > 1:
            print("Warning: multiple files provided without --merge, using first file only")
        data = load_data(args.input[0])

    # Summary statistics
    summary = print_summary_stats(data)

    # Transform samples
    X = transform_samples(data['samples'], data['param_log_scale'])
    y = data['survived']

    param_names = [str(p) for p in data['param_names']]

    # Fit Random Forest
    rf, rf_importances, rf_metrics = fit_random_forest(X, y, param_names)
    summary.update(rf_metrics)
    summary['feature_importances'] = rf_importances

    # Fit GP Classifier (optional — O(N³) memory-heavy)
    if not args.skip_gp:
        gp, gp_metrics = fit_gp_classifier(X, y, param_names)
        summary.update(gp_metrics)
    else:
        gp = None
        gp_metrics = {'gp_accuracy': None, 'gp_auc': None, 'gp_length_scale_importances': {}}
        print("\nSkipping GP Classifier (--skip-gp)", flush=True)

    # Fit GP Regressor for A* (on alive samples only)
    if not args.skip_gp:
        alive_mask = data['ode_alive'] == 1
        if alive_mask.sum() > 10:
            X_alive = X[alive_mask]
            y_astar = data['a_star_ode'][alive_mask]
            gp_reg = fit_gp_regressor(X_alive, y_astar, param_names)
        else:
            gp_reg = None
            print("\nToo few alive samples for GP regressor")
    else:
        gp_reg = None
        print("Skipping GP Regressor (--skip-gp)", flush=True)

    # Top parameters for plotting
    top_params = [p for p, _ in sorted(rf_importances.items(), key=lambda x: x[1], reverse=True)]

    # Generate plots
    plot_2d_survival_maps(gp, rf, data, output_dir, top_params, PARAM_DEFAULTS)
    plot_marginal_survival_curves(gp, rf, data, output_dir, PARAM_DEFAULTS)
    plot_astar_maps(gp_reg, data, output_dir, top_params, PARAM_DEFAULTS)
    plot_feature_importance(rf_importances, gp_metrics.get('gp_length_scale_importances', {}), output_dir)
    plot_survival_NA_vs_T(gp, rf, data, output_dir)
    critical_vals = find_critical_values(gp, rf, data, output_dir, PARAM_DEFAULTS)

    summary['critical_values_25C'] = critical_vals[25]
    summary['critical_values_100C'] = critical_vals[100]

    # Save models
    if gp is not None:
        joblib.dump(gp, output_dir / 'oracle_gp_classifier.pkl')
        print(f"\nSaved GP classifier to {output_dir / 'oracle_gp_classifier.pkl'}")
    joblib.dump(rf, output_dir / 'oracle_rf_classifier.pkl')
    print(f"Saved RF classifier to {output_dir / 'oracle_rf_classifier.pkl'}")

    if gp_reg is not None:
        joblib.dump(gp_reg, output_dir / 'oracle_gp_regressor.pkl')
        print(f"Saved GP regressor to {output_dir / 'oracle_gp_regressor.pkl'}")

    # Save summary JSON
    summary_path = output_dir / 'oracle_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    print("\n" + "="*60)
    print("ORACLE ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
