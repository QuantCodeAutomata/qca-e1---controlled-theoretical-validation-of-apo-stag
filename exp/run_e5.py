"""E5: Field Rolling-Panel APO (Synthetic Data).

Synthetic design:
  - 12 assets, 3 latent factors, 20 candidate drivers
  - 2268 total observations (3 × 756)
  - Rolling windows: size=756, step=63 → ~25 windows
  - Window split: 50% certification, 25% estimation, 25% evaluation
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.field_pipeline import (
    greedy_forward_select,
    estimate_factor_model,
    conditional_cov_from_factor_model,
    gmv_portfolio_unconstrained,
    benchmark_covariances,
    portfolio_stats,
)

RESULTS_DIR = Path(_PROJECT_ROOT) / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_ASSETS = 12
N_FACTORS = 3
N_DRIVERS = 20
N_OBS_TOTAL = 756 * 3      # 2268
WINDOW_SIZE = 756
STEP = 63
CERT_SIZE = WINDOW_SIZE // 2          # 378
EST_SIZE = WINDOW_SIZE // 4           # 189
EVAL_SIZE = WINDOW_SIZE - CERT_SIZE - EST_SIZE  # 189
MAX_DRIVERS = 8
TOL = 0.27
RIDGE = 1e-6

# ---------------------------------------------------------------------------
# Synthetic DGP
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

B_true = rng.normal(0, 0.5, (N_ASSETS, N_FACTORS))
sigma_true = np.ones(N_ASSETS) * 0.7

factors_all = rng.standard_normal((N_OBS_TOTAL, N_FACTORS))
epsilon_all = rng.standard_normal((N_OBS_TOTAL, N_ASSETS))
R_all = factors_all @ B_true.T + epsilon_all * sigma_true

# Drivers 0-2: true factors (recoverable)
# Drivers 3-9: noisy versions of factors (7 drivers, partially useful)
# Drivers 10-19: pure noise (10 drivers, useless)
noisy_drivers = np.column_stack([
    factors_all[:, j % N_FACTORS] + rng.standard_normal(N_OBS_TOTAL) * 0.5
    for j in range(7)
])
pure_noise = rng.standard_normal((N_OBS_TOTAL, 10))
drivers_all = np.column_stack([factors_all, noisy_drivers, pure_noise])

DRIVER_NAMES = (
    [f'F{i+1}' for i in range(N_FACTORS)] +
    [f'NF{i+1}' for i in range(7)] +
    [f'X{i+1}' for i in range(10)]
)


# ---------------------------------------------------------------------------
# Helper: max absolute off-diagonal correlation
# ---------------------------------------------------------------------------

def _max_corr(R: np.ndarray) -> float:
    """Max |off-diagonal| entry of the sample correlation matrix of R."""
    R_dm = R - R.mean(0)
    std = R_dm.std(axis=0, ddof=1)
    std = np.where(std == 0, 1e-12, std)
    R_norm = R_dm / std
    corr = R_norm.T @ R_norm / (len(R_dm) - 1)
    np.fill_diagonal(corr, 0.0)
    return float(np.max(np.abs(corr)))


# ---------------------------------------------------------------------------
# Single-window processing
# ---------------------------------------------------------------------------

def _process_window(R_window: np.ndarray, D_window: np.ndarray, window_idx: int) -> dict:
    R_cert = R_window[:CERT_SIZE]
    R_est  = R_window[CERT_SIZE:CERT_SIZE + EST_SIZE]
    R_eval = R_window[CERT_SIZE + EST_SIZE:]
    D_cert = D_window[:CERT_SIZE]
    D_est  = D_window[CERT_SIZE:CERT_SIZE + EST_SIZE]

    # Stage I: greedy driver selection on certification block
    defect_before = _max_corr(R_cert)
    selected_names, defect_after = greedy_forward_select(
        R_cert, D_cert, DRIVER_NAMES, max_drivers=MAX_DRIVERS, tol=TOL,
    )

    # Stage II: estimate factor model on estimation block
    if selected_names:
        sel_idx = [DRIVER_NAMES.index(n) for n in selected_names]
        D_est_sel = D_est[:, sel_idx]
        B_hat, V_hat, Omega_hat = estimate_factor_model(R_est, D_est_sel)
        Sigma_apo = conditional_cov_from_factor_model(B_hat, V_hat, Omega_hat)
    else:
        Sigma_apo = np.cov(R_est, rowvar=False)

    Sigma_apo = Sigma_apo + RIDGE * np.eye(N_ASSETS)
    w_apo = gmv_portfolio_unconstrained(Sigma_apo)

    # Benchmark portfolios
    bench = benchmark_covariances(R_est, n_components=N_FACTORS)
    bench_weights = {
        name: gmv_portfolio_unconstrained(S + RIDGE * np.eye(N_ASSETS))
        for name, S in bench.items()
    }

    # Evaluate on out-of-sample eval block
    stats_apo = portfolio_stats(w_apo, R_eval)

    row = {
        'window':         window_idx,
        'n_selected':     len(selected_names),
        'defect_before':  defect_before,
        'defect_after':   defect_after,
        'is_admissible':  int(defect_after <= TOL),
        'apo_vol':        stats_apo['realized_vol'],
        'apo_sharpe':     stats_apo['sharpe'],
        'apo_leverage':   stats_apo['gross_leverage'],
    }
    for name, w in bench_weights.items():
        st = portfolio_stats(w, R_eval)
        row[f'{name}_vol']    = st['realized_vol']
        row[f'{name}_sharpe'] = st['sharpe']

    return row


# ---------------------------------------------------------------------------
# Rolling loop
# ---------------------------------------------------------------------------

starts = list(range(0, N_OBS_TOTAL - WINDOW_SIZE + 1, STEP))
print(f"E5: {len(starts)} rolling windows  "
      f"(window={WINDOW_SIZE}, step={STEP}, n_total={N_OBS_TOTAL})")

rows = []
for idx, start in enumerate(starts):
    end = start + WINDOW_SIZE
    row = _process_window(R_all[start:end], drivers_all[start:end], idx)
    rows.append(row)
    if (idx + 1) % 5 == 0 or idx == len(starts) - 1:
        print(f"  window {idx+1:2d}/{len(starts)}  "
              f"defect {row['defect_before']:.3f} → {row['defect_after']:.3f}  "
              f"apo_vol={row['apo_vol']:.3f}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / 'e5_rolling_results.csv', index=False)
print(f"\nSaved {len(df)} rows → results/e5_rolling_results.csv")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

summary = {
    'n_windows':                   int(len(df)),
    'admissible_window_fraction':  float(df['is_admissible'].mean()),
    'median_n_selected':           float(df['n_selected'].median()),
    'median_defect_before':        float(df['defect_before'].median()),
    'median_defect_after':         float(df['defect_after'].median()),
    'defect_reduction_pct':        float(
        (1 - df['defect_after'].median() / df['defect_before'].median()) * 100
    ),
    'apo_median_vol':              float(df['apo_vol'].median()),
    'sample_median_vol':           float(df['sample_vol'].median()),
    'pca_median_vol':              float(df['pca_diagonal_vol'].median()),
    'lw_linear_median_vol':        float(df['ledoit_wolf_linear_vol'].median()),
    'apo_median_sharpe':           float(df['apo_sharpe'].median()),
    'sample_median_sharpe':        float(df['sample_sharpe'].median()),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(RESULTS_DIR / 'e5_summary.csv', index=False)

print("\nE5 Summary:")
for k, v in summary.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Panel 1: rolling defect
ax = axes[0]
ax.plot(df['window'], df['defect_before'], 'b--', alpha=0.7, label='Before APO')
ax.plot(df['window'], df['defect_after'],  'r-',  alpha=0.9, label='After APO')
ax.axhline(TOL, color='gray', linestyle=':', linewidth=1, label=f'Tolerance ({TOL})')
ax.set_xlabel('Window index')
ax.set_ylabel('Defect  (max |off-diag corr|)')
ax.set_title('E5: Rolling Defect Before / After APO Selection')
ax.legend()

# Panel 2: realized GMV volatility
ax = axes[1]
vol_cols  = ['apo_vol', 'sample_vol', 'pca_diagonal_vol', 'ledoit_wolf_linear_vol']
vol_labels = ['APO Conditional', 'Sample Cov', 'PCA Factor', 'Ledoit-Wolf']
styles = [('r-', 2.0), ('b--', 1.2), ('g:', 1.2), ('m-.', 1.2)]
for col, lbl, (ls, lw) in zip(vol_cols, vol_labels, styles):
    ax.plot(df['window'], df[col], ls, linewidth=lw, label=lbl, alpha=0.85)
ax.set_xlabel('Window index')
ax.set_ylabel('Annualised Realised Volatility')
ax.set_title('E5: GMV Realised Volatility by Covariance Estimator')
ax.legend()

plt.tight_layout()
fig.savefig(RESULTS_DIR / 'e5_plots.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved plot → results/e5_plots.png")
