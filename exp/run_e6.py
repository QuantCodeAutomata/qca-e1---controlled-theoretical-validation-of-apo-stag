"""E6: Cross-Universe and Aggregation Extensions (Synthetic).

Three sub-experiments:
  1. Cross-universe: Universe types A (sigma=0.8), B (sigma=0.5), C (sigma=0.2)
  2. Search-size: Attainable defect vs candidate pool size {4, 8, 20}
  3. Aggregation: Defect vs equal-weight aggregation level k ∈ {1,2,4,8,12}
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

from src.field_pipeline import greedy_forward_select

RESULTS_DIR = Path(_PROJECT_ROOT) / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
N_ASSETS = 12
N_FACTORS = 3
N_DRIVERS = 20
N_CERT = 500          # certification block size for all sub-experiments
MAX_DRIVERS = 8
TOL = 0.27

DRIVER_NAMES = (
    [f'F{i+1}' for i in range(N_FACTORS)] +
    [f'NF{i+1}' for i in range(7)] +
    [f'X{i+1}' for i in range(10)]
)


def _max_corr(R: np.ndarray) -> float:
    R_dm = R - R.mean(0)
    std = R_dm.std(axis=0, ddof=1)
    std = np.where(std == 0, 1e-12, std)
    R_norm = R_dm / std
    corr = R_norm.T @ R_norm / (len(R_dm) - 1)
    np.fill_diagonal(corr, 0.0)
    return float(np.max(np.abs(corr)))


def _build_drivers(rng: np.random.Generator, factors: np.ndarray, n: int,
                   proxy_noise: float = 0.8) -> np.ndarray:
    """Build 20-driver panel: NO true factors — only noisy proxies and pure noise.

    Using imperfect proxies (instead of the true factors) is critical for the
    aggregation experiment: as assets become more diversified their returns are
    dominated by common factor variation, and because the drivers only *partially*
    capture the factors, residual correlation grows with k (harder to separate).

    proxy_noise=0.8 makes each driver a moderately noisy proxy of one factor,
    giving measurable residual dependence in all sub-experiments.
    """
    # 10 noisy proxies of the 3 factors (each factor proxied ~3-4 times)
    proxies = np.column_stack([
        factors[:, j % N_FACTORS] + rng.standard_normal(n) * proxy_noise
        for j in range(10)
    ])
    pure_noise = rng.standard_normal((n, 10))
    return np.column_stack([proxies, pure_noise])


# ---------------------------------------------------------------------------
# Experiment 1: Cross-universe
# ---------------------------------------------------------------------------
print("=" * 60)
print("E6 Experiment 1: Cross-Universe")
print("=" * 60)

UNIVERSE_SIGMAS = {'A': 0.8, 'B': 0.5, 'C': 0.2}

rng_cu = np.random.default_rng(100)
B_true = rng_cu.normal(0, 0.5, (N_ASSETS, N_FACTORS))
factors_cu = rng_cu.standard_normal((N_CERT, N_FACTORS))
drivers_cu = _build_drivers(rng_cu, factors_cu, N_CERT)

cross_rows = []
for utype, sigma in UNIVERSE_SIGMAS.items():
    eps = rng_cu.standard_normal((N_CERT, N_ASSETS))
    R_cert = factors_cu @ B_true.T + eps * sigma

    defect_before = _max_corr(R_cert)
    selected, defect_after = greedy_forward_select(
        R_cert, drivers_cu, DRIVER_NAMES, max_drivers=MAX_DRIVERS, tol=TOL,
    )
    reduction_pct = (1 - defect_after / defect_before) * 100

    row = {
        'universe_type':    utype,
        'sigma':            sigma,
        'defect_before':    round(defect_before, 4),
        'defect_after':     round(defect_after, 4),
        'defect_reduction_pct': round(reduction_pct, 2),
        'n_selected':       len(selected),
        'is_admissible':    int(defect_after <= TOL),
        'selected_drivers': '+'.join(selected),
    }
    cross_rows.append(row)
    print(f"  Type {utype} (sigma={sigma}): "
          f"defect {defect_before:.3f} → {defect_after:.3f}  "
          f"({reduction_pct:.1f}% reduction)  "
          f"admissible={bool(row['is_admissible'])}")

df_cross = pd.DataFrame(cross_rows)
df_cross.to_csv(RESULTS_DIR / 'e6_cross_universe.csv', index=False)
print(f"Saved → results/e6_cross_universe.csv\n")


# ---------------------------------------------------------------------------
# Experiment 2: Search-size
# ---------------------------------------------------------------------------
print("=" * 60)
print("E6 Experiment 2: Search-Size")
print("=" * 60)

# True factor-proxy drivers are at indices 0-9 (10 proxies of 3 factors).
# With pool_size=4, often misses the best proxies → higher defect.
# With pool_size=20 (full set), greedy always has access to best proxies → lower defect.
POOL_SIZES = [4, 8, 20]
N_DRAWS = 8  # random draws per pool size

rng_ss = np.random.default_rng(200)
B_ss = rng_ss.normal(0, 0.5, (N_ASSETS, N_FACTORS))
factors_ss = rng_ss.standard_normal((N_CERT, N_FACTORS))
eps_ss = rng_ss.standard_normal((N_CERT, N_ASSETS))
R_ss = factors_ss @ B_ss.T + eps_ss * 0.8   # Type A sigma

# Drivers: 10 noisy proxies + 10 pure noise (produced by updated _build_drivers)
drivers_ss = _build_drivers(rng_ss, factors_ss, N_CERT)

search_rows = []
for pool_size in POOL_SIZES:
    defects = []
    for draw in range(N_DRAWS):
        # Completely random draw from the full driver pool (no forced inclusion).
        # Larger pools give greedy more options → generally lower attainable defect.
        pool_idx = sorted(rng_ss.choice(N_DRIVERS, size=pool_size, replace=False).tolist())
        pool_drivers = drivers_ss[:, pool_idx]
        pool_names = [DRIVER_NAMES[i] for i in pool_idx]

        _, defect_after = greedy_forward_select(
            R_ss, pool_drivers, pool_names,
            max_drivers=min(MAX_DRIVERS, pool_size), tol=TOL,
        )
        defects.append(defect_after)

    mean_d = float(np.mean(defects))
    min_d  = float(np.min(defects))
    max_d  = float(np.max(defects))

    row = {
        'pool_size':        pool_size,
        'mean_defect_after': round(mean_d, 4),
        'min_defect_after':  round(min_d, 4),
        'max_defect_after':  round(max_d, 4),
        'fraction_admissible': round(
            sum(d <= TOL for d in defects) / N_DRAWS, 3
        ),
    }
    search_rows.append(row)
    print(f"  Pool size {pool_size:2d}: defect_after mean={mean_d:.3f}  "
          f"[{min_d:.3f}, {max_d:.3f}]  "
          f"admissible_frac={row['fraction_admissible']:.2f}")

df_search = pd.DataFrame(search_rows)
df_search.to_csv(RESULTS_DIR / 'e6_search_size.csv', index=False)
print(f"Saved → results/e6_search_size.csv\n")


# ---------------------------------------------------------------------------
# Experiment 3: Aggregation
# ---------------------------------------------------------------------------
print("=" * 60)
print("E6 Experiment 3: Aggregation")
print("=" * 60)

K_VALUES = [1, 2, 4, 8, 12]
K_MAX = 12
N_GROUPS = 12             # always 12 portfolio assets
N_IND_PER_GROUP = K_MAX   # 12 base assets per group → always enough for k ≤ 12

rng_agg = np.random.default_rng(300)
# Factor loadings for 12 portfolio groups
B_agg = rng_agg.normal(0, 0.5, (N_GROUPS, N_FACTORS))
# Generate individual base assets: N_IND_PER_GROUP assets per group,
# each group shares the same factor loading (different idiosyncratic noise).
B_ind = np.repeat(B_agg, N_IND_PER_GROUP, axis=0)  # (144, N_FACTORS)
N_IND_TOTAL = N_GROUPS * N_IND_PER_GROUP           # 144

sigma_base = 0.7
factors_agg = rng_agg.standard_normal((N_CERT, N_FACTORS))
drivers_agg = _build_drivers(rng_agg, factors_agg, N_CERT)
eps_ind = rng_agg.standard_normal((N_CERT, N_IND_TOTAL)) * sigma_base
R_ind_all = factors_agg @ B_ind.T + eps_ind  # (N_CERT, 144)

agg_rows = []
for k in K_VALUES:
    # Construct 12 portfolio assets: portfolio g = mean of k base assets in group g
    R_port = np.column_stack([
        R_ind_all[:, g * N_IND_PER_GROUP: g * N_IND_PER_GROUP + k].mean(axis=1)
        for g in range(N_GROUPS)
    ])

    defect_before = _max_corr(R_port)
    selected, defect_after = greedy_forward_select(
        R_port, drivers_agg, DRIVER_NAMES, max_drivers=MAX_DRIVERS, tol=TOL,
    )
    reduction_pct = (1 - defect_after / defect_before) * 100
    sigma_effective = sigma_base / np.sqrt(k)

    row = {
        'k':                   k,
        'sigma_effective':     round(sigma_effective, 4),
        'defect_before':       round(defect_before, 4),
        'defect_after':        round(defect_after, 4),
        'defect_reduction_pct': round(reduction_pct, 2),
        'n_selected':          len(selected),
        'is_admissible':       int(defect_after <= TOL),
    }
    agg_rows.append(row)
    print(f"  k={k:2d} (sigma_eff={sigma_effective:.3f}): "
          f"defect {defect_before:.3f} → {defect_after:.3f}  "
          f"({reduction_pct:.1f}% reduction)  "
          f"admissible={bool(row['is_admissible'])}")

df_agg = pd.DataFrame(agg_rows)
df_agg.to_csv(RESULTS_DIR / 'e6_aggregation.csv', index=False)
print(f"Saved → results/e6_aggregation.csv\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: cross-universe defect comparison
ax = axes[0]
types = df_cross['universe_type'].tolist()
x = np.arange(len(types))
width = 0.35
bars_b = ax.bar(x - width/2, df_cross['defect_before'], width, label='Before APO', color='steelblue', alpha=0.8)
bars_a = ax.bar(x + width/2, df_cross['defect_after'],  width, label='After APO',  color='tomato',   alpha=0.8)
ax.axhline(TOL, color='gray', linestyle=':', linewidth=1.2, label=f'Tolerance ({TOL})')
ax.set_xticks(x)
ax.set_xticklabels([f'Type {t}\n(σ={s})' for t, s in zip(types, df_cross['sigma'])])
ax.set_ylabel('Defect')
ax.set_title('Cross-Universe Defect')
ax.legend(fontsize=8)

# Panel 2: search-size defect
ax = axes[1]
ax.plot(df_search['pool_size'], df_search['mean_defect_after'], 'bo-', linewidth=2, label='Mean defect after')
ax.fill_between(
    df_search['pool_size'],
    df_search['min_defect_after'],
    df_search['max_defect_after'],
    alpha=0.25, color='blue', label='Min–Max range',
)
ax.axhline(TOL, color='gray', linestyle=':', linewidth=1.2, label=f'Tolerance ({TOL})')
ax.set_xlabel('Candidate pool size')
ax.set_ylabel('Defect after APO')
ax.set_title('Attainable Defect vs Pool Size')
ax.legend(fontsize=8)
ax.set_xticks(POOL_SIZES)

# Panel 3: aggregation defect
ax = axes[2]
ax.plot(df_agg['k'], df_agg['defect_before'], 'b--o', linewidth=1.8, label='Before APO', alpha=0.8)
ax.plot(df_agg['k'], df_agg['defect_after'],  'r-o',  linewidth=1.8, label='After APO',  alpha=0.8)
ax.axhline(TOL, color='gray', linestyle=':', linewidth=1.2, label=f'Tolerance ({TOL})')
ax.set_xlabel('Aggregation level k')
ax.set_ylabel('Defect')
ax.set_title('Defect vs Aggregation Level k')
ax.legend(fontsize=8)
ax.set_xticks(K_VALUES)

plt.tight_layout()
fig.savefig(RESULTS_DIR / 'e6_plots.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved plot → results/e6_plots.png")
print("\nE6 complete.")
