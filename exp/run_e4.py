"""E4: Sequential Hysteresis Reselection Experiment.

DGP: 8 assets, 2 latent factors that shift mid-sequence.
  Nodes 1-19:  R = B_Z1*Z1 + B_Z2*Z2 + sigma*eps   →  optimal H = {Z1, Z2}
  Nodes 20-40: R = B_Z1*Z1 + B_Z3*Z3 + sigma*eps   →  optimal H = {Z1, Z3}

DGP is calibrated with strong loadings (B_Z2 magnitudes ~0.8-1.0) and low
idiosyncratic noise (sigma=0.3) so the true separator classes have analytically
zero defect while all competing candidates have large defects (>0.5), making
the true separator identifiable even with N_CERT=100 per node.

Candidate variables: {Z1, Z2, Z3, P, nu}  (all nonempty subsets of size ≤ 2 → 15 candidates)

Certification score:  ŝ_t(H) = ε̂_t(H) + 0.02 * |H|
Hysteresis rule: keep incumbent H_prev unless  ŝ_t(H_prev) > min_H ŝ_t(H) + η

Metrics (per η ∈ {0, 0.02, 0.05, 0.10, 0.20}):
  n_reselections   : number of incumbent changes across nodes 2..40
  fraction_correct : fraction of nodes where selected H equals node-optimal
  lag_to_adopt     : nodes after t=20 until {Z1,Z3} is first selected
"""

from __future__ import annotations

import itertools
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.admissibility import compute_separator_defect_sample

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DGP parameters
# ---------------------------------------------------------------------------

N_ASSETS = 8
B_Z1 = np.array([0.6] * 8)
# Strong loadings so that conditioning on Z2 (or Z3) completely removes factor residuals.
# Large magnitudes mean all-but-{Z1,Z2} candidates have large defect, making the
# true separator class easily identifiable even at N_CERT=100.
B_Z2 = np.array([ 1.0, -0.9,  0.8, -1.0,  0.9, -0.8,  1.0, -0.9])
B_Z3 = np.array([-0.9,  1.0, -0.8,  0.9, -1.0,  0.8, -0.9,  1.0])
SIGMA = np.array([0.3] * 8)    # low idiosyncratic noise → large signal-to-noise
ETA_P = 0.20  # P = Z1 + ETA_P * e

N_NODES = 40
N_OBS   = 200   # observations generated per node
N_CERT  = 100   # first N_CERT observations used for certification scores
PENALTY = 0.02  # per-variable complexity penalty in scoring rule

ETAS = [0.0, 0.02, 0.05, 0.10, 0.20]
N_REPS = 100

# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

_CAND_VARS = ['Z1', 'Z2', 'Z3', 'P', 'nu']

CANDIDATES: list[frozenset] = []
for _size in [1, 2]:
    for _combo in itertools.combinations(_CAND_VARS, _size):
        CANDIDATES.append(frozenset(_combo))
# 15 candidates: C(5,1)+C(5,2) = 5+10 = 15

KAPPA: dict[frozenset, int] = {H: len(H) for H in CANDIDATES}

# Node-optimal representation (population-level)
OPTIMAL: dict[int, frozenset] = {}
for _t in range(1, N_NODES + 1):
    OPTIMAL[_t] = frozenset({'Z1', 'Z2'}) if _t <= 19 else frozenset({'Z1', 'Z3'})


# ---------------------------------------------------------------------------
# Node simulation
# ---------------------------------------------------------------------------

def simulate_node(t: int, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Simulate N_OBS observations for node t.

    Returns (R, var_data) where var_data maps variable name → array of length N_OBS.
    At nodes 1-19:  Z2 is an active factor; Z3 is independent noise.
    At nodes 20-40: Z3 is an active factor; Z2 is independent noise.
    """
    rng = np.random.default_rng(seed)
    Z1  = rng.standard_normal(N_OBS)
    Z2  = rng.standard_normal(N_OBS)
    Z3  = rng.standard_normal(N_OBS)
    eps = rng.standard_normal((N_OBS, N_ASSETS))
    e_P = rng.standard_normal(N_OBS)
    nu  = rng.standard_normal(N_OBS)

    P = Z1 + ETA_P * e_P

    if t <= 19:
        R = np.outer(Z1, B_Z1) + np.outer(Z2, B_Z2) + eps * SIGMA
    else:
        R = np.outer(Z1, B_Z1) + np.outer(Z3, B_Z3) + eps * SIGMA

    var_data = {'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'P': P, 'nu': nu}
    return R, var_data


# ---------------------------------------------------------------------------
# Certification score
# ---------------------------------------------------------------------------

def certification_score(
    R_cert: np.ndarray,
    var_cert: dict[str, np.ndarray],
    H: frozenset,
) -> tuple[float, float]:
    """Return (score, defect) for candidate H on certification data."""
    H_data = np.column_stack([var_cert[v] for v in sorted(H)])
    defect = compute_separator_defect_sample(R_cert, H_data)
    score  = defect + PENALTY * KAPPA[H]
    return score, defect


# ---------------------------------------------------------------------------
# Hysteresis algorithm
# ---------------------------------------------------------------------------

def run_hysteresis(
    scores_by_node: list[dict[frozenset, float]],
    eta: float,
) -> tuple[int, float, float]:
    """Run the hysteresis reselection algorithm.

    Parameters
    ----------
    scores_by_node : list of length N_NODES; each element is {H: score_t(H)}.
    eta            : hysteresis margin.

    Returns
    -------
    n_reselections   : int
    fraction_correct : float
    lag_to_adopt     : float (∞ replaced by N_NODES - 19 + 1 = 22 if never adopted)
    """
    n_nodes = len(scores_by_node)

    # Initialise at node 1
    H_prev = min(CANDIDATES, key=lambda H: (scores_by_node[0][H], sorted(H)))
    selected: list[frozenset] = [H_prev]
    n_reselections = 0

    for t_idx in range(1, n_nodes):
        scores_t = scores_by_node[t_idx]
        incumb_score = scores_t[H_prev]
        H_best = min(CANDIDATES, key=lambda H: (scores_t[H], sorted(H)))
        min_score = scores_t[H_best]

        if incumb_score <= min_score + eta:
            selected.append(H_prev)
        else:
            H_prev = H_best
            selected.append(H_prev)
            n_reselections += 1

    # Fraction of nodes where selected == node-optimal
    n_correct = sum(
        1 for t_idx, H in enumerate(selected)
        if H == OPTIMAL[t_idx + 1]
    )
    fraction_correct = n_correct / n_nodes

    # Lag to adopt {Z1, Z3}: nodes 20-40 (t_idx 19-39) before first selection
    new_opt = frozenset({'Z1', 'Z3'})
    lag: float = float(n_nodes - 19 + 1)  # default = never adopted (22)
    for t_idx in range(19, n_nodes):       # node 20 → t_idx=19
        if selected[t_idx] == new_opt:
            lag = float(t_idx - 19)        # 0 if adopted at node 20
            break

    return n_reselections, fraction_correct, lag


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

print("=" * 62)
print("E4: Sequential Hysteresis Reselection")
print("=" * 62)
print(f"  N_NODES={N_NODES}, N_OBS={N_OBS}, N_CERT={N_CERT}, "
      f"N_REPS={N_REPS}, penalty={PENALTY}")
print(f"  Candidates: {len(CANDIDATES)}  "
      f"(all nonempty subsets size ≤ 2 of {_CAND_VARS})")
print(f"  Hysteresis margins η: {ETAS}")
print()

# Store per-replication, per-eta metrics
records: list[dict] = []

for rep in range(N_REPS):
    if (rep + 1) % 20 == 0:
        print(f"  Replication {rep + 1}/{N_REPS} …")

    # ── Compute certification scores for every node ─────────────────────
    scores_by_node: list[dict[frozenset, float]] = []

    for t in range(1, N_NODES + 1):
        node_seed = rep * N_NODES + (t - 1)
        R_full, var_full = simulate_node(t, seed=node_seed)

        R_cert   = R_full[:N_CERT]
        var_cert = {v: arr[:N_CERT] for v, arr in var_full.items()}

        scores_t: dict[frozenset, float] = {}
        for H in CANDIDATES:
            scores_t[H], _ = certification_score(R_cert, var_cert, H)
        scores_by_node.append(scores_t)

    # ── Run hysteresis for each η ────────────────────────────────────────
    for eta in ETAS:
        n_resel, frac_corr, lag = run_hysteresis(scores_by_node, eta)
        records.append({
            'rep':               rep,
            'eta':               eta,
            'n_reselections':    n_resel,
            'fraction_correct':  frac_corr,
            'lag_to_adopt':      lag,
        })

df_all = pd.DataFrame(records)
csv_path = os.path.join(RESULTS_DIR, 'e4_results.csv')
df_all.to_csv(csv_path, index=False)
print(f"\n  Results saved → {csv_path}\n")

# ---------------------------------------------------------------------------
# Aggregate over replications
# ---------------------------------------------------------------------------

agg = (
    df_all.groupby('eta')[['n_reselections', 'fraction_correct', 'lag_to_adopt']]
    .agg(['mean', 'std'])
)

print("=" * 62)
print("E4 Summary Statistics (aggregated over replications)")
print("=" * 62)
print(f"  {'η':>6}  {'n_resel (mean±sd)':>20}  "
      f"{'frac_correct (mean±sd)':>24}  {'lag (mean±sd)':>18}")
print("  " + "-" * 74)
for eta in ETAS:
    row = agg.loc[eta]
    nr_m, nr_s = row[('n_reselections', 'mean')], row[('n_reselections', 'std')]
    fc_m, fc_s = row[('fraction_correct', 'mean')], row[('fraction_correct', 'std')]
    lg_m, lg_s = row[('lag_to_adopt', 'mean')], row[('lag_to_adopt', 'std')]
    print(f"  {eta:>6.2f}  {nr_m:>9.3f}±{nr_s:<8.3f}  "
          f"{fc_m:>11.4f}±{fc_s:<11.4f}  "
          f"{lg_m:>7.3f}±{lg_s:<7.3f}")

print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

agg_flat = df_all.groupby('eta').agg(
    n_resel_mean=('n_reselections', 'mean'),
    n_resel_std=('n_reselections', 'std'),
    frac_mean=('fraction_correct', 'mean'),
    frac_std=('fraction_correct', 'std'),
    lag_mean=('lag_to_adopt', 'mean'),
    lag_std=('lag_to_adopt', 'std'),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('E4 — Sequential Hysteresis Reselection', fontsize=13)

etas_arr = agg_flat['eta'].values

# Panel 1: n_reselections vs η
ax = axes[0]
ax.errorbar(etas_arr, agg_flat['n_resel_mean'], yerr=agg_flat['n_resel_std'],
            fmt='o-', color='steelblue', capsize=4, linewidth=1.8)
ax.set_xlabel('Hysteresis margin η')
ax.set_ylabel('Mean reselections (nodes 2-40)')
ax.set_title('Reselections vs η')
ax.grid(True, alpha=0.3)
ax.set_xticks(etas_arr)

# Panel 2: fraction_correct vs η
ax = axes[1]
ax.errorbar(etas_arr, agg_flat['frac_mean'], yerr=agg_flat['frac_std'],
            fmt='s-', color='seagreen', capsize=4, linewidth=1.8)
ax.set_xlabel('Hysteresis margin η')
ax.set_ylabel('Fraction of nodes with correct H')
ax.set_title('Accuracy vs η')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.set_xticks(etas_arr)

# Panel 3: lag_to_adopt vs η
ax = axes[2]
ax.errorbar(etas_arr, agg_flat['lag_mean'], yerr=agg_flat['lag_std'],
            fmt='^-', color='tomato', capsize=4, linewidth=1.8)
ax.set_xlabel('Hysteresis margin η')
ax.set_ylabel('Lag to adopt {Z1,Z3}  (nodes after t=20)')
ax.set_title('Adoption lag vs η')
ax.grid(True, alpha=0.3)
ax.set_xticks(etas_arr)

plt.tight_layout()
png_path = os.path.join(RESULTS_DIR, 'e4_plots.png')
plt.savefig(png_path, dpi=130)
plt.close()
print(f"  Plots saved → {png_path}")
print()
print("E4 complete.")
