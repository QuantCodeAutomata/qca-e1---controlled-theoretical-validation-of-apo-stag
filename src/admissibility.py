"""Admissibility checking and Stage I candidate selection for the APO framework."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .dgp import GaussianAPOSystem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANDIDATE_VARS: list[str] = ['Z1', 'Z2', 'phi_Z1', 'P', 'nu', 'X_plus']
FUTURE_VARS: frozenset[str] = frozenset(['X_plus'])


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

def enumerate_candidates(max_size: int = 3) -> list[frozenset]:
    """Generate all non-empty subsets of CANDIDATE_VARS with size <= max_size.

    Returns 41 subsets for max_size=3 over 6 candidate variables.
    """
    result = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(CANDIDATE_VARS, size):
            result.append(frozenset(combo))
    return result


# ---------------------------------------------------------------------------
# Admissibility check
# ---------------------------------------------------------------------------

def check_chronology(candidate: frozenset) -> bool:
    """Return True if candidate is chronologically admissible (no future variables)."""
    return candidate.isdisjoint(FUTURE_VARS)


# ---------------------------------------------------------------------------
# Sample separator defect
# ---------------------------------------------------------------------------

def compute_separator_defect_sample(R: np.ndarray, H_data: np.ndarray) -> float:
    """Estimate separator defect from data.

    Regresses each column of R on H_data with intercept, computes residual
    correlation matrix, and returns max absolute off-diagonal entry.
    """
    n = R.shape[0]
    if H_data.ndim == 1:
        H_data = H_data[:, None]

    ones = np.ones((n, 1))
    X = np.hstack([ones, H_data])

    # OLS: solve (X'X) beta = X'R for all assets simultaneously
    coeffs, _, _, _ = np.linalg.lstsq(X, R, rcond=None)
    residuals = R - X @ coeffs  # (n x n_assets)

    # Correlation matrix of residuals
    std = residuals.std(axis=0, ddof=1)
    std = np.where(std == 0, 1e-12, std)
    resid_norm = (residuals - residuals.mean(axis=0)) / std
    corr = resid_norm.T @ resid_norm / (n - 1)
    np.fill_diagonal(corr, 0.0)
    return float(np.max(np.abs(corr)))


# ---------------------------------------------------------------------------
# Candidate table
# ---------------------------------------------------------------------------

def compute_candidate_table(
    system: 'GaussianAPOSystem',
    candidates: list[frozenset] | None = None,
) -> pd.DataFrame:
    """Build the candidate table with admissibility and epsilon metrics.

    Columns: candidate_id, variables, n_vars, chronology_pass,
             epsilon_analytical, kappa, selected_flag.
    Sorted by epsilon_analytical ascending (NaN last).
    """
    if candidates is None:
        candidates = enumerate_candidates()

    rows = []
    for cid, cand in enumerate(candidates):
        chron = check_chronology(cand)
        var_list = sorted(cand)  # deterministic ordering
        var_str = '+'.join(var_list)
        kappa = len(cand)

        if chron:
            eps = system.separator_defect_analytical(list(cand))
        else:
            eps = float('nan')

        rows.append({
            'candidate_id':     cid,
            'variables':        var_str,
            'n_vars':           kappa,
            'chronology_pass':  chron,
            'epsilon_analytical': eps,
            'kappa':            kappa,
            'selected_flag':    False,
        })

    df = pd.DataFrame(rows)

    # Sort: admissible (finite eps) ascending first, then NaN
    df.sort_values('epsilon_analytical', ascending=True, na_position='last', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Mark selected: minimum epsilon among admissible, tie-break by minimum kappa
    admissible = df[df['chronology_pass']].copy()
    if not admissible.empty:
        min_eps = admissible['epsilon_analytical'].min()
        min_kappa = admissible.loc[
            admissible['epsilon_analytical'] == min_eps, 'kappa'
        ].min()
        mask = (
            (df['chronology_pass']) &
            (df['epsilon_analytical'] == min_eps) &
            (df['kappa'] == min_kappa)
        )
        df.loc[mask, 'selected_flag'] = True

    return df


# ---------------------------------------------------------------------------
# Tolerance and Stage I selection
# ---------------------------------------------------------------------------

def tolerance_epsilon(n_cert: int, base: float = 0.10) -> float:
    """Return admissibility tolerance: base + 2/sqrt(n_cert)."""
    return base + 2.0 / math.sqrt(n_cert)


def stage1_select(
    candidate_table: pd.DataFrame,
    n_cert: int | None = None,
    custom_tol: float | None = None,
) -> frozenset:
    """Select the admissibility-optimal class from the candidate table.

    Among admissible candidates, picks minimum epsilon with tie-break on
    minimum kappa.  If n_cert or custom_tol is given, filters to candidates
    with epsilon <= tolerance first.
    """
    pool = candidate_table[candidate_table['chronology_pass']].copy()

    if custom_tol is not None:
        tol = custom_tol
        filtered = pool[pool['epsilon_analytical'] <= tol]
        if not filtered.empty:
            pool = filtered
    elif n_cert is not None:
        tol = tolerance_epsilon(n_cert)
        filtered = pool[pool['epsilon_analytical'] <= tol]
        if not filtered.empty:
            pool = filtered

    # Primary: min epsilon, secondary: min kappa
    pool = pool.sort_values(['epsilon_analytical', 'kappa'])
    best_row = pool.iloc[0]
    var_str = best_row['variables']
    return frozenset(var_str.split('+'))
