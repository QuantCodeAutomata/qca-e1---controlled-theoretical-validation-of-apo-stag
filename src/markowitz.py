"""Stage II Markowitz portfolio optimization for the APO framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .dgp import GaussianAPOSystem


# ---------------------------------------------------------------------------
# Scalar parameters
# ---------------------------------------------------------------------------

def markowitz_params(mu: np.ndarray, Sigma: np.ndarray) -> dict:
    """Compute scalar Markowitz parameters a, b, c, d and Sigma_inv.

    a = 1'S^{-1}1,  b = 1'S^{-1}mu,  c = mu'S^{-1}mu,  d = ac - b^2.
    """
    n = len(mu)
    ones = np.ones(n)

    # Solve Sigma @ x = ones  and  Sigma @ y = mu  (avoids explicit inversion)
    Sinv1 = linalg.solve(Sigma, ones, assume_a='pos')
    Sinvmu = linalg.solve(Sigma, mu, assume_a='pos')
    Sigma_inv = linalg.inv(Sigma)

    a = float(ones @ Sinv1)
    b = float(ones @ Sinvmu)
    c = float(mu @ Sinvmu)
    d = a * c - b ** 2

    return {'a': a, 'b': b, 'c': c, 'd': d, 'Sigma_inv': Sigma_inv}


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------

def efficient_frontier_variance(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_returns: np.ndarray,
) -> np.ndarray:
    """Compute frontier variance rho(m) = (a*m^2 - 2*b*m + c) / d (closed-form)."""
    p = markowitz_params(mu, Sigma)
    a, b, c, d = p['a'], p['b'], p['c'], p['d']
    m = np.asarray(target_returns)
    return (a * m ** 2 - 2.0 * b * m + c) / d


# ---------------------------------------------------------------------------
# Portfolio weights
# ---------------------------------------------------------------------------

def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Global minimum variance portfolio: w = S^{-1}1 / (1'S^{-1}1)."""
    n = Sigma.shape[0]
    ones = np.ones(n)
    Sinv1 = linalg.solve(Sigma, ones, assume_a='pos')
    return Sinv1 / (ones @ Sinv1)


def tangency_weights(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Tangency portfolio weights: w ∝ S^{-1}(mu - rf*1), normalised to sum=1."""
    excess = mu - rf
    n = len(mu)
    Sinv_exc = linalg.solve(Sigma, excess, assume_a='pos')
    denom = np.ones(n) @ Sinv_exc
    return Sinv_exc / denom


def frontier_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float,
) -> np.ndarray:
    """Minimum-variance frontier weights at a given target return (two-fund theorem).

    w(m) = g + h * m  where
        g = (c * S^{-1}1  - b * S^{-1}mu) / d
        h = (-b * S^{-1}1 + a * S^{-1}mu) / d
    """
    n = len(mu)
    ones = np.ones(n)
    Sinv1 = linalg.solve(Sigma, ones, assume_a='pos')
    Sinvmu = linalg.solve(Sigma, mu, assume_a='pos')

    a = float(ones @ Sinv1)
    b = float(ones @ Sinvmu)
    c = float(mu @ Sinvmu)
    d = a * c - b ** 2

    g = (c * Sinv1 - b * Sinvmu) / d
    h = (-b * Sinv1 + a * Sinvmu) / d
    return g + h * target_return


# ---------------------------------------------------------------------------
# Frontier comparison (APO closed-form vs. classical SLSQP)
# ---------------------------------------------------------------------------

def compare_frontiers(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_points: int = 41,
) -> pd.DataFrame:
    """Compare APO closed-form and classical SLSQP frontiers.

    Grids target returns from the GMV return (b/a) to max(mu).
    Returns DataFrame: target_return, apo_variance, classical_variance, abs_diff.
    """
    p = markowitz_params(mu, Sigma)
    a, b = p['a'], p['b']
    m_min = b / a  # GMV return
    m_max = float(np.max(mu))
    targets = np.linspace(m_min, m_max, n_points)

    apo_vars = efficient_frontier_variance(mu, Sigma, targets)

    n = len(mu)
    ones = np.ones(n)
    classical_vars = np.empty(n_points)

    for i, m in enumerate(targets):
        w0 = frontier_weights(mu, Sigma, m)

        def objective(w):
            return w @ Sigma @ w

        def jac(w):
            return 2.0 * Sigma @ w

        constraints = [
            {'type': 'eq', 'fun': lambda w: ones @ w - 1.0,
             'jac': lambda w: ones},
            {'type': 'eq', 'fun': lambda w: mu @ w - m,
             'jac': lambda w: mu},
        ]
        res = minimize(
            objective, w0, jac=jac,
            method='SLSQP', constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000},
        )
        classical_vars[i] = res.fun

    return pd.DataFrame({
        'target_return':    targets,
        'apo_variance':     apo_vars,
        'classical_variance': classical_vars,
        'abs_diff':         np.abs(apo_vars - classical_vars),
    })


# ---------------------------------------------------------------------------
# Conditional GMV (given information set H)
# ---------------------------------------------------------------------------

def compute_conditional_gmv(
    system: 'GaussianAPOSystem',
    H_vars: list[str],
) -> np.ndarray:
    """Compute GMV weights using the analytical conditional covariance Sigma_{R|H}."""
    Sigma_cond = system.conditional_cov_analytical(H_vars)
    return gmv_weights(Sigma_cond)
