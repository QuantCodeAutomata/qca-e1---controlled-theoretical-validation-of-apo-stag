"""Rolling-window APO pipeline utilities (used in E5/E6 experiments)."""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, OAS
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Greedy forward driver selection
# ---------------------------------------------------------------------------

def _residual_max_corr(R: np.ndarray, X: np.ndarray) -> float:
    """OLS-regress each column of R on X (with intercept), return max |off-diag corr|."""
    n = R.shape[0]
    ones = np.ones((n, 1))
    X_aug = np.hstack([ones, X]) if X.shape[1] > 0 else ones

    coeffs, _, _, _ = np.linalg.lstsq(X_aug, R, rcond=None)
    residuals = R - X_aug @ coeffs

    std = residuals.std(axis=0, ddof=1)
    std = np.where(std == 0.0, 1e-12, std)
    resid_norm = (residuals - residuals.mean(axis=0)) / std
    n_eff = resid_norm.shape[0] - 1
    corr = resid_norm.T @ resid_norm / n_eff
    np.fill_diagonal(corr, 0.0)
    return float(np.max(np.abs(corr)))


def greedy_forward_select(
    R: np.ndarray,
    drivers: np.ndarray,
    driver_names: list[str],
    max_drivers: int = 8,
    tol: float = 0.27,
) -> tuple[list[str], float]:
    """Greedy forward selection of drivers to minimise max absolute residual correlation.

    Parameters
    ----------
    R:            (n_samples, n_assets) asset returns matrix.
    drivers:      (n_samples, n_drivers) driver (conditioning variable) matrix.
    driver_names: list of driver names corresponding to columns of `drivers`.
    max_drivers:  maximum number of drivers to select.
    tol:          stop early when defect falls at or below this tolerance.

    Returns
    -------
    (selected_names, final_defect)
    """
    n_drivers = drivers.shape[1]
    remaining = list(range(n_drivers))
    selected_idx: list[int] = []
    n = R.shape[0]

    current_X = np.empty((n, 0))
    current_defect = _residual_max_corr(R, current_X)

    while remaining and len(selected_idx) < max_drivers:
        if current_defect <= tol:
            break

        best_idx = None
        best_defect = current_defect

        for idx in remaining:
            candidate_X = np.hstack([current_X, drivers[:, [idx]]])
            defect = _residual_max_corr(R, candidate_X)
            if defect < best_defect:
                best_defect = defect
                best_idx = idx

        if best_idx is None:
            # No improvement; still add the first remaining driver to continue
            best_idx = remaining[0]
            best_defect = _residual_max_corr(
                R, np.hstack([current_X, drivers[:, [best_idx]]])
            )

        selected_idx.append(best_idx)
        remaining.remove(best_idx)
        current_X = np.hstack([current_X, drivers[:, [best_idx]]])
        current_defect = best_defect

    selected_names = [driver_names[i] for i in selected_idx]
    return selected_names, current_defect


# ---------------------------------------------------------------------------
# Factor model estimation
# ---------------------------------------------------------------------------

def estimate_factor_model(
    R: np.ndarray,
    drivers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate factor model via OLS.

    Model:  R_t = alpha + B * f_t + e_t

    Parameters
    ----------
    R:       (n_samples, n_assets)
    drivers: (n_samples, n_drivers)  — the factor realizations

    Returns
    -------
    B_hat:     (n_assets, n_drivers)  — factor loading matrix (excl. intercept)
    V_hat:     (n_drivers, n_drivers) — sample covariance of drivers
    Omega_hat: (n_assets, n_assets)   — diagonal residual covariance
    """
    n = R.shape[0]
    ones = np.ones((n, 1))
    X = np.hstack([ones, drivers])  # (n, 1+n_drivers)

    # OLS for all assets simultaneously
    coeffs, _, _, _ = np.linalg.lstsq(X, R, rcond=None)  # (1+n_drivers, n_assets)
    B_hat = coeffs[1:].T  # (n_assets, n_drivers)

    residuals = R - X @ coeffs  # (n, n_assets)

    V_hat = np.cov(drivers, rowvar=False) if drivers.ndim > 1 and drivers.shape[1] > 1 \
        else np.atleast_2d(np.var(drivers, ddof=1))

    resid_var = np.var(residuals, axis=0, ddof=1)  # (n_assets,)
    Omega_hat = np.diag(resid_var)

    return B_hat, V_hat, Omega_hat


# ---------------------------------------------------------------------------
# Conditional covariance from factor model
# ---------------------------------------------------------------------------

def conditional_cov_from_factor_model(
    B_hat: np.ndarray,
    V_hat: np.ndarray,
    Omega_hat: np.ndarray,
) -> np.ndarray:
    """Compute Sigma_hat_cond = B_hat @ V_hat @ B_hat.T + diag(diag(Omega_hat))."""
    return B_hat @ V_hat @ B_hat.T + np.diag(np.diag(Omega_hat))


# ---------------------------------------------------------------------------
# GMV portfolios
# ---------------------------------------------------------------------------

def gmv_portfolio_unconstrained(Sigma: np.ndarray) -> np.ndarray:
    """Unconstrained GMV weights: w = S^{-1}1 / (1'S^{-1}1)."""
    n = Sigma.shape[0]
    ones = np.ones(n)
    Sinv1 = linalg.solve(Sigma, ones, assume_a='pos')
    return Sinv1 / (ones @ Sinv1)


def gmv_portfolio_longonly(Sigma: np.ndarray, n_assets: int) -> np.ndarray:
    """Long-only GMV weights: min w'Sigma w  s.t. w >= 0, sum(w)=1."""
    ones = np.ones(n_assets)
    w0 = ones / n_assets

    def objective(w):
        return w @ Sigma @ w

    def jac(w):
        return 2.0 * Sigma @ w

    constraints = [{'type': 'eq', 'fun': lambda w: ones @ w - 1.0, 'jac': lambda w: ones}]
    bounds = [(0.0, 1.0)] * n_assets

    res = minimize(
        objective, w0, jac=jac,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-12, 'maxiter': 1000},
    )
    return res.x


# ---------------------------------------------------------------------------
# Benchmark covariances
# ---------------------------------------------------------------------------

def benchmark_covariances(
    R: np.ndarray,
    n_components: int = 2,
) -> dict[str, np.ndarray]:
    """Compute benchmark covariance matrices for comparison.

    Keys: 'sample', 'pca_diagonal', 'ledoit_wolf_linear', 'ledoit_wolf_nonlinear'.
    """
    n, p = R.shape

    # Sample covariance
    sample = np.cov(R, rowvar=False)

    # PCA: loadings @ loadings.T + diag(residual_var)
    pca = PCA(n_components=n_components)
    pca.fit(R)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # (p, n_components)
    reconstructed = loadings @ loadings.T
    residual_var = np.diag(sample) - np.diag(reconstructed)
    residual_var = np.maximum(residual_var, 0.0)
    pca_diag = reconstructed + np.diag(residual_var)

    # Ledoit-Wolf (linear shrinkage)
    lw = LedoitWolf().fit(R)
    lw_linear = lw.covariance_

    # OAS (nonlinear shrinkage approximation)
    oas = OAS().fit(R)
    lw_nonlinear = oas.covariance_

    return {
        'sample':                 sample,
        'pca_diagonal':           pca_diag,
        'ledoit_wolf_linear':     lw_linear,
        'ledoit_wolf_nonlinear':  lw_nonlinear,
    }


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------

def portfolio_stats(
    weights: np.ndarray,
    returns: np.ndarray,
    annualization: int = 252,
) -> dict:
    """Compute realized portfolio statistics.

    Parameters
    ----------
    weights:       1-D (n_assets,) static weights, or 2-D (T, n_assets) time-varying.
    returns:       2-D (T, n_assets) asset return matrix.
    annualization: periods per year.

    Returns
    -------
    dict with keys: realized_vol, sharpe, turnover, gross_leverage, annualized_return.
    """
    T, n_assets = returns.shape
    weights = np.asarray(weights)

    if weights.ndim == 1:
        port_returns = returns @ weights
        turnover = 0.0
        gross_leverage = float(np.sum(np.abs(weights)))
    else:
        # Time-varying: row t of weights applies to row t of returns
        port_returns = np.sum(returns * weights, axis=1)
        diffs = np.diff(weights, axis=0)
        turnover = float(np.mean(np.sum(np.abs(diffs), axis=1)))
        gross_leverage = float(np.mean(np.sum(np.abs(weights), axis=1)))

    ann_return = float(np.mean(port_returns) * annualization)
    realized_vol = float(np.std(port_returns, ddof=1) * np.sqrt(annualization))
    sharpe = ann_return / realized_vol if realized_vol > 0 else float('nan')

    return {
        'annualized_return': ann_return,
        'realized_vol':      realized_vol,
        'sharpe':            sharpe,
        'turnover':          turnover,
        'gross_leverage':    gross_leverage,
    }
