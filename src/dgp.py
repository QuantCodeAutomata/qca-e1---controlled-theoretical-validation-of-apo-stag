"""Data-generating process for the APO (Admissibility and Portfolio Optimization) framework."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DGP parameters
# ---------------------------------------------------------------------------

N_ASSETS = 8
N_FACTORS = 2

B = np.array([
    [0.8,  0.50],
    [0.8, -0.30],
    [0.8,  0.40],
    [0.8, -0.20],
    [0.8,  0.60],
    [0.8, -0.40],
    [0.8,  0.30],
    [0.8, -0.50],
])

SIGMA = np.array([0.7] * 8)
PROXY_NOISE_STD = 0.7

# All non-future candidate variables (ordered)
_NON_FUTURE_VARS = ['Z1', 'Z2', 'phi_Z1', 'P', 'nu']
_RETURN_COLS = [f'R_{i+1}' for i in range(N_ASSETS)]
_ALL_COLS = _RETURN_COLS + _NON_FUTURE_VARS + ['X_plus']


class GaussianAPOSystem:
    """Gaussian APO system with 8 assets, 2 latent factors, and 6 candidate variables."""

    def __init__(self):
        self.n_assets = N_ASSETS
        self.n_factors = N_FACTORS
        self.B = B.copy()
        self.sigma = SIGMA.copy()
        self.proxy_noise_std = PROXY_NOISE_STD

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, n: int, seed: int) -> pd.DataFrame:
        """Simulate n observations. Returns DataFrame with columns:
        R_1,...,R_8, Z1, Z2, phi_Z1, P, nu, X_plus."""
        rng = np.random.default_rng(seed)

        Z1 = rng.standard_normal(n)
        Z2 = rng.standard_normal(n)
        epsilon = rng.standard_normal((n, self.n_assets))

        R = Z1[:, None] * self.B[:, 0] + Z2[:, None] * self.B[:, 1] + epsilon * self.sigma

        phi_Z1 = 3.0 * Z1 - 2.0
        e = rng.standard_normal(n)
        P = Z1 + self.proxy_noise_std * e
        nu = rng.standard_normal(n)

        # X_plus = equal-weighted sum of realised returns at the current horizon.
        # This is the "sum of horizon shocks" in the paper: it is strongly
        # correlated with every individual asset return (common-factor channel
        # plus shared idiosyncratic direction), so any candidate containing it
        # looks excellent in-sample but is unavailable at the decision node —
        # hence chronologically inadmissible.
        X_plus = R.sum(axis=1)

        data = np.column_stack([R, Z1, Z2, phi_Z1, P, nu, X_plus])
        return pd.DataFrame(data, columns=_ALL_COLS)

    # ------------------------------------------------------------------
    # Analytical covariance helpers
    # ------------------------------------------------------------------

    def _sigma_r(self) -> np.ndarray:
        """8x8 marginal covariance of R."""
        return self.B @ self.B.T + np.diag(self.sigma ** 2)

    def _cov_ri_var(self, var: str) -> np.ndarray:
        """Return 8-vector Cov(R, var)."""
        if var == 'Z1':
            return self.B[:, 0].copy()
        if var == 'Z2':
            return self.B[:, 1].copy()
        if var == 'phi_Z1':
            return 3.0 * self.B[:, 0]
        if var == 'P':
            return self.B[:, 0].copy()
        if var == 'nu':
            return np.zeros(self.n_assets)
        raise ValueError(f"Unknown variable: {var}")

    def _cov_var_var(self, v1: str, v2: str) -> float:
        """Scalar covariance between two non-future candidate variables."""
        _cov_table = {
            ('Z1',     'Z1'):     1.0,
            ('Z1',     'Z2'):     0.0,
            ('Z1',     'phi_Z1'): 3.0,
            ('Z1',     'P'):      1.0,
            ('Z1',     'nu'):     0.0,
            ('Z2',     'Z2'):     1.0,
            ('Z2',     'phi_Z1'): 0.0,
            ('Z2',     'P'):      0.0,
            ('Z2',     'nu'):     0.0,
            ('phi_Z1', 'phi_Z1'): 9.0,
            ('phi_Z1', 'P'):      3.0,
            ('phi_Z1', 'nu'):     0.0,
            ('P',      'P'):      1.49,
            ('P',      'nu'):     0.0,
            ('nu',     'nu'):     1.0,
        }
        key = (v1, v2)
        if key in _cov_table:
            return _cov_table[key]
        if (v2, v1) in _cov_table:
            return _cov_table[(v2, v1)]
        raise ValueError(f"Unknown variable pair: {v1}, {v2}")

    # ------------------------------------------------------------------
    # Conditional covariance / mean
    # ------------------------------------------------------------------

    def conditional_cov_analytical(self, H_vars: list) -> np.ndarray:
        """Compute 8x8 conditional covariance Sigma_{R|H} analytically.

        H_vars must not include 'X_plus'.
        """
        Sigma_R = self._sigma_r()

        if not H_vars:
            return Sigma_R

        k = len(H_vars)
        # Sigma_RH: (8 x k)
        Sigma_RH = np.column_stack([self._cov_ri_var(v) for v in H_vars])

        # Sigma_H: (k x k)
        Sigma_H = np.array(
            [[self._cov_var_var(H_vars[i], H_vars[j]) for j in range(k)]
             for i in range(k)]
        )

        # Sigma_{R|H} = Sigma_R - Sigma_RH @ Sigma_H^+ @ Sigma_RH.T
        # Use pseudoinverse to handle collinear / singular Sigma_H cases
        # (e.g. Z1 and phi_Z1 together, which are perfectly correlated)
        Sigma_H_pinv = np.linalg.pinv(Sigma_H)
        return Sigma_R - Sigma_RH @ Sigma_H_pinv @ Sigma_RH.T

    def conditional_mean_analytical(self, H_vars: list, h_vals: np.ndarray) -> np.ndarray:
        """Compute E[R | H=h_vals] analytically (zero-mean model). Returns 8-vector."""
        if not H_vars:
            return np.zeros(self.n_assets)

        k = len(H_vars)
        Sigma_RH = np.column_stack([self._cov_ri_var(v) for v in H_vars])
        Sigma_H = np.array(
            [[self._cov_var_var(H_vars[i], H_vars[j]) for j in range(k)]
             for i in range(k)]
        )
        # E[R|H=h] = Sigma_RH @ Sigma_H^+ @ h  (pseudoinverse handles collinear H)
        Sigma_H_pinv = np.linalg.pinv(Sigma_H)
        return Sigma_RH @ (Sigma_H_pinv @ h_vals.ravel())

    # ------------------------------------------------------------------
    # Separator defect
    # ------------------------------------------------------------------

    def separator_defect_analytical(self, H_vars: list) -> float:
        """Compute epsilon_t(H) = max_{i!=j} |corr(R_i, R_j | H)|."""
        Sigma_cond = self.conditional_cov_analytical(H_vars)
        std = np.sqrt(np.diag(Sigma_cond))
        # Avoid division by zero
        std = np.where(std == 0, 1e-12, std)
        corr = Sigma_cond / np.outer(std, std)
        np.fill_diagonal(corr, 0.0)
        return float(np.max(np.abs(corr)))

    # ------------------------------------------------------------------
    # Joint covariance of all observable non-future variables
    # ------------------------------------------------------------------

    def get_joint_cov_all_vars(self) -> pd.DataFrame:
        """Return (8+5)x(8+5) joint covariance matrix of R and non-future candidates."""
        labels = _RETURN_COLS + _NON_FUTURE_VARS
        n = len(labels)  # 13
        C = np.zeros((n, n))

        Sigma_R = self._sigma_r()
        C[:8, :8] = Sigma_R

        for j, v in enumerate(_NON_FUTURE_VARS):
            cov_rv = self._cov_ri_var(v)
            C[:8, 8 + j] = cov_rv
            C[8 + j, :8] = cov_rv

        for i, v1 in enumerate(_NON_FUTURE_VARS):
            for j, v2 in enumerate(_NON_FUTURE_VARS):
                C[8 + i, 8 + j] = self._cov_var_var(v1, v2)

        return pd.DataFrame(C, index=labels, columns=labels)


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def create_default_system() -> GaussianAPOSystem:
    """Create and return a GaussianAPOSystem with default parameters."""
    return GaussianAPOSystem()
