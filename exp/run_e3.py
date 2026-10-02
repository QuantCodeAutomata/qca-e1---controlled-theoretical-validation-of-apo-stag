"""E3: Causal Admissibility / Intervention Invariance Experiment.

DGP: 6 assets, 1 latent factor F (unobserved).
  R_i = b * F + sigma * eps_i,  b = 0.681, sigma = 0.6
  Z   = F + eta_Z * eta,        eta_Z = 0.173  (structural measurement)
  P   = F + eta_P * e,          eta_P = 0.173  (proxy measurement)

Both Z and P are noisy measurements of F with the same noise level, so their
pre-intervention separator defects are analytically equal (~0.036).

Pre-intervention analytical defects:
  defect({Z}) = defect({P}) = 0.036  (symmetric, equal observational quality)

Post-intervention do(P := P_hat), P_hat independent of F:
  defect_post({Z}) = 0.036  (Z still tracks F, stable)
  defect_post({P}) = 0.563  (P_hat carries no information about F)

Conditional-law selection (based only on sample defect):
  With equal population defects, ~50% Z and ~50% P selected.

Interventional admissibility (Wald test across regimes):
  Z: regression R~Z is invariant (F is structural) -> admitted
  P: regression R~P breaks after intervention (P_hat is independent) -> rejected
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.admissibility import compute_separator_defect_sample
from src.markowitz import gmv_weights  # noqa: F401 (available for extensions)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

class E3DGP:
    """Single-factor Gaussian DGP calibrated for E3.

    True latent factor F is unobserved.  Both Z and P are noisy measurements
    of F with the same noise level (eta_Z = eta_P = 0.173), so their
    pre-intervention separator defects are analytically identical (~0.036).

    The crucial difference is structural:
      Z = F + eta_Z * noise   (structurally stable measurement)
      P = F + eta_P * e       (proxy whose link to F breaks under intervention)
    """

    n_assets: int = 6
    b: np.ndarray = np.array([0.681] * 6)
    sigma: np.ndarray = np.array([0.6] * 6)
    eta_Z: float = 0.173   # noise on Z measurement
    eta_P: float = 0.173   # noise on P measurement (same level → equal defect)

    # Pre-compute the noise-shrinkage factor  1 / (1 + eta^2)
    def _shrinkage(self, eta: float) -> float:
        return 1.0 / (1.0 + eta ** 2)

    def _defect_noisy_proxy(self, eta: float) -> float:
        """Analytical separator defect for a noisy measurement H = F + eta*noise."""
        # E[F | H] = H * shrinkage,  Var(F | H) = eta^2 * shrinkage
        v = eta ** 2 * self._shrinkage(eta)   # = eta^2 / (1 + eta^2)
        b2 = self.b ** 2
        cov = np.outer(self.b, self.b) * v
        var = b2 * v + self.sigma ** 2
        np.fill_diagonal(cov, var)
        corr = cov / np.sqrt(np.outer(var, var))
        np.fill_diagonal(corr, 0.0)
        return float(np.max(np.abs(corr)))

    def simulate_baseline(self, n: int, seed: int):
        """Returns (R, Z, P) under the observational distribution.

        F ~ N(0,1) is the latent factor; both Z and P are independent noisy
        measurements of F with the same noise SD.
        """
        rng = np.random.default_rng(seed)
        F   = rng.standard_normal(n)                         # latent factor
        eta = rng.standard_normal(n)                         # noise for Z
        e   = rng.standard_normal(n)                         # noise for P
        eps = rng.standard_normal((n, self.n_assets))
        Z = F + self.eta_Z * eta
        P = F + self.eta_P * e
        R = np.outer(F, self.b) + eps * self.sigma
        return R, Z, P

    def simulate_intervened(self, n: int, seed: int):
        """Returns (R, Z_post, P_hat) after do(P := P_hat).

        P_hat is an independent draw with the same marginal as P.
        R and Z still follow their structural equations driven by F.
        """
        rng = np.random.default_rng(seed)
        F   = rng.standard_normal(n)                         # structural factor
        eta = rng.standard_normal(n)                         # noise for Z
        eps = rng.standard_normal((n, self.n_assets))
        F_hat = rng.standard_normal(n)                       # independent factor
        e_hat = rng.standard_normal(n)
        Z     = F + self.eta_Z * eta                         # stable
        P_hat = F_hat + self.eta_P * e_hat                   # independent of F
        R     = np.outer(F, self.b) + eps * self.sigma       # depends on F
        return R, Z, P_hat

    def defect_Z_analytical(self) -> float:
        """Pre-intervention defect of Z = F + eta_Z * noise."""
        return self._defect_noisy_proxy(self.eta_Z)

    def defect_P_analytical(self) -> float:
        """Pre-intervention defect of P = F + eta_P * noise."""
        return self._defect_noisy_proxy(self.eta_P)

    def defect_P_post_analytical(self) -> float:
        """After do(P := P_hat): P_hat is independent of F → defect = max|corr(R)|."""
        b2    = self.b ** 2
        var_R = b2 + self.sigma ** 2
        cov_R = np.outer(self.b, self.b)
        corr_R = cov_R / np.sqrt(np.outer(var_R, var_R))
        np.fill_diagonal(corr_R, 0.0)
        return float(np.max(np.abs(corr_R)))


# ---------------------------------------------------------------------------
# OLS utilities
# ---------------------------------------------------------------------------

def _ols(R: np.ndarray, H: np.ndarray):
    """Fit R = [1, H] @ coef via OLS. Returns (coef, residuals, XtX_inv)."""
    n = R.shape[0]
    if H.ndim == 1:
        H = H[:, None]
    X = np.hstack([np.ones((n, 1)), H])
    coef, _, _, _ = np.linalg.lstsq(X, R, rcond=None)
    residuals = R - X @ coef
    XtX_inv = np.linalg.inv(X.T @ X)
    return coef, residuals, XtX_inv


def wald_test_equal_slopes(
    R1: np.ndarray, H1: np.ndarray,
    R2: np.ndarray, H2: np.ndarray,
) -> tuple[float, float, int]:
    """Wald chi2 test of H0: slope coefficients equal across two samples.

    Sums contribution (beta1_ja - beta2_ja)^2 / Var(beta1_ja - beta2_ja) over
    all slope indices j and all assets a.  Returns (chi2, p_value, dof).
    """
    n1, n2 = R1.shape[0], R2.shape[0]
    n_assets = R1.shape[1]

    coef1, res1, XtX1_inv = _ols(R1, H1)
    coef2, res2, XtX2_inv = _ols(R2, H2)

    k = coef1.shape[0]          # intercept + n_slopes
    s2_1 = np.sum(res1**2, axis=0) / max(n1 - k, 1)  # (n_assets,)
    s2_2 = np.sum(res2**2, axis=0) / max(n2 - k, 1)

    diag1 = np.diag(XtX1_inv)[1:]  # slope-block diagonal entries
    diag2 = np.diag(XtX2_inv)[1:]

    W, dof = 0.0, 0
    for j in range(k - 1):
        for a in range(n_assets):
            var_diff = s2_1[a] * diag1[j] + s2_2[a] * diag2[j]
            if var_diff > 1e-16:
                diff = coef1[j + 1, a] - coef2[j + 1, a]
                W += diff**2 / var_diff
                dof += 1

    p_val = float(1.0 - stats.chi2.cdf(W, df=max(dof, 1)))
    return W, p_val, dof


def oos_mse(
    R_test: np.ndarray, H_test: np.ndarray,
    R_train: np.ndarray, H_train: np.ndarray,
) -> float:
    """Out-of-sample mean squared prediction error (averaged over obs and assets)."""
    if H_train.ndim == 1:
        H_train = H_train[:, None]
    if H_test.ndim == 1:
        H_test = H_test[:, None]
    n_tr, n_te = R_train.shape[0], R_test.shape[0]
    X_tr = np.hstack([np.ones((n_tr, 1)), H_train])
    X_te = np.hstack([np.ones((n_te, 1)), H_test])
    coef, _, _, _ = np.linalg.lstsq(X_tr, R_train, rcond=None)
    return float(np.mean((R_test - X_te @ coef) ** 2))


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

N_HALF = 2000        # observations per regime
N_REPS_NULL = 300    # replications for null size check
N_REPS_MAIN = 200    # replications for main experiment
ALPHA = 0.05         # significance level

dgp = E3DGP()

# ---------------------------------------------------------------------------
# Print analytical benchmarks
# ---------------------------------------------------------------------------

print("=" * 62)
print("E3: Causal Admissibility / Intervention Invariance")
print("=" * 62)
print(f"  DGP:  n_assets={dgp.n_assets}, b={dgp.b[0]}, "
      f"sigma={dgp.sigma[0]}, eta_Z={dgp.eta_Z}, eta_P={dgp.eta_P}")
print(f"  Z=F+eta_Z*noise, P=F+eta_P*e  (same noise level -> equal defect)")
print(f"  Analytical defect(Z) [pre]        = {dgp.defect_Z_analytical():.4f}")
print(f"  Analytical defect(P) [pre]        = {dgp.defect_P_analytical():.4f}")
print(f"  Analytical defect(Z) [post]       = {dgp.defect_Z_analytical():.4f}  (stable)")
print(f"  Analytical defect(P) [post]       = {dgp.defect_P_post_analytical():.4f}  (inflated)")
print()

# ---------------------------------------------------------------------------
# Null size check (both regimes observational — H0 is true)
# ---------------------------------------------------------------------------

print("-" * 62)
print(f"Null size check  (N_REPS={N_REPS_NULL}, N_half={N_HALF})")
print("-" * 62)

null_rej_Z: list[bool] = []
null_rej_P: list[bool] = []

for rep in range(N_REPS_NULL):
    R1, Z1, P1 = dgp.simulate_baseline(N_HALF, seed=rep)
    R2, Z2, P2 = dgp.simulate_baseline(N_HALF, seed=N_REPS_NULL + rep)

    _, pz, _ = wald_test_equal_slopes(R1, Z1, R2, Z2)
    _, pp, _ = wald_test_equal_slopes(R1, P1, R2, P2)
    null_rej_Z.append(pz < ALPHA)
    null_rej_P.append(pp < ALPHA)

    if (rep + 1) % 100 == 0:
        print(f"  null check {rep + 1}/{N_REPS_NULL} …")

null_size_Z = float(np.mean(null_rej_Z))
null_size_P = float(np.mean(null_rej_P))
print(f"  Rejection rate (Z): {null_size_Z:.3f}   (target ≈ {ALPHA:.2f})")
print(f"  Rejection rate (P): {null_size_P:.3f}   (target ≈ {ALPHA:.2f})")
print()

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

print("-" * 62)
print(f"Main experiment  (N_REPS={N_REPS_MAIN}, N_half={N_HALF})")
print("-" * 62)

SEED_PRE  = 10_000
SEED_POST = 20_000

records: list[dict] = []

for rep in range(N_REPS_MAIN):
    R_pre, Z_pre, P_pre     = dgp.simulate_baseline(N_HALF, seed=SEED_PRE  + rep)
    R_post, Z_post, P_hat   = dgp.simulate_intervened(N_HALF, seed=SEED_POST + rep)

    # ── Pre-intervention sample defects ────────────────────────────────────
    d_Z_pre = compute_separator_defect_sample(R_pre, Z_pre)
    d_P_pre = compute_separator_defect_sample(R_pre, P_pre)

    # ── Conditional-law selection (lower sample defect wins) ───────────────
    select_Z_cond = int(d_Z_pre <= d_P_pre)

    # ── Interventional invariance test (Wald, 5 % level) ──────────────────
    _, pval_Z, _ = wald_test_equal_slopes(R_pre, Z_pre,  R_post, Z_post)
    _, pval_P, _ = wald_test_equal_slopes(R_pre, P_pre,  R_post, P_hat)
    Z_admitted = int(pval_Z >= ALPHA)   # not rejected  →  admitted
    P_admitted = int(pval_P >= ALPHA)

    # ── Post-intervention sample defects ───────────────────────────────────
    d_Z_post = compute_separator_defect_sample(R_post, Z_post)
    d_P_post = compute_separator_defect_sample(R_post, P_hat)

    # ── Post-intervention prediction loss (out-of-sample MSE) ─────────────
    loss_Z = oos_mse(R_post, Z_post, R_pre, Z_pre)
    loss_P = oos_mse(R_post, P_hat,  R_pre, P_pre)

    records.append({
        'rep':                     rep,
        'defect_Z_pre':            d_Z_pre,
        'defect_P_pre':            d_P_pre,
        'select_Z_conditional':    select_Z_cond,
        'pval_Z':                  pval_Z,
        'pval_P':                  pval_P,
        'Z_admitted_interventional': Z_admitted,
        'P_admitted_interventional': P_admitted,
        'defect_Z_post':           d_Z_post,
        'defect_P_post':           d_P_post,
        'loss_Z_post':             loss_Z,
        'loss_P_post':             loss_P,
    })

    if (rep + 1) % 50 == 0:
        print(f"  main rep {rep + 1}/{N_REPS_MAIN} …")

df = pd.DataFrame(records)
csv_path = os.path.join(RESULTS_DIR, 'e3_results.csv')
df.to_csv(csv_path, index=False)
print(f"\n  Results saved → {csv_path}")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

print()
print("=" * 62)
print("E3 Summary Statistics")
print("=" * 62)

summary_cols = {
    'defect_Z_pre':              'Pre defect(Z)',
    'defect_P_pre':              'Pre defect(P)',
    'select_Z_conditional':      'Cond-law: Z selected  (fraction)',
    'Z_admitted_interventional': 'Interv. admission: Z  (fraction)',
    'P_admitted_interventional': 'Interv. admission: P  (fraction)',
    'defect_Z_post':             'Post defect(Z)',
    'defect_P_post':             'Post defect(P)',
    'loss_Z_post':               'Post OOS-MSE loss (Z)',
    'loss_P_post':               'Post OOS-MSE loss (P)',
}
for col, label in summary_cols.items():
    m, s = df[col].mean(), df[col].std()
    print(f"  {label:42s}  mean={m:.4f}  sd={s:.4f}")

print()
print(f"  Null rejection rate (Z): {null_size_Z:.3f}  "
      f"Null rejection rate (P): {null_size_P:.3f}")
print()

# Inline comparison
sel_P_cond = 1.0 - df['select_Z_conditional'].mean()
loss_ratio  = df['loss_P_post'].mean() / df['loss_Z_post'].mean()
print(f"  Cond-law:      Z={df['select_Z_conditional'].mean():.3f}  "
      f"P={sel_P_cond:.3f}")
print(f"  Post-MSE ratio P/Z = {loss_ratio:.3f}  (>1 means Z is better post-intervention)")

# ---------------------------------------------------------------------------
# Diagnostics plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle('E3 Diagnostics — Causal Admissibility / Intervention Invariance', fontsize=13)

# --- Top-left: pre-intervention sample defects ---
ax = axes[0, 0]
ax.boxplot(
    [df['defect_Z_pre'].values, df['defect_P_pre'].values],
    tick_labels=['Z (pre)', 'P (pre)'],
    patch_artist=True,
    boxprops=dict(facecolor='steelblue', alpha=0.6),
)
ax.axhline(dgp.defect_Z_analytical(), color='navy',   ls='--', lw=1.2, label='analytical Z')
ax.axhline(dgp.defect_P_analytical(), color='tomato', ls='--', lw=1.2, label='analytical P')
ax.set_title('Pre-intervention sample defects')
ax.set_ylabel('defect (max |corr| of residuals)')
ax.legend(fontsize=8)

# --- Top-right: post-intervention sample defects ---
ax = axes[0, 1]
ax.boxplot(
    [df['defect_Z_post'].values, df['defect_P_post'].values],
    tick_labels=['Z (post)', 'P_hat (post)'],
    patch_artist=True,
    boxprops=dict(facecolor='steelblue', alpha=0.6),
)
ax.axhline(dgp.defect_Z_analytical(),    color='navy',   ls='--', lw=1.2, label='analytical Z post')
ax.axhline(dgp.defect_P_post_analytical(), color='tomato', ls='--', lw=1.2, label='analytical P post')
ax.set_title('Post-intervention sample defects  do(P := P\u0302)')
ax.set_ylabel('defect')
ax.legend(fontsize=8)

# --- Bottom-left: invariance test p-values ---
ax = axes[1, 0]
bins = np.linspace(0, 1, 21)
ax.hist(df['pval_Z'], bins=bins, alpha=0.6, label='Z (structural)', color='steelblue')
ax.hist(df['pval_P'], bins=bins, alpha=0.6, label='P (proxy)',      color='tomato')
ax.axvline(ALPHA, color='k', ls='--', lw=1.2, label=f'α={ALPHA}')
ax.set_xlabel('Wald test p-value')
ax.set_ylabel('frequency')
ax.set_title('Invariance test p-values  (Z stable, P unstable)')
ax.legend(fontsize=8)

# --- Bottom-right: post-intervention OOS-MSE ---
ax = axes[1, 1]
ax.boxplot(
    [df['loss_Z_post'].values, df['loss_P_post'].values],
    tick_labels=['Z (post)', 'P\u0302 (post)'],
    patch_artist=True,
    boxprops=dict(facecolor='steelblue', alpha=0.6),
)
ax.set_title('Post-intervention OOS-MSE  (Z \u00ab P)')
ax.set_ylabel('mean squared prediction error')

plt.tight_layout()
png_path = os.path.join(RESULTS_DIR, 'e3_diagnostics.png')
plt.savefig(png_path, dpi=130)
plt.close()
print(f"  Plot saved     → {png_path}")
print()
print("E3 complete.")
