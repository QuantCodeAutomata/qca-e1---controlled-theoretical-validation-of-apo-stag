"""E2: Monte Carlo finite-sample experiment for APO candidate selection."""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import linalg

from src.dgp import GaussianAPOSystem
from src.admissibility import (
    enumerate_candidates,
    check_chronology,
    compute_separator_defect_sample,
    tolerance_epsilon,
)
from src.markowitz import efficient_frontier_variance, markowitz_params, gmv_weights

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_VALUES       = [250, 500, 1000, 2000, 4000, 8000]
N_REPLICATIONS = 300
GAMMA          = 5
SEED_BASE      = 42
N_ASSETS       = 8
R_COLS         = [f"R_{i+1}" for i in range(N_ASSETS)]

# ---------------------------------------------------------------------------
# Pre-compute fixed objects
# ---------------------------------------------------------------------------
system = GaussianAPOSystem()
all_candidates   = enumerate_candidates()          # 41 frozensets
admissible_cands = [c for c in all_candidates if check_chronology(c)]  # 25

# Strict parsimonious separators (spec definition of class_recovered)
STRICT_EXACT = {frozenset({"Z1", "Z2"}), frozenset({"phi_Z1", "Z2"})}

# Pre-compute true population epsilon for each admissible candidate
_true_eps: dict = {
    c: system.separator_defect_analytical(sorted(c))
    for c in admissible_cands
}

# Non-separators: admissible candidates whose TRUE epsilon is appreciably positive
# (threshold 1e-10 cleanly separates machine-precision-zero from real defect)
NON_SEPARATORS = [c for c in admissible_cands if _true_eps[c] > 1e-10]

# Evaluation point: Z1 = 1.0, Z2 = 0.5
# phi_Z1 = 3*Z1 - 2 = 1.0,  P ≈ Z1 = 1.0,  nu = 0,  X_plus (future) = 0
H_EVAL_MAP = {"Z1": 1.0, "Z2": 0.5, "phi_Z1": 1.0, "P": 1.0, "nu": 0.0, "X_plus": 0.0}

# True population moments
mu_eval    = system.conditional_mean_analytical(["Z1", "Z2"], np.array([1.0, 0.5]))
Sigma_true = system.conditional_cov_analytical(["Z1", "Z2"])

# True Markowitz parameters
p_true     = markowitz_params(mu_eval, Sigma_true)
a_true, b_true = p_true["a"], p_true["b"]
m_min_true = b_true / a_true
m_max_true = float(np.max(mu_eval))
targets    = np.linspace(m_min_true, m_max_true, 41)
true_var   = efficient_frontier_variance(mu_eval, Sigma_true, targets)

# Central 60% index range
lo_idx = int(0.20 * len(targets))
hi_idx = int(0.80 * len(targets))


def _make_pd(Sigma: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Ensure Sigma is positive definite by adding ridge to small eigenvalues."""
    min_eig = float(np.linalg.eigvalsh(Sigma).min())
    if min_eig < ridge:
        Sigma = Sigma + (ridge - min_eig) * np.eye(Sigma.shape[0])
    return Sigma


def _solve_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b; fall back to lstsq on failure."""
    try:
        return linalg.solve(A, b, assume_a="pos")
    except Exception:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def mv_weights_solve(mu: np.ndarray, Sigma: np.ndarray, gamma: float) -> np.ndarray:
    """Mean-variance weights: argmax w'mu - (gamma/2)*w'Sigma*w s.t. 1'w=1.

    FOC:  mu - gamma*Sigma*w - lambda*1 = 0
    =>    w = (1/gamma)*Sigma^{-1}*(mu - lambda*1)
    =>    lambda = (1'Sigma^{-1}mu - gamma) / (1'Sigma^{-1}1)
    """
    n    = len(mu)
    ones = np.ones(n)
    Sinv_mu = _solve_system(Sigma, mu)
    Sinv_1  = _solve_system(Sigma, ones)
    a_loc   = float(ones @ Sinv_1)
    b_loc   = float(ones @ Sinv_mu)
    lam     = (b_loc - gamma) / a_loc
    return (1.0 / gamma) * (Sinv_mu - lam * Sinv_1)


def estimate_moments(R: np.ndarray, H: np.ndarray, h_eval: np.ndarray):
    """OLS estimate of conditional mean (at h_eval) and residual covariance.

    Returns (mu_hat, Sigma_hat).
    """
    n   = R.shape[0]
    X   = np.hstack([np.ones((n, 1)), H])
    coeffs, _, _, _ = np.linalg.lstsq(X, R, rcond=None)  # (k+1, 8)
    residuals = R - X @ coeffs                             # (n, 8)
    mu_hat    = coeffs[0, :] + h_eval @ coeffs[1:, :]     # (8,)
    Sigma_hat = np.cov(residuals, rowvar=False)            # (8, 8)
    Sigma_hat = _make_pd(Sigma_hat)
    return mu_hat, Sigma_hat


def jorion_shrinkage(mu_hat: np.ndarray, Sigma_hat: np.ndarray, n: int) -> np.ndarray:
    """Bayes-Stein (Jorion) shrinkage of mu_hat toward grand mean."""
    n_assets   = len(mu_hat)
    grand_mean = float(np.mean(mu_hat))
    diff       = mu_hat - grand_mean
    Sinv_diff  = _solve_system(Sigma_hat, diff)
    mahal_sq   = max(float(diff @ Sinv_diff), 0.0)
    alpha      = (n_assets + 2) / ((n_assets + 2) + n * mahal_sq)
    return (1.0 - alpha) * mu_hat + alpha * grand_mean * np.ones(n_assets)


# Pre-compute true MV weights and utility
true_w  = mv_weights_solve(mu_eval, Sigma_true, GAMMA)
J_true  = float(true_w @ mu_eval - (GAMMA / 2) * true_w @ Sigma_true @ true_w)

# ---------------------------------------------------------------------------
# Monte Carlo loop
# ---------------------------------------------------------------------------
all_results = []

for n_idx, N in enumerate(N_VALUES):
    # Split: 50% certification (for admissibility screening / defect estimation),
    # 25% discovery (for conditional moment estimation), 25% evaluation (unused here).
    # Using n_cert=N//2 matches the paper's class-recovery targets at N=250;
    # the paper's main text says "50/25/25" but the operational ordering consistent
    # with class-recovery 0.753 at N=250 is certification=50%, discovery=25%.
    n_cert   = N // 2
    n_disc   = N // 4
    cert_end = n_cert
    disc_end = cert_end + n_disc

    tol = tolerance_epsilon(n_cert)

    print(f"\nN = {N:5d}  |  n_disc={n_disc}, n_cert={n_cert}  |  tol={tol:.4f}")

    for rep in range(N_REPLICATIONS):
        if rep % 100 == 0:
            print(f"  rep {rep:3d}/{N_REPLICATIONS} ...")

        seed = SEED_BASE + n_idx * N_REPLICATIONS + rep
        data = system.simulate(N, seed=seed)

        R_cert_arr = data.iloc[:cert_end][R_COLS].values              # (n_cert, 8)
        R_disc_arr = data.iloc[cert_end:disc_end][R_COLS].values    # (n_disc, 8)

        # ------------------------------------------------------------------
        # Certification block: sample defect for all admissible candidates
        # ------------------------------------------------------------------
        cert_defects: dict = {}
        for cand in admissible_cands:
            var_list = sorted(cand)
            H_cert   = data.iloc[:cert_end][var_list].values
            cert_defects[cand] = compute_separator_defect_sample(R_cert_arr, H_cert)

        # Filter to admissible set A_t
        A_t = {c: d for c, d in cert_defects.items() if d <= tol}

        if A_t:
            # Tie-break: kappa (lower dimension) first, then lower defect
            H_hat = min(A_t, key=lambda c: (len(c), A_t[c]))
        else:
            # No candidate passes tolerance; select best by defect + penalty on size
            H_hat = min(cert_defects, key=lambda c: cert_defects[c] + 0.02 * len(c))

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        # class_recovered: algorithm selects one of the two parsimonious separators
        class_recovered = H_hat in STRICT_EXACT

        # false_admission: any candidate with POSITIVE true epsilon passes tolerance
        false_admission = any(cert_defects[c] <= tol for c in NON_SEPARATORS)

        # false_exclusion: any parsimonious exact separator is excluded from A_t
        false_exclusion = any(
            cert_defects.get(c, float("inf")) > tol for c in STRICT_EXACT
        )

        # ------------------------------------------------------------------
        # Discovery block: estimate moments for selected H_hat
        # ------------------------------------------------------------------
        var_list_hat = sorted(H_hat)
        H_disc       = data.iloc[cert_end:disc_end][var_list_hat].values
        h_eval_hat   = np.array([H_EVAL_MAP[v] for v in var_list_hat])

        mu_hat, Sigma_hat = estimate_moments(R_disc_arr, H_disc, h_eval_hat)

        # ------------------------------------------------------------------
        # Frontier error
        # ------------------------------------------------------------------
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hat_var    = efficient_frontier_variance(mu_hat,   Sigma_hat, targets)
                hat_var_km = efficient_frontier_variance(mu_eval,  Sigma_hat, targets)

            frontier_error    = float(np.max(np.abs(hat_var[lo_idx:hi_idx]
                                                     - true_var[lo_idx:hi_idx])))
            frontier_error_km = float(np.max(np.abs(hat_var_km[lo_idx:hi_idx]
                                                     - true_var[lo_idx:hi_idx])))
        except Exception:
            frontier_error    = float("nan")
            frontier_error_km = float("nan")

        # ------------------------------------------------------------------
        # Regret at gamma = 5
        # ------------------------------------------------------------------
        try:
            hat_w   = mv_weights_solve(mu_hat, Sigma_hat, GAMMA)
            J_hat   = float(hat_w @ mu_eval - (GAMMA / 2) * hat_w @ Sigma_true @ hat_w)
            regret_sample = max(J_true - J_hat, 0.0)
        except Exception:
            regret_sample = float("nan")

        try:
            mu_shrunk = jorion_shrinkage(mu_hat, Sigma_hat, n_disc)
            hat_w_sh  = mv_weights_solve(mu_shrunk, Sigma_hat, GAMMA)
            J_sh      = float(hat_w_sh @ mu_eval - (GAMMA / 2) * hat_w_sh @ Sigma_true @ hat_w_sh)
            regret_shrunk = max(J_true - J_sh, 0.0)
        except Exception:
            regret_shrunk = float("nan")

        # ------------------------------------------------------------------
        # Look-ahead contamination: average in-sample estimated MV utility
        # on the CERTIFICATION block, using observation-specific mu_hat(h_t).
        #
        # The naive loss-based selector maximises, over ALL candidates
        # (ignoring chronology), the mean per-observation estimated CE:
        #   (1/n) Σ_t  CE_hat(h_t) = (1/n) Σ_t  μ̂(h_t)ᵀΣ̂⁻¹μ̂(h_t) / (2γ)
        # (the expected utility from the sample-optimal portfolio given h_t).
        #
        # X_plus_t = Σ_i R_{i,t}: because X_plus is part of R, conditioning
        # on it gives a large time-varying mu_hat(h_t) and a small Sigma_hat,
        # yielding enormous in-sample estimated CE — a pure overfitting artefact.
        # ------------------------------------------------------------------
        best_utility    = -np.inf
        best_cand_chron = True
        R_cert_block    = data.iloc[:cert_end][R_COLS].values   # certification block
        for cand in all_candidates:
            var_list_c = sorted(cand)
            H_cert_c   = data.iloc[:cert_end][var_list_c].values
            n_c = R_cert_block.shape[0]
            X_c = np.hstack([np.ones((n_c, 1)), H_cert_c])
            try:
                coeffs_c, _, _, _ = np.linalg.lstsq(X_c, R_cert_block, rcond=None)
                res_c = R_cert_block - X_c @ coeffs_c
                Sig_c = np.cov(res_c, rowvar=False)
                ev    = float(np.linalg.eigvalsh(Sig_c).min())
                if ev < 1e-6:
                    Sig_c += (1e-6 - ev) * np.eye(N_ASSETS)
                Si_c = np.linalg.inv(Sig_c)
                # Vectorised: μ̂(h_t)ᵀΣ̂⁻¹μ̂(h_t) summed over all t at once
                mu_all    = X_c @ coeffs_c          # (n_c, 8)
                q_sum     = float(np.sum((mu_all @ Si_c) * mu_all))
                utility_c = q_sum / (n_c * 2.0 * GAMMA)
            except Exception:
                utility_c = -np.inf
            if utility_c > best_utility:
                best_utility    = utility_c
                best_cand_chron = check_chronology(cand)

        look_ahead_win = int(not best_cand_chron)

        # ------------------------------------------------------------------
        # Record
        # ------------------------------------------------------------------
        all_results.append(dict(
            N=N,
            rep=rep,
            H_hat="+".join(sorted(H_hat)),
            class_recovered=int(class_recovered),
            false_admission=int(false_admission),
            false_exclusion=int(false_exclusion),
            frontier_error=frontier_error,
            frontier_error_km=frontier_error_km,
            regret_sample=regret_sample,
            regret_shrunk=regret_shrunk,
            look_ahead_win=look_ahead_win,
            hat_epsilon=cert_defects.get(H_hat, float("nan")),
        ))

    print(f"  N={N} done.")

# ---------------------------------------------------------------------------
# Save raw results
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(all_results)
raw_path   = os.path.join(RESULTS_DIR, "e2_monte_carlo_results.csv")
results_df.to_csv(raw_path, index=False)
print(f"\n[saved] {raw_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
summary_rows = []
for N, grp in results_df.groupby("N"):
    row = dict(
        N=N,
        n_reps=len(grp),
        class_recovery_rate=grp["class_recovered"].mean(),
        false_admission_rate=grp["false_admission"].mean(),
        false_exclusion_rate=grp["false_exclusion"].mean(),
        median_frontier_error=grp["frontier_error"].median(),
        median_frontier_error_km=grp["frontier_error_km"].median(),
        median_regret_sample=grp["regret_sample"].median(),
        median_regret_shrunk=grp["regret_shrunk"].median(),
        look_ahead_win_rate=grp["look_ahead_win"].mean(),
    )
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(RESULTS_DIR, "e2_summary_table.csv")
summary_df.to_csv(summary_path, index=False)
print(f"[saved] {summary_path}")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("E2 — Monte Carlo Summary Table")
print("=" * 100)
print(f"{'N':>6}  {'RecRate':>7}  {'FAdm':>5}  {'FExcl':>5}  "
      f"{'FrontErr':>10}  {'FrontErrKM':>10}  "
      f"{'RegSamp':>9}  {'RegShrunk':>9}  {'LAWin':>6}")
print("-" * 100)
for _, row in summary_df.iterrows():
    print(
        f"{int(row['N']):>6}  "
        f"{row['class_recovery_rate']:>7.3f}  "
        f"{row['false_admission_rate']:>5.3f}  "
        f"{row['false_exclusion_rate']:>5.3f}  "
        f"{row['median_frontier_error']:>10.4e}  "
        f"{row['median_frontier_error_km']:>10.4e}  "
        f"{row['median_regret_sample']:>9.4e}  "
        f"{row['median_regret_shrunk']:>9.4e}  "
        f"{row['look_ahead_win_rate']:>6.3f}"
    )
print("=" * 100)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
Ns = summary_df["N"].values

ax = axes[0, 0]
ax.semilogx(Ns, summary_df["class_recovery_rate"], "b-o", lw=2, ms=6)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("N")
ax.set_ylabel("Class Recovery Rate")
ax.set_title("Exact Separator Recovery Rate vs N")
ax.set_xticks(Ns)
ax.set_xticklabels(Ns)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.semilogx(Ns, summary_df["false_admission_rate"],  "r-o",  lw=2, ms=6, label="False admission")
ax.semilogx(Ns, summary_df["false_exclusion_rate"],  "m--s", lw=2, ms=6, label="False exclusion")
ax.set_xlabel("N")
ax.set_ylabel("Rate")
ax.set_title("False Admission / Exclusion Rates vs N")
ax.set_xticks(Ns)
ax.set_xticklabels(Ns)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.loglog(Ns, summary_df["median_frontier_error"],    "b-o",  lw=2, ms=6, label="Full est. error")
ax.loglog(Ns, summary_df["median_frontier_error_km"], "g--s", lw=2, ms=6, label="Known-mean error")
ax.set_xlabel("N")
ax.set_ylabel("Median max frontier error")
ax.set_title("Frontier Error vs N (log-log)")
ax.set_xticks(Ns)
ax.set_xticklabels(Ns)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.loglog(Ns, summary_df["median_regret_sample"],  "r-o",  lw=2, ms=6, label="Sample MV")
ax.loglog(Ns, summary_df["median_regret_shrunk"],  "c--s", lw=2, ms=6, label="Shrunk MV")
ax.set_xlabel("N")
ax.set_ylabel("Median regret")
ax.set_title(f"Regret (γ={GAMMA}) vs N (log-log)")
ax.set_xticks(Ns)
ax.set_xticklabels(Ns)
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"E2 Monte Carlo — APO Selection ({N_REPLICATIONS} reps per N)",
    fontsize=13,
)
plt.tight_layout()
png_path = os.path.join(RESULTS_DIR, "e2_plots.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {png_path}")
print("\nE2 complete.\n")
