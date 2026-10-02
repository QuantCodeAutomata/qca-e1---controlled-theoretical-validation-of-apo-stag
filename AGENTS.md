# APO Framework — Repository Notes

## Project Structure
```
src/
  __init__.py        — re-exports from dgp and admissibility
  dgp.py             — GaussianAPOSystem DGP with 8 assets, 2 factors, 6 candidates
  admissibility.py   — Stage I candidate enumeration and admissibility selection
  markowitz.py       — Stage II closed-form and SLSQP Markowitz optimization
  field_pipeline.py  — Rolling-window utilities: greedy selection, factor model, benchmarks
data/
  __init__.py        — empty
  generate.py        — generates data/synthetic_e1.csv (2000 obs, seed=42)
  synthetic_e1.csv   — 2000 rows × 14 cols (R_1..R_8, Z1, Z2, phi_Z1, P, nu, X_plus)
exp/                 — experiment scripts (E1–E6)
results/             — output artifacts
tests/               — test suite
```

## Key DGP Facts
- 8 assets, 2 factors Z1, Z2 ~ N(0,1)
- R_i = B[i,0]*Z1 + B[i,1]*Z2 + 0.7*epsilon_i
- All B[:,0] = 0.8 (common factor loading); B[:,1] alternates ±
- phi_Z1 = 3*Z1 - 2 (perfect recoding of Z1 — same information content)
- P = Z1 + 0.7*e (noisy proxy for Z1)
- nu ~ N(0,1) independent noise
- X_plus = Z1_future + Z2_future (chronologically inadmissible)
- Cov(P,P) = 1 + 0.49 = 1.49; Cov(phi_Z1, phi_Z1) = 9; Cov(Z1, phi_Z1) = 3

## Key Mathematical Results
- epsilon({Z1, Z2}) = 0 exactly (full separation — perfect admissible set)
- epsilon({phi_Z1}) = epsilon({Z1}) (same residual correlation, identical info)
- epsilon({nu}) ≈ 0.656 (noise adds no information)
- 41 candidate subsets with max_size=3 over 6 variables: C(6,1)+C(6,2)+C(6,3)=41
- Collinear H (Z1+phi_Z1): Sigma_H is singular → handled via np.linalg.pinv

## Design Decisions
- `conditional_cov_analytical` uses Moore-Penrose pseudoinverse (np.linalg.pinv)
  to handle singular Sigma_H when collinear variables (e.g. Z1 + phi_Z1) are combined.
- `markowitz.py` uses scipy.linalg.solve for non-singular Sigma; compare_frontiers
  verifies closed-form vs SLSQP to machine precision (~1e-16 abs diff).
- `field_pipeline.greedy_forward_select` always adds the best driver even with no
  improvement when tol not yet met; stops early when defect <= tol.
- `portfolio_stats` handles both static (1-D weights) and time-varying (2-D weights).

## E5/E6 Experiment Design (Synthetic Field Experiments)
- N_ASSETS=12, N_FACTORS=3, N_DRIVERS=20; drivers 0-2 = true factors, 3-9 = noisy, 10-19 = pure noise
- E5 rolling: N_OBS_TOTAL=2268 (756×3), WINDOW=756, STEP=63 → 25 windows; CERT/EST/EVAL = 378/189/189
- Ridge=1e-6 added to all covariance matrices before gmv_portfolio_unconstrained (ensures PD)
- `benchmark_covariances` called with n_components=N_FACTORS (overrides default 2)
- E5 Result: 100% windows admissible, median defect 0.582→0.209 (64% reduction), APO vol < all benchmarks
- E6 cross-universe: Type C (sigma=0.2, factor-dominated) fails admissibility — high factor loading makes
  greedy certification harder (the 0.89 raw correlation overwhelms 8-driver greedy search)
- E6 search-size: smaller pool (4 drivers including all 3 true factors) beats larger pool (20) because
  greedy with noise drivers can be misdirected in early steps (greedy "curse of dimensionality")
- E6 aggregation: defect_before increases with k (more factor-driven), defect_after decreases sharply at
  k=4+ (factor signal cleaner → greedy reliably identifies true factors → >83% reduction at k≥4)

## Python Dependencies
numpy 1.26.4, pandas 2.2.2, scipy 1.14.1, sklearn 1.5.1, matplotlib 3.11.1

## Experiment Notes

### E3 — Causal Admissibility (exp/run_e3.py)
- Custom DGP: 6 assets, 1 factor Z, b=0.681, sigma=0.6, eta_P=0.173
- Analytical: defect(Z)=0.000, defect(P)=0.036, post-intervention defect(P)=0.563
- Null size check (300 reps): Wald test of OLS slope equality across two baseline halves; target ≈ 0.05
- Main experiment (200 reps, N_half=2000):
  - Conditional-law: selects Z ~96.5% (Z has lower sample defect than P consistently)
  - Interventional admission: Z ~92.5% (95% expected from size), P = 0% (always rejected)
  - Post-intervention OOS-MSE: Z≈0.36 (≈sigma²), P≈1.27 (ratio ≈3.5×)
- Outputs: results/e3_results.csv, results/e3_diagnostics.png
- Known: `defect_P_analytical` uses `np.outer(self.b, self.b)` NOT `np.outer(b²,b²)`
- Matplotlib 3.9+: boxplot uses `tick_labels=` not `labels=`

### E4 — Sequential Hysteresis (exp/run_e4.py)
- DGP: 8 assets, nodes 1-19 use (Z1,Z2), nodes 20-40 use (Z1,Z3)
- 15 candidates: nonempty subsets of {Z1,Z2,Z3,P,nu} with size ≤ 2
- Score: sample_defect + 0.02*|H|; hysteresis margins: {0, 0.02, 0.05, 0.10, 0.20}
- With N_CERT=100 (first 100 of 200 per node), {P,Z2} (pop score=0.068) can beat {Z1,Z2}
  (pop score=0.04) in sample, causing wrong initial incumbent and biased fraction_correct
- Qualitative results are correct: n_reselections decreases and lag increases with η
- n_resel=1.000 for η=0.10/0.20 (exactly one structural break switch is always triggered)
- fraction_correct and lag_to_adopt are bimodal: either lag=0 (switch at node 20) or lag=22 (never)
- Outputs: results/e4_results.csv, results/e4_plots.png

## Experiment Scripts (exp/)

### run_e1.py — Analytical Certification
- Always use `sorted(cand)` (not `list(cand)`) when calling `separator_defect_analytical`;
  frozenset iteration order is arbitrary and causes non-deterministic epsilon values.
- Build `display_df` from the *updated* `df` (after writing `selected_flag`) not from
  stale `admissible_df` captured before the flag update.
- `conditional_mean_analytical(['Z1','Z2'], [1.0, 0.5])` → mu_H = B@[1,0.5] (non-zero),
  giving a non-degenerate efficient frontier for `compare_frontiers`.
- X_plus diagnostic: `separator_defect_analytical` raises ValueError for 'X_plus';
  compute the defect empirically via `compute_separator_defect_sample` on a large sim.

### run_e2.py — Monte Carlo
- `NON_SEPARATORS`: admissible candidates with TRUE epsilon > 1e-10.
  Super-separators ({Z1,Z2,nu}, {P+Z1+Z2}, etc.) have true epsilon ≈ 0 and must NOT
  be counted as false admissions; their population defect is machine-precision zero.
- `STRICT_EXACT` = {{Z1,Z2}, {phi_Z1,Z2}} for `class_recovered` (parsimonious separators).
- H_EVAL_MAP maps each variable to its value at the evaluation point (Z1=1, Z2=0.5):
  phi_Z1 = 3*1-2 = 1.0; P ≈ 1.0; nu = 0; X_plus = 0 (future, use population mean).
- MV weights formula (1'w=1 constraint): lambda = (1'Σ⁻¹μ - γ)/(1'Σ⁻¹1),
  w = (1/γ)·Σ⁻¹·(μ - λ·1). Derived from Lagrangian FOC.
- Jorion shrinkage: alpha = (p+2)/((p+2) + n·(μ̂-μ̄)'Σ̂⁻¹(μ̂-μ̄)).
- Ensure hat_Sigma is PD: add (ridge - min_eig)*I when min_eig < ridge=1e-6.
- Seed scheme: seed = SEED_BASE + n_idx * N_REPLICATIONS + rep to get unique seeds
  per (N, replication) pair.

### Observed Monte Carlo Patterns (300 reps, seed=42)
- Class recovery rate ~20-30%, stable across N: algorithm favors 3-element super-
  separators (lower sample defect) over the parsimonious 2-element ones.
- False admission rate drops to ~0% for N≥500: non-separators (true eps≈0.30)
  reliably exceed tolerance tol = 0.10 + 2/sqrt(n_cert) for large n_cert.
- False exclusion rate → 0 quickly as N grows.
- Frontier error and regret decay at roughly 1/sqrt(N) rate.
- Look-ahead win rate ~25-30%: X_plus (independent of current R) wins on sample
  separator defect by chance at roughly the chance level (since it provides no true
  signal about current residual correlations).
