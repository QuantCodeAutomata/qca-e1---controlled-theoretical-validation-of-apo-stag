"""E1: Analytical APO candidate certification experiment."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dgp import GaussianAPOSystem
from src.admissibility import (
    enumerate_candidates,
    check_chronology,
    compute_separator_defect_sample,
)
from src.markowitz import compare_frontiers, markowitz_params

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Create DGP system
# ---------------------------------------------------------------------------
system = GaussianAPOSystem()

# ---------------------------------------------------------------------------
# 2. Enumerate all 41 candidates
# ---------------------------------------------------------------------------
candidates = enumerate_candidates()
assert len(candidates) == 41, f"Expected 41 candidates, got {len(candidates)}"

# ---------------------------------------------------------------------------
# 3. Chronology check — confirm 16 rejected, 25 admissible
# ---------------------------------------------------------------------------
chron_flags = [check_chronology(c) for c in candidates]
n_rejected   = sum(1 for f in chron_flags if not f)
n_admissible = sum(chron_flags)
assert n_rejected   == 16, f"Expected 16 rejected, got {n_rejected}"
assert n_admissible == 25, f"Expected 25 admissible, got {n_admissible}"

# ---------------------------------------------------------------------------
# 4. Build candidate table with epsilon for admissible candidates
# ---------------------------------------------------------------------------
rows = []
for cid, (cand, chron) in enumerate(zip(candidates, chron_flags)):
    var_list = sorted(cand)
    kappa    = len(cand)
    # Use sorted variable order for deterministic epsilon computation
    eps      = system.separator_defect_analytical(var_list) if chron else float("nan")
    rows.append(
        dict(candidate_id=cid, variables="+".join(var_list), n_vars=kappa,
             chronology_pass=chron, epsilon=eps, kappa=kappa, selected_flag=False)
    )

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 5. Rank admissible candidates: primary epsilon asc, secondary kappa asc
# ---------------------------------------------------------------------------
admissible_df = (
    df[df["chronology_pass"]]
    .sort_values(["epsilon", "kappa"])
    .reset_index(drop=True)
)

# Mark selected (global minimum epsilon, tie-break kappa)
min_eps   = admissible_df["epsilon"].iloc[0]
min_kappa = admissible_df.loc[
    np.isclose(admissible_df["epsilon"], min_eps, atol=1e-14), "kappa"
].min()
for idx in df.index:
    row = df.loc[idx]
    if (row["chronology_pass"]
            and np.isclose(row["epsilon"], min_eps, atol=1e-14)
            and row["kappa"] == min_kappa):
        df.at[idx, "selected_flag"] = True

# ---------------------------------------------------------------------------
# 6. Verify {Z1,Z2} and {phi_Z1,Z2} are top-ranked
# ---------------------------------------------------------------------------
OPTIMAL_CLASSES = {frozenset({"Z1", "Z2"}), frozenset({"phi_Z1", "Z2"})}
top_vars = [frozenset(r["variables"].split("+")) for _, r in admissible_df.iterrows()
            if np.isclose(r["epsilon"], min_eps, atol=1e-14)]
assert any(s in OPTIMAL_CLASSES for s in top_vars), \
    f"Top-ranked sets {top_vars} do not include exact separators"

# ---------------------------------------------------------------------------
# 7. Verify {P,Z2} has epsilon >= 0.299
# ---------------------------------------------------------------------------
pz2_mask = df["variables"].isin(["P+Z2", "Z2+P"])
assert pz2_mask.any(), "Candidate {P,Z2} not found"
pz2_eps = df.loc[pz2_mask, "epsilon"].values[0]
assert pz2_eps >= 0.299, f"Expected {{P,Z2}} defect >= 0.299, got {pz2_eps:.4f}"

# ---------------------------------------------------------------------------
# 8. Recoding invariance check
# ---------------------------------------------------------------------------
cov_z1z2   = system.conditional_cov_analytical(["Z1", "Z2"])
cov_phi_z2 = system.conditional_cov_analytical(["phi_Z1", "Z2"])
max_cov_diff = float(np.max(np.abs(cov_z1z2 - cov_phi_z2)))
assert max_cov_diff < 1e-14, f"Recoding invariance violated: max diff = {max_cov_diff:.3e}"

# ---------------------------------------------------------------------------
# 9. Stage II vs classical Markowitz comparison
# ---------------------------------------------------------------------------
H_STAR  = ["Z1", "Z2"]
Sigma_H = system.conditional_cov_analytical(H_STAR)
h_vals  = np.array([1.0, 0.5])
mu_H    = system.conditional_mean_analytical(H_STAR, h_vals)

frontier_df     = compare_frontiers(mu_H, Sigma_H, n_points=41)
max_frontier_diff = float(frontier_df["abs_diff"].max())
assert max_frontier_diff < 1e-10, f"Frontier max diff too large: {max_frontier_diff:.3e}"

# ---------------------------------------------------------------------------
# 10. X+ diagnostic — sample defect for one X_plus-containing candidate
# ---------------------------------------------------------------------------
# Simulate large dataset and compute sample defect for {Z1, X_plus}
big_data    = system.simulate(20_000, seed=12345)
R_cols      = [f"R_{i+1}" for i in range(8)]
R_big       = big_data[R_cols].values
H_xplus     = big_data[["Z1", "X_plus"]].values
defect_xplus = compute_separator_defect_sample(R_big, H_xplus)

# ---------------------------------------------------------------------------
# 11. Print candidate certification table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("E1 — APO Candidate Certification Table")
print("=" * 80)
print(f"{'ID':>4}  {'Variables':<28}  {'k':>2}  {'Chron':>5}  "
      f"{'epsilon':>10}  {'kappa':>5}  {'Selected':>8}")
print("-" * 80)

# Sort for display using the updated df (so selected_flag is current)
admissible_display = (
    df[df["chronology_pass"]]
    .sort_values(["epsilon", "kappa"])
    .reset_index(drop=True)
)
display_df = pd.concat([
    admissible_display,
    df[~df["chronology_pass"]].sort_values("kappa"),
], ignore_index=True)

for _, row in display_df.iterrows():
    eps_str  = f"{row['epsilon']:.4e}" if not np.isnan(row["epsilon"]) else "  n/a (X+)"
    sel_str  = "  *" if row["selected_flag"] else ""
    chron_str = "  yes" if row["chronology_pass"] else "   no"
    print(f"{row['candidate_id']:>4}  {row['variables']:<28}  {row['n_vars']:>2}  "
          f"{chron_str}  {eps_str:>10}  {row['kappa']:>5}{sel_str}")

print("=" * 80)

# ---------------------------------------------------------------------------
# 12. Save candidate table CSV
# ---------------------------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, "e1_candidate_table.csv")
# Combine admissible (sorted) and inadmissible
out_df = df.copy()
out_df["epsilon_str"] = out_df["epsilon"].apply(
    lambda x: f"{x:.6e}" if not np.isnan(x) else "inadmissible"
)
out_df.sort_values(["chronology_pass", "epsilon", "kappa"],
                   ascending=[False, True, True]).to_csv(csv_path, index=False)
print(f"\n[saved] {csv_path}")

# ---------------------------------------------------------------------------
# 13. Save frontier comparison plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

targets  = frontier_df["target_return"].values
apo_var  = frontier_df["apo_variance"].values
cls_var  = frontier_df["classical_variance"].values
abs_diff = frontier_df["abs_diff"].values

ax = axes[0]
ax.plot(np.sqrt(apo_var),  targets, "b-",  lw=2, label="APO (closed-form)")
ax.plot(np.sqrt(cls_var),  targets, "r--", lw=1.5, label="Classical (SLSQP)")
ax.set_xlabel("Portfolio Std Dev")
ax.set_ylabel("Expected Return")
ax.set_title(r"Efficient Frontier: APO vs Classical ($H^*=\{Z_1,Z_2\}$)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(targets, abs_diff, "k-", lw=1.5)
ax.axhline(1e-10, color="r", ls="--", lw=1, label=r"$10^{-10}$ threshold")
ax.set_xlabel("Target Return")
ax.set_ylabel("|APO variance − Classical variance|")
ax.set_title("Frontier Difference (log scale)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"APO Stage II: Closed-form matches SLSQP to {max_frontier_diff:.2e}",
    fontsize=12
)
plt.tight_layout()
png_path = os.path.join(RESULTS_DIR, "e1_frontier_comparison.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {png_path}")

# ---------------------------------------------------------------------------
# 14. Validation summary
# ---------------------------------------------------------------------------
eps_z1z2   = system.separator_defect_analytical(["Z1",     "Z2"])
eps_phi_z2 = system.separator_defect_analytical(["phi_Z1", "Z2"])

print("\n" + "=" * 60)
print("E1 Validation Summary")
print("=" * 60)
print(f"  Total candidates:                  {len(candidates)}")
print(f"  Chronology rejects:                {n_rejected}")
print(f"  Admissible:                        {n_admissible}")
print(f"  epsilon({{Z1,Z2}}):                  {eps_z1z2:.4e}  [target: ~0]")
print(f"  epsilon({{phi_Z1,Z2}}):              {eps_phi_z2:.4e}  [target: ~4.5e-16]")
print(f"  epsilon({{P,Z2}}):                   {pz2_eps:.4f}    [target: 0.300]")
print(f"  Max cov diff (recoding inv.):       {max_cov_diff:.4e}  [target: ~2.2e-16]")
print(f"  Max frontier diff (APO vs SLSQP):   {max_frontier_diff:.4e}  [target: <1e-10]")
print(f"  X+ diagnostic defect ({{Z1,X+}},N=20k): {defect_xplus:.4f}  [flagged infeasible]")
print("=" * 60)
print("\nAll assertions passed. E1 complete.\n")
