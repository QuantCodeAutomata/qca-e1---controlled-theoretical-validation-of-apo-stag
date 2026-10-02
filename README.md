# APO (Admissibility and Portfolio Optimization) Framework Experiments

## Overview

This repository implements and validates the **APO methodology** — a two-stage approach to
portfolio construction that first identifies an *admissible* set of conditioning variables
(Stage I) and then applies conditional Markowitz optimization (Stage II).

### APO Methodology

**Stage I — Admissibility Selection (Separator Certification)**

A set of drivers `H` is called *admissible* (or a *separator*) for returns `R` if conditioning
on `H` renders all assets conditionally uncorrelated:

```
epsilon(H) = max_{i≠j} |Corr(R_i, R_j | H)| ≤ ε̄
```

where `ε̄(n) = 0.10 + 2/√n` is a sample-size-adaptive tolerance.  Chronological admissibility
requires that `H` contains no future information (`X_plus` is excluded).

Candidate subsets are enumerated (up to size 3 over the 6 candidate variables), each is scored
by its defect `epsilon`, and the minimum-defect admissible set with smallest cardinality is
selected.  In the field (E5/E6), a greedy forward selection algorithm approximates this over
large driver panels.

**Stage II — Conditional Markowitz**

Given the selected separator `H = h`, the conditional covariance `Σ_{R|H}` replaces the
unconditional covariance in the standard Markowitz program.  The closed-form efficient frontier
is:

```
ρ(m) = (a·m² − 2b·m + c) / d
```

where `a = 1'Σ⁻¹1`, `b = 1'Σ⁻¹μ`, `c = μ'Σ⁻¹μ`, `d = ac − b²`.

The Global Minimum Variance (GMV) portfolio `w = Σ⁻¹1 / (1'Σ⁻¹1)` requires no return
forecast and is used throughout the field experiments.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── AGENTS.md                    # Developer notes and key DGP facts
├── data/
│   ├── generate.py              # Generates synthetic_e1.csv
│   └── synthetic_e1.csv         # 2000 obs × 14 cols for E1 experiments
├── src/
│   ├── __init__.py              # Re-exports core API
│   ├── dgp.py                   # GaussianAPOSystem: 8 assets, 2 factors, 6 candidates
│   ├── admissibility.py         # Stage I: enumerate_candidates, check_chronology, stage1_select
│   ├── markowitz.py             # Stage II: efficient_frontier_variance, gmv_weights, compare_frontiers
│   └── field_pipeline.py        # Rolling utilities: greedy_forward_select, estimate_factor_model
├── exp/
│   ├── run_e5.py                # E5: Field rolling-panel APO (synthetic, 12 assets, 20 drivers)
│   └── run_e6.py                # E6: Cross-universe and aggregation extensions
├── results/
│   ├── RESULTS.md               # Key numerical results summary
│   ├── e5_rolling_results.csv   # Per-window metrics from E5
│   ├── e5_summary.csv           # Median metrics summary for E5
│   ├── e5_plots.png             # Defect and volatility panels for E5
│   ├── e6_cross_universe.csv    # Cross-universe defect comparison
│   ├── e6_search_size.csv       # Attainable defect vs candidate pool size
│   ├── e6_aggregation.csv       # Defect vs portfolio aggregation level k
│   └── e6_plots.png             # Summary panels for E6
└── tests/
    ├── __init__.py
    ├── test_dgp.py              # GaussianAPOSystem unit tests
    ├── test_admissibility.py    # Candidate enumeration and Stage I tests
    ├── test_markowitz.py        # Efficient frontier and GMV tests
    ├── test_field_pipeline.py   # Greedy selection and factor model tests
    └── test_e1_methodology.py   # End-to-end E1 methodology checks
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running Experiments

All scripts should be run from the **project root** directory.

### E5 — Field Rolling-Panel APO (Synthetic)

Simulates 12 equities driven by 3 latent factors, with 20 candidate drivers.
Runs ~25 rolling windows (756-obs each, 50/25/25 cert/est/eval split).
Compares APO conditional GMV against sample, PCA, and Ledoit-Wolf benchmarks.

```bash
python exp/run_e5.py
```

**Outputs** (in `results/`):
- `e5_rolling_results.csv` — per-window defect, portfolio volatility, Sharpe
- `e5_summary.csv` — median metrics across all windows
- `e5_plots.png` — rolling defect panel + GMV volatility comparison panel

### E6 — Cross-Universe and Aggregation Extensions (Synthetic)

Three sub-experiments:
1. **Cross-universe**: Types A (high idiosyncratic σ=0.8), B (medium σ=0.5), C (low σ=0.2)
2. **Search-size**: Attainable defect vs candidate pool size {4, 8, 20}
3. **Aggregation**: Defect vs equal-weight aggregation level k ∈ {1, 2, 4, 8, 12}

```bash
python exp/run_e6.py
```

**Outputs** (in `results/`):
- `e6_cross_universe.csv` — defect metrics per universe type
- `e6_search_size.csv` — attainable defect per pool size (5 draws each)
- `e6_aggregation.csv` — defect per aggregation level k
- `e6_plots.png` — three-panel summary figure

### Running All Tests

```bash
python -m pytest tests/ -v --tb=short
```

---

## Expected Outputs

### E5 Key Results

| Metric | Expected range |
|--------|---------------|
| Admissible window fraction | ≥ 0.5 |
| Median defect before | 0.55 – 0.80 |
| Median defect after APO | ≤ 0.27 (tolerance) |
| APO median vol vs Sample | APO ≤ Sample |

### E6 Key Results

| Experiment | Finding |
|-----------|---------|
| Cross-universe | Defect reduction is largest for Type C (factor-dominated) |
| Search size | Larger candidate pool → lower attainable defect |
| Aggregation | Increasing k reduces remaining defect (diversification aids separability) |

---

## Key Results Summary

See `results/RESULTS.md` for full numerical results after running experiments.

---

## DGP Reference

The canonical (E1) system has:
- **8 assets**, **2 factors** (Z1, Z2 ~ N(0,1))
- `R_i = 0.8·Z1 + B[i,1]·Z2 + 0.7·εᵢ`
- `phi(Z1) = 3·Z1 − 2` (perfect recoding — same separator power as Z1)
- `P = Z1 + 0.7·e` (noisy proxy: Corr(P, Z1) ≈ 0.819)
- `nu ~ N(0,1)` independent noise
- `X_plus = Z1' + Z2'` (future — chronologically inadmissible)
- **Exact separator**: `{Z1, Z2}` and `{phi_Z1, Z2}` both achieve `ε = 0`
- **41 candidates** (size ≤ 3 over 6 vars), **25 chronologically admissible**
