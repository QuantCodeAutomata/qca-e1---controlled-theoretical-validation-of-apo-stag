"""Test E1 methodology adherence."""
import numpy as np
import pytest
from src.dgp import create_default_system
from src.admissibility import enumerate_candidates, check_chronology, compute_candidate_table
from src.markowitz import compare_frontiers, gmv_weights


def test_e1_candidate_count():
    cands = enumerate_candidates()
    assert len(cands) == 41


def test_e1_chronology_rejects():
    cands = enumerate_candidates()
    rejected = sum(1 for c in cands if not check_chronology(c))
    assert rejected == 16


def test_e1_exact_separators():
    sys = create_default_system()
    tbl = compute_candidate_table(sys)
    admissible = tbl[tbl['chronology_pass'] == True]

    z1z2_rows = admissible[
        admissible['variables'].str.contains('Z1') &
        admissible['variables'].str.contains('Z2') &
        ~admissible['variables'].str.contains('phi')
    ]
    phi_z2_rows = admissible[
        admissible['variables'].str.contains('phi_Z1') &
        admissible['variables'].str.contains('Z2')
    ]

    if len(z1z2_rows) > 0:
        assert z1z2_rows['epsilon_analytical'].iloc[0] < 1e-10
    if len(phi_z2_rows) > 0:
        assert phi_z2_rows['epsilon_analytical'].iloc[0] < 1e-10


def test_e1_p_z2_defect_geq_0299():
    sys = create_default_system()
    defect = sys.separator_defect_analytical(['P', 'Z2'])
    assert defect >= 0.299, f'defect={defect}'


def test_e1_recoding_invariance_machine_precision():
    sys = create_default_system()
    cov1 = sys.conditional_cov_analytical(['Z1', 'Z2'])
    cov2 = sys.conditional_cov_analytical(['phi_Z1', 'Z2'])
    max_diff = np.max(np.abs(cov1 - cov2))
    assert max_diff < 1e-10, f'Max cov diff = {max_diff:.2e}'


def test_e1_stage_ii_equals_classical_markowitz():
    sys = create_default_system()
    Sigma_H = sys.conditional_cov_analytical(['Z1', 'Z2'])
    mu_H = np.array([0.05, 0.03, 0.04, 0.02, 0.06, 0.03, 0.04, 0.05])
    result = compare_frontiers(mu_H, Sigma_H, n_points=41)
    assert result['abs_diff'].max() < 1e-6, f'Max diff: {result["abs_diff"].max():.2e}'
