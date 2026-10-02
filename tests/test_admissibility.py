import pytest
import numpy as np
from src.admissibility import (
    enumerate_candidates,
    check_chronology,
    FUTURE_VARS,
    CANDIDATE_VARS,
    tolerance_epsilon,
    stage1_select,
    compute_candidate_table,
)
from src.dgp import create_default_system


def test_total_candidates_41():
    cands = enumerate_candidates()
    assert len(cands) == 41


def test_chronology_rejects_16():
    cands = enumerate_candidates()
    rejected = [c for c in cands if not check_chronology(c)]
    assert len(rejected) == 16


def test_admissible_candidates_25():
    cands = enumerate_candidates()
    admissible = [c for c in cands if check_chronology(c)]
    assert len(admissible) == 25


def test_all_rejected_contain_X_plus():
    cands = enumerate_candidates()
    for c in cands:
        if not check_chronology(c):
            assert 'X_plus' in c, f'{c} rejected but no X_plus'


def test_tolerance_formula():
    assert abs(tolerance_epsilon(100) - (0.10 + 2 / 10)) < 1e-10
    assert abs(tolerance_epsilon(400) - (0.10 + 2 / 20)) < 1e-10


def test_candidate_table_shape():
    sys = create_default_system()
    tbl = compute_candidate_table(sys)
    assert len(tbl) == 41


def test_selected_is_exact_separator():
    sys = create_default_system()
    tbl = compute_candidate_table(sys)
    selected = tbl[tbl['selected_flag'] == True]
    assert len(selected) >= 1
    assert selected['epsilon_analytical'].iloc[0] < 1e-10


def test_stage1_select_returns_frozenset():
    sys = create_default_system()
    tbl = compute_candidate_table(sys)
    result = stage1_select(tbl)
    assert isinstance(result, frozenset)


def test_enumerate_sizes():
    cands = enumerate_candidates(max_size=3)
    for c in cands:
        assert 1 <= len(c) <= 3
