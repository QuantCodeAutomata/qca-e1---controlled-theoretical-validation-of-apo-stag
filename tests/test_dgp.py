import numpy as np
import pandas as pd
import pytest
from src.dgp import GaussianAPOSystem, create_default_system


def test_simulate_shape():
    sys = create_default_system()
    df = sys.simulate(100, seed=0)
    assert df.shape == (100, 14), f'Expected (100,14), got {df.shape}'


def test_simulate_columns():
    sys = create_default_system()
    df = sys.simulate(50, seed=0)
    expected_cols = [f'R_{i+1}' for i in range(8)] + ['Z1', 'Z2', 'phi_Z1', 'P', 'nu', 'X_plus']
    for col in expected_cols:
        assert col in df.columns


def test_phi_z1_is_recoding():
    sys = create_default_system()
    df = sys.simulate(1000, seed=42)
    np.testing.assert_allclose(df['phi_Z1'].values, 3 * df['Z1'].values - 2, rtol=1e-10)


def test_proxy_p_structure():
    sys = create_default_system()
    df = sys.simulate(5000, seed=42)
    # P = Z1 + 0.7*e, so Corr(P, Z1) = 1/sqrt(1 + 0.49) = 1/sqrt(1.49)
    expected_corr = 1 / np.sqrt(1.49)
    actual_corr = np.corrcoef(df['P'], df['Z1'])[0, 1]
    assert abs(actual_corr - expected_corr) < 0.05, (
        f'Corr(P,Z1)={actual_corr:.3f}, expected ~{expected_corr:.3f}'
    )


def test_conditional_cov_exact_separator():
    sys = create_default_system()
    Sigma_H = sys.conditional_cov_analytical(['Z1', 'Z2'])
    Sigma_off = Sigma_H.copy()
    np.fill_diagonal(Sigma_off, 0)
    assert np.max(np.abs(Sigma_off)) < 1e-10, 'Off-diagonal should be 0 for exact separator'


def test_recoding_invariance():
    sys = create_default_system()
    cov_z1z2 = sys.conditional_cov_analytical(['Z1', 'Z2'])
    cov_phi_z2 = sys.conditional_cov_analytical(['phi_Z1', 'Z2'])
    diff = np.max(np.abs(cov_z1z2 - cov_phi_z2))
    assert diff < 1e-10, f'Recoding invariance violated: max diff = {diff}'


def test_separator_defect_zero_for_exact_separator():
    sys = create_default_system()
    assert sys.separator_defect_analytical(['Z1', 'Z2']) < 1e-10
    assert sys.separator_defect_analytical(['phi_Z1', 'Z2']) < 1e-10


def test_separator_defect_p_z2():
    sys = create_default_system()
    defect = sys.separator_defect_analytical(['P', 'Z2'])
    assert defect >= 0.299, f'Expected >= 0.299, got {defect:.4f}'


def test_separator_defect_bounds():
    sys = create_default_system()
    from src.admissibility import enumerate_candidates, check_chronology
    for cand in enumerate_candidates():
        if check_chronology(cand):
            defect = sys.separator_defect_analytical(list(cand))
            assert 0 <= defect <= 1, f'Defect {defect} out of [0,1] for {cand}'


def test_get_joint_cov():
    sys = create_default_system()
    df = sys.get_joint_cov_all_vars()
    n = len(df)
    assert n == 13, f'Expected 13x13, got {n}x{n}'
    assert abs(df.loc['P', 'P'] - 1.49) < 1e-10
    assert abs(df.loc['Z1', 'phi_Z1'] - 3.0) < 1e-10
