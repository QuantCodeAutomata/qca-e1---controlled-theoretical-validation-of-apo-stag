import numpy as np
import pytest
from src.markowitz import markowitz_params, efficient_frontier_variance, gmv_weights, compare_frontiers
from src.dgp import create_default_system


def make_test_params():
    rng = np.random.default_rng(0)
    n = 5
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T + np.eye(n) * 0.5
    mu = rng.standard_normal(n)
    return mu, Sigma


def test_markowitz_params_d_positive():
    mu, Sigma = make_test_params()
    params = markowitz_params(mu, Sigma)
    assert params['d'] > 0, 'ac-b^2 should be positive (Cauchy-Schwarz)'


def test_markowitz_params_a_positive():
    mu, Sigma = make_test_params()
    params = markowitz_params(mu, Sigma)
    assert params['a'] > 0, '1^T Sigma^-1 1 should be positive'


def test_gmv_weights_sum_to_one():
    _, Sigma = make_test_params()
    w = gmv_weights(Sigma)
    assert abs(np.sum(w) - 1.0) < 1e-8, f'GMV weights sum={np.sum(w)}'


def test_efficient_frontier_nonnegative_variance():
    mu, Sigma = make_test_params()
    params = markowitz_params(mu, Sigma)
    b = params['b']
    a = params['a']
    targets = np.linspace(b / a - 0.5, b / a + 0.5, 20)
    variances = efficient_frontier_variance(mu, Sigma, targets)
    assert np.all(variances >= 0), 'Frontier variance should be non-negative'


def test_frontier_minimum_at_gmv():
    mu, Sigma = make_test_params()
    params = markowitz_params(mu, Sigma)
    gmv_return = params['b'] / params['a']
    variances = efficient_frontier_variance(mu, Sigma, np.array([gmv_return]))
    gmv_var = 1 / params['a']
    assert abs(variances[0] - gmv_var) < 1e-8, (
        f'Min variance at GMV: {variances[0]:.6f} vs {gmv_var:.6f}'
    )


def test_compare_frontiers_equality():
    sys = create_default_system()
    Sigma_H = sys.conditional_cov_analytical(['Z1', 'Z2'])
    mu_H = np.array([0.01, 0.02, 0.015, 0.008, 0.025, 0.012, 0.018, 0.022])
    result = compare_frontiers(mu_H, Sigma_H, n_points=21)
    assert result['abs_diff'].max() < 1e-6, f'Max frontier diff: {result["abs_diff"].max():.2e}'


def test_conditional_moments_dimension():
    sys = create_default_system()
    Sigma_H = sys.conditional_cov_analytical(['Z1', 'Z2'])
    assert Sigma_H.shape == (8, 8)
    assert np.all(np.linalg.eigvalsh(Sigma_H) > -1e-10), 'Conditional cov should be PSD'
