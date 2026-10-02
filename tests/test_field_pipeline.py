import numpy as np
import pytest
from src.field_pipeline import (
    greedy_forward_select,
    estimate_factor_model,
    conditional_cov_from_factor_model,
    gmv_portfolio_unconstrained,
    benchmark_covariances,
    portfolio_stats,
)


def make_factor_data(n=200, n_assets=8, n_drivers=5, seed=0):
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n, 2))
    B = rng.standard_normal((n_assets, 2)) * 0.5 + 0.3
    eps = rng.standard_normal((n, n_assets)) * 0.7
    R = factors @ B.T + eps
    drivers = np.column_stack([
        factors,
        rng.standard_normal((n, n_drivers - 2)),
    ])
    driver_names = [f'd{i}' for i in range(n_drivers)]
    return R, drivers, driver_names


def test_greedy_select_returns_drivers():
    R, drivers, names = make_factor_data()
    selected, defect = greedy_forward_select(R, drivers, names, max_drivers=3, tol=0.5)
    assert isinstance(selected, list)
    assert len(selected) <= 3
    assert defect >= 0


def test_estimate_factor_model_shapes():
    R, drivers, names = make_factor_data(n=300)
    B_hat, V_hat, Omega_hat = estimate_factor_model(R, drivers[:, :2])
    assert B_hat.shape == (8, 2)
    assert V_hat.shape == (2, 2)
    assert Omega_hat.shape == (8, 8)


def test_conditional_cov_positive_definite():
    R, drivers, names = make_factor_data(n=300)
    B_hat, V_hat, Omega_hat = estimate_factor_model(R, drivers[:, :2])
    Sigma = conditional_cov_from_factor_model(B_hat, V_hat, Omega_hat)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals > -1e-6), f'Min eigenvalue: {eigvals.min():.2e}'


def test_gmv_weights_sum_to_one():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 8))
    Sigma = A @ A.T + np.eye(8)
    w = gmv_portfolio_unconstrained(Sigma)
    assert abs(np.sum(w) - 1.0) < 1e-6


def test_benchmark_covariances_psd():
    rng = np.random.default_rng(42)
    R = rng.standard_normal((200, 8))
    benchmarks = benchmark_covariances(R)
    for name, cov in benchmarks.items():
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > -1e-6), f'{name} cov is not PSD'


def test_portfolio_stats_keys():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((100, 8)) * 0.01
    weights = np.ones(8) / 8
    stats = portfolio_stats(weights, returns)
    for key in ['realized_vol', 'sharpe', 'gross_leverage']:
        assert key in stats, f'Missing key: {key}'
