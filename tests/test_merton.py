import numpy as np
import pandas as pd
import pytest

from adaptive_ltv.merton import MertonParams, calibrate, expected_growth, log_return_variance, simulate_returns

P = MertonParams(mu=0.002, sigma=0.012, lam=0.05, mu_j=-0.06, sigma_j=0.03)


def test_monte_carlo_mean_matches_closed_form():
    sims = simulate_returns(P, months=60, n_paths=200_000, seed=1)
    mc = np.exp(sims.sum(axis=1)).mean()
    assert mc == pytest.approx(expected_growth(P, 60), rel=0.005)


def test_monte_carlo_variance_matches_closed_form():
    sims = simulate_returns(P, months=60, n_paths=200_000, seed=2)
    assert sims.sum(axis=1).var() == pytest.approx(log_return_variance(P, 60), rel=0.02)


def test_no_jumps_reduces_to_gbm():
    p = MertonParams(0.001, 0.01, 0.0, 0.0, 0.0)
    assert expected_growth(p, 12) == pytest.approx(np.exp(12 * (0.001 + 0.01 ** 2 / 2)))


def test_calibration_recovers_parameters():
    sims = simulate_returns(P, months=20_000, n_paths=1, seed=3)[0]
    est = calibrate(pd.Series(sims))
    assert est.sigma == pytest.approx(P.sigma, rel=0.1)
    assert est.lam == pytest.approx(P.lam, rel=0.3)
    assert est.mu_j == pytest.approx(P.mu_j, abs=0.015)
