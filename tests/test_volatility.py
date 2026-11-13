import numpy as np
import pandas as pd
import pytest

from adaptive_ltv.inputs import desmooth, rolling_volatility, volatility_signal
from adaptive_ltv.membership import volatility_variable

from adaptive_ltv.inputs import DATA

CS = DATA / "cache" / "dexrnsa.csv"


def test_desmooth_recovers_known_series():
    """A series smoothed with a known rho is recovered exactly by the Geltner formula."""
    rng = np.random.default_rng(0)
    true = pd.Series(rng.normal(0, 0.02, 2000))
    rho = 0.7
    obs = [true.iloc[0]]
    for x in true.iloc[1:]:
        obs.append(rho * obs[-1] + (1 - rho) * x)
    obs = pd.Series(obs)
    recovered, rho_hat = desmooth(obs)
    assert rho_hat == pytest.approx(rho, abs=0.05)
    # recovered returns are far closer to the true ones than the smoothed series is
    err_obs = (obs - true).abs().mean()
    err_rec = (recovered - true.loc[recovered.index]).abs().mean()
    assert err_rec < err_obs / 2


def test_desmooth_rejects_invalid_rho():
    alternating = pd.Series([0.01, -0.01] * 50)
    with pytest.raises(ValueError):
        desmooth(alternating)


def test_rolling_volatility_annualizes():
    r = pd.Series([0.01, -0.01] * 24)
    vol = rolling_volatility(r, window=12)
    assert vol.iloc[-1] == pytest.approx(r.iloc[:12].std() * np.sqrt(12))


@pytest.fixture(scope="module")
def vol():
    try:
        series, _ = volatility_signal(CS)
    except FileNotFoundError:
        pytest.skip("Case-Shiller cache missing: run 00_data_test.ipynb first")
    return series


def test_desmoothing_raises_volatility():
    try:
        _, info = volatility_signal(CS)
    except FileNotFoundError:
        pytest.skip("Case-Shiller cache missing")
    assert 0 < info["rho"] < 1
    assert info["vol_desmoothed"] > info["vol_observed"]


def test_volatility_terms_cover_universe(vol):
    var = volatility_variable(vol)
    assert np.allclose(var.coverage(), 1.0)


def test_volatility_terms_are_ordered(vol):
    var = volatility_variable(vol)
    assert var.memberships(0.01)["low"] == 1.0
    p50, p75 = vol.quantile([0.50, 0.75])
    assert var.memberships(float((p50 + p75) / 2))["medium"] == 1.0
    assert var.memberships(float(vol.max()))["high"] == 1.0
