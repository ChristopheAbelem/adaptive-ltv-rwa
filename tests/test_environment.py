import numpy as np
import pandas as pd
import pytest

from adaptive_ltv import environment as E


def flat_scenario(n=36, rent=1.0, price=1.0):
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    return E.Scenario("flat", pd.Series(price, index=idx), pd.Series(0.05, index=idx), pd.Series(rent, index=idx),
                      pd.Series(0.0, index=range(12)))


def test_borrow_rate_is_continuous_and_increasing():
    u = np.linspace(0, 1, 101)
    r = np.array([E.borrow_rate(x) for x in u])
    assert np.all(np.diff(r) >= 0)
    assert E.borrow_rate(0.9) == pytest.approx(0.05)
    assert E.borrow_rate(1.0) == pytest.approx(0.65)


def test_calm_market_has_no_losses():
    res = E.run(flat_scenario(), E.fixed_policy())
    assert res["liquidations"].sum() == 0
    assert res["loss_to_date"].iloc[-1] == pytest.approx(0, abs=1e-9)


def test_same_seed_same_result():
    a = E.run(flat_scenario(rent=0.5), E.fixed_policy(), seed=3)
    b = E.run(flat_scenario(rent=0.5), E.fixed_policy(), seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_accounting_stays_consistent():
    """Debt, supply and bad debt never go negative; utilization stays in [0, 1]."""
    idx = pd.date_range("2020-01-01", periods=36, freq="MS")
    falling = pd.Series(np.linspace(1, 0.4, 36), index=idx)
    sc = E.Scenario("crash", falling, pd.Series(0.2, index=idx), pd.Series(np.linspace(1, 0, 36), index=idx),
                    pd.Series(0.0, index=range(12)))
    res = E.run(sc, E.fixed_policy())
    assert (res["total_debt"] >= 0).all() and (res["supply"] > 0).all() and (res["bad_debt"] >= 0).all()
    assert res["utilization"].between(0, 1).all()
    assert res["liquidations"].sum() > 0


def test_liquidation_only_above_threshold():
    sc = flat_scenario(n=3)
    s = E.initial_state(sc, 0.60)
    s.supply = 1000.0                     # plenty of idle cash: no interest pressure
    s.debt[:] = 0.69                      # below 70% of the appraisal: safe
    s.debt[0] = 0.80                      # one position well above (still above after 10% deleveraging): liquidated
    nxt = E.update_market(s, 0.60, sc)
    assert nxt.liquidations == 1 and nxt.collateral[0] == 0


def test_lower_limit_means_less_debt():
    sc = flat_scenario(n=24)
    high = E.run(sc, E.fixed_policy(0.60))
    low = E.run(sc, E.fixed_policy(0.30))
    assert low["total_debt"].iloc[-1] < high["total_debt"].iloc[-1]


def test_volatility_is_endogenous():
    """Liquidation sales lower the market price, which raises the volatility the controller reads next month."""
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    falling = pd.Series(np.linspace(1, 0.6, 24), index=idx)
    sc = E.Scenario("crash", falling, pd.Series(0.0, index=idx), pd.Series(1.0, index=idx), pd.Series(0.0, index=range(12)))
    res = E.run(sc, E.fixed_policy())
    month = res.index[res["liquidations"] > 0][0]
    nxt = res.index.get_loc(month)
    assert res["market_price"].iloc[nxt] < res["token_value"].iloc[nxt]          # selling pressure
    assert res["volatility"].iloc[nxt] > res["volatility"].iloc[nxt - 1]


def test_lender_run_only_when_rent_stops():
    calm = E.run(flat_scenario(n=12, rent=0.5), E.fixed_policy())
    run = E.run(flat_scenario(n=12, rent=0.05), E.fixed_policy())
    assert calm["supply"].iloc[-1] >= calm["supply"].iloc[0] - 1e-9
    assert run["utilization"].iloc[-1] > 0.95
