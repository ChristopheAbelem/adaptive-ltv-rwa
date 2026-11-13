"""The two test scenarios, built from the data."""

import numpy as np
import pandas as pd

from .environment import Scenario
from .inputs import desmooth, load_case_shiller, rent_coverage_stylized, seasonally_adjusted_returns, volatility_signal


def _true_returns() -> pd.Series:
    r_true, _ = desmooth(seasonally_adjusted_returns(load_case_shiller()))
    return r_true


def _house_path(start: str, end: str) -> tuple[pd.Series, pd.Series]:
    """De-smoothed Detroit house value over a window (1.0 at the start) and the 12 returns before it."""
    r_true = _true_returns()
    r = r_true.loc[start:end]
    pre = r_true.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)].iloc[-11:]
    pre = pd.concat([pre, r.iloc[:1]])          # the window's first return completes the 12 before month 1
    return np.exp(r.cumsum() - r.iloc[0]).rename("house_index"), pre


def market_shock() -> Scenario:
    """Detroit 2006-2012: prices fall by about half; rent and the pool are normal at the start."""
    house, pre = _house_path("2006-01-01", "2012-12-01")
    vol, _ = volatility_signal()
    return Scenario("market shock (Detroit 2006-2012)", house, vol.reindex(house.index),
                    pd.Series(1.0, index=house.index, name="rent_coverage"), pre)


def operational_collapse() -> Scenario:
    """RealT 2024-2026: prices near their peak, rent stops (stylized path), lenders leave."""
    house, pre = _house_path("2024-02-01", "2026-07-01")
    vol, _ = volatility_signal()
    rent = rent_coverage_stylized().reindex(house.index)
    return Scenario("operational collapse (RealT 2024-2026)", house, vol.reindex(house.index), rent, pre)


def merton_market_scenarios(n_paths: int = 300, months: int = 84, seed: int = 0):
    """Random futures for the market-shock test: Merton paths calibrated on 1992-2026 Detroit returns.

    Rent is paid throughout; each path starts from the same 12 months of history as the 2006 scenario."""
    from .merton import calibrate, simulate_returns

    r_true = _true_returns()
    params = calibrate(r_true)
    sims = simulate_returns(params, months, n_paths, seed)
    _, pre = _house_path("2006-01-01", "2012-12-01")
    idx = pd.date_range("2006-01-01", periods=months, freq="MS")
    out = []
    for k in range(n_paths):
        r = pd.Series(sims[k], index=idx)
        house = np.exp(r.cumsum() - r.iloc[0]).rename("house_index")
        pre_k = pd.concat([pre.iloc[:-1], r.iloc[:1]])
        out.append(Scenario(f"merton path {k}", house, pd.Series(np.nan, index=idx),
                            pd.Series(1.0, index=idx, name="rent_coverage"), pre_k))
    return params, out
