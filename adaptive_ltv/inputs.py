"""Turn the frozen raw data into the monthly signals the controller reads."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")


def load_case_shiller(path: Path = DATA / "cache" / "dexrnsa.csv") -> pd.Series:
    """Monthly S&P Cotality Case-Shiller MI-Detroit index (FRED DEXRNSA, not seasonally adjusted)."""
    cs = pd.read_csv(path, parse_dates=["date"], index_col="date")["dexrnsa"]
    return cs.astype(float).sort_index()


def seasonally_adjusted_returns(index: pd.Series) -> pd.Series:
    """Monthly log returns with each calendar month's mean removed (overall mean kept)."""
    r = np.log(index).diff().dropna()
    return r - r.groupby(r.index.month).transform("mean") + r.mean()


def desmooth(returns: pd.Series) -> tuple[pd.Series, float]:
    """Geltner de-smoothing: r_true = (r_obs - rho * r_obs[t-1]) / (1 - rho), rho = AR(1) coefficient."""
    rho = float(returns.autocorr(lag=1))
    if not 0 <= rho < 1:
        raise ValueError(f"AR(1) coefficient outside [0, 1): {rho:.3f}")
    r_true = ((returns - rho * returns.shift(1)) / (1 - rho)).dropna()
    return r_true, rho


def rolling_volatility(returns: pd.Series, window: int = 12) -> pd.Series:
    """Rolling standard deviation of monthly returns, annualized by sqrt(12)."""
    return (returns.rolling(window).std() * np.sqrt(12)).dropna().rename("vol")


def volatility_signal(path: Path = DATA / "cache" / "dexrnsa.csv", window: int = 12) -> tuple[pd.Series, dict]:
    """Full pipeline: index -> seasonally adjusted returns -> de-smoothed returns -> rolling volatility."""
    r_sa = seasonally_adjusted_returns(load_case_shiller(path))
    r_true, rho = desmooth(r_sa)
    vol = rolling_volatility(r_true, window)
    info = {
        "rho": rho,
        "vol_observed": float(r_sa.std() * np.sqrt(12)),
        "vol_desmoothed": float(r_true.std() * np.sqrt(12)),
    }
    return vol, info
