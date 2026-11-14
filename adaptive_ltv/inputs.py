"""Turn the frozen raw data into the monthly signals the controller reads."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"   # repo-root data folder, whatever the working directory


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXRNSA"
CS_END = "2026-07-01"   # last month used in the report: later FRED releases are cut here so results do not move


def download_case_shiller(path: Path = DATA / "cache" / "dexrnsa.csv") -> Path:
    """Download the licensed series from FRED into the local cache (git-ignored, never redistributed)."""
    raw = pd.read_csv(FRED_URL)
    raw.columns = ["date", "dexrnsa"]
    raw["dexrnsa"] = pd.to_numeric(raw["dexrnsa"], errors="coerce")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.dropna().to_csv(path, index=False)
    return path


def load_case_shiller(path: Path = DATA / "cache" / "dexrnsa.csv", download: bool = True) -> pd.Series:
    """Monthly S&P Cotality Case-Shiller MI-Detroit index (FRED DEXRNSA, not seasonally adjusted), up to CS_END.

    If the local cache is missing it is downloaded from FRED first (download=False raises FileNotFoundError)."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        download_case_shiller(path)
    cs = pd.read_csv(path, parse_dates=["date"], index_col="date")["dexrnsa"]
    return cs.astype(float).sort_index().loc[:CS_END]


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


# --- pool utilization -------------------------------------------------------------------------------------------

CRUNCH_START = "2026-01-01"   # first month of the documented USDC crunch (CoinPaprika; DefiLlama data)


def utilization_daily(path: Path = DATA / "raw" / "rmm_liquidity_daily.csv") -> pd.Series:
    """Daily stablecoin utilization of the RMM pool: borrowed / (borrowed + idle), from DefiLlama."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df["utilization"].clip(0, 1).sort_index()


def utilization_signal(path: Path = DATA / "raw" / "rmm_liquidity_daily.csv") -> pd.Series:
    """Monthly mean utilization (the controller runs monthly)."""
    return utilization_daily(path).resample("MS").mean().rename("utilization")


# --- rent coverage (stylized) -----------------------------------------------------------------------------------

# Documented sequence (Outlier Media, BridgeDetroit, Lofty): rent escrow from Jul 2025, distributions largely
# halted by the end of 2025, formal suspension in Feb 2026. No public time series exists, so the path between
# those dates is a stylized assumption: full coverage until Jun 2025, linear fall to 20% in Dec 2025, 10% in
# Jan 2026, zero from Feb 2026.
RENT_PATH = {
    "2025-06-01": 1.0,
    "2025-12-01": 0.2,
    "2026-01-01": 0.1,
    "2026-02-01": 0.0,
}


def rent_coverage_stylized(start: str = "2024-02-01", end: str = "2026-11-01") -> pd.Series:
    """Monthly share of expected rent actually paid, stylized from the documented RealT dates."""
    idx = pd.date_range(start, end, freq="MS")
    anchors = pd.Series(RENT_PATH, dtype=float)
    anchors.index = pd.to_datetime(anchors.index)
    s = pd.Series(np.nan, index=idx)
    s.loc[s.index <= anchors.index[0]] = 1.0
    s.loc[s.index >= anchors.index[-1]] = 0.0
    s.update(anchors)
    return s.interpolate("time").clip(0, 1).rename("rent_coverage")
