"""Merton jump-diffusion for monthly house price returns: calibration, simulation and closed forms.

Monthly log return = mu + sigma * Z + sum of N jumps, N ~ Poisson(lam), each jump ~ Normal(mu_j, sigma_j).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MertonParams:
    mu: float          # monthly drift of the diffusion part
    sigma: float       # monthly volatility of the diffusion part
    lam: float         # expected number of jumps per month
    mu_j: float        # mean jump size (log)
    sigma_j: float     # standard deviation of jump size (log)

    def as_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "parameter": ["mu (monthly drift)", "sigma (monthly diffusion vol)", "lambda (jumps per year)",
                          "mean jump", "jump std"],
            "value": [self.mu, self.sigma, self.lam * 12, self.mu_j, self.sigma_j]})


def calibrate(returns: pd.Series, threshold: float = 3.0) -> MertonParams:
    """Jumps = returns more than `threshold` robust standard deviations from the median (MAD estimate);
    the rest is the diffusion. Simple, transparent and reproducible."""
    r = returns.dropna().astype(float)
    robust_sd = 1.4826 * (r - r.median()).abs().median()
    is_jump = (r - r.median()).abs() > threshold * robust_sd
    normal, jumps = r[~is_jump], r[is_jump] - r[~is_jump].mean()
    return MertonParams(mu=float(normal.mean()), sigma=float(normal.std(ddof=1)), lam=float(is_jump.mean()),
                        mu_j=float(jumps.mean()) if len(jumps) else 0.0,
                        sigma_j=float(jumps.std(ddof=1)) if len(jumps) > 1 else 0.0)


def simulate_returns(p: MertonParams, months: int, n_paths: int, seed: int = 0) -> np.ndarray:
    """Array (n_paths, months) of monthly log returns."""
    rng = np.random.default_rng(seed)
    diffusion = p.mu + p.sigma * rng.standard_normal((n_paths, months))
    n_jumps = rng.poisson(p.lam, (n_paths, months))
    jump_sum = p.mu_j * n_jumps + p.sigma_j * np.sqrt(n_jumps) * rng.standard_normal((n_paths, months))
    return diffusion + jump_sum


def expected_growth(p: MertonParams, months: int) -> float:
    """Closed form: E[S_T / S_0] = exp(T * (mu + sigma^2/2 + lam * (exp(mu_j + sigma_j^2/2) - 1)))."""
    return float(np.exp(months * (p.mu + p.sigma ** 2 / 2 + p.lam * (np.exp(p.mu_j + p.sigma_j ** 2 / 2) - 1))))


def log_return_variance(p: MertonParams, months: int) -> float:
    """Closed form: Var[log(S_T / S_0)] = T * (sigma^2 + lam * (sigma_j^2 + mu_j^2))."""
    return float(months * (p.sigma ** 2 + p.lam * (p.sigma_j ** 2 + p.mu_j ** 2)))
