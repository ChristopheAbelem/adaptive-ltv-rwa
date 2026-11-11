"""Fuzzy variables: universes, terms and trapezoidal membership functions."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import skfuzzy as fuzz


@dataclass
class FuzzyVariable:
    """A linguistic variable: a universe and trapezoid breakpoints per term, in order."""

    name: str
    universe: np.ndarray
    terms: dict[str, list[float]]
    mf: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.mf = {term: fuzz.trapmf(self.universe, pts) for term, pts in self.terms.items()}

    def memberships(self, x: float) -> dict[str, float]:
        """Degree of membership of a crisp value in each term."""
        x = float(np.clip(x, self.universe[0], self.universe[-1]))
        return {t: float(fuzz.interp_membership(self.universe, mf, x)) for t, mf in self.mf.items()}

    def coverage(self) -> np.ndarray:
        """Sum of all memberships at each point of the universe (1 everywhere = full coverage, no gaps)."""
        return sum(self.mf.values())


def volatility_variable(vol: pd.Series, vmax: float = 0.30) -> FuzzyVariable:
    """Volatility terms from the historical percentiles of the series: low, medium, high."""
    p25, p50, p75, p90 = (round(float(q), 3) for q in vol.quantile([0.25, 0.50, 0.75, 0.90]))
    return FuzzyVariable(
        name="volatility",
        universe=np.round(np.arange(0, vmax + 1e-9, 0.001), 3),
        terms={
            "low": [0.0, 0.0, p25, p50],
            "medium": [p25, p50, p75, p90],
            "high": [p75, p90, vmax, vmax],
        },
    )


# Utilization: the normal series is too narrow (84% to 88% before 2026) for percentile-based terms, so the upper
# breakpoints follow observed behaviour: 90% sits above every month before the crunch (max 87.6%), 99% means the
# pool has practically no idle cash.
UTIL_TIGHT_FULL = 0.90
UTIL_TIGHT_END = 0.95
UTIL_CRUNCH_FULL = 0.99


def utilization_variable(util_daily: pd.Series, crunch_start: str = "2026-01-01") -> FuzzyVariable:
    """Utilization terms: healthy, tight, crunch. Healthy is full up to the pre-crunch median."""
    p50 = round(float(util_daily.loc[:pd.Timestamp(crunch_start) - pd.Timedelta(days=1)].median()), 3)
    return FuzzyVariable(
        name="utilization",
        universe=np.round(np.arange(0, 1 + 1e-9, 0.001), 3),
        terms={
            "healthy": [0.0, 0.0, p50, UTIL_TIGHT_FULL],
            "tight": [p50, UTIL_TIGHT_FULL, UTIL_TIGHT_END, UTIL_CRUNCH_FULL],
            "crunch": [UTIL_TIGHT_END, UTIL_CRUNCH_FULL, 1.0, 1.0],
        },
    )


# Rent coverage: no historical distribution exists, so the breakpoints are stated assumptions (hypotheses):
# below 30% the asset is not producing income, 60% to 80% is a partial payer, from 95% rent is paid as expected.
def rent_variable() -> FuzzyVariable:
    """Rent coverage terms: low, partial, full."""
    return FuzzyVariable(
        name="rent_coverage",
        universe=np.round(np.arange(0, 1 + 1e-9, 0.001), 3),
        terms={
            "low": [0.0, 0.0, 0.30, 0.60],
            "partial": [0.30, 0.60, 0.80, 0.95],
            "full": [0.80, 0.95, 1.0, 1.0],
        },
    )


# Output: max LTV between 30% and 60%. Triangles centred on 30%, 45% and 60% with a 15-point half-width; the
# universe runs from 15% to 75% so that, when a single term fires, the centroid lands exactly on its centre
# (a calm market gives exactly RealT's 60%, keeping the comparison with the fixed rule fair). The crisp output is
# then clipped to [30%, 60%].
LTV_MIN, LTV_MAX = 0.30, 0.60


def max_ltv_variable() -> FuzzyVariable:
    """Output terms: low (30%), medium (45%), high (60%)."""
    return FuzzyVariable(
        name="max_ltv",
        universe=np.round(np.arange(0.15, 0.75 + 1e-9, 0.001), 3),
        terms={
            "low": [0.15, 0.30, 0.30, 0.45],
            "medium": [0.30, 0.45, 0.45, 0.60],
            "high": [0.45, 0.60, 0.60, 0.75],
        },
    )
