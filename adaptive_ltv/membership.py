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
