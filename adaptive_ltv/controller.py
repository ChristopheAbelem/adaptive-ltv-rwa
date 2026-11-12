"""Mamdani fuzzy controller for the max LTV: 27 rules, min for AND, min implication, max aggregation, centroid."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import skfuzzy as fuzz

from .membership import LTV_MAX, LTV_MIN, FuzzyVariable

# Each input term gets a severity: how much it worsens the lending risk.
# Medium volatility is the normal market (it contains the median), so it scores less than a tight pool or partial
# rent. A crunch or low rent scores 4: enough on its own to cut the LTV to its minimum, whatever the volatility
# (a calm price does not offset an empty pool or an asset that stopped paying).
SEVERITY = {
    "volatility": {"low": 0, "medium": 1, "high": 2},
    "utilization": {"healthy": 0, "tight": 2, "crunch": 4},
    "rent_coverage": {"full": 0, "partial": 2, "low": 4},
}


def output_term(score: int) -> str:
    """Total severity -> output term: 0-1 high (60%), 2-3 medium (45%), 4+ low (30%)."""
    if score <= 1:
        return "high"
    if score <= 3:
        return "medium"
    return "low"


def build_rules() -> list[tuple[str, str, str, str]]:
    """All 27 combinations (volatility, utilization, rent) -> max LTV term."""
    rules = []
    for v, u, r in product(SEVERITY["volatility"], SEVERITY["utilization"], SEVERITY["rent_coverage"]):
        score = SEVERITY["volatility"][v] + SEVERITY["utilization"][u] + SEVERITY["rent_coverage"][r]
        rules.append((v, u, r, output_term(score)))
    return rules


@dataclass
class Decision:
    """Everything the controller did for one input: used by the internal-workings panel."""

    inputs: dict[str, float]
    memberships: dict[str, dict[str, float]]
    activations: list[tuple[tuple[str, str, str, str], float]]   # (rule, firing strength)
    aggregated: np.ndarray
    centroid: float
    max_ltv: float


class FuzzyLTVController:
    """Three inputs (volatility, utilization, rent coverage) -> max LTV, Mamdani inference."""

    def __init__(self, vol: FuzzyVariable, util: FuzzyVariable, rent: FuzzyVariable, ltv: FuzzyVariable):
        self.vars = {"volatility": vol, "utilization": util, "rent_coverage": rent}
        self.ltv = ltv
        self.rules = build_rules()

    def decide(self, volatility: float, utilization: float, rent_coverage: float) -> Decision:
        x = {"volatility": volatility, "utilization": utilization, "rent_coverage": rent_coverage}
        # 1. fuzzification
        mu = {name: var.memberships(x[name]) for name, var in self.vars.items()}
        # 2. inference: AND = min of the three degrees; implication clips the output term at that strength
        aggregated = np.zeros_like(self.ltv.universe, dtype=float)
        activations = []
        for rule in self.rules:
            v, u, r, out = rule
            strength = min(mu["volatility"][v], mu["utilization"][u], mu["rent_coverage"][r])
            activations.append((rule, strength))
            if strength > 0:
                # 3. aggregation: max over all clipped output terms
                aggregated = np.fmax(aggregated, np.fmin(strength, self.ltv.mf[out]))
        # 4. defuzzification: centroid, then clipped to the allowed LTV range
        centroid = float(fuzz.defuzz(self.ltv.universe, aggregated, "centroid"))
        return Decision(x, mu, activations, aggregated, centroid, float(np.clip(centroid, LTV_MIN, LTV_MAX)))

    def __call__(self, volatility: float, utilization: float, rent_coverage: float) -> float:
        return self.decide(volatility, utilization, rent_coverage).max_ltv
