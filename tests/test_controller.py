from itertools import product

import numpy as np
import pytest

from adaptive_ltv.controller import FuzzyLTVController, build_rules
from adaptive_ltv.membership import LTV_MAX, LTV_MIN, FuzzyVariable, max_ltv_variable, rent_variable

# synthetic volatility and utilization variables with the same shapes as the real ones (no data files needed)
VOL = FuzzyVariable("volatility", np.round(np.arange(0, 0.3001, 0.001), 3),
                    {"low": [0, 0, 0.042, 0.055], "medium": [0.042, 0.055, 0.091, 0.166], "high": [0.091, 0.166, 0.3, 0.3]})
UTIL = FuzzyVariable("utilization", np.round(np.arange(0, 1.0001, 0.001), 3),
                     {"healthy": [0, 0, 0.858, 0.90], "tight": [0.858, 0.90, 0.95, 0.99], "crunch": [0.95, 0.99, 1, 1]})


@pytest.fixture(scope="module")
def ctrl():
    return FuzzyLTVController(VOL, UTIL, rent_variable(), max_ltv_variable())


def test_rule_base_is_complete():
    rules = build_rules()
    assert len(rules) == 27
    assert len({r[:3] for r in rules}) == 27          # every input combination exactly once


def test_calm_market_gives_benchmark(ctrl):
    """Normal volatility, healthy pool, full rent: exactly RealT's fixed 60%."""
    assert ctrl(0.07, 0.85, 1.0) == pytest.approx(LTV_MAX, abs=1e-3)


def test_worst_case_gives_minimum(ctrl):
    assert ctrl(0.25, 1.0, 0.0) == pytest.approx(LTV_MIN, abs=1e-3)


def test_output_always_in_range(ctrl):
    for v, u, r in product(np.linspace(0, 0.3, 7), np.linspace(0.8, 1, 7), np.linspace(0, 1, 7)):
        assert LTV_MIN <= ctrl(v, u, r) <= LTV_MAX


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_monotone_worse_input_never_raises_ltv(ctrl, axis):
    """Coherence: making one input worse (others fixed) never increases the max LTV by more than 0.5 point.

    Mamdani with centroid is not strictly monotone: where two clipped output terms overlap, the centroid can move
    up by a fraction of a point (measured max: 0.32 point). The tolerance documents that known artifact."""
    grids = [np.linspace(0, 0.3, 13), np.linspace(0.8, 1, 13), np.linspace(1, 0, 13)]   # rent: worse = lower
    for a, b in product(*[g[::4] for i, g in enumerate(grids) if i != axis]):
        others = iter([a, b])
        point = [None if i == axis else next(others) for i in range(3)]
        ltvs = []
        for x in grids[axis]:
            p = list(point)
            p[axis] = x
            ltvs.append(ctrl(*p))
        assert all(l2 <= l1 + 0.005 for l1, l2 in zip(ltvs, ltvs[1:]))


def test_decision_exposes_internals(ctrl):
    d = ctrl.decide(0.12, 0.93, 0.5)
    assert len(d.activations) == 27
    assert any(s > 0 for _, s in d.activations)
    assert d.aggregated.shape == ctrl.ltv.universe.shape


@pytest.mark.parametrize("vol", [0.01, 0.07, 0.25])
def test_crunch_or_failed_rent_alone_cut_to_minimum(ctrl, vol):
    """A calm price never offsets an empty pool or an asset that stopped paying."""
    assert ctrl(vol, 1.0, 1.0) == pytest.approx(LTV_MIN, abs=1e-3)
    assert ctrl(vol, 0.85, 0.0) == pytest.approx(LTV_MIN, abs=1e-3)
