import numpy as np
import pytest

from adaptive_ltv.analysis import REFERENCE, active_rules, static_cases, surface
from adaptive_ltv.membership import LTV_MAX, LTV_MIN


def test_static_cases_report(ctrl):
    out = static_cases(ctrl, {"calm": REFERENCE})
    assert out.loc["calm", "max_ltv"] == pytest.approx(LTV_MAX, abs=1e-3)
    assert out.loc["calm", "active_rules"] == 1


@pytest.mark.parametrize("pair", [("volatility", "utilization"), ("volatility", "rent_coverage"),
                                  ("utilization", "rent_coverage")])
def test_surface_shape_and_range(ctrl, pair):
    xs, ys, z, (third, value) = surface(ctrl, *pair, n=9)
    assert z.shape == (9, 9)
    assert third not in pair and value == REFERENCE[third]
    assert np.all((z >= LTV_MIN - 1e-9) & (z <= LTV_MAX + 1e-9))


def test_active_rules_lists_only_fired_rules(ctrl):
    d = ctrl.decide(0.065, 0.93, 0.5)
    rules = active_rules(d)
    assert len(rules) == sum(s > 0 for _, s in d.activations)
    assert rules["strength"].is_monotonic_decreasing and (rules["strength"] > 0).all()
