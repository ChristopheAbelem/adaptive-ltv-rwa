import numpy as np
import pytest

from adaptive_ltv.analysis import REFERENCE, static_cases, surface
from adaptive_ltv.controller import FuzzyLTVController
from adaptive_ltv.membership import LTV_MAX, LTV_MIN, max_ltv_variable, rent_variable
from test_controller import UTIL, VOL


@pytest.fixture(scope="module")
def ctrl():
    return FuzzyLTVController(VOL, UTIL, rent_variable(), max_ltv_variable())


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
