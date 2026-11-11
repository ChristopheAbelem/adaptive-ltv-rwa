import numpy as np
import pandas as pd
import pytest
import skfuzzy as fuzz

from adaptive_ltv.inputs import rent_coverage_stylized, utilization_daily, utilization_signal
from adaptive_ltv.membership import LTV_MAX, LTV_MIN, max_ltv_variable, rent_variable, utilization_variable

LIQ = "data/raw/rmm_liquidity_daily.csv"


@pytest.fixture(scope="module")
def util_daily():
    try:
        return utilization_daily(LIQ)
    except FileNotFoundError:
        pytest.skip("liquidity data missing: run 00_data_test.ipynb first")


def test_utilization_in_unit_interval(util_daily):
    monthly = utilization_signal(LIQ)
    assert monthly.between(0, 1).all()


def test_utilization_terms(util_daily):
    var = utilization_variable(util_daily)
    assert np.allclose(var.coverage(), 1.0)
    monthly = utilization_signal(LIQ)
    assert var.memberships(monthly.loc["2025-06-01"])["healthy"] > 0.5
    assert var.memberships(monthly.loc["2026-03-01"])["crunch"] == 1.0


def test_rent_path_shape():
    rent = rent_coverage_stylized()
    assert (rent.loc[:"2025-06-01"] == 1.0).all()
    assert (rent.loc["2026-02-01":] == 0.0).all()
    assert (rent.diff().dropna() <= 1e-12).all()   # never increases


def test_rent_terms_cover_universe():
    assert np.allclose(rent_variable().coverage(), 1.0)


def test_output_terms_cover_ltv_range():
    var = max_ltv_variable()
    inside = (var.universe >= LTV_MIN) & (var.universe <= LTV_MAX)
    assert np.allclose(var.coverage()[inside], 1.0)


@pytest.mark.parametrize("term,centre", [("low", 0.30), ("medium", 0.45), ("high", 0.60)])
def test_single_term_centroid_is_its_centre(term, centre):
    var = max_ltv_variable()
    assert fuzz.defuzz(var.universe, var.mf[term], "centroid") == pytest.approx(centre, abs=1e-3)
