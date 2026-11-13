import numpy as np
import pytest

from adaptive_ltv.controller import FuzzyLTVController
from adaptive_ltv.membership import FuzzyVariable, max_ltv_variable, rent_variable

# Synthetic volatility and utilization variables with the same shapes as the real ones, so the controller tests do
# not depend on the data files.
SYNTH_VOL = FuzzyVariable("volatility", np.round(np.arange(0, 0.3001, 0.001), 3),
                          {"low": [0, 0, 0.042, 0.055], "medium": [0.042, 0.055, 0.091, 0.166],
                           "high": [0.091, 0.166, 0.3, 0.3]})
SYNTH_UTIL = FuzzyVariable("utilization", np.round(np.arange(0, 1.0001, 0.001), 3),
                           {"healthy": [0, 0, 0.858, 0.90], "tight": [0.858, 0.90, 0.95, 0.99],
                            "crunch": [0.95, 0.99, 1, 1]})


@pytest.fixture(scope="session")
def ctrl():
    return FuzzyLTVController(SYNTH_VOL, SYNTH_UTIL, rent_variable(), max_ltv_variable())
