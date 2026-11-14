"""Static analysis of the controller: test cases and control surfaces."""

import numpy as np
import pandas as pd

from .controller import FuzzyLTVController


def default_controller() -> FuzzyLTVController:
    """The controller used in the report: volatility and utilization terms built from the frozen data."""
    from .inputs import utilization_daily, volatility_signal
    from .membership import max_ltv_variable, rent_variable, utilization_variable, volatility_variable

    vol, _ = volatility_signal()
    return FuzzyLTVController(volatility_variable(vol), utilization_variable(utilization_daily()),
                              rent_variable(), max_ltv_variable())


INPUTS = ("volatility", "utilization", "rent_coverage")

# Reference values for the input held fixed in a surface: a normal month (median volatility, healthy pool, full rent).
REFERENCE = {"volatility": 0.055, "utilization": 0.85, "rent_coverage": 1.0}

RANGES = {"volatility": (0.0, 0.30), "utilization": (0.80, 1.0), "rent_coverage": (0.0, 1.0)}


def static_cases(ctrl: FuzzyLTVController, cases: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Run named input cases through the controller; report output and the number of active rules."""
    rows = []
    for name, x in cases.items():
        d = ctrl.decide(x["volatility"], x["utilization"], x["rent_coverage"])
        active = [rule for rule, s in d.activations if s > 0]
        top = max(d.activations, key=lambda a: a[1])[0]
        rows.append({"case": name, **x, "max_ltv": d.max_ltv, "active_rules": len(active),
                     "strongest_rule": f"{top[0]} / {top[1]} / {top[2]} -> {top[3]}"})
    return pd.DataFrame(rows).set_index("case")


def surface(ctrl: FuzzyLTVController, x_name: str, y_name: str, n: int = 41,
            fixed: dict[str, float] | None = None):
    """Max LTV over a grid of two inputs, the third held at its reference value."""
    fixed = {**REFERENCE, **(fixed or {})}
    xs = np.linspace(*RANGES[x_name], n)
    ys = np.linspace(*RANGES[y_name], n)
    z = np.empty((n, n))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            point = dict(fixed, **{x_name: xv, y_name: yv})
            z[i, j] = ctrl(point["volatility"], point["utilization"], point["rent_coverage"])
    third = next(k for k in INPUTS if k not in (x_name, y_name))
    return xs, ys, z, (third, fixed[third])


def active_rules(decision) -> pd.DataFrame:
    """The rules that fired for one decision, strongest first (firing strength = min of the three degrees)."""
    rows = [{"volatility": r[0], "utilization": r[1], "rent_coverage": r[2], "max_ltv_term": r[3], "strength": s}
            for r, s in decision.activations if s > 0]
    return pd.DataFrame(rows).sort_values("strength", ascending=False, kind="stable").reset_index(drop=True)
