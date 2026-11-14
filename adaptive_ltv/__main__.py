"""Command line: run one scenario under both rules and print each month as it happens.

    python -m adaptive_ltv collapse          # operational collapse (RealT 2024-2026)
    python -m adaptive_ltv shock --delay 0.1 # market shock (Detroit 2006-2012)
"""

import argparse
import time

from . import environment as env
from .analysis import default_controller
from .scenarios import market_shock, operational_collapse


def main():
    p = argparse.ArgumentParser(description="Adaptive LTV: fixed 60% vs fuzzy controller, month by month")
    p.add_argument("scenario", choices=["collapse", "shock"])
    p.add_argument("--delay", type=float, default=0.0, help="seconds between months")
    p.add_argument("--deleverage", type=float, default=env.ASSUMPTIONS["deleverage_rate"][0])
    p.add_argument("--discount", type=float, default=env.ASSUMPTIONS["token_discount"][0])
    a = p.parse_args()

    scenario = operational_collapse() if a.scenario == "collapse" else market_shock()
    ctrl = default_controller()
    with env.assumption("deleverage_rate", a.deleverage), env.assumption("token_discount", a.discount):
        supply0 = env.initial_state(scenario, env.START_LTV).supply
        loss = {"fixed": 0.0, "fuzzy": 0.0}
        print(f"{scenario.name}\n{'month':8} {'vol':>6} {'util':>6} {'rent':>6} {'fuzzy LTV':>10} "
              f"{'loss fixed':>11} {'loss fuzzy':>11}")
        for (_, f), (_, z) in zip(env.simulate(scenario, env.fixed_policy()), env.simulate(scenario, ctrl)):
            loss["fixed"] += f["bad_debt"]
            loss["fuzzy"] += z["bad_debt"]
            lf = (loss["fixed"] + f["latent_bad_debt"]) / supply0
            lz = (loss["fuzzy"] + z["latent_bad_debt"]) / supply0
            print(f"{z['date']:%Y-%m}  {z['volatility']:6.1%} {z['utilization']:6.1%} {z['rent_coverage']:6.0%} "
                  f"{z['max_ltv']:10.1%} {lf:11.1%} {lz:11.1%}", flush=True)
            time.sleep(a.delay)


if __name__ == "__main__":
    main()
