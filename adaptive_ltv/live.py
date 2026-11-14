"""Real-time display inside a Jupyter notebook: both rules run month by month and the figures are redrawn each month.

The Streamlit app (app.py) shows the same thing in a browser."""

import time

import matplotlib.pyplot as plt
import pandas as pd

from . import environment as env
from .plots import LINE, pct_axis, plot_decision


def live_comparison(scenario, ctrl, delay: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Run RealT's fixed 60% and the fuzzy controller side by side; after every month redraw the paths and the
    controller's four inference steps for that month. Returns the monthly history of both runs."""
    from IPython.display import clear_output, display

    supply0 = env.initial_state(scenario, env.START_LTV, seed).supply
    last = {}

    def fuzzy(v, u, r):
        last["d"] = ctrl.decide(v, u, r)
        return last["d"].max_ltv

    hist, realized = [], {"fixed": 0.0, "fuzzy": 0.0}
    runs = zip(env.simulate(scenario, env.fixed_policy(), seed), env.simulate(scenario, fuzzy, seed))
    for (_, f), (_, z) in runs:
        realized["fixed"] += f["bad_debt"]
        realized["fuzzy"] += z["bad_debt"]
        hist.append({"date": z["date"], "ltv_fixed": f["max_ltv"], "ltv_fuzzy": z["max_ltv"],
                     "loss_fixed": (realized["fixed"] + f["latent_bad_debt"]) / supply0,
                     "loss_fuzzy": (realized["fuzzy"] + z["latent_bad_debt"]) / supply0})
        df = pd.DataFrame(hist).set_index("date")

        top, axes = plt.subplots(1, 2, figsize=(14, 3.2))
        for ax, col, title in [(axes[0], "ltv", "max LTV"), (axes[1], "loss", "lenders' loss, % of deposits")]:
            ax.plot(df.index, df[f"{col}_fixed"], color="#9a9a9a", lw=2, ls="--", label="fixed 60%")
            ax.plot(df.index, df[f"{col}_fuzzy"], color=LINE, lw=2, label="fuzzy")
            ax.set_xlim(scenario.house_index.index[0], scenario.house_index.index[-1])
            pct_axis(ax.yaxis)
            ax.set_title(title, loc="left")
        axes[0].set_ylim(0.25, 0.65)
        axes[1].set_ylim(0, max(0.05, df[["loss_fixed", "loss_fuzzy"]].to_numpy().max() * 1.1))
        axes[0].legend(frameon=False)
        top.suptitle(f"{scenario.name}: {z['date']:%Y-%m}", x=0.01, ha="left", fontsize=11)
        top.tight_layout()
        bottom = plot_decision(ctrl, last["d"])
        clear_output(wait=True)
        display(top, bottom)
        plt.close(top)
        plt.close(bottom)
        time.sleep(delay)
    return df
