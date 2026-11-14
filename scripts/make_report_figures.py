"""Regenerate the figures used in docs/report.md:  python scripts/make_report_figures.py"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from adaptive_ltv import environment as env
from adaptive_ltv.analysis import default_controller
from adaptive_ltv.inputs import DATA
from adaptive_ltv.plots import LINE, STYLE, pct_axis
from adaptive_ltv.scenarios import market_shock, operational_collapse

DOCS = DATA.parent / "docs"


def main():
    plt.rcParams.update(STYLE)
    ctrl = default_controller()
    for sc, name in [(operational_collapse(), "fig_collapse.png"), (market_shock(), "fig_shock.png")]:
        runs = [(env.run(sc, env.fixed_policy()), "fixed 60%", "#9a9a9a", "--"), (env.run(sc, ctrl), "fuzzy", LINE, "-")]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        for r, label, color, ls in runs:
            axes[0].plot(r.index, r["max_ltv"], color=color, ls=ls, lw=2, label=label)
            axes[1].plot(r.index, r["utilization"], color=color, ls=ls, lw=2)
            axes[2].plot(r.index, r["loss_to_date"] / r.attrs["initial_supply"], color=color, ls=ls, lw=2)
        titles = ["max LTV (control action)", "pool utilization", "lenders' loss to date, % of deposits"]
        for ax, title in zip(axes, titles):
            ax.set_title(title, loc="left", fontsize=10)
            loc = mdates.YearLocator() if len(sc) > 40 else mdates.MonthLocator(bymonth=[1, 7])
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        pct_axis(axes[0].yaxis)
        pct_axis(axes[1].yaxis)
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        axes[0].set_ylim(0.25, 0.65)
        axes[0].legend(frameon=False, loc="lower left")
        fig.suptitle(sc.name, x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(DOCS / name, dpi=130)
        plt.close(fig)
        print(DOCS / name)


if __name__ == "__main__":
    main()
