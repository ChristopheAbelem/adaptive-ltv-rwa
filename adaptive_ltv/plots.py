"""Shared chart helpers (matplotlib)."""

import matplotlib.pyplot as plt

TERM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # term order 1 / 2 / 3 (validated categorical slots 1-3)
LINE = "#2a6fdb"
SHADE = "#d9d9d9"

STYLE = {
    "figure.figsize": (10, 4),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#9a9a9a",
    "axes.labelcolor": "#444444",
    "xtick.color": "#666666",
    "ytick.color": "#666666",
}


def pct_axis(axis):
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))


def plot_memberships(var, title, xlabel, band=None, band_label=None, label_x=None, xlim=None):
    """Membership functions of a FuzzyVariable, terms labelled above their plateau; optional shaded band."""
    fig, ax = plt.subplots()
    if band is not None:
        ax.axvspan(band[0], band[1], color=SHADE, alpha=0.6, lw=0)
        if band_label:
            ax.text((band[0] + band[1]) / 2, 0.03, band_label, ha="center", color="#555555", fontsize=9)
    label_x = label_x or {}
    for (term, mf), color in zip(var.mf.items(), TERM_COLORS):
        ax.plot(var.universe, mf, color=color, lw=2)
        x = label_x.get(term, var.universe[mf == 1].mean())
        ax.text(x, 1.04, term, color="#333333", fontsize=10, ha="center")
    pct_axis(ax.xaxis)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(*(xlim or (var.universe[0], var.universe[-1])))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("membership degree")
    ax.set_title(title, loc="left")
    return fig, ax
