"""Shared chart helpers (matplotlib)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

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


def plot_control_surface(xs, ys, z, xlabel, ylabel, title, fixed_text, ax=None):
    """3D control surface (max LTV over two inputs), single-hue sequential colour for magnitude."""
    if ax is None:
        fig = plt.figure(figsize=(7, 5.2))
        ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, z, cmap=cm.Blues, vmin=0.2, vmax=0.62, edgecolor="#ffffff", linewidth=0.2,
                    antialiased=True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        pct_axis(axis)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_zlabel("max LTV", labelpad=8)
    ax.set_zlim(0.28, 0.62)
    ax.set_title(f"{title}\n{fixed_text}", loc="left", fontsize=11)
    ax.view_init(elev=25, azim=-55)
    return ax


INPUT_LABELS = {"volatility": "1. volatility", "utilization": "2. utilization", "rent_coverage": "3. rent coverage"}


def plot_decision(ctrl, decision, fig=None):
    """The four Mamdani steps for one month: fuzzified inputs (crisp value and degrees) and the aggregated output
    with its centroid."""
    if fig is None:
        fig = plt.figure(figsize=(14, 3.4))
    fig.clear()
    axes = fig.subplots(1, 4)
    for ax, (name, var) in zip(axes[:3], ctrl.vars.items()):
        x = decision.inputs[name]
        for (term, mf), color in zip(var.mf.items(), TERM_COLORS):
            ax.plot(var.universe, mf, color=color, lw=1.8, label=term)
            mu = decision.memberships[name][term]
            if mu > 0:
                ax.plot([x], [mu], "o", color=color, ms=7, mec="#ffffff", mew=1.5)
                right = x > (var.universe[0] + var.universe[-1]) / 2 if name != "utilization" else x > 0.85
                ax.text(x, mu + 0.06, f"{term} {mu:.2f} ", color="#333333", fontsize=8,
                        ha="right" if right else "left")
        ax.axvline(x, color="#333333", lw=1, ls=":")
        pct_axis(ax.xaxis)
        if name == "volatility":
            ax.set_xlim(0, 0.30)
        elif name == "utilization":
            ax.set_xlim(0.70, 1.0)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
        ax.set_title(f"{INPUT_LABELS[name]}: {x:.1%}", loc="left", fontsize=10)
    ax = axes[3]
    for (term, mf), color in zip(ctrl.ltv.mf.items(), TERM_COLORS):
        ax.plot(ctrl.ltv.universe, mf, color=color, lw=1, alpha=0.35)
    ax.fill_between(ctrl.ltv.universe, decision.aggregated, color=LINE, alpha=0.35, lw=0)
    ax.plot(ctrl.ltv.universe, decision.aggregated, color=LINE, lw=1.8)
    ax.axvline(decision.max_ltv, color="#333333", lw=1.5)
    ax.text(decision.max_ltv, 1.06, f"centroid {decision.max_ltv:.1%}", fontsize=9, ha="center")
    pct_axis(ax.xaxis)
    ax.set_ylim(0, 1.15)
    ax.set_title("4. aggregated output -> max LTV", loc="left", fontsize=10)
    for a in axes:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    return fig
