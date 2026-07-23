# -*- coding: utf-8 -*-
"""House style for the two-background figures (fig5).

Colours are Okabe-Ito (colour-blind safe), and colour never carries a
distinction on its own: whatever the colour says, a shape, line style, or
direct label says too. save() writes a 450 dpi PNG into ../../figures.

Yes, there are two style modules -- fig1/3/4/C1 predate this one and their
published pixels are frozen, so natstyle.py stays as it was and newer figures
use this file.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.abspath(os.path.join(HERE, "..", "..", "figures"))

# column widths for a generic double-column journal
W_SINGLE = 88 / 25.4     # 3.46 in
W_DOUBLE = 183 / 25.4    # 7.20 in

# Okabe-Ito
BLUE = "#0072B2"     # primary accent (new / balanced / main estimate)
VERM = "#D55E00"     # secondary accent (factory / caution)
ORANGE = "#E69F00"   # parameters / intermediate class
SKY = "#56B4E9"      # light accent (newly gated groups)
GREEN = "#009E73"    # spares, no assigned role yet
PURPLE = "#CC79A7"
INK = "#1a1a1a"      # text / axes
GRAY60 = "#8C8C8C"   # secondary text / not significant
GRAY85 = "#D9D9D9"   # grid / null silhouettes
BG = "#F5F5F5"       # block background (very low saturation)

FS_BASE = 7.0
FS_SMALL = 6.0
FS_PANEL = 8.5

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": FS_BASE,
    "axes.labelsize": FS_BASE, "axes.titlesize": FS_BASE,
    "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
    "legend.fontsize": FS_SMALL,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.unicode_minus": False,   # avoid U+2212 tofu boxes on CJK systems
    "figure.dpi": 110,
})


def panel_label(ax, letter, dx=-0.08, dy=1.04):
    """Bold panel letter, hung just outside the axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="bottom", ha="right", color=INK)


def refline(ax, x=None, y=None, kind="chance", label=None, label_dy=0.02):
    """Reference line in one of three roles: "chance" is the solid grey 0.50
    line, "null" the dotted empirical null, "anchor" the coloured main
    estimate. Give x or y depending on which way it should run."""
    styles = {"chance": dict(color=GRAY60, lw=0.7, ls="-"),
              "null": dict(color=INK, lw=0.7, ls=":"),
              "anchor": dict(color=BLUE, lw=0.9, ls="-")}
    st = styles[kind]
    if x is not None:
        ax.axvline(x, zorder=1, **st)
        if label:
            ax.text(x, ax.get_ylim()[1], " " + label, fontsize=FS_SMALL,
                    color=st["color"], ha="left", va="top", rotation=0)
    if y is not None:
        ax.axhline(y, zorder=1, **st)
        if label:
            ax.text(ax.get_xlim()[1], y, " " + label, fontsize=FS_SMALL,
                    color=st["color"], ha="left", va="center")


def save(fig, name):
    """Write the figure as a 450 dpi PNG."""
    os.makedirs(FIGS, exist_ok=True)
    png = os.path.join(FIGS, f"{name}.png")
    fig.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    print("saved", name + ".png")
