# -*- coding: utf-8 -*-
"""
Figure 7 -- the balanced-background control next to the original result,
as a single-column square.

Each row is drawn in the dialect of Fig. 3 panel A: a light-grey silhouette
of the permutation null around the row line, a coloured whisker for the 95%
CI, and the AUROC + p printed above the marker. A filled marker means
p < 0.05, an open one means n.s. (the paper's convention throughout).

The balanced rerun sits directly under the original number it checks, first
for the 17 primary groups and then with the cross-chain actor added; each
pair carries its own header. Numbers come from main_results.csv and
permutation_null.csv of the two packages.

    python fig7_dualcontrol.py
"""
import csv
import io
import os

import numpy as np
from scipy.stats import gaussian_kde

import natstyle as ns
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
SWAPD = os.path.join(PKG, "balanced_background", "data")
CAND = os.path.join(PKG, "data")
OUT = os.path.join(PKG, "figures", "fig7_dualcontrol.png")
DOTGRAY = "#666666"


def load_main(d):
    out = {}
    for r in csv.DictReader(io.open(os.path.join(d, "main_results.csv"), encoding="utf-8-sig")):
        out[r["crew_set"]] = dict(auroc=float(r["auroc"]), lo=float(r["ci_low"]),
                                  hi=float(r["ci_high"]), p=float(r["p_value"]))
    return out


def load_null(d):
    out = {"17_crews": [], "18_crews": []}
    for r in csv.DictReader(io.open(os.path.join(d, "permutation_null.csv"), encoding="utf-8-sig")):
        out[r["crew_set"]].append(float(r["shuffled_auroc"]))
    return out


canon, swap = load_main(CAND), load_main(SWAPD)
null_c, null_s = load_null(CAND), load_null(SWAPD)


def fp(p):
    return f"{p:.3f}"[1:]


fig = ns.apply((ns.W_SINGLE, 3.15))
fig.set_constrained_layout(False)
ax = fig.add_axes([0.185, 0.145, 0.79, 0.80])

# rows top->bottom: (results dict, null dict, key, condition, y)
ROWS = [(canon, null_c, "17_crews", "orig", 3.3),
        (swap, null_s, "17_crews", "bal", 2.3),
        (canon, null_c, "18_crews", "orig", 0.9),
        (swap, null_s, "18_crews", "bal", -0.1)]
VIO = 0.30


def violin(vals, y0):
    """Fig.3-style symmetric null silhouette around the row line."""
    kde = gaussian_kde(vals)
    xs = np.linspace(min(vals) - 0.02, max(vals) + 0.02, 220)
    ys = kde(xs)
    ys = ys / ys.max() * VIO
    ax.fill_between(xs, y0 - ys, y0 + ys, color=ns.LGRAY, alpha=0.75, lw=0, zorder=1)


for res, nul, key, kind, y in ROWS:
    r = res[key]
    violin(nul[key], y)
    sig = r["p"] < 0.05
    col = ns.INK if kind == "orig" else ns.BLUE
    ax.errorbar([r["auroc"]], [y], xerr=[[r["auroc"] - r["lo"]], [r["hi"] - r["auroc"]]],
                fmt="none", ecolor=col, elinewidth=1.8, capsize=3.0, capthick=1.8, zorder=4)
    mk = "o" if kind == "orig" else "D"
    ax.plot([r["auroc"]], [y], marker=mk, ms=(9.0 if kind == "orig" else 8.2),
            mew=1.5, mec=col, mfc=(col if sig else "white"), ls="none", zorder=5)
    txt = f"{r['auroc']:.3f},  p = {fp(r['p'])}" + ("" if sig else "  (n.s.)")
    ax.text(r["auroc"], y + 0.40, txt, ha="center", va="bottom", fontsize=6.8,
            color=(col if sig else DOTGRAY), fontweight=("bold" if sig else "normal"),
            bbox=dict(fc="white", ec="none", pad=0.5), zorder=6)

ax.text(0.385, 4.18, "17 actor-groups (primary)", fontsize=7.0, fontweight="bold",
        color=ns.INK, ha="left", va="center")
ax.text(0.385, 1.62, "+ cross-chain actor (18 groups)", fontsize=7.0, fontweight="bold",
        color=ns.INK, ha="left", va="center")
ax.axhline(1.78, color=ns.LGRAY, lw=0.7, zorder=0)

# reference lines (fig3 style): dashed chance, dotted empirical null mean
ax.axvline(0.50, ls=(0, (5, 3)), lw=1.0, color=ns.GRAY, zorder=0)
ax.axvline(0.53, ls=(0, (1, 2)), lw=1.2, color=DOTGRAY, zorder=0)
ax.text(0.537, -0.72, "null mean ≈ 0.53", fontsize=6.2, color=DOTGRAY, ha="left", va="center")

ax.set_yticks([3.3, 2.3, 0.9, -0.1])
ax.set_yticklabels(["original\nbackground", "balanced\nbackground",
                    "original\nbackground", "balanced\nbackground"], fontsize=6.8)
# colour the y labels to echo the markers
for tick, colr in zip(ax.get_yticklabels(), [ns.INK, ns.BLUE, ns.INK, ns.BLUE]):
    tick.set_color(colr)
ax.set_ylim(-0.92, 4.42)
ax.set_xlim(0.38, 0.97)
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax.tick_params(axis="x", labelsize=6.8)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("cross-actor AUROC  (95% CI)", fontsize=7.4)
ax.spines[["top", "right", "left"]].set_visible(False)

fig.savefig(OUT, dpi=600, facecolor="white")
plt.close(fig)
print("saved -> " + os.path.basename(OUT))
for res, _n, key, kind, _y in ROWS:
    r = res[key]
    print(f"  {key} {kind:<5} {r['auroc']:.3f}  p={r['p']:.3f}")
