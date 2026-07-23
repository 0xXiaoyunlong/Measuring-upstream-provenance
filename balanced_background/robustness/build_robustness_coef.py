# -*- coding: utf-8 -*-
"""
Combined robustness overview (economics coefficient-plot style): every perturbation
on one axis -- baseline (diamond), leave-one-actor-out (circles), Phi-extraction
parameters (triangles), feature families (squares) -- as a point estimate with a
vertical 95% CI whisker, against the empirical-null (0.53) and main (canonical 0.703)
reference lines.

Reads robustness_forest_data.json, the frozen recompute from
robustness_permutation.py (scikit-learn 1.3.2); the title and annotation carry
the main AUROC (0.703) and the group-aware permutation p from that file.

    python build_robustness_coef.py       # renders robustness/figures/robustness_coef.png
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "03_make_figures"))
import natstyle as N
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_JSON = os.path.join(HERE, "robustness_forest_data.json")
FIGDIR = os.path.join(HERE, "figures")
NULL = 0.53
_META = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}


def main():
    if not os.path.exists(DATA_JSON):
        raise SystemExit(f"missing {os.path.basename(DATA_JSON)} -- run robustness_permutation.py first.")
    data = json.load(io.open(DATA_JSON, encoding="utf-8"))
    bm = data["main"]
    main_p = data["none"]["p"]
    loo = sorted(data["loo"], key=lambda d: -d["auroc"])
    params = [d for d in data["params"] if d["label"] != "Main (7d/2hop/10 ETH)"]
    feats = [d for d in data["feats"] if d["label"] != "Full Φ"]

    CIRC, TRI, SQ, DIA = N.DARK, N.VERM, "#009E73", N.BLUE
    fig = N.apply((N.W_DOUBLE, 4.9))
    fig.set_constrained_layout(False)
    ax = fig.add_axes([0.065, 0.30, 0.925, 0.55])
    xs, lab, x = [], [], 0

    def plot_block(items, color, marker, ms=5.0):
        nonlocal x
        for it in items:
            m, lo, hi = it["auroc"], it["lo"], it["hi"]
            yerr = None if lo is None else [[m - lo], [hi - m]]
            ax.errorbar(x, m, yerr=yerr, fmt=marker, ms=ms, color=color, mfc=color, mec=color,
                        mew=0.9, ecolor=color, elinewidth=0.9, capsize=2.2, zorder=5)
            xs.append(x)
            lab.append(it["label"].replace("Main (7d/2hop/10 ETH)", "Main (Φ)"))
            x += 1

    plot_block([data["none"]], DIA, "D", ms=6.2)
    x += 1; d1 = x - 0.5
    plot_block(loo, CIRC, "o")
    x += 1; d2 = x - 0.5
    plot_block(params, TRI, "^")
    x += 1; d3 = x - 0.5
    plot_block(feats, SQ, "s")

    ax.axhline(NULL, ls=(0, (4, 2)), lw=0.8, color=N.GRAY, zorder=1)
    ax.axhline(bm, ls="-", lw=0.7, color=N.BLUE, alpha=0.5, zorder=1)
    for dv in (d1, d2, d3):
        ax.axvline(dv, lw=0.5, color="#DDDDDD", zorder=0)
    blocks = [(0, "Baseline"), ((d1 + d2) / 2, "Leave-one-actor-out"),
              ((d2 + d3) / 2, "Extraction parameters"), ((d3 + x) / 2, "Feature families")]
    for bx, bt in blocks:
        ax.text(bx, 0.955, bt, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6.6, fontweight="bold", color=N.INK)
    ax.set_xlim(-1, x)
    ax.set_ylim(0.15, 0.95)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, rotation=55, ha="right", fontsize=5.6)
    ax.set_ylabel("Cross-actor AUROC  (leave-one-group-out)", fontsize=7.4)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.text(-0.7, NULL + 0.006, "empirical null 0.53", ha="left", va="bottom", fontsize=5.8, color=N.GRAY)
    ax.text(-0.7, bm + 0.006, f"main {bm:.3f}", ha="left", va="bottom", fontsize=5.8, color=N.BLUE)
    ax.text(0.5, 0.20, f"Main AUROC = {bm:.3f}  (group-aware permutation p = {main_p:.3f})\n"
                       "Whiskers: 95% CI, bootstrap over actor groups",
            transform=ax.transAxes, ha="center", va="center", fontsize=6.4, color=N.INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", lw=0.6))
    leg = [Line2D([0], [0], marker="D", ls="none", mfc=DIA, mec=DIA, ms=6, label="Main (Φ)"),
           Line2D([0], [0], marker="o", ls="none", mfc=CIRC, mec=CIRC, ms=5, label="Leave-one-actor-out"),
           Line2D([0], [0], marker="^", ls="none", mfc=TRI, mec=TRI, ms=5, label="Extraction parameters"),
           Line2D([0], [0], marker="s", ls="none", mfc=SQ, mec=SQ, ms=5, label="Feature families")]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.005, 0.02), fontsize=6.0,
              frameon=False, ncol=2, handletextpad=0.3, columnspacing=1.0)
    fig.suptitle("Robustness of the cross-actor signal: perturbed AUROCs cluster near 0.70, with wide small-sample CIs",
                 fontsize=8.2, fontweight="bold", x=0.012, ha="left", y=0.975, color=N.INK)
    os.makedirs(FIGDIR, exist_ok=True)
    base = os.path.join(FIGDIR, "robustness_coef")
    fig.savefig(base + ".png", dpi=600, facecolor="white", metadata=_META["png"])
    plt.close(fig)
    print(f"done -> robustness/figures/robustness_coef.png (main {bm:.3f}, p={main_p:.3f})")


if __name__ == "__main__":
    main()
