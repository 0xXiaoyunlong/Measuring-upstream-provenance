"""
Figure 2 -- a schematic of the reverse funding cone (the "Phi" representation).

This is a diagram, not a data plot: it explains how the backward walk works. The
parameters printed on it (7-day window, 10 ETH threshold, depth 2, 25-node cap,
0.1*V main-path rule) are imported from the extraction code rather than typed in.

    python figure2_provenance_cone.py   ->  ../../figures/figure2_provenance_cone.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, RegularPolygon, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)

# import the real parameters from the extraction code
sys.path.insert(0, os.path.join(HERE, "..", "01_collect_and_build_features"))
import build_upstream_features as phi

WINDOW_DAYS = phi.WINDOW_SECONDS // (24 * 3600)
THETA_ETH = int(phi.MIN_TRANSFER_ETH)
MIN_SHARE = phi.MIN_FUNDER_SHARE
COVER = phi.MIN_TRACED_SHARE
MAX_HOPS = phi.MAX_HOPS_BACK
NODE_CAP = 25

BLUE, ORANGE, GREEN, RED, PURPLE, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"
GRAY = "#7F7F7F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "savefig.dpi": 300})

fig, ax = plt.subplots(figsize=(7.48, 4.55))
ax.set_xlim(0, 10.6); ax.set_ylim(-2.10, 4.42); ax.axis("off")
t0_x, window_x = 8.15, 0.55

ax.add_patch(Rectangle((t0_x, 0.02), 10.6 - t0_x, 3.58, facecolor="#EDEDED", edgecolor="none", zorder=0))
ax.text(9.35, 3.22, "after the deposit:\nnot looked at", ha="center", va="center",
        fontsize=7.2, color="#555555", style="italic")

ax.annotate("", xy=(10.38, 0.0), xytext=(0.12, 0.0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="black"))
ax.text(10.44, 0.0, "time", fontsize=8, va="center")
for x, lab in ((window_x, f"$t_0-{WINDOW_DAYS}\\,$d"), (t0_x, "$t_0$ (deposit)")):
    ax.plot([x, x], [-0.09, 0.09], color="black", lw=1.1)
    ax.text(x, -0.30, lab, ha="center", va="top", fontsize=8.5)
ax.plot([window_x, window_x], [0.09, 3.62], color=GRAY, lw=0.9, ls=(0, (4, 3)))
ax.plot([t0_x, t0_x], [0.09, 3.62], color="black", lw=1.2)
ax.annotate("", xy=(t0_x - 0.03, 3.92), xytext=(window_x + 0.03, 3.92),
            arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color=GRAY))
ax.text((window_x + t0_x) / 2, 4.07,
        f"only transfers in this {WINDOW_DAYS}-day window are followed",
        ha="center", va="bottom", fontsize=8.2, color="#333333")


def node(x, y, label, kind="eoa", face="white", edge="black", dashed=False, r=0.30):
    if kind == "eoa":
        p = Circle((x, y), r, facecolor=face, edgecolor=edge, lw=1.3,
                   linestyle=(0, (3, 2)) if dashed else "solid", zorder=4)
    elif kind == "contract":
        p = Rectangle((x - r, y - r * 0.92), 2 * r, 1.84 * r, facecolor=face, edgecolor=edge, lw=1.3, hatch="////", zorder=4)
    else:
        p = RegularPolygon((x, y), numVertices=4, radius=r * 1.3, orientation=0.785398,
                           facecolor=face, edgecolor=edge, lw=1.3, zorder=4)
    ax.add_patch(p)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.8, zorder=5)


def edge(x1, y1, x2, y2, main=False, dashed=False, color=None, shrink=14):
    col = color or (ORANGE if main else GRAY)
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11 if main else 8,
                 lw=2.2 if main else 1.0, color=col, linestyle=(0, (4, 3)) if dashed else "solid",
                 shrinkA=shrink, shrinkB=shrink, zorder=3))


ax.add_patch(FancyBboxPatch((9.02, 1.18), 1.30, 0.95, boxstyle="round,pad=0.05",
             facecolor=SKY, edgecolor="black", lw=1.2, zorder=4))
ax.text(9.67, 1.78, "mixer\npool", ha="center", va="center", fontsize=7.8, zorder=5)
ax.text(9.67, 1.33, "deposit $V$", ha="center", va="center", fontsize=6.8, style="italic", zorder=5)

node(1.70, 2.80, "$s$", face="#D7EEDD", edge=GREEN)
ax.text(1.70, 2.38, "origin reached\n(no qualifying funder)", ha="center", va="top", fontsize=6.8, color="#333333")
node(4.35, 2.30, "$m$", face="#FFE8C2", edge=ORANGE)
ax.text(4.35, 1.88, "intermediary (hop 1)", ha="center", va="top", fontsize=6.8, color="#333333")
node(6.95, 1.62, "$a_0$", face="#FFE8C2", edge=ORANGE)
ax.text(6.95, 1.20, "depositor", ha="center", va="top", fontsize=6.8, color="#333333")
node(1.70, 1.32, "C", kind="contract", face="#F3F3F3", edge=PURPLE)
ax.text(2.14, 1.32, "contract $\\Rightarrow$ censored (trail cut off)", ha="left", va="center", fontsize=6.8, color="#333333")
node(1.70, 0.50, "DEX", kind="dex", face="#F3F3F3", edge=PURPLE, r=0.30)
ax.text(2.20, 0.50, "swap router $\\Rightarrow$ asset converted, stop", ha="left", va="center", fontsize=6.8, color="#333333")
node(5.70, 0.62, "$f$", face="white", edge=GRAY, r=0.26)
ax.text(5.70, 0.27, f"small funder ($<{MIN_SHARE:g}\\,V$: not followed)", ha="center", va="top", fontsize=6.6, color="#333333")
node(6.30, 3.30, "$b$", face="white", edge=RED, dashed=True, r=0.26)
ax.text(5.92, 3.30, "record not found $\\Rightarrow$\nmarked missing, never 0", ha="right", va="center", fontsize=6.8, color=RED)

edge(6.95, 1.62, 9.07, 1.62, main=True, color=BLUE)
ax.text(7.98, 1.80, "deposit at $t_0$", ha="center", fontsize=6.9, color="#1a4f75")
edge(1.70, 2.80, 4.35, 2.30, main=True)
ax.text(3.02, 2.78, f"largest funder $\\geq {MIN_SHARE:g}\\,V$", ha="center", fontsize=6.9, color="#8a5a00")
edge(4.35, 2.30, 6.95, 1.62, main=True)
ax.text(5.66, 2.22, f"largest funder $\\geq {MIN_SHARE:g}\\,V$", ha="center", fontsize=6.9, color="#8a5a00")
edge(1.70, 1.32, 4.35, 2.30)
edge(1.70, 0.50, 4.35, 2.30)
edge(5.70, 0.62, 6.95, 1.62)
ax.text(6.62, 0.86, f"$\\geq {THETA_ETH}$ ETH", ha="center", fontsize=6.8, color="#444444")
edge(6.30, 3.30, 6.95, 1.62, dashed=True, color=RED)

ax.annotate("dwell at $m$ = time held before passing on", xy=(4.42, 2.62), xytext=(2.95, 3.42),
            fontsize=6.8, color="#333333", ha="center", arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666"))

rules = (f"cone: transfers $\\geq {THETA_ETH}$ ETH within the window;  depth $\\leq {MAX_HOPS}$;  "
         f"$\\leq {NODE_CAP}$ nodes;  contracts & swap routers = boundary\n"
         f"trail: follow the largest funder $\\geq {MIN_SHARE:g}\\,V$ per hop;  stop at origin / boundary / "
         f"coverage $< {COVER:g}$ / depth limit\n"
         f"two thresholds: {THETA_ETH} ETH decides what enters the cone;  {MIN_SHARE:g}$\\,V$ decides the main trail followed\n"
         "output: 15 features (hops, elapsed time, dwell, coverage, ambiguity, source context) + explicit missingness flags")
ax.text(0.15, -0.62, rules, fontsize=7.1, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7F7F7", edgecolor="#999999", lw=0.8))

out = os.path.join(OUT, "figure2_provenance_cone.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"figure 2 written -> {out}   (parameters imported from build_upstream_features.py: "
      f"window={WINDOW_DAYS}d, theta={THETA_ETH} ETH, depth<={MAX_HOPS})")
