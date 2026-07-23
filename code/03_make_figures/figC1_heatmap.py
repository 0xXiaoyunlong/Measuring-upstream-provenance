# -*- coding: utf-8 -*-
"""Figure C1 -- which features could not be observed, per group.

Presence/absence matrix in the style of genomics panels: a filled square marks
"not observed for the group's event(s)", absence of a mark means observed.
The 17 positive-group rows are strictly binary (asserted); the background pool
is the one fractional row and keeps shaded squares with the share printed.
Columns are grouped by feature family so missingness reads as blocks; the
right margin counts each group's unobserved features.
                                          <- data/missingness_by_crew.csv
"""
import csv, io, os
import numpy as np
import natstyle as ns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

rows = list(csv.DictReader(io.open(os.path.join(ns.DATA, "missingness_by_crew.csv"), encoding="utf-8-sig")))

FAMILIES = [
    ("path skeleton", ["observed_hops", "coverage", "attribution_ambiguous",
                       "boundary_type", "censoring_reason", "observation_confidence"]),
    ("path timing", ["path_elapsed_time", "dwell_mean_h", "dwell_max_h"]),
    ("source context", ["node_context_out_degree", "node_context_in_degree",
                        "wallet_age_min_h", "node_context_resid_Rrange_max",
                        "node_context_resid_nin_mean", "node_context_resid_nout_mean"]),
]
FEAT_LABEL = {
    "observed_hops": "hops", "coverage": "coverage", "attribution_ambiguous": "ambiguity flag",
    "boundary_type": "boundary type", "censoring_reason": "censor reason",
    "observation_confidence": "obs. confidence", "path_elapsed_time": "elapsed time",
    "dwell_mean_h": "dwell mean", "dwell_max_h": "dwell max",
    "node_context_out_degree": "out-degree", "node_context_in_degree": "in-degree",
    "wallet_age_min_h": "wallet age", "node_context_resid_Rrange_max": "balance range",
    "node_context_resid_nin_mean": "n transfers in", "node_context_resid_nout_mean": "n transfers out",
}
cols = [c for _f, cc in FAMILIES for c in cc]
assert set(cols) == {c for c in rows[0] if c not in ("crew", "n_events")}

known = sorted((r for r in rows if not r["crew"].startswith("N") and r["crew"].lower() != "background"),
               key=lambda r: r["crew"])
newg = sorted((r for r in rows if r["crew"].startswith("N")), key=lambda r: r["crew"])
bg = [r for r in rows if r["crew"].lower() == "background"][0]
ordered = known + newg + [bg]
NR, NC = len(ordered), len(cols)

def val(r, c):
    v = float(r[c])
    return v * 100 if v <= 1 else v

M = np.array([[val(r, c) for c in cols] for r in ordered])
assert np.isin(M[:-1], (0.0, 100.0)).all(), "positive rows are not binary"

SIZE = (ns.W_DOUBLE, 3.75)
fig = ns.apply(SIZE)
ax = fig.add_subplot(111)
ax.set_xlim(-0.5, NC + 1.3)          # +margin column
ax.set_ylim(NR - 0.5, -0.5)          # inverted, rows top-down
ax.set_aspect("auto")

blue = np.array(to_rgb(ns.BLUE))
white = np.array([1.0, 1.0, 1.0])
SQ = 0.72

# light row banding for traceability
for i in range(NR):
    if i % 2 == 1:
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), NC + 1.0, 1.0,
                                   fc="#F5F5F5", ec="none", zorder=0))

# marks -- squares in display space (compute the unit aspect after layout)
fig.canvas.draw()
bb = ax.get_window_extent()
x_per_unit = bb.width / (NC + 1.8)
y_per_unit = bb.height / (NR + 0.0)
SQW = min(SQ * y_per_unit / x_per_unit, SQ)     # square marks
for i in range(NR - 1):              # binary rows: filled square = not observed
    for j in range(NC):
        if M[i, j] == 100:
            ax.add_patch(plt.Rectangle((j - SQW / 2, i - SQ / 2), SQW, SQ,
                                       fc=ns.BLUE, ec="none", zorder=3))
i = NR - 1                           # background row: shaded cell + printed share
for j in range(NC):
    f = M[i, j] / 100
    if f > 0:
        ax.add_patch(plt.Rectangle((j - 0.47, i - SQ / 2), 0.94, SQ,
                                   fc=tuple(white * (1 - f) + blue * f), ec="none", zorder=3))
        ax.annotate(f"{M[i, j]:.0f}%", xy=(j, i), ha="center", va="center",
                    fontsize=6.0, color=("white" if f > 0.55 else ns.INK), zorder=4)

# right margin: unobserved-feature count per group
for i in range(NR):
    n_miss = int((M[i] > 0).sum())
    ax.annotate(str(n_miss) if i < NR - 1 else f"{n_miss}", xy=(NC + 0.55, i),
                ha="center", va="center", fontsize=7.0,
                color=(ns.GRAY if n_miss == 0 else ns.INK))
ax.annotate("not obs.\n(of 15)", xy=(NC + 0.55, -1.05), ha="center", va="bottom",
            fontsize=6.4, color=ns.GRAY, linespacing=1.25, annotation_clip=False)

# frame around the matrix proper + separators
ax.add_patch(plt.Rectangle((-0.5, -0.5), NC, NR, fc="none", ec="#9A9A9A",
                           lw=0.7, zorder=5))
ax.plot([-0.5, NC - 0.5], [len(known) - 0.5] * 2, ls=(0, (4, 2.4)),
        color=ns.INK, lw=0.8, zorder=6)
ax.plot([-0.5, NC - 0.5], [len(known) + len(newg) - 0.5] * 2,
        color=ns.INK, lw=1.0, zorder=6)

# family brackets
j0 = 0
for fam, cc in FAMILIES:
    j1 = j0 + len(cc)
    ax.plot([j0 - 0.35, j1 - 0.65], [-0.95, -0.95], color=ns.INK, lw=0.8,
            clip_on=False)
    ax.annotate(fam, xy=((j0 + j1 - 1) / 2, -1.25), ha="center", va="bottom",
                fontsize=9.2, fontweight="bold", color=ns.INK, annotation_clip=False)
    j0 = j1

ax.set_xticks(range(NC))
ax.set_xticklabels([FEAT_LABEL[c] for c in cols], rotation=40, ha="right",
                   fontsize=7.4, rotation_mode="anchor")
ax.set_yticks(range(NR))
ax.set_yticklabels([f"{r['crew']}  (n={r['n_events']})" for r in ordered], fontsize=7.4)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

# one terse legend line, below the rotated column labels
LY = NR + 4.6
ax.add_patch(plt.Rectangle((0.0, LY - 0.31), SQW * 0.9, 0.62, fc=ns.BLUE, ec="none",
                           clip_on=False))
ax.annotate("not observed for the group's event(s);  background row: share of its 139 deposits",
            xy=(0.8, LY), ha="left", va="center", fontsize=7.2,
            color=ns.INK, annotation_clip=False)

ns.save_all(fig, "figC1_missingness", SIZE)
print("figC1 done")
