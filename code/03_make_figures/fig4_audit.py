# -*- coding: utf-8 -*-
"""Figure 4 -- the missingness audit.

Same canvas and type scale as fig3 (8.2 x 3.35 in, square panels,
8.4/8.8/9.4 pt type). Colors follow the set-wide semantics: Φ blue, factory
orange, combined dark. Category labels stay horizontal, every point prints
its value, and the increment labels in panel C sit on white pads so they
survive crossing the whiskers.
Data: missingness_audit_metrics.csv, missingness_audit_increments.csv.
"""
import csv, io, os
import natstyle as ns
import matplotlib.pyplot as plt

def rd(name):
    return list(csv.DictReader(io.open(os.path.join(ns.DATA, name), encoding="utf-8-sig")))

metrics = rd("missingness_audit_metrics.csv")
increments = rd("missingness_audit_increments.csv")

ORDER = ["provenance (Phi)", "factory-pattern", "provenance + factory"]
LABEL = ["Φ", "factory", "Φ + factory"]
COLOR = [ns.BLUE, ns.ORANGE, ns.DARK]

SIZE = (8.2, 3.35)
fig = ns.apply(SIZE)
plt.rcParams.update({"font.size": 10})
(a, b, c) = fig.subplots(1, 3)

def stage_panel(ax, stage, title):
    by = {r["feature_set"]: r for r in metrics if r["stage"] == stage}
    ax.axhline(0.5, ls="--", color=ns.GRAY, lw=1.0, zorder=1)
    for i, key in enumerate(ORDER):
        v, lo, hi = (float(by[key][k]) for k in ("auroc", "ci_low", "ci_high"))
        ax.errorbar([i], [v], yerr=[[v - lo], [hi - v]], fmt="o", ms=8,
                    color=COLOR[i], capsize=4, lw=1.8, zorder=4)
        ax.annotate(f"{v:.2f}", xy=(i, hi), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.2, color=COLOR[i])
    ax.set_xticks(range(3))
    ax.set_xticklabels(LABEL, fontsize=9.0)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.28, 1.04)
    ax.set_ylabel("cross-actor AUROC (95% CI)", fontsize=9.4)
    ax.annotate("chance", xy=(2.55, 0.507), fontsize=7.8, color=ns.GRAY,
                ha="right", va="bottom")
    ax.set_title(title, fontsize=10)

stage_panel(a, "before_recovery", "A   apparent\n(outflow records missing)")
stage_panel(b, "after_recovery", "B   after outflow recovery\n(same split, same models)")

# ---- C: increment forest --------------------------------------------------------
combined = next(r for r in increments if r["comparison"].startswith("(provenance + factory)"))
factory = next(r for r in increments if r["comparison"].startswith("factory-pattern"))
c.axvline(0, color=ns.INK, lw=1.0, zorder=2)
for y, r, col in ((1.0, combined, ns.DARK), (0.0, factory, ns.ORANGE)):
    v, lo, hi = (float(r[k]) for k in ("auroc_change", "ci_low", "ci_high"))
    c.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt="o", ms=8, color=col,
               capsize=4, lw=1.8, zorder=4)
    c.annotate(f"{v:+.4f}  [{lo:+.3f}, {hi:+.3f}]", xy=((lo + hi) / 2, y),
               xytext=(0, 12), textcoords="offset points", ha="center",
               fontsize=8.2, color=ns.INK, zorder=6,
               bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
c.set_yticks([0, 1])
c.set_yticklabels(["factory\n− Φ", "Φ + factory\n− Φ"], fontsize=9.0)
c.set_ylim(-0.75, 1.85)
c.set_xlim(-0.42, 0.24)
c.set_xlabel("AUROC difference (95% CI)", fontsize=9.4)
c.annotate("no gain", xy=(0.012, -0.70), fontsize=7.8, color=ns.GRAY,
           ha="left", va="bottom")
c.set_title("C   increment after audit\n(both intervals span zero)", fontsize=10)

for ax in (a, b, c):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9.0)
    ax.set_box_aspect(1)

fig.set_layout_engine("tight")
ns.save_all(fig, "fig4_audit", SIZE)
print("fig4 done")
