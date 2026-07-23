# -*- coding: utf-8 -*-
"""Figure 3 -- the cross-actor signal, three panels.

The frame keeps the accepted V1 layout: 8.2 x 3.35 in canvas, square panels
via set_box_aspect(1), 8.4/8.8/9.4 pt type, printing compact at .9\\linewidth.
Panel A is the primary-vs-sensitivity forest over each estimate's own null;
B ranks the per-group dot plot with names and event-count sizing; C shows the
permutation tail at/above the observed value in the accent color, with its
(count / N) tally read from the data. In all three panels dashed = 0.5 and
dotted = the empirical null mean 0.53, same as the caption says.
Data: main_results.csv, per_crew_results.csv, permutation_null.csv, permutation_summary.csv.
"""
import csv, io, os
import numpy as np
import natstyle as ns
import matplotlib.pyplot as plt

def rd(name):
    return list(csv.DictReader(io.open(os.path.join(ns.DATA, name), encoding="utf-8-sig")))

main = {r["crew_set"]: r for r in rd("main_results.csv")}
crews = rd("per_crew_results.csv")
null17 = [float(r["shuffled_auroc"]) for r in rd("permutation_null.csv")
          if r["crew_set"] == "17_crews" and r["model"] == "logistic"]
null18 = [float(r["shuffled_auroc"]) for r in rd("permutation_null.csv")
          if r["crew_set"] == "18_crews" and r["model"] == "logistic"]
psum = next(r for r in rd("permutation_summary.csv")
            if r["crew_set"] == "17_crews" and r["model"] == "logistic")
obs, null_mean, pval = float(psum["observed_auroc"]), float(psum["null_mean"]), float(psum["p_value"])
N_PERM = len(null17)               # read the permutation count from the data, don't hard-code
assert N_PERM >= 100

SIZE = (8.2, 3.35)
fig = ns.apply(SIZE)
plt.rcParams.update({"font.size": 10})          # V1 type scale on the V1 canvas
(a, b, c) = fig.subplots(1, 3)

def refs(ax, vert=True, z=1):
    f = ax.axvline if vert else ax.axhline
    f(0.5, ls="--", color=ns.GRAY, lw=1.0, zorder=z)
    f(null_mean, ls=":", color="#666666", lw=1.2, zorder=z)

# ---- A: forest, primary vs sensitivity ----------------------------------------
# each estimate over its own group-preserving permutation null (light silhouette,
# drawn first so the reference lines and markers sit on top): the 17-group point
# lands in its null's right tail, the 18-group point inside its null's bulk
for nulls_a, y in ((null17, 1), (null18, 0)):
    cnts_a, edges_a = np.histogram(nulls_a, bins=30, range=(0.40, 0.95))
    xc_a = (edges_a[:-1] + edges_a[1:]) / 2
    h_a = 0.30 * cnts_a / cnts_a.max()
    a.fill_between(xc_a, y - h_a, y + h_a, color=ns.LGRAY, alpha=0.45, lw=0, zorder=1)
refs(a)
for key, y, filled in (("17_crews", 1, True), ("18_crews", 0, False)):
    r = main[key]
    v, lo, hi, p = (float(r[k]) for k in ("auroc", "ci_low", "ci_high", "p_value"))
    a.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt="o", color=ns.BLUE, ms=8,
               capsize=4, lw=1.8, mfc=(ns.BLUE if filled else "white"), zorder=4)
    a.text(v, y + 0.22, f"{v:.4f},  p = {p:.3f}", fontsize=8.0, ha="center",
           bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=5)
a.text(0.545, 1.72, "null mean 0.53", fontsize=7.2, color="#444444")
a.set_yticks([0, 1])
a.set_yticklabels(["18 groups\n(sensitivity)", "17 groups\n(primary)"], fontsize=8.4)
a.set_ylim(-0.75, 1.95)
a.set_xlim(0.40, 0.95)
a.set_xlabel("cross-actor AUROC (95% CI)", fontsize=8.8)
a.set_title("A   primary vs sensitivity\n(filled = significant, open = not)", fontsize=9.4)

# ---- B: ranked per-group dot plot ----------------------------------------------
crews_sorted = sorted(crews, key=lambda r: (float(r["auroc_logistic"]), r["crew"]))
refs(b)
for i, r in enumerate(crews_sorted):
    v, n = float(r["auroc_logistic"]), int(r["n_events"])
    known = r["crew_type"] == "known"
    b.scatter([v], [i], s=18 + 24 * n, color=(ns.BLUE if known else ns.SKY),
              edgecolor="white", lw=0.6, zorder=4)
b.set_yticks(range(len(crews_sorted)))
b.set_yticklabels([r["crew"] for r in crews_sorted], fontsize=5.6)
b.tick_params(axis="y", pad=3)
b.set_ylim(-1.3, len(crews_sorted) - 0.0)
b.set_xlim(-0.12, 1.08)
b.set_xlabel("per-group held-out AUROC", fontsize=8.8)
big = next(i for i, r in enumerate(crews_sorted) if int(r["n_events"]) > 1)
b.annotate(f"{crews_sorted[big]['n_events']} events",
           xy=(float(crews_sorted[big]["auroc_logistic"]), big),
           xytext=(14, -2), textcoords="offset points", fontsize=7.2,
           color="#333333", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.6, color="#666666", shrinkA=0, shrinkB=6))
hnd = [plt.scatter([], [], s=34, color=ns.BLUE, ec="white", lw=0.6),
       plt.scatter([], [], s=34, color=ns.SKY, ec="white", lw=0.6)]
b.legend(hnd, ["pre-existing", "newly gated"], loc="lower right", frameon=False,
         fontsize=7.2, borderpad=0.1, handletextpad=0.2, borderaxespad=0.2,
         labelspacing=0.3)
b.set_title("B   17 held-out groups, ranked\n(single-event groups polarize to 0/1)", fontsize=9.4)

# ---- C: permutation null with the p-tail in accent -----------------------------
refs(c, z=3)
cnts, edges = np.histogram(null17, bins=28)
for cc, e0, e1 in zip(cnts, edges[:-1], edges[1:]):
    tail = e0 >= obs
    c.bar((e0 + e1) / 2, cc, width=e1 - e0,
          color=(ns.VERM if tail else "#BBBBBB"),
          alpha=(0.6 if tail else 1.0), edgecolor="white", lw=0.4, zorder=2)
n_ge = sum(1 for v in null17 if v >= obs)
c.axvline(obs, color=ns.VERM, lw=2.0, zorder=4)
ymax = cnts.max() * 1.12
c.text(0.515, ymax * 0.97, "null mean ≈ 0.53\n(empirical)", fontsize=7.6,
       color="#444444", ha="right", va="top")
c.text(obs + 0.012, ymax * 0.72,
       f"observed {obs:.4f}\np = {pval:.3f}\n({n_ge}/{N_PERM} permutations\n≥ observed)",
       fontsize=7.2, color=ns.VERM, va="top", linespacing=1.35)
c.set_ylim(0, ymax)
c.set_xlim(0.18, 0.84)
c.set_xlabel("permuted AUROC", fontsize=8.8)
c.set_ylabel("count", fontsize=8.8)
c.set_title(f"C   permutation null\n(group-preserving, N = {N_PERM})", fontsize=9.4)

for ax in (a, b, c):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.4)
    ax.set_box_aspect(1)            # square drawing areas, as in the accepted V1

fig.set_layout_engine("tight")
ns.save_all(fig, "fig3_signal", SIZE)
print("fig3 done")
