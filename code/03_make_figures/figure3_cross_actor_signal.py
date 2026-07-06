"""
Figure 3 -- the cross-crew signal is real but fragile. Three panels, all read
from the public data files:

  A  primary (17 crews) vs sensitivity (18 crews)   <- data/main_results.csv
  B  the per-crew AUROCs, which polarise to 0 or 1   <- data/per_crew_results.csv
  C  the permutation null distribution               <- data/permutation_null.csv
                                                         + data/permutation_summary.csv

    python figure3_cross_actor_signal.py   ->  ../../figures/figure3_cross_actor_signal.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, GRAY = "#0072B2", "#7F7F7F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "axes.linewidth": 0.8})


def read_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# --- panel A: primary vs sensitivity, from main_results.csv ---
main = {r["crew_set"]: r for r in read_csv("main_results.csv")}
primary, sensitivity = main["17_crews"], main["18_crews"]

# --- panel B: per-crew AUROCs, from per_crew_results.csv ---
per_crew = read_csv("per_crew_results.csv")
fold_aurocs = [float(r["auroc_logistic"]) for r in per_crew]
largest = max(per_crew, key=lambda r: int(r["n_events"]))     # the one big crew
largest_auroc = float(largest["auroc_logistic"])
largest_events = int(largest["n_events"])

# --- panel C: null distribution, from permutation_null.csv + summary ---
null_scores = [float(r["shuffled_auroc"]) for r in read_csv("permutation_null.csv")
               if r["crew_set"] == "17_crews" and r["model"] == "logistic"]
perm = {(r["crew_set"], r["model"]): r for r in read_csv("permutation_summary.csv")}
observed = float(perm[("17_crews", "logistic")]["observed_auroc"])
null_mean = float(perm[("17_crews", "logistic")]["null_mean"])

fig, (a, b, c) = plt.subplots(1, 3, figsize=(8.2, 3.35))

# ---- A ----
def draw_point(ax, row, y, filled):
    v, lo, hi = float(row["auroc"]), float(row["ci_low"]), float(row["ci_high"])
    ax.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt="o", color=BLUE,
                ms=8, capsize=4, lw=1.8, mfc=(BLUE if filled else "white"))
    ax.text(v + 0.02, y + 0.22, f"{v:.2f},  p = {float(row['p_value']):.3f}",
            fontsize=8.2, ha="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=5)

draw_point(a, primary, 1, filled=True)
draw_point(a, sensitivity, 0, filled=False)
a.axvline(0.5, ls="--", color=GRAY, lw=1, zorder=1)
a.axvline(null_mean, ls=":", color="#666666", lw=1.2, zorder=1)
a.text(null_mean + 0.01, 1.72, f"null mean {null_mean:.2f}", fontsize=7.2, color="#444444")
a.set_yticks([0, 1])
a.set_yticklabels(["18 crews\n(sensitivity)", "17 crews\n(primary)"], fontsize=8.4)
a.set_ylim(-0.75, 1.95)
a.set_xlim(0.40, 0.95)
a.set_xlabel("cross-crew AUROC (95% CI)", fontsize=8.8)
a.set_title("A   primary vs sensitivity\n(filled = significant, open = not)", fontsize=9.4)

# ---- B ----
# spread the points across the panel deterministically (golden-ratio sequence)
xs = [0.08 + 0.84 * (((i + 1) * 0.6180339887) % 1.0) for i in range(len(fold_aurocs))]
sizes = [150 if abs(v - largest_auroc) < 1e-6 else 46 for v in fold_aurocs]
b.scatter(xs, fold_aurocs, s=sizes, color=BLUE, alpha=0.9, edgecolor="white", lw=0.6, zorder=3)
bx = xs[fold_aurocs.index(largest_auroc)]
b.annotate(f"largest crew\n({largest_events} events)", xy=(bx, largest_auroc),
           xytext=(bx + 0.14, largest_auroc + 0.14), fontsize=7.6, color="#333333",
           arrowprops=dict(arrowstyle="-", lw=0.6, color="#666666"))
b.axhline(0.5, ls="--", color=GRAY, lw=1)
b.set_xlim(0, 1)
b.set_xticks([])
b.set_ylabel("per-crew AUROC", fontsize=8.8)
b.set_ylim(-0.06, 1.06)
b.set_title("B   17 held-out crews\n(single-event crews polarize to 0/1)", fontsize=9.4)

# ---- C ----
c.hist(null_scores, bins=28, color="#BBBBBB", edgecolor="white", lw=0.4)
c.axvline(null_mean, ls=":", color="#666666", lw=1.4)
c.axvline(observed, color=BLUE, lw=2.2)
c.text(null_mean - 0.02, c.get_ylim()[1] * 0.97, f"null mean ≈ {null_mean:.2f}",
       fontsize=7.6, color="#444444", ha="right", va="top")
c.text(observed + 0.01, c.get_ylim()[1] * 0.70,
       f"observed {observed:.3f}\np = {float(perm[('17_crews','logistic')]['p_value']):.3f}",
       fontsize=7.6, color=BLUE, va="top")
c.set_xlabel("permuted AUROC", fontsize=8.8)
c.set_ylabel("count", fontsize=8.8)
c.set_title("C   permutation null\n(crew-preserving, N = %d)" % len(null_scores), fontsize=9.4)

for ax in (a, b, c):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.4)
    ax.set_box_aspect(1)

fig.tight_layout(w_pad=1.1)
out = os.path.join(OUT, "figure3_cross_actor_signal.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"figure 3 written -> {out}")
print(f"  A: {primary['auroc']} vs {sensitivity['auroc']}   B: {len(fold_aurocs)} crews, "
      f"largest={largest['crew']}   C: {len(null_scores)} shuffles, observed {observed}")
