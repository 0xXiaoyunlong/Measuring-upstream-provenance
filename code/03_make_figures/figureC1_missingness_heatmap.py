"""
Figure C1 -- per-crew missingness heatmap. For every crew and every provenance
feature, the share of that crew's events for which the feature was not observed.
Read from data/missingness_by_crew.csv. For provenance features an empty value is
usually structural (a funding path with no intermediary has no dwell to measure)
and enters the model as an explicit flag; see build_upstream_features.py.

    python figureC1_missingness_heatmap.py   ->  ../../figures/figureC1_missingness_heatmap.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.linewidth": 0.8, "savefig.dpi": 300})

# short, readable labels for the 15 feature columns
SHORT = {"observed_hops": "hops", "path_elapsed_time": "elapsed time",
         "dwell_mean_h": "dwell mean", "dwell_max_h": "dwell max",
         "coverage": "coverage", "attribution_ambiguous": "ambiguity flag",
         "node_context_out_degree": "ctx out-degree", "node_context_in_degree": "ctx in-degree",
         "wallet_age_min_h": "wallet age", "node_context_resid_Rrange_max": "resid. range",
         "node_context_resid_nin_mean": "resid. n-in", "node_context_resid_nout_mean": "resid. n-out",
         "boundary_type": "boundary type", "censoring_reason": "censor reason",
         "observation_confidence": "obs. confidence"}

with open(os.path.join(DATA, "missingness_by_crew.csv"), encoding="utf-8-sig") as fh:
    rows = list(csv.reader(fh))
header, body = rows[0], rows[1:]
features = header[2:]                       # first two columns are crew, n_events
crew_labels = [r[0] for r in body]
n_events = [int(r[1]) for r in body]
matrix = np.array([[float(v) for v in r[2:]] for r in body])

fig, ax = plt.subplots(figsize=(7.6, 4.6))
image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(features)))
ax.set_xticklabels([SHORT.get(f, f) for f in features], rotation=45, ha="right", fontsize=8.2)
ax.set_yticks(range(len(crew_labels)))
ax.set_yticklabels([f"{c}  (n={n})" for c, n in zip(crew_labels, n_events)], fontsize=8.2)

# a solid line above the background row, a dashed line between known and new crews
ax.axhline(len(crew_labels) - 1.5, color="white", lw=2.2)
n_known = sum(1 for c in crew_labels if c in ("Audius", "Beanstalk", "Convergence", "Harmony", "Monkey"))
ax.axhline(n_known - 0.5, color="white", lw=1.2, ls=(0, (3, 2)))

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        share = matrix[i, j]
        if share > 0.005:
            ax.text(j, i, f"{int(round(share * 100))}", ha="center", va="center",
                    fontsize=6.2, color=("black" if share > 0.55 else "white"))

bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
bar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
bar.set_ticklabels(["0", "25", "50", "75", "100"])
bar.set_label("share of events with no observed value (%)", fontsize=8.5)
bar.ax.tick_params(labelsize=7.8)
ax.set_title("Per-crew share of events with no observed value, by provenance feature",
             fontsize=8.8, pad=8)

fig.tight_layout()
out = os.path.join(OUT, "figureC1_missingness_heatmap.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"figure C1 written -> {out}   ({len(crew_labels)} rows x {len(features)} features)")
