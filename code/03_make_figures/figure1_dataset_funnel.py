"""
Figure 1 -- how the dataset was filtered, and what it ends up containing.

Left: the evidence-gate funnel, read from data/dataset_funnel.csv.
Right: the composition of the frozen dataset, counted from data/entrance_events.csv.

Every number is read from the public data files rather than typed in.

    python figure1_dataset_funnel.py   ->  ../../figures/figure1_dataset_funnel.png
"""
import csv
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#7F7F7F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "savefig.dpi": 300, "axes.linewidth": 0.8})

# --- read the funnel counts from data ---
STAGE_LABELS = {
    "0_census": "Full census of candidate addresses",
    "1_gate_eligible": "Gate-eligible after census bucketing",
    "2_on_chain_verified": "Gate 1  on-chain verification",
    "3_attributed": "Gate 2  attribution (independent sources)",
    "4_independent": "Gate 3  independence audit",
    "5_measurable": "Gate 4  measurability (frozen extraction)",
}
with open(os.path.join(DATA, "dataset_funnel.csv"), encoding="utf-8-sig") as fh:
    funnel = list(csv.DictReader(fh))
stages = [(STAGE_LABELS[r["stage"]], int(r["n_candidates"])) for r in funnel]

# --- count the composition straight from the event list ---
with open(os.path.join(DATA, "entrance_events.csv"), encoding="utf-8-sig") as fh:
    events = list(csv.DictReader(fh))
illicit = [e for e in events if e["is_illicit"] == "1"]
n_illicit = len(illicit)
n_background = len(events) - n_illicit
crews = sorted({e["crew"] for e in illicit})
n_crews = len(crews)
n_new = len({e["crew"] for e in illicit if e["crew_type"] == "newly_gated"})
n_known = n_crews - n_new
events_per_crew = Counter(e["crew"] for e in illicit)
largest_crew, largest_count = events_per_crew.most_common(1)[0]

fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.3),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

# ---- left: funnel ----
ys = np.arange(len(stages))[::-1]
values = [v for _, v in stages]
colors = [BLUE] + ["#7EB6D9"] * (len(stages) - 1)
left.barh(ys, values, height=0.48, color=colors, edgecolor="#33627E", lw=0.8)
for y, (label, v) in zip(ys, stages):
    left.text(0.5, y + 0.30, label, fontsize=7.4, va="bottom", color="#222222")
    left.text(v + 1.5, y, f"n = {v}", fontsize=8.6, va="center", fontweight="bold")
left.set_xlim(0, 100)
left.set_ylim(-0.45, len(stages) - 0.10)
left.axis("off")
left.set_title("Evidence-gate funnel\n(census to primary set)", fontsize=8.8, loc="left")

# ---- right: composition (three units that must not be added together) ----
right.axis("off")
right.set_title("Frozen dataset composition\n(three distinct units — not additive)",
                fontsize=8.8, loc="left")
boxes = [
    (f"{n_new} newly gated + {n_known} pre-existing\n= {n_crews} independent actor groups", BLUE),
    (f"{n_illicit} illicit entrance events\n(largest crew {largest_count}; most others single)", "#333333"),
    (f"{n_background} background depositors\n(unlabeled — not clean)", GRAY),
    ("+ 1 cross-chain crew = 18 crews / 31 events\n— sensitivity analysis only", ORANGE),
]
for i, (text, color) in enumerate(boxes):
    y0 = 0.98 - i * 0.25
    right.add_patch(FancyBboxPatch((0.02, y0 - 0.20), 0.96, 0.19,
                    boxstyle="round,pad=0.012", facecolor="white", edgecolor=color,
                    lw=1.6, linestyle="--" if i == 3 else "solid", transform=right.transAxes))
    right.text(0.50, y0 - 0.105, text, ha="center", va="center", fontsize=7.7,
               transform=right.transAxes)

fig.tight_layout()
out = os.path.join(OUT, "figure1_dataset_funnel.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"figure 1 written -> {out}")
print(f"  (read {len(stages)} funnel stages; counted {n_crews} crews, "
      f"{n_illicit} illicit events, {n_background} background; largest crew = {largest_crew} with {largest_count})")
