"""Figure 4: the missingness audit -- factory-pattern AUROC before vs after
recovering the outflow records, plus the increment over provenance (spans zero).
Reads missingness_audit_metrics.csv and missingness_audit_increments.csv."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#7F7F7F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "axes.linewidth": 0.8})


def read_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


metrics = read_csv("missingness_audit_metrics.csv")
increments = read_csv("missingness_audit_increments.csv")

# organise the before/after AUROCs by feature set
FEATURE_ORDER = ["provenance (Phi)", "factory-pattern", "provenance + factory"]
def values_for(stage):
    by = {r["feature_set"]: r for r in metrics if r["stage"] == stage}
    return [(float(by[f]["auroc"]), float(by[f]["ci_low"]), float(by[f]["ci_high"])) for f in FEATURE_ORDER]

fig, (a, b, c) = plt.subplots(1, 3, figsize=(8.2, 3.35))


def panel(ax, triples, title):
    labels = ["Φ", "factory", "Φ + factory"]
    colors = [BLUE, ORANGE, "#333333"]
    for i, ((v, lo, hi), col) in enumerate(zip(triples, colors)):
        ax.errorbar([i], [v], yerr=[[v - lo], [hi - v]], fmt="o", color=col,
                    ms=8, capsize=4, lw=1.8)
    ax.axhline(0.5, ls="--", color=GRAY, lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.4, rotation=18, ha="right")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0.30, 1.0)
    ax.set_ylabel("cross-crew AUROC", fontsize=8.8)
    ax.set_title(title, fontsize=9.4)


panel(a, values_for("before_recovery"), "A   apparent\n(outflow records missing)")
panel(b, values_for("after_recovery"), "B   after outflow recovery")

# ---- C: increment forest, from the increments file ----
combined = next(r for r in increments if r["comparison"].startswith("(provenance + factory)"))
factory = next(r for r in increments if r["comparison"].startswith("factory-pattern"))


def draw_increment(y, row, color):
    v, lo, hi = float(row["auroc_change"]), float(row["ci_low"]), float(row["ci_high"])
    c.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt="o", color=color, ms=8, capsize=4, lw=1.8)
    c.text(v, y + 0.30, f"{v:+.3f}  [{lo:+.3f}, {hi:+.3f}]", fontsize=7.9, ha="center",
           bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=5)

draw_increment(1, combined, "#333333")
draw_increment(0, factory, ORANGE)
c.axvline(0, color="#333333", lw=1.2, zorder=1)
c.set_yticks([0, 1])
c.set_yticklabels(["factory\n− Φ", "Φ + factory\n− Φ"], fontsize=8.4)
c.set_ylim(-0.75, 1.95)
c.set_xlim(-0.44, 0.34)
c.set_xlabel("AUROC difference (95% CI)", fontsize=8.8)
c.set_title("C   increment after audit\n(both intervals span zero)", fontsize=9.4)

for ax in (a, b, c):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.4)
    ax.set_box_aspect(1)

fig.tight_layout(w_pad=1.1)
out = os.path.join(OUT, "figure4_missingness_audit.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"figure 4 written -> {out}")
print(f"  combined increment {combined['auroc_change']} [{combined['ci_low']}, {combined['ci_high']}]")
