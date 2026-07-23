# -*- coding: utf-8 -*-
"""
Robustness forest-and-table figures, one per family (A leave-one-actor-out,
B Phi-extraction parameters, C feature-family ablation). Each is a forest plot
of the perturbed cross-actor AUROCs with a small stats table underneath; the
unperturbed estimate gets a reference diamond, markers are filled where the
group-aware permutation p < 0.05, and the empirical-null (0.53) and main
(0.703) reference lines run through all of them.

Every number is read from robustness_forest_data.json, the frozen recompute
produced by robustness_permutation.py (scikit-learn 1.3.2, N=1000, seed
20260629; its main row reproduces permutation_test.py's p = 0.026973).

    python build_forest_tables.py         # renders robustness/figures/forest_table_{groups,params,features}.png
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "03_make_figures"))   # package natstyle
import natstyle as N                                                       # sets Agg + determinism
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_JSON = os.path.join(HERE, "robustness_forest_data.json")
FIGDIR = os.path.join(HERE, "figures")
NULL = 0.53

SIG, NOSIG, INK, GRY = N.BLUE, N.DARK, N.INK, N.GRAY
_META = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}


def stars(p):
    return "" if p is None else ("***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "")


def ci_txt(d):
    return "—" if d["lo"] is None else f"({d['lo']:.3f}, {d['hi']:.3f})"


def p_txt(d):
    return "—" if d["p"] is None else f"{d['p']:.3f}"


def is_main(lb):
    return lb in ("None (all 17)", "Main (7d/2hop/10 ETH)", "Full Φ")


def make_fig(data, sections, title, sub, fname):
    bm = data["main"]
    items = []
    for st, rws in sections:
        if st is not None:
            items.append(("hdr", st))
        for r in rws:
            items.append(("row", r))
    NS = len(items)
    H = 0.315 * NS + 2.15
    fig = N.apply((7.4, H))
    fig.set_constrained_layout(False)
    b = 1.02 / H
    h = 1 - (2.02 / H)
    axf = fig.add_axes([0.255, b, 0.345, h])
    axt = fig.add_axes([0.615, b, 0.378, h])
    YL = (-0.9, NS + 1.15)
    for ax in (axf, axt):
        ax.set_ylim(*YL)
    axt.set_xlim(0, 1)
    axt.axis("off")
    ypos = [NS - 1 - i for i in range(NS)]

    def figy(dy):
        return b + h * (dy - YL[0]) / (YL[1] - YL[0])

    def rule(dy, x0=0.045, x1=0.995, lw=1.0, color=INK, z=12):
        fig.add_artist(Line2D([x0, x1], [figy(dy), figy(dy)], transform=fig.transFigure,
                              color=color, lw=lw, zorder=z, solid_capstyle="butt"))

    axf.axvline(NULL, ls=(0, (4, 2.5)), lw=0.8, color=GRY, zorder=1)
    axf.axvline(bm, ls="-", lw=0.8, color=SIG, alpha=0.35, zorder=1)
    HY = NS + 0.55
    axf.text(-0.02, HY, "Perturbation", transform=axf.get_yaxis_transform(), ha="right",
             va="center", fontsize=6.8, fontweight="bold", color=INK)
    CX = {"AUROC": 0.19, "95% CI": 0.52, "p": 0.82, "N": 0.985}
    hd = {"AUROC": "AUROC", "95% CI": "95% CI", "p": "p", "N": "folds"}
    for k, x in CX.items():
        axt.text(x, HY, hd[k], ha=("center" if k == "95% CI" else "right"), va="center",
                 fontsize=6.8, fontweight="bold", color=INK)
    rule(NS + 1.02, lw=1.1)
    rule(NS + 0.05, lw=0.8)
    rule(-0.85, lw=1.1)
    first_hdr = True
    for (kind, payload), y in zip(items, ypos):
        if kind == "hdr":
            if not first_hdr:
                rule(y + 0.55, lw=0.4, color="#CCCCCC")
            first_hdr = False
            axf.text(-0.235, y, payload, transform=axf.get_yaxis_transform(), ha="left",
                     va="center", fontsize=6.2, fontstyle="italic", color="#777777")
            continue
        d = payload
        sig = (d["p"] is not None and d["p"] < 0.05)
        col = SIG if sig else NOSIG
        mk = "D" if is_main(d["label"]) else "o"
        msz = 5.6 if mk == "D" else 4.7
        xerr = None if d["lo"] is None else [[d["auroc"] - d["lo"]], [d["hi"] - d["auroc"]]]
        axf.errorbar(d["auroc"], y, xerr=xerr, fmt=mk, ms=msz, color=col,
                     mfc=(col if (sig or mk == "D") else "white"), mec=col, mew=1.0,
                     ecolor=col, elinewidth=1.0, capsize=2.4, capthick=0.9, zorder=6)
        axf.text(-0.02, y, d["label"], transform=axf.get_yaxis_transform(), ha="right",
                 va="center", fontsize=6.5, fontweight=("bold" if is_main(d["label"]) else "normal"), color=INK)
        axt.text(CX["AUROC"], y, f"{d['auroc']:.3f}{stars(d['p'])}", ha="right", va="center",
                 fontsize=6.4, fontweight=("bold" if is_main(d["label"]) else "normal"), color=INK)
        axt.text(CX["95% CI"], y, ci_txt(d), ha="center", va="center", fontsize=6.1, color="#3A3A3A")
        axt.text(CX["p"], y, p_txt(d), ha="right", va="center", fontsize=6.1, color=(SIG if sig else "#707070"))
        axt.text(CX["N"], y, str(d["folds"]), ha="right", va="center", fontsize=6.1, color="#707070")
    axf.set_yticks([])
    axf.set_xlim(0.38, 0.92)
    for s in ("left", "right", "top"):
        axf.spines[s].set_visible(False)
    axf.spines["bottom"].set_bounds(0.4, 0.9)
    axf.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axf.tick_params(length=2.6, labelsize=6.2)
    axf.set_xlabel("Cross-actor AUROC  (95% CI)", fontsize=7.0)
    axf.annotate("null 0.53", xy=(NULL, YL[0]), xytext=(NULL, YL[0] + 0.15), ha="center",
                 va="bottom", fontsize=5.6, color=GRY)
    axf.annotate(f"main {bm:.3f}", xy=(bm, YL[0]), xytext=(bm, YL[0] + 0.15), ha="center",
                 va="bottom", fontsize=5.6, color=SIG)
    fig.text(0.045, 1 - 0.34 / H, title, fontsize=9.0, fontweight="bold", ha="left", color=INK)
    fig.text(0.045, 1 - 0.74 / H, sub, fontsize=6.4, color="#555555", ha="left")
    fig.text(0.045, 0.30 / H, "Diamond = reference estimate.   Filled marker = group-aware permutation p<0.05;  open marker = not significant.   Stars: *** p<.01  ** p<.05  * p<.1.",
             fontsize=5.6, color="#666666", ha="left")
    fig.text(0.045, 0.11 / H, "AUROC = leave-one-group-out (paper protocol);  CI = bootstrap over actor groups;  p = group-aware permutation (Ojala–Garriga, N=1000, scikit-learn 1.3.2).",
             fontsize=5.4, color="#888888", ha="left")
    os.makedirs(FIGDIR, exist_ok=True)
    bpth = os.path.join(FIGDIR, fname)
    fig.savefig(bpth + ".png", dpi=600, facecolor="white", metadata=_META["png"])
    plt.close(fig)
    print("->", fname)


def main():
    if not os.path.exists(DATA_JSON):
        raise SystemExit(f"missing {os.path.basename(DATA_JSON)} -- run robustness_permutation.py first "
                         "(it is the frozen canonical recompute the forest tables render from).")
    data = json.load(io.open(DATA_JSON, encoding="utf-8"))
    none_row, loo, params, feats = data["none"], data["loo"], data["params"], data["feats"]

    lm = {d["label"]: d for d in loo}
    est_names = ["− Harmony", "− Beanstalk", "− Audius", "− Monkey", "− Convergence"]
    established = sorted([lm[n] for n in est_names if n in lm], key=lambda d: -d["auroc"])
    newly = sorted([d for d in loo if d["label"] not in est_names], key=lambda d: -d["auroc"])
    secA = [(None, [none_row]), ("Established actors (5)", established), ("Newly-gated actors (12)", newly)]
    pm = {d["label"]: d for d in params}
    secB = [(None, [params[0]]),
            ("Observation window", [pm["Window 3 d"], pm["Window 5 d"], pm["Window 14 d"]]),
            ("Cone depth", [pm["Depth 1 hop"], pm["Depth 3 hops"]]),
            ("Value / share floor", [pm["Value floor 15 ETH"], pm["Value floor 20 ETH"], pm["Share floor 0.2 V"]])]
    secC = [(None, [feats[0]]), ("Feature family removed", feats[1:])]

    make_fig(data, secA, "Robustness A · Leave-one-actor-out",
             "Remove one actor group from training and evaluation; refit on the remaining groups.",
             "forest_table_groups")
    make_fig(data, secB, "Robustness B · Φ extraction parameters",
             "Re-extract Φ under an alternative observation window, cone depth, or value / share floor.",
             "forest_table_params")
    make_fig(data, secC, "Robustness C · Feature-family ablation",
             "Drop one feature family from the model; all 17 evaluation folds retained.",
             "forest_table_features")
    print("done -> robustness/figures/forest_table_{groups,params,features}")


if __name__ == "__main__":
    main()
