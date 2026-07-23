# -*- coding: utf-8 -*-
"""
Figure 5 -- the robustness forest for the main text.

Every variant shows up twice, once against each background cohort, and the two
lines sit close enough that the pair reads as a single unit: dark gray is the
original background (139 deposits), orange the balanced one (140). A filled
square means the permutation p came in under .05; open means it did not. The
main-model anchor and the 17 leave-one-group-out runs take up the left panel.
The parameter sweeps and feature-family ablations go on the right, which ends
up shorter, so the legend lives in the gap that leaves below it. Both panels
share one Macro-AUROC axis, with a dashed line at each background's own
main-model score (0.703 and 0.802).

Nothing here computes anything. The numbers arrive through figdata, which
reads the forest JSONs of both frozen packages -- if a value looks off,
run `python figdata.py` on its own and make sure its oracle comes back green
before suspecting the plotting.
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import figdata as F
import research_style as S

XLIM = (0.40, 0.92)
X_LABEL = -0.07      # variant label, right-aligned, hugging the box
X_LEFT = -1.05       # left extent of the full-width black section rules
X_VAL = 1.07         # AUROC [95% CI]
X_P = 1.82           # p  (close to where the CI text ends)
SEP_R = 2.18         # right extent of the black section rules
BASE = "#454545"
BAL = "#fda501"
FS_ROW = 5.3
FS_HEAD = 6.6
IN_PER_UNIT = 0.128
SLOT = 1.42
PH = 0.38            # half-offset of each line within a pair


def _d3(x):
    s = f"{x:.3f}"
    return s[1:] if s.startswith("0.") else s


def _fmt_val(a, lo, hi):
    return f"{_d3(a)} [{_d3(lo)}, {_d3(hi)}]"


def _fmt_p(p):
    return "<.001" if p < 0.001 else f"{p:.3f}"[1:]


def _row_ci(ax, y, lo, hi, color, lw):
    xlo, xhi = XLIM
    x0, x1 = max(lo, xlo), min(hi, xhi)
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=3,
            path_effects=[pe.Stroke(linewidth=lw + 0.5, foreground="black"), pe.Normal()])
    if lo < xlo:
        ax.plot([xlo], [y], marker="<", ms=3.0, mfc=color, mec="black", mew=0.4, zorder=3)
    if hi > xhi:
        ax.plot([xhi], [y], marker=">", ms=3.0, mfc=color, mec="black", mew=0.4, zorder=3)


def _series(ax, y, r, color, blend):
    sig = r["p"] < 0.05
    _row_ci(ax, y, r["lo"], r["hi"], color, lw=1.2 if sig else 0.6)
    ax.plot([r["auroc"]], [y], marker="s", ms=4.0,
            mfc=(color if sig else "white"), mec="black", mew=0.5, zorder=4)
    ax.text(X_VAL, y, _fmt_val(r["auroc"], r["lo"], r["hi"]), transform=blend,
            ha="left", va="center", fontsize=FS_ROW, color=S.INK, clip_on=False)
    ax.text(X_P, y, _fmt_p(r["p"]), transform=blend, ha="left", va="center",
            fontsize=FS_ROW, fontweight=("bold" if sig else "normal"),
            color=S.INK, clip_on=False)


def _layout(blocks, anchor):
    """Work out y positions for one panel, walking top to bottom.

    Kept separate from rendering because the figure height has to be known
    before the axes exist -- draw() sizes the canvas from what comes back here.
    """
    y_head = 0.0
    box_top = -0.62
    entries, seps, black_lines, sec_heads = [], [], [], []
    y = -1.34
    if anchor:
        entries.append((y, anchor[0], anchor[1], True))
        y -= SLOT * 0.72
        black_lines.append(y + 0.38)
        y -= 0.38
    for bi, (title, prs) in enumerate(blocks):
        bsig = sum(1 for c, _s in prs if c["p"] < 0.05)
        wsig = sum(1 for _c, s in prs if s["p"] < 0.05)
        if bi > 0:
            y -= 0.26
            black_lines.append(y + 0.05)
            y -= 0.40
        sec_heads.append((y, title, bsig, wsig, len(prs)))
        y -= 1.02
        for i, (cr, sr) in enumerate(prs):
            entries.append((y, cr, sr, False))
            if i < len(prs) - 1:
                seps.append(y - SLOT / 2)
            y -= SLOT
        y -= 0.26
    box_bottom = y + 0.26 - 0.22
    y_axislab = box_bottom - 1.05
    return dict(y_head=y_head, box_top=box_top, box_bottom=box_bottom,
                y_axislab=y_axislab, entries=entries, seps=seps,
                black_lines=black_lines, sec_heads=sec_heads,
                y_top=y_head + 0.5, y_bottom=y_axislab - 0.15)


def _render(ax, lay, a703, a802):
    ax.set_xlim(*XLIM)
    ax.set_ylim(lay["y_bottom"], lay["y_top"])
    blend = ax.get_yaxis_transform()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    bt, bb = lay["box_top"], lay["box_bottom"]

    ax.add_patch(Rectangle((XLIM[0], bb), XLIM[1] - XLIM[0], bt - bb, fill=False,
                           edgecolor=S.INK, lw=0.9, zorder=5))
    for xv, col in ((a703, BASE), (a802, BAL)):
        ax.plot([xv, xv], [bb, bt], color=col, lw=1.0, ls=(0, (5, 2)), zorder=1)
    for xt in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        ax.plot([xt, xt], [bb, bb - 0.26], color=S.INK, lw=0.6, zorder=5, clip_on=False)
        ax.text(xt, bb - 0.52, f"{xt:.1f}", ha="center", va="top",
                fontsize=S.FS_SMALL, color=S.INK)
    ax.text(0.66, lay["y_axislab"], "Macro AUROC", ha="center", va="top",
            fontsize=S.FS_SMALL + 0.5, color=S.INK)

    for sy in lay["seps"]:
        ax.plot([XLIM[0], XLIM[1]], [sy, sy], color="#B4B4B4", lw=0.9, zorder=2)
    for by in lay["black_lines"]:
        ax.plot([X_LEFT, SEP_R], [by, by], transform=blend, color=S.INK, lw=0.9,
                zorder=6, clip_on=False)

    ax.text(X_LABEL, lay["y_head"], "Robustness variant", transform=blend, ha="right",
            va="center", fontsize=FS_HEAD, fontweight="bold", color=S.INK, clip_on=False)
    ax.text(X_VAL, lay["y_head"], "AUROC [CI]", transform=blend, ha="left",
            va="center", fontsize=FS_HEAD, fontweight="bold", color=S.INK, clip_on=False)
    ax.text(X_P, lay["y_head"], "p", transform=blend, ha="left", va="center",
            fontsize=FS_HEAD, fontweight="bold", color=S.INK, clip_on=False)

    for y, title, bsig, wsig, n in lay["sec_heads"]:
        ax.text(X_LABEL, y, title, transform=blend, ha="right", va="center",
                fontsize=FS_ROW + 0.8, fontweight="bold", color=S.INK, clip_on=False)
        ax.text(X_VAL, y, f"{bsig}/{n} base → {wsig}/{n} bal  (p<.05)",
                transform=blend, ha="left", va="center", fontsize=S.FS_SMALL - 0.3,
                style="italic", color=S.INK, clip_on=False)

    for cy, cr, sr, is_anchor in lay["entries"]:
        _series(ax, cy + PH, cr, BASE, blend)
        _series(ax, cy - PH, sr, BAL, blend)
        label = "Main model (all 17)" if is_anchor else cr["label"]
        ax.text(X_LABEL, cy, label, transform=blend, ha="right", va="center",
                fontsize=FS_ROW, fontweight=("bold" if is_anchor else "normal"),
                color=S.INK, clip_on=False)


def draw():
    pairs = F.forest_pair()
    anchor = pairs[0]
    a703, a802 = anchor[0]["auroc"], anchor[1]["auroc"]
    by_sec = {k: [p for p in pairs if p[0]["section"] == k] for k in ("loo", "params", "feats")}

    left = _layout([("Leave-one-group-out", by_sec["loo"])], anchor)
    right = _layout([("Parameters", by_sec["params"]),
                     ("Feature families", by_sec["feats"])], None)

    top_in = 0.12
    bot_in = 0.22
    left_h = (left["y_top"] - left["y_bottom"]) * IN_PER_UNIT
    right_h = (right["y_top"] - right["y_bottom"]) * IN_PER_UNIT
    fig_h = top_in + left_h + bot_in
    fig = plt.figure(figsize=(7.2, fig_h))

    top_fig = 1 - top_in / fig_h
    axL = fig.add_axes([0.155, top_fig - left_h / fig_h, 0.135, left_h / fig_h])
    axR = fig.add_axes([0.575, top_fig - right_h / fig_h, 0.135, right_h / fig_h])
    _render(axL, left, a703, a802)
    _render(axR, right, a703, a802)

    handles = [
        Line2D([0], [0], color=BASE, lw=1.2, marker="s", mfc=BASE, mec="black",
               mew=0.5, ms=4.0, label="original background (139)"),
        Line2D([0], [0], color=BAL, lw=1.2, marker="s", mfc=BAL, mec="black",
               mew=0.5, ms=4.0, label="balanced background (140)"),
        Line2D([0], [0], color=S.INK, marker="s", ls="none", mfc="white", mec="black",
               mew=0.5, ms=4.0, label="open = not significant"),
        Line2D([0], [0], color=S.INK, lw=1.0, ls=(0, (5, 2)),
               label="dashed = each background's\nmain model"),
    ]
    axR_bottom = top_fig - right_h / fig_h
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.70, axR_bottom * 0.62),
               ncol=2, frameon=False, fontsize=S.FS_SMALL, handletextpad=0.5,
               columnspacing=2.0, labelspacing=1.0)

    S.save(fig, "fig5_forest")
    plt.close(fig)


if __name__ == "__main__":
    draw()
