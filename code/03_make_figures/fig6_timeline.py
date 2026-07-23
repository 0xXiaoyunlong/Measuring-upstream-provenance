# -*- coding: utf-8 -*-
"""
Figure 6 -- deposit timeline: the temporally balanced background next to the
illicit crews.

Illicit deposits are drawn per crew (17 actor-groups), not per event. Harmony's
14 deposits collapse into a single diamond at the crew's median deposit time,
each crew gets its own colour, and the colour->crew legend sits below the plot.
The bottom strip is the 140 balanced background deposits, jittered a little
(random y plus a few days of x) so they do not pile up on top of each other;
the original 139-deposit background keeps a strip of its own at the top for
comparison. Two vertical markers flag the OFAC designation (2022-08-08) and
the delisting (2025-03-21).

Illicit events and the original background come from data/entrance_events.csv,
the balanced background from balanced_background/data/. The jitter runs off a
fixed seed, so the PNG comes out the same on every run.

    python fig6_timeline.py
"""
import colorsys
import csv
import collections
import io
import os
import random
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(PKG, "figures", "fig6_timeline.png")

DESIGNATION = datetime(2022, 8, 8, tzinfo=timezone.utc)
DELISTING = datetime(2025, 3, 21, tzinfo=timezone.utc)
GRAY, VERM = "#9AA0A6", "#E8820C"

# Soft designer pastels. Hues fill the wheel evenly (uniform 21 deg gaps), but are
# handed out in a spread order (step 8 of 17) so time-adjacent crews land on
# opposite sides of the wheel; lightness alternates by hue slot, so any two
# neighbouring hues also differ in brightness.
def pastel(i, n):
    slot = (i * 8) % n                       # spread consecutive crews far apart
    h = (((slot * 360.0 / n) + 14) % 360) / 360.0
    light = 0.63 if slot % 2 == 0 else 0.77  # adjacent hues differ in brightness
    r, g, b = colorsys.hls_to_rgb(h, light, 0.64)
    return (r, g, b)

random.seed(20260715)

# group the 30 illicit events into crews; each crew is plotted once, at its median time
rows = list(csv.DictReader(io.open(os.path.join(PKG, "data", "entrance_events.csv"), encoding="utf-8-sig")))
by_crew = collections.defaultdict(list)
for r in rows:
    if r["is_illicit"] == "1":
        by_crew[r["crew"]].append(datetime.fromisoformat(r["deposit_time_utc"].replace("Z", "+00:00")))
crews = sorted(by_crew, key=lambda c: sorted(by_crew[c])[len(by_crew[c]) // 2])   # by median time
n_events = sum(len(v) for v in by_crew.values())

# the two backgrounds: canonical (all 2022) and balanced (140 deposits, 2022-2026)
old_bg = sorted(datetime.fromisoformat(r["deposit_time_utc"].replace("Z", "+00:00"))
                for r in rows if r["is_illicit"] == "0")
bal = list(csv.DictReader(io.open(os.path.join(PKG, "balanced_background", "data", "entrance_events.csv"),
                                  encoding="utf-8-sig")))
bg = sorted(datetime.fromisoformat(r["deposit_time_utc"].replace("Z", "+00:00"))
            for r in bal if r["is_illicit"] == "0")

fig, ax = plt.subplots(figsize=(12.5, 4.0))
fig.subplots_adjust(left=0.135, right=0.985, top=0.88, bottom=0.30)

XJIT = 7.0   # days of horizontal jitter


def jx(d):
    return mdates.date2num(d) + random.uniform(-XJIT, XJIT)


def bg_strip(dates, y0):
    xs = [jx(d) for d in dates]
    ys = [y0 + random.uniform(-0.18, 0.18) for _ in dates]
    ax.scatter(xs, ys, s=26, color=GRAY, marker="o", edgecolor="white",
               linewidth=0.5, alpha=0.85, zorder=3)


bg_strip(bg, 0.0)        # new balanced background: bottom strip
bg_strip(old_bg, 1.9)    # original background: top strip

# illicit crew markers (one per crew, distinct colour)
handles = []
n_crew = len(crews)
for i, c in enumerate(crews):
    ts = sorted(by_crew[c])
    med = ts[len(ts) // 2]
    col = pastel(i, n_crew)
    x = mdates.date2num(med)
    y = 1.0 + random.uniform(-0.09, 0.09)
    ax.scatter([x], [y], s=50 + 8 * len(ts), color=col, marker="D", edgecolor="#3a3a3a",
               linewidth=0.6, zorder=5)      # marker size scales with the group's event count
    lbl = f"{c} ({len(ts)})" if len(ts) > 1 else c
    handles.append(Line2D([0], [0], marker="D", linestyle="none", markersize=8,
                          markerfacecolor=col, markeredgecolor="black",
                          markeredgewidth=0.5, label=lbl))

# year bars: one between each pair of strips (symmetric above/below the crews row)
for ybar in (0.5, 1.42):
    ax.axhline(ybar, color="black", lw=1.1, zorder=1)
    for yr in range(2022, 2027):
        ax.text(mdates.date2num(datetime(yr, 7, 2, tzinfo=timezone.utc)), ybar, str(yr),
                fontsize=12, ha="center", va="center", zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.5))

# OFAC designation + delisting lines (dashes stop just below the label text)
for when, label in [(DESIGNATION, "OFAC designation\n8 Aug 2022"),
                    (DELISTING, "OFAC delisting\n21 Mar 2025")]:
    sx = mdates.date2num(when)
    ax.plot([sx, sx], [-0.42, 2.28], ls=(0, (5, 3)), lw=1.2, color=VERM, zorder=2,
            solid_capstyle="butt")
    ax.text(sx, 2.34, label, fontsize=8.5, color=VERM, ha="center", va="bottom",
            linespacing=1.15, fontweight="bold")

ax.set_yticks([0, 1, 1.9])
ax.set_yticklabels([f"Balanced background\n(n = {len(bg)}, 2022–2026)",
                    f"Illicit crews\n({len(crews)} groups, {n_events} events)",
                    f"Original background\n(n = {len(old_bg)}, 2022 only)"], fontsize=9.0)
ax.set_ylim(-0.48, 2.78)
ax.set_xlim(mdates.date2num(datetime(2021, 12, 1, tzinfo=timezone.utc)),
            mdates.date2num(datetime(2026, 8, 1, tzinfo=timezone.utc)))
ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.tick_params(axis="x", labelsize=7)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("deposit month", fontsize=9.5)
ax.spines[["top", "right", "left"]].set_visible(False)

ax.legend(handles=handles, ncol=9, fontsize=8, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), frameon=False, handletextpad=0.3,
          columnspacing=0.9, title="Illicit crew (events)", title_fontsize=8.5)

fig.savefig(OUT, dpi=300, facecolor="white")
plt.close(fig)
print("saved -> " + os.path.basename(OUT))
print(f"crews={len(crews)}  events={n_events}  background={len(bg)}")
print("crew order (left->right by median time):", ", ".join(crews))
