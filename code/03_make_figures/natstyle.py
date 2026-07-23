# -*- coding: utf-8 -*-
"""Shared design system for the manuscript figures (Nature-grade restyle).

Shared palette, type sizes and export code so the figures come out as one set.
No numbers are computed here: every figure script reads the CSVs this
package's analysis produced, in ../../data. Self-contained; nothing beyond
matplotlib + pillow is needed.
"""
import os

# --- deterministic rendering ------------------------------------------------
# Make the figure bytes reproducible across runs: a fixed SOURCE_DATE_EPOCH removes
# the wall-clock timestamp matplotlib would otherwise embed in PDF/SVG metadata, and
# a fixed svg.hashsalt makes SVG clip-path/gradient ids stable instead of random.
# (Set before importing matplotlib so the PDF/SVG backends pick them up.)
os.environ.setdefault("SOURCE_DATE_EPOCH", "1735689600")   # 2025-01-01T00:00:00Z, fixed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.hashsalt"] = "upstream-provenance-canonical"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")     # this package's data/
FIGS = os.path.join(HERE, "..", "..", "figures")  # this package's figures/

# ---- semantic palette (Okabe-Ito, colorblind-safe) --------------------------
BLUE = "#0072B2"      # Φ / primary / pre-existing groups
SKY = "#56B4E9"       # newly gate-admitted groups
ORANGE = "#E69F00"    # factory-pattern
DARK = "#2B2B2B"      # Φ + factory combined
VERM = "#D55E00"      # observed / highlight accent
GRAY = "#8C8C8C"      # neutral lines & secondary text
LGRAY = "#C9C9C9"     # light neutral fills
INK = "#1A1A1A"       # primary text

# Nature double column: 183 mm = 7.2 in. Max height 247 mm.
W_DOUBLE = 7.2
W_SINGLE = 3.5


def apply(figsize):
    """Set the shared style and return a new figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlepad": 4.0,
        "svg.fonttype": "none",     # keep SVG text editable
        "pdf.fonttype": 42,         # embed TrueType in PDF (editable text)
        "figure.constrained_layout.use": True,
        "mathtext.fontset": "dejavusans",
    })
    return plt.figure(figsize=figsize)


def panel_label(ax, letter, dx=-18.0, dy=1.0):
    """Uppercase bold panel letter, top-left outside the axes."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=8, fontweight="bold", ha="left", va="bottom", color=INK)


def save_all(fig, name, size_inches=None, preview_only=False):
    """Write the figure as a 600 dpi PNG into ../../figures."""
    os.makedirs(FIGS, exist_ok=True)
    base = os.path.join(FIGS, name)
    if preview_only:
        prev = base + "_preview.png"
        fig.savefig(prev, dpi=150, bbox_inches="tight")
        return prev
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", facecolor="white",
                metadata={})
    plt.close(fig)
    return f"{base}.png"
