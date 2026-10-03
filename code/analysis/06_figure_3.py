"""Figure 3: LLMs shift more than clinical adjudicators on the same case versions.

(A) Cue effect in each case family for the clinical adjudicators (x) and the frontier LLMs (y). Plotted positions
carry a fixed-seed uniform jitter of up to 1.5 percentage points on each axis so that coinciding families separate.
(B) LLM cue effect (95% CI) by the adjudicators' support for the intended diagnosis, for the frontier LLMs and all 22
LLMs, against the adjudicators' cue effect on all families.

Reads results/tables/table_s16.csv, table_s17.csv and results/estimates/05_physician_validation.csv.
Writes results/figures/figure_3.png.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from common import ESTIMATES, FIGURES, TABLES

FIG_W, FIG_H, DPI = 6.25, 3.1, 300
SPLIT = 0.42 * FIG_W  # boundary between panels A and B, inches

BLUE = "#2166AC"       # physicians
BAND = "#D1E5F0"       # physicians' 95% CI
DARK_RED = "#B2182B"   # frontier LLMs
LIGHT_RED = "#EF8A62"  # all 22 LLMs
REF_GREY = "#888888"
ZERO_GREY = "#D0D0D0"
MARKER_AREA = 14       # pt^2
OPEN_EDGE = 0.6
FILL_RING = 0.4
LIM_A = (-105, 105)
JITTER, JITTER_SEED = 1.5, 7
XLIM_B = (-10, 100)
ROWS = [("All families", "All families"),
        ("Intended diagnosis led in both versions", "Intended diagnosis led in both"),
        ("More than half selected it in both", "Majority chose intended in both"),
        ("At least two thirds selected it in both", "Two-thirds chose intended in both"),
        ("At least three quarters selected it in both", "Three-quarters chose intended in both"),
        ("All selected it in both", "All chose intended in both")]


def pick_font():
    for name in ("Arial", "Helvetica"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "DejaVu Sans"


class PhysicianKey:
    """Legend placeholder for the physician line and its 95% CI band."""


class BandLineHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x0, y0 = -xdescent, -ydescent
        lo, hi = y0 - 0.3 * height, y0 + 1.3 * height
        band = Rectangle((x0, lo), width, hi - lo, facecolor=BAND, edgecolor="none", transform=trans)
        line = Line2D([x0 + width / 2] * 2, [lo, hi], color=BLUE, lw=1.0, transform=trans)
        return [band, line]


fam = pd.read_csv(TABLES / "table_s17.csv")
x = (fam.cue_associated_first_choice_cue_pct - fam.cue_associated_first_choice_no_cue_pct).to_numpy(float)
y = fam.frontier_llms_cue_effect_pp.to_numpy(float)
lead = ((fam.leading_diagnosis_no_cue == "intended") & (fam.leading_diagnosis_cue == "intended")).to_numpy()
sub = pd.read_csv(TABLES / "table_s16.csv").set_index("subset").loc[[k for k, _ in ROWS]]
est = pd.read_csv(ESTIMATES / "05_physician_validation.csv").set_index("estimate")
phys, phys_lo, phys_hi = est.loc["physician_cue_effect_selection_pp", ["value", "ci_low", "ci_high"]].astype(float)

jit = np.random.default_rng(JITTER_SEED).uniform(-JITTER, JITTER, size=(len(x), 2))
xp, yp = x + jit[:, 0], y + jit[:, 1]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [pick_font()], "font.size": 7, "axes.unicode_minus": False,
    "axes.linewidth": 0.6, "axes.labelsize": 8, "axes.labelpad": 2, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "legend.fontsize": 7, "figure.facecolor": "white",
    "savefig.facecolor": "white"})

row_labels = [f"{short}\n({sub.loc[k, 'frontier_families']:d}; {sub.loc[k, 'all_llms_families']:d})" for k, short in ROWS]
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
renderer = fig.canvas.get_renderer()


def text_width_in(s, size):
    t = fig.text(0, 0, s, fontsize=size)
    w = t.get_window_extent(renderer).width / DPI
    t.remove()
    return w


# Layout in inches: panel A is square; panel B shares its top and bottom edges
top = FIG_H - 0.36
a_left = 0.58
side = SPLIT - 0.07 - a_left
bottom = top - side
b_left = SPLIT + 0.05 + max(text_width_in(s, 7) for s in row_labels) + 3 / 72
b_right = FIG_W - 0.12
ax_a = fig.add_axes((a_left / FIG_W, bottom / FIG_H, side / FIG_W, side / FIG_H))
ax_b = fig.add_axes((b_left / FIG_W, bottom / FIG_H, (b_right - b_left) / FIG_W, side / FIG_H))

# Panel A
ax_a.axhline(0, color=ZERO_GREY, lw=0.5, zorder=0)
ax_a.axvline(0, color=ZERO_GREY, lw=0.5, zorder=0)
ax_a.plot(LIM_A, LIM_A, color=REF_GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
d_open = np.sqrt(MARKER_AREA) - OPEN_EDGE
d_fill = np.sqrt(MARKER_AREA) + FILL_RING
ax_a.scatter(xp[~lead], yp[~lead], s=d_open ** 2, facecolors="white", edgecolors=DARK_RED, linewidths=OPEN_EDGE,
             alpha=0.85, zorder=2)
ax_a.scatter(xp[lead], yp[lead], s=d_fill ** 2, facecolors=DARK_RED, edgecolors="white", linewidths=FILL_RING,
             alpha=0.85, zorder=3)
ax_a.set_xlim(LIM_A)
ax_a.set_ylim(LIM_A)
ax_a.set_aspect("equal", adjustable="box")
ax_a.set_xticks([-100, -50, 0, 50, 100])
ax_a.set_yticks([-100, -50, 0, 50, 100])
ax_a.set_xlabel("Clinical adjudicators:\ncue effect (percentage points)")
ax_a.set_ylabel("Frontier LLMs:\ncue effect (percentage points)")
for s in ("top", "right"):
    ax_a.spines[s].set_visible(False)
ax_a.legend(
    [Line2D([], [], ls="none", marker="o", ms=d_fill, mfc=DARK_RED, mec="white", mew=FILL_RING, alpha=0.85),
     Line2D([], [], ls="none", marker="o", ms=d_open, mfc="white", mec=DARK_RED, mew=OPEN_EDGE, alpha=0.85)],
    ["Intended diagnosis led\nin both versions", "Other families"], loc="lower right", frameon=True,
    facecolor="white", edgecolor="none", framealpha=1, fancybox=False, borderpad=0.3, borderaxespad=0.2,
    handlelength=1.0, handletextpad=0.3, labelspacing=0.35)

# Panel B
rows = np.arange(len(ROWS))
offset = 0.17  # frontier above the row centre, all 22 LLMs below
ax_b.axvspan(phys_lo, phys_hi, facecolor=BAND, edgecolor="none", zorder=0)
ax_b.axvline(0, color=REF_GREY, lw=0.6, zorder=1)
ax_b.axvline(phys, color=BLUE, lw=1.0, zorder=2)
for prefix, dy, color, marker, size in (("frontier", -offset, DARK_RED, "o", 4.0),
                                        ("all_llms", offset, LIGHT_RED, "s", 3.4)):
    yy = rows + dy
    ax_b.hlines(yy, sub[f"{prefix}_ci_low"], sub[f"{prefix}_ci_high"], color=color, lw=1.0, zorder=3)
    ax_b.plot(sub[f"{prefix}_cue_effect"], yy, ls="none", marker=marker, ms=size, mfc=color, mec=color, zorder=4)
ax_b.set_ylim(len(ROWS) - 0.4, -0.6)
ax_b.set_yticks(rows, labels=row_labels)
ax_b.tick_params(axis="y", length=0, pad=3)
for t in ax_b.get_yticklabels():
    t.set_verticalalignment("center")
    t.set_linespacing(1.1)
ax_b.set_xlim(XLIM_B)
ax_b.set_xticks([0, 20, 40, 60, 80, 100])
ax_b.set_xlabel("LLM cue effect (percentage points)")
for s in ("top", "right", "left"):
    ax_b.spines[s].set_visible(False)

# Legend below panel B's axis label, right-aligned with the axis
fig.canvas.draw()
label_bottom = ax_b.xaxis.label.get_window_extent(fig.canvas.get_renderer()).y0 / DPI
fig.legend([Line2D([], [], color=DARK_RED, lw=1.0, marker="o", ms=4.0),
            Line2D([], [], color=LIGHT_RED, lw=1.0, marker="s", ms=3.4), PhysicianKey()],
           ["Frontier LLMs (100 families)", "All 22 LLMs (100 families)", "Clinical adjudicators, all families"],
           handler_map={PhysicianKey: BandLineHandler()}, ncol=2, loc="upper right",
           bbox_to_anchor=(b_right / FIG_W, (label_bottom - 0.06) / FIG_H), frameon=False, borderpad=0,
           borderaxespad=0, handlelength=1.8, handletextpad=0.5, columnspacing=1.2, labelspacing=0.4)

for letter, x_in in (("A", 0.03), ("B", SPLIT + 0.02)):
    fig.text(x_in / FIG_W, (FIG_H - 0.15) / FIG_H, letter, fontsize=10, fontweight="bold", ha="left", va="top")

fig.savefig(FIGURES / "figure_3.png", dpi=DPI, facecolor="white")
plt.close(fig)
print("wrote figure_3.png")
