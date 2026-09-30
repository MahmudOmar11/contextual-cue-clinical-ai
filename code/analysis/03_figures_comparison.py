"""Figures for the clinician-versus-LLM comparison: Figures 1 and 2 and Supplementary Figures S1 to S3.

Clinician participants are blue and LLMs red; light fill is no cue and dark fill is cue. Error bars are
95% Wilson intervals.
Run from the repository root: python3 code/analysis/03_figures_comparison.py
"""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import FIGURES, ROOT, cases, clinician_responses, form_families, model_responses, wilson

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
})
DPI = 300
BLUE_L, BLUE_D = "#a6cee3", "#1f78b4"
RED_L, RED_D = "#f4a582", "#b2182b"
SELECTION = "Selection of the cue-associated diagnosis"

clin = clinician_responses()
models = model_responses()
families = form_families()
cue_dx = cases().set_index("family").cue_associated_diagnosis
baseline = models[models.prompt == "baseline"]
baseline_matched = baseline[baseline.family.isin(families)]
matched_models = models[models.family.isin(families)]
repeated = cases().query("repeated_on_clinician_form == 1").family.tolist()
with open(ROOT / "config" / "models.json") as f:
    MODEL_NAME = {m["openrouter_id"]: m["name"] for m in json.load(f)["models"]}


def rate(d, version, outcome):
    """Percentage with its 95% Wilson interval: (value, low, high, n)."""
    s = d.loc[d.version == version, outcome]
    p, lo, hi = wilson(int(s.sum()), len(s))
    return 100 * p, 100 * min(max(lo, 0.0), p), 100 * max(min(hi, 1.0), p), len(s)


def grouped(ax, labels, groups, outcome, ylabel, panel, refline=None):
    """Paired no-cue and cue bars per group; groups are (data, light colour, dark colour)."""
    width = 0.38
    for i, (d, light, dark) in enumerate(groups):
        for version, colour, offset in [("no_cue", light, -0.20), ("cue", dark, 0.20)]:
            val, lo, hi, _ = rate(d, version, outcome)
            x = i + offset
            ax.bar(x, val, width, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
            ax.errorbar(x, val, yerr=[[val - lo], [hi - val]], fmt="none", ecolor="#222222", elinewidth=1.0,
                        capsize=2.5, zorder=4)
            ax.annotate(f"{val:.1f}", (x, hi), textcoords="offset points", xytext=(0, 3), ha="center",
                        va="bottom", fontsize=8.5, zorder=5)
    if refline is not None:
        ax.axhline(refline, ls="--", lw=1.2, color=BLUE_D, zorder=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.yaxis.grid(True, color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.04, panel, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def figure_1():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5))
    labels = ["Clinician\nparticipants", "LLMs\n(all cases)", "LLMs\n(matched cases)"]
    groups = [(clin, BLUE_L, BLUE_D), (baseline, RED_L, RED_D), (baseline_matched, RED_L, RED_D)]
    grouped(ax_a, labels, groups, "cue_associated", f"{SELECTION} (%)", "A")
    grouped(ax_b, labels, groups, "intended", "Accuracy (%)", "B")
    handles = [Patch(facecolor=BLUE_L, label="Clinician participants, no cue"),
               Patch(facecolor=BLUE_D, label="Clinician participants, cue"),
               Patch(facecolor=RED_L, label="LLMs, no cue"),
               Patch(facecolor=RED_D, label="LLMs, cue")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "figure_1.png")


# ---------------------------------------------------------------- Figure 2
def figure_2():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5))
    labels = ["Baseline", "Base-rate\nprompt", "Rare-or-serious\nprompt"]
    groups = [(matched_models[matched_models.prompt == p], RED_L, RED_D)
              for p in ["baseline", "base_rate", "rare_or_serious"]]
    grouped(ax_a, labels, groups, "cue_associated", f"{SELECTION} (%)", "A",
            refline=rate(clin, "cue", "cue_associated")[0])
    grouped(ax_b, labels, groups, "intended", "Accuracy (%)", "B", refline=rate(clin, "cue", "intended")[0])
    handles = [Patch(facecolor=RED_L, label="No cue"), Patch(facecolor=RED_D, label="Cue"),
               Line2D([0], [0], color=BLUE_D, ls="--", lw=1.2, label="Clinician participants (cue)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "figure_2.png")


def horizontal_bars(ax, ys, rates, colour, offset, errors=True):
    vals = [r[0] for r in rates]
    kw = {}
    if errors:
        kw = dict(xerr=[[r[0] - r[1] for r in rates], [r[2] - r[0] for r in rates]],
                  error_kw=dict(ecolor="#333333", elinewidth=0.8, capsize=1.8))
    ax.barh([y + offset for y in ys], vals, height=0.38, color=colour, edgecolor="white", linewidth=0.5,
            zorder=3, **kw)
    return vals


# ---------------------------------------------------------------- Figure S1 (per case family)
def figure_s1():
    rows = []
    for fam in families:
        c = rate(clin[clin.family == fam], "cue", "cue_associated")
        m = rate(baseline_matched[baseline_matched.family == fam], "cue", "cue_associated")
        rows.append((c[0], fam, c, m))
    rows.sort(key=lambda r: r[0])  # by clinician rate; ties keep family order
    labels = [cue_dx[r[1]].strip() for r in rows]
    labels = [x if len(x) <= 30 else x[:29] + "…" for x in labels]
    ys = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8.2, 9.6))
    horizontal_bars(ax, ys, [r[3] for r in rows], RED_D, 0.19)
    horizontal_bars(ax, ys, [r[2] for r in rows], BLUE_D, -0.19)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel(f"{SELECTION} in cue cases (%)")
    ax.set_ylabel("Cue-associated diagnosis")
    ax.xaxis.grid(True, color="#e9e9e9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=BLUE_D, label="Clinician participants"),
                       Patch(facecolor=RED_D, label="Large language models (baseline)")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "figure_s1.png")


# ---------------------------------------------------------------- Figure S2 (per model)
def figure_s2():
    rows = []
    for model_id in sorted(baseline_matched.model_id.unique()):
        d = baseline_matched[baseline_matched.model_id == model_id]
        rows.append((MODEL_NAME[model_id], rate(d, "cue", "cue_associated"), rate(d, "no_cue", "cue_associated")))
    rows.sort(key=lambda r: r[1][0])  # ascending, so the highest rate is at the top; ties keep model order
    ys = list(range(len(rows)))
    clinician_cue = rate(clin, "cue", "cue_associated")[0]
    fig, ax = plt.subplots(figsize=(8.2, 9.6))
    cue = horizontal_bars(ax, ys, [r[1] for r in rows], RED_D, 0.19)
    horizontal_bars(ax, ys, [r[2] for r in rows], RED_L, -0.19, errors=False)
    for y, v in zip(ys, cue):
        ax.annotate(f"{v:.0f}", (v, y + 0.19), textcoords="offset points", xytext=(3, 0), va="center", ha="left",
                    fontsize=7.5)
    ax.axvline(clinician_cue, ls="--", lw=1.3, color=BLUE_D, zorder=4)
    ax.annotate(f"Clinician participants (cue) {clinician_cue:.1f}%", (clinician_cue, len(rows) - 0.4),
                textcoords="offset points", xytext=(5, 0), va="top", ha="left", fontsize=8.5, color=BLUE_D)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 108)
    ax.set_xlabel(f"{SELECTION} (%)")
    ax.set_ylabel("Model")
    ax.xaxis.grid(True, color="#e9e9e9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=RED_D, label="Cue cases"), Patch(facecolor=RED_L, label="No-cue cases"),
                       Patch(facecolor="none", edgecolor=BLUE_D, label="Clinician cue reference")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "figure_s2.png")


# ---------------------------------------------------------------- Figure S3 (sensitivity analyses)
def figure_s3():
    analyses = [
        ("Primary (matched 21 cases)", clin, baseline_matched),
        (f"Excluding repeated cases ({', '.join(map(str, repeated))})",
         clin[~clin.family.isin(repeated)], baseline_matched[~baseline_matched.family.isin(repeated)]),
        ("Excluding medical students", clin[clin.training_level != "medical_student"], baseline_matched),
        ("Frontier-3 comparator only", clin, baseline_matched[baseline_matched.panel == "frontier"]),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for i, (_, c_data, m_data) in enumerate(analyses):
        y = len(analyses) - 1 - i
        cv, cl, ch, _ = rate(c_data, "cue", "cue_associated")
        lv, ll, lh, _ = rate(m_data, "cue", "cue_associated")
        ax.errorbar(lv, y + 0.12, xerr=[[lv - ll], [lh - lv]], fmt="s", color=RED_D, ecolor=RED_D, elinewidth=1.4,
                    capsize=3, markersize=7, zorder=3)
        ax.errorbar(cv, y - 0.12, xerr=[[cv - cl], [ch - cv]], fmt="o", color=BLUE_D, ecolor=BLUE_D, elinewidth=1.4,
                    capsize=3, markersize=7, zorder=3)
        ax.annotate(f"+{lv - cv:.0f} points", (max(lh, ch) + 2, y), va="center", ha="left", fontsize=8,
                    color="#555555")
    ax.set_yticks([len(analyses) - 1 - i for i in range(len(analyses))])
    ax.set_yticklabels([a[0] for a in analyses], fontsize=9)
    ax.set_ylim(-0.6, len(analyses) - 0.4)
    ax.set_xlim(0, 108)
    ax.set_xlabel(f"{SELECTION} in cue cases (%)")
    ax.xaxis.grid(True, color="#e9e9e9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=BLUE_D, label="Clinician participants"),
                       Patch(facecolor=RED_D, label="Large language models (baseline)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "figure_s3.png")


figure_1()
figure_2()
with plt.rc_context({"font.size": 10}):
    figure_s1()
    figure_s2()
    figure_s3()
print("Wrote Figures 1, 2 and S1 to S3.")
