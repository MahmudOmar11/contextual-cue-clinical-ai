"""Figures for the clinician-versus-LLM comparison: Figures 1 and 2 and Extended Data Figures 1 to 3.

Clinician participants are blue and LLMs red; light fill is no cue and dark fill is cue. Bars are the pooled
percentage of responses, error bars 95% Wilson intervals, and the dots in Figures 1 and 2 the percentage of each
clinician participant or each unique LLM. Each figure is written as PNG and PDF, with its source data in
results/source_data/.
Run from the repository root: python3 code/analysis/03_figures_comparison.py
"""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import source_data
from common import FIGURES, ROOT, bold_font, cases, clinician_responses, form_families, model_responses, wilson

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
})
DPI = 300
PANEL_FONT = bold_font(plt.rcParams["font.sans-serif"])
BLUE_L, BLUE_D = "#a6cee3", "#1f78b4"
RED_L, RED_D = "#f4a582", "#b2182b"
DOT = "#333333"
DOT_SEED = 3
SELECTION = "Selection of the cue-associated diagnosis"
CHALLENGE = "selected challenge set"

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
VERSION_LABEL = {"no_cue": "No cue", "cue": "Cue"}


def rate(d, version, outcome):
    """Percentage with its 95% Wilson interval: (value, low, high, n)."""
    s = d.loc[d.version == version, outcome]
    p, lo, hi = wilson(int(s.sum()), len(s))
    return 100 * p, 100 * min(max(lo, 0.0), p), 100 * max(min(hi, 1.0), p), len(s)


def rater_rates(d, version, outcome):
    """Percentage for each clinician participant or each unique LLM (all its runs and answer orders)."""
    rater = "participant" if "participant" in d.columns else "model_id"
    g = d[d.version == version].groupby(rater)[outcome].agg(["sum", "size"]).reset_index()
    names = g[rater].map(MODEL_NAME) if rater == "model_id" else g[rater]
    return pd.DataFrame({"rater": names, "numerator": g["sum"].astype(int), "n": g["size"].astype(int),
                         "percentage": 100 * g["sum"] / g["size"]})


def bar_row(group, condition, d, version, outcome):
    val, lo, hi, n = rate(d, version, outcome)
    k = int(d.loc[d.version == version, outcome].sum())
    return {"element": "bar", "group": group, "condition": condition, "rater": "", "numerator": k, "n": n,
            "percentage": round(val, 2), "wilson_low": round(lo, 2), "wilson_high": round(hi, 2)}


def grouped(ax, labels, groups, outcome, ylabel, panel, refline=None, dots=False, rng=None):
    """Paired no-cue and cue bars per group; groups are (data, light colour, dark colour). Returns source rows."""
    width = 0.38
    bars, rows = [], []
    for i, (d, light, dark) in enumerate(groups):
        group = " ".join(labels[i].split())
        for version, colour, offset in [("no_cue", light, -0.20), ("cue", dark, 0.20)]:
            val, lo, hi, _ = rate(d, version, outcome)
            x = i + offset
            ax.bar(x, val, width, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
            top = hi
            if dots:
                r = rater_rates(d, version, outcome)
                xs = x + rng.uniform(-0.12, 0.12, len(r))
                ax.scatter(xs, r.percentage, s=10, facecolors="white", edgecolors=DOT, linewidths=0.6, alpha=0.85,
                           zorder=4)
                top = max(hi, r.percentage.max())
                rows += [{"element": "dot", "group": group, "condition": VERSION_LABEL[version], "rater": t.rater,
                          "numerator": t.numerator, "n": t.n, "percentage": round(t.percentage, 2),
                          "wilson_low": np.nan, "wilson_high": np.nan} for t in r.itertuples()]
            ax.errorbar(x, val, yerr=[[val - lo], [hi - val]], fmt="none", ecolor="#222222", elinewidth=1.0,
                        capsize=2.5, zorder=5)
            ax.annotate(f"{val:.1f}", (x, top), textcoords="offset points", xytext=(0, 3), ha="center",
                        va="bottom", fontsize=8.5, zorder=6)
            bars.append(bar_row(group, VERSION_LABEL[version], d, version, outcome))
    if refline is not None:
        ax.axhline(refline, ls="--", lw=1.2, color=BLUE_D, zorder=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 110 if dots else 100)
    if dots:
        ax.set_yticks(range(0, 101, 20))
        ax.spines["left"].set_bounds(0, 100)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.yaxis.grid(True, color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.04, panel, transform=ax.transAxes, fontsize=14, fontweight="bold", fontfamily=PANEL_FONT,
            va="bottom")
    return pd.DataFrame(bars + rows)


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=DPI, bbox_inches="tight")
    source_data.save_pdf(fig, FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


DOT_KEY = Line2D([0], [0], ls="none", marker="o", ms=4, mfc="white", mec=DOT, mew=0.6, alpha=0.85)


# ---------------------------------------------------------------- Figure 1
def figure_1():
    rng = np.random.default_rng(DOT_SEED)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5.2))
    labels = [f"Clinician participants\n({CHALLENGE})", "LLMs\n(all 100 cases)", f"LLMs\n({CHALLENGE})"]
    groups = [(clin, BLUE_L, BLUE_D), (baseline, RED_L, RED_D), (baseline_matched, RED_L, RED_D)]
    a = grouped(ax_a, labels, groups, "cue_associated", f"{SELECTION} (%)", "A", dots=True, rng=rng)
    b = grouped(ax_b, labels, groups, "intended", "Accuracy (%)", "B", dots=True, rng=rng)
    for ax in (ax_a, ax_b):
        ax.tick_params(axis="x", labelsize=9.5)
    handles = [Patch(facecolor=BLUE_L, label="Clinician participants, no cue"),
               Patch(facecolor=BLUE_D, label="Clinician participants, cue"),
               Patch(facecolor=RED_L, label="LLMs, no cue"),
               Patch(facecolor=RED_D, label="LLMs, cue"),
               DOT_KEY]
    names = [h.get_label() for h in handles[:4]] + ["Each participant or LLM"]
    fig.legend(handles=handles, labels=names, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02),
               fontsize=9.5, handletextpad=0.5, columnspacing=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "figure_1")
    source_data.write("source_data_figure_1.xlsx", {"Panel A": a, "Panel B": b})


# ---------------------------------------------------------------- Figure 2
def figure_2():
    rng = np.random.default_rng(DOT_SEED)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5.2))
    labels = ["Baseline", "Base-rate\nprompt", "Rare-or-serious\nprompt"]
    groups = [(matched_models[matched_models.prompt == p], RED_L, RED_D)
              for p in ["baseline", "base_rate", "rare_or_serious"]]
    ref_a, ref_b = rate(clin, "cue", "cue_associated")[0], rate(clin, "cue", "intended")[0]
    a = grouped(ax_a, labels, groups, "cue_associated", f"{SELECTION} (%)", "A", refline=ref_a, dots=True, rng=rng)
    b = grouped(ax_b, labels, groups, "intended", "Accuracy (%)", "B", refline=ref_b, dots=True, rng=rng)
    for ax in (ax_a, ax_b):
        ax.set_xlabel("Selected challenge set (21 case families)", fontsize=10)
    handles = [Patch(facecolor=RED_L, label="No cue"), Patch(facecolor=RED_D, label="Cue"), DOT_KEY,
               Line2D([0], [0], color=BLUE_D, ls="--", lw=1.2, label="Clinician participants (cue)")]
    names = ["No cue", "Cue", "Each LLM", "Clinician participants (cue)"]
    fig.legend(handles=handles, labels=names, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02),
               fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "figure_2")
    clin_cue = clin[clin.version == "cue"]
    for frame, outcome, ref in [(a, "cue_associated", ref_a), (b, "intended", ref_b)]:
        frame.loc[len(frame)] = {"element": "reference line", "group": "Clinician participants", "condition": "Cue",
                                 "rater": "", "numerator": int(clin_cue[outcome].sum()), "n": len(clin_cue),
                                 "percentage": round(ref, 2), "wilson_low": np.nan, "wilson_high": np.nan}
    source_data.write("source_data_figure_2.xlsx", {"Panel A": a, "Panel B": b})


def horizontal_bars(ax, ys, rates, colour, offset, errors=True):
    vals = [r[0] for r in rates]
    kw = {}
    if errors:
        kw = dict(xerr=[[r[0] - r[1] for r in rates], [r[2] - r[0] for r in rates]],
                  error_kw=dict(ecolor="#333333", elinewidth=0.8, capsize=1.8))
    ax.barh([y + offset for y in ys], vals, height=0.38, color=colour, edgecolor="white", linewidth=0.5,
            zorder=3, **kw)
    return vals


def rate_row(label, d, version, outcome, interval=True):
    val, lo, hi, n = rate(d, version, outcome)
    return {"numerator": int(d.loc[d.version == version, outcome].sum()), "n": n, "percentage": round(val, 2),
            "wilson_low": round(lo, 2) if interval else np.nan, "wilson_high": round(hi, 2) if interval else np.nan}


# ---------------------------------------------------------------- Extended Data Figure 1 (per case family)
def extended_data_figure_1():
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
    ax.set_title("Selected challenge set (21 case families)", fontsize=10, loc="left")
    ax.xaxis.grid(True, color="#e9e9e9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=BLUE_D, label="Clinician participants"),
                       Patch(facecolor=RED_D, label="Large language models (baseline)")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "extended_data_figure_1")
    out = []
    for position, (_, fam, _, _) in enumerate(reversed(rows), 1):
        for group, d in [("Clinician participants", clin), ("Large language models (baseline)", baseline_matched)]:
            out.append({"row_from_top": position, "family": fam, "cue_associated_diagnosis": cue_dx[fam].strip(),
                        "group": group, "condition": "Cue", **rate_row(group, d[d.family == fam], "cue", "cue_associated")})
    source_data.write("source_data_extended_data_figure_1.xlsx", {"Extended Data Fig. 1": pd.DataFrame(out)})


# ---------------------------------------------------------------- Extended Data Figure 2 (per model)
def extended_data_figure_2():
    rows = []
    for model_id in sorted(baseline_matched.model_id.unique()):
        d = baseline_matched[baseline_matched.model_id == model_id]
        rows.append((MODEL_NAME[model_id], rate(d, "cue", "cue_associated"), rate(d, "no_cue", "cue_associated"), d))
    rows.sort(key=lambda r: r[1][0])  # ascending, so the highest rate is at the top; ties keep model order
    ys = list(range(len(rows)))
    clinician_cue = rate(clin, "cue", "cue_associated")[0]
    fig, ax = plt.subplots(figsize=(8.2, 9.6))
    cue = horizontal_bars(ax, ys, [r[1] for r in rows], RED_D, 0.19)
    horizontal_bars(ax, ys, [r[2] for r in rows], RED_L, -0.19, errors=False)
    for y, v, r in zip(ys, cue, rows):  # the value sits just past the interval, clear of the error bar
        ax.annotate(f"{v:.0f}", (r[1][2], y + 0.19), textcoords="offset points", xytext=(3, 0), va="center",
                    ha="left", fontsize=7.5)
    ax.axvline(clinician_cue, ls="--", lw=1.3, color=BLUE_D, zorder=4)
    ax.set_ylim(-0.7, len(rows) + 0.2)  # room above the top bar for the reference label
    ax.annotate(f"Clinician participants (cue) {clinician_cue:.1f}%", (clinician_cue, len(rows) - 0.45),
                textcoords="offset points", xytext=(5, 0), va="bottom", ha="left", fontsize=8.5, color=BLUE_D)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 108)
    ax.set_xlabel(f"{SELECTION} (%)")
    ax.set_ylabel("Model")
    ax.set_title("Selected challenge set (21 case families)", fontsize=10, loc="left")
    ax.xaxis.grid(True, color="#e9e9e9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=RED_D, label="Cue cases"), Patch(facecolor=RED_L, label="No-cue cases"),
                       Patch(facecolor="none", edgecolor=BLUE_D, label="Clinician cue reference")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "extended_data_figure_2")
    out = []
    for position, (name, _, _, d) in enumerate(reversed(rows), 1):
        for version in ["cue", "no_cue"]:
            out.append({"row_from_top": position, "model": name, "condition": VERSION_LABEL[version],
                        **rate_row(name, d, version, "cue_associated", interval=version == "cue")})
    clin_cue = clin[clin.version == "cue"]
    out.append({"row_from_top": np.nan, "model": "Clinician participants (reference line)", "condition": "Cue",
                "numerator": int(clin_cue.cue_associated.sum()), "n": len(clin_cue), "percentage": round(clinician_cue, 2),
                "wilson_low": np.nan, "wilson_high": np.nan})
    source_data.write("source_data_extended_data_figure_2.xlsx", {"Extended Data Fig. 2": pd.DataFrame(out)})


# ---------------------------------------------------------------- Extended Data Figure 3 (sensitivity analyses)
def extended_data_figure_3():
    analyses = [
        ("Primary (matched 21 cases)", clin, baseline_matched),
        (f"Excluding repeated cases ({', '.join(map(str, repeated))})",
         clin[~clin.family.isin(repeated)], baseline_matched[~baseline_matched.family.isin(repeated)]),
        ("Excluding medical students", clin[clin.training_level != "medical_student"], baseline_matched),
        ("Frontier-3 comparator only", clin, baseline_matched[baseline_matched.panel == "frontier"]),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    out = []
    for i, (label, c_data, m_data) in enumerate(analyses):
        y = len(analyses) - 1 - i
        cv, cl, ch, _ = rate(c_data, "cue", "cue_associated")
        lv, ll, lh, _ = rate(m_data, "cue", "cue_associated")
        ax.errorbar(lv, y + 0.12, xerr=[[lv - ll], [lh - lv]], fmt="s", color=RED_D, ecolor=RED_D, elinewidth=1.4,
                    capsize=3, markersize=7, zorder=3)
        ax.errorbar(cv, y - 0.12, xerr=[[cv - cl], [ch - cv]], fmt="o", color=BLUE_D, ecolor=BLUE_D, elinewidth=1.4,
                    capsize=3, markersize=7, zorder=3)
        ax.annotate(f"+{lv - cv:.0f} points", (max(lh, ch) + 2, y), va="center", ha="left", fontsize=8,
                    color="#555555")
        for group, d in [("Clinician participants", c_data), ("Large language models (baseline)", m_data)]:
            out.append({"analysis": label, "group": group, "condition": "Cue",
                        **rate_row(group, d, "cue", "cue_associated"),
                        "difference_points": round(lv - cv, 2) if group.startswith("Large") else np.nan})
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
    save(fig, "extended_data_figure_3")
    source_data.write("source_data_extended_data_figure_3.xlsx", {"Extended Data Fig. 3": pd.DataFrame(out)})


figure_1()
figure_2()
with plt.rc_context({"font.size": 10}):
    extended_data_figure_1()
    extended_data_figure_2()
    extended_data_figure_3()
print("Wrote Figures 1 and 2 and Extended Data Figures 1 to 3, with their source data.")
