"""Explanation-text screen of the explanation run (Table S14, Figure S4).

Five models answered every case in both versions under the three prompts and
added a one-sentence explanation and a 0-100 confidence. Each output is parsed
and its explanation is screened with fixed keyword and regular-expression
patterns for six markers:

- cue mentioned: the explanation uses a word that appears in the cue version of
  the vignette but not in the no-cue version
- base-rate language, risk language, uncertainty language: keyword patterns
- cue-associated justification: the model chose the cue-associated diagnosis
  and its explanation mentions the cue
- high-confidence wrong answer: the model chose a diagnosis other than the
  intended one with confidence of 80 or more

Outputs: table_s14.csv, table_s19.csv (two markers by model), figure_s4.png and
estimates/07_explanation_screen.csv (the Figure S4 rates with 95% Wilson intervals).
"""
import re
import string
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from common import DATA, ESTIMATES, FIGURES, PROMPT_LABEL, PROMPTS, ROOT, TABLES, VERSIONS, explanation_responses, wilson

sys.path.insert(0, str(ROOT / "code" / "collect"))
from parse_output import parse_response, selected_role  # noqa: E402

HIGH_CONFIDENCE = 80

# Words ignored when finding the words the cue adds to a vignette.
STOPWORDS = {
    "a", "about", "after", "and", "are", "day", "days", "during", "for", "from", "had", "has", "have",
    "her", "his", "man", "old", "one", "patient", "presents", "recent", "recently", "reports", "she",
    "symptoms", "the", "to", "week", "weeks", "with", "woman", "year", "years",
}

BASE_RATE_PATTERNS = [
    r"\bbase[- ]?rate", r"\bcommon\b", r"\bcommoner\b", r"\bmore common\b", r"\bmost common\b",
    r"\bprevalen", r"\btypical\b", r"\bmost likely\b", r"\bmost probable\b", r"\blikely\b",
    r"\bprobable\b", r"\bconsistent with\b", r"\bshort duration\b", r"\bself[- ]limited\b",
    r"\btime course\b",
]

RISK_PATTERNS = [
    r"\bserious\b", r"\bsevere\b", r"\bfatal\b", r"\bdanger", r"\blife[- ]threat", r"\bmust not miss\b",
    r"\bdo not miss\b", r"\brule out\b", r"\bred flag", r"\bhigh[- ]consequence\b", r"\bpotentially\b",
    r"\bcomplication", r"\bemergency\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bpossible\b", r"\bpossibly\b", r"\bcould\b", r"\bmay\b", r"\bmight\b", r"\bplausible\b",
    r"\bconcern for\b", r"\bcannot exclude\b", r"\bcan present\b",
]

MARKERS = [
    "cue_mentioned",
    "cue_associated_justification",
    "base_rate_language",
    "risk_language",
    "uncertainty_language",
    "high_confidence_wrong",
]
TABLE_MARKERS = [m for m in MARKERS if m != "uncertainty_language"]  # the columns of Table S14


def normalize(text):
    """Lower case, punctuation removed, single spaces."""
    return " ".join(str(text).lower().translate(str.maketrans("", "", string.punctuation)).split())


def word_tokens(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower()))


def has_pattern(text, patterns):
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)


def cue_words(vignettes):
    """For each family, the words (4+ letters, not stopwords) in the cue vignette but not the no-cue vignette."""
    words = {}
    for family, v in vignettes.groupby("family"):
        text = dict(zip(v.version, v.vignette))
        added = word_tokens(text.get("cue", "")) - word_tokens(text.get("no_cue", ""))
        words[family] = {t for t in added if len(t) >= 4 and t not in STOPWORDS and not t.isdigit()}
    return words


def mentions_cue(explanation, words):
    padded = f" {normalize(explanation)} "
    return any(re.search(rf"\b{re.escape(w)}\b", padded) for w in words)


def response_markers():
    """One row per explanation-run output with the parsed answer and the six markers."""
    d = explanation_responses()
    items = pd.read_csv(DATA / "items.csv", keep_default_na=False)
    keys = ["family", "version", "answer_order"]
    d = d.merge(items[keys + ["vignette", "intended_option", "cue_associated_option"]], on=keys, how="left")
    words = cue_words(d.drop_duplicates(["family", "version"])[["family", "version", "vignette"]])

    rows = []
    for r in d.itertuples(index=False):
        parsed = parse_response(r.raw_output)
        explanation = " ".join(str(parsed.get("short_explanation", "")).strip().split())
        confidence = parsed.get("confidence")
        role = selected_role(parsed["selected_option"], r.intended_option, r.cue_associated_option)
        cue_seen = mentions_cue(explanation, words[r.family])
        wrong = role in ("cue_associated", "other")
        rows.append({
            "model_id": r.model_id,
            "prompt": r.prompt,
            "family": r.family,
            "version": r.version,
            "answer_order": r.answer_order,
            "selected_option": parsed["selected_option"],
            "selected_role": role,
            "confidence": confidence,
            "explanation": explanation,
            "cue_words": ";".join(sorted(words[r.family])),
            "cue_mentioned": cue_seen,
            "cue_associated_justification": role == "cue_associated" and cue_seen,
            "base_rate_language": has_pattern(explanation, BASE_RATE_PATTERNS),
            "risk_language": has_pattern(explanation, RISK_PATTERNS),
            "uncertainty_language": has_pattern(explanation, UNCERTAINTY_PATTERNS),
            "high_confidence_wrong": wrong and isinstance(confidence, int) and confidence >= HIGH_CONFIDENCE,
        })
    return pd.DataFrame(rows)


def table_s14(markers):
    rows = []
    for prompt in PROMPTS:
        for version in VERSIONS:
            g = markers[(markers.prompt == prompt) & (markers.version == version)]
            row = {"prompt": prompt, "version": version, "n": len(g)}
            for m in TABLE_MARKERS:
                row[m] = round(100 * g[m].mean(), 1)
            rows.append(row)
    return pd.DataFrame(rows)


def table_s19(markers):
    """Cue-associated justification and high-confidence wrong answers by model, prompt and version."""
    rows = []
    for model in sorted(markers.model_id.unique()):
        for prompt in PROMPTS:
            row = {"model_id": model, "prompt": prompt}
            for version in VERSIONS:
                g = markers[(markers.model_id == model) & (markers.prompt == prompt) & (markers.version == version)]
                row[f"n_{version}"] = len(g)
                for m in ("cue_associated_justification", "high_confidence_wrong"):
                    row[f"{m}_{version}"] = round(100 * g[m].mean(), 1)
            rows.append(row)
    return pd.DataFrame(rows)


def figure_estimates(markers):
    """Rates with 95% Wilson intervals for the two Figure S4 markers."""
    rows = []
    for m in ["cue_associated_justification", "high_confidence_wrong"]:
        for prompt in PROMPTS:
            for version in VERSIONS:
                g = markers[(markers.prompt == prompt) & (markers.version == version)]
                p, lo, hi = wilson(int(g[m].sum()), len(g))
                rows.append({"estimate": f"{m}_{prompt}_{version}", "value": round(100 * p, 1),
                             "ci_low": round(100 * lo, 1), "ci_high": round(100 * hi, 1), "n": len(g)})
    return pd.DataFrame(rows)


LIGHT, DARK = "#f4a582", "#b2182b"


def figure_s4(markers):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
    })
    panels = [
        ("cue_associated_justification", "Justification of the cue-associated diagnosis (%)", "A"),
        ("high_confidence_wrong", "High-confidence wrong answers (%)", "B"),
    ]
    labels = [PROMPT_LABEL[p].replace(" prompt", "\nprompt") for p in PROMPTS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (marker, ylabel, letter) in zip(axes, panels):
        for gi, prompt in enumerate(PROMPTS):
            for version, color, offset in [("no_cue", LIGHT, -0.20), ("cue", DARK, 0.20)]:
                g = markers[(markers.prompt == prompt) & (markers.version == version)]
                p, lo, hi = (100 * v for v in wilson(int(g[marker].sum()), len(g)))
                x = gi + offset
                ax.bar(x, p, 0.38, color=color, edgecolor="white", linewidth=0.6, zorder=3)
                ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="#222222",
                            elinewidth=1.0, capsize=2.5, zorder=4)
                ax.annotate(f"{p:.1f}", (x, hi), textcoords="offset points", xytext=(0, 3),
                            ha="center", va="bottom", fontsize=8.5, zorder=5)
        ax.set_xticks(range(len(PROMPTS)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 60)
        ax.set_xlim(-0.6, len(PROMPTS) - 0.4)
        ax.yaxis.grid(True, color="#e6e6e6", zorder=0)
        ax.set_axisbelow(True)
        ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    handles = [Patch(facecolor=LIGHT, label="No cue"), Patch(facecolor=DARK, label="Cue")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES / "figure_s4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    markers = response_markers()
    table_s14(markers).to_csv(TABLES / "table_s14.csv", index=False)
    table_s19(markers).to_csv(TABLES / "table_s19.csv", index=False)
    figure_estimates(markers).to_csv(ESTIMATES / "07_explanation_screen.csv", index=False)
    figure_s4(markers)
    print("wrote table_s14.csv, table_s19.csv, figure_s4.png and estimates/07_explanation_screen.csv")


if __name__ == "__main__":
    main()
