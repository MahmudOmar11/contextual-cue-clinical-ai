"""Irrelevant-control experiment: Tables S20 and S21 and the numbers reported in the text.

Every case family was presented in three versions: no cue, cue, and a control version in which a clinically
irrelevant detail, matched to the cue in position and form, takes the place of the cue. The three frontier LLMs
answered all 1,200 items once with the baseline prompt (data/control_responses.csv.gz).

Responses are averaged over the four answer orders within each model, family and version; families are weighted
equally within each model and the models equally. Intervals come from 10,000 bootstrap resamples of case families
(multinomial weights) with the three models held fixed, percentile 2.5 and 97.5, paired across contrasts. Each scope
(all families, one model, a subset of families) draws its weights afresh from the same seed.

Writes results/tables/table_s20.csv, table_s21.csv and results/estimates/09_control_experiment.csv.
"""
import difflib
import re

import numpy as np
import pandas as pd

from common import DATA, ESTIMATES, FRONTIER, TABLES, cases, form_families, physician_ratings

B = 10000
SEED = 20260928
VERSIONS = ["no_cue", "cue", "control"]
CONTRASTS = {"cue_minus_no_cue": ("cue", "no_cue"), "control_minus_no_cue": ("control", "no_cue"),
             "cue_minus_control": ("cue", "control")}
MODEL_NAME = {"anthropic/claude-opus-4.8": "Claude Opus 4.8", "openai/gpt-5.5": "GPT-5.5",
              "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro"}
CUE_TYPES = ["travel or residence", "occupation or exposure", "family history", "lifestyle", "other"]

items = pd.read_csv(DATA / "control_items.csv")
resp = pd.read_csv(DATA / "control_responses.csv.gz", keep_default_na=False)
assert len(items) == 1200 and len(resp) == 3600 and sorted(resp.model_id.unique()) == sorted(FRONTIER)
resp["cue_associated"] = 100.0 * (resp.selected_role == "cue_associated")
resp["intended"] = 100.0 * (resp.selected_role == "intended")
CELLS = resp.groupby(["model_id", "family", "version"], as_index=False)[["cue_associated", "intended"]].mean()
assert len(CELLS) == 900

# Adjudicators' cue effect per family (share selecting the cue-associated diagnosis first), for the plan's comparison
R = physician_ratings()
share = (R.assign(first=100.0 * (R.top_choice == "cue_associated")).groupby(["family", "version"])["first"].mean().unstack())
ADJ_SHIFT = share["cue"] - share["no_cue"]


def scope(cells, extra=None):
    """Point estimates and intervals for every level and contrast of both outcomes in one scope."""
    models, families = sorted(cells.model_id.unique()), sorted(cells.family.unique())
    rng = np.random.default_rng(SEED)
    rng.multinomial(len(models), np.full(len(models), 1 / len(models)), size=B)  # model weights: drawn, then held fixed
    fw = rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=B) / len(families)
    wide = cells.pivot_table(index=["model_id", "family"], columns="version", values=["cue_associated", "intended"])
    series = {}
    for outcome in ("cue_associated", "intended"):
        for v in VERSIONS:
            series[outcome, f"level_{v}"] = wide[(outcome, v)]
        for name, (a, b) in CONTRASTS.items():
            series[outcome, name] = wide[(outcome, a)] - wide[(outcome, b)]
    for name, values in (extra or {}).items():
        series["cue_associated", name] = values(wide)
    out = {}
    for key, s in series.items():
        m = s.unstack("family").reindex(index=models, columns=families).to_numpy(float)
        point = float(np.mean(m.mean(axis=1)))
        draws = (m @ fw.T).mean(axis=0)  # per-model family means under each weight vector, then equal model weights
        lo, hi = np.quantile(draws, [0.025, 0.975])
        out[key] = (point, float(lo), float(hi), len(families))
    return out


def tokens(text):
    return re.findall(r"\s+|[\w'’-]+|[^\w\s]", text)


def detail(base, variant):
    """The words a variant adds to or changes in the no-cue vignette; separate parts joined by semicolons."""
    a, b = tokens(base), tokens(variant)
    spans = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op in ("insert", "replace") and "".join(b[j1:j2]).strip(" ,;."):
            if spans and len([t for t in b[spans[-1][1]:j1] if t.strip()]) <= 2:  # nearby changes read as one phrase
                spans[-1][1] = j2
            else:
                spans.append([j1, j2])
    return "; ".join("".join(b[j1:j2]).strip(" ,;.") for j1, j2 in spans)


def r1(x):
    return round(round(x, 9), 1)  # floating-point noise removed first, so exact ties such as 0.75 round consistently


ctype = cases().set_index("family").cue_type
adj = {"model_cue_effect_minus_adjudicators_cue_effect":
       lambda w: (w[("cue_associated", "cue")] - w[("cue_associated", "no_cue")]) - w.index.get_level_values("family").map(ADJ_SHIFT).to_numpy()}
SCOPES = [("Selection of the cue-associated diagnosis", "Frontier LLMs", CELLS, adj)]
SCOPES += [("Selection of the cue-associated diagnosis", MODEL_NAME[m], CELLS[CELLS.model_id == m], None) for m in FRONTIER]
SCOPES += [("Selection of the cue-associated diagnosis", "Without Gemini 3.1 Pro",
            CELLS[CELLS.model_id != "google/gemini-3.1-pro-preview"], None),
           ("Selection of the cue-associated diagnosis", "21 clinician-form families",
            CELLS[CELLS.family.isin(form_families())], None)]
SCOPES += [("By cue type (exploratory)", t.capitalize(), CELLS[CELLS.family.map(ctype) == t], None) for t in CUE_TYPES]

est, rows20, results = [], [], {}
for group, label, cells, extra in SCOPES:
    results[label] = scope(cells, extra)
for group, label, _, _ in SCOPES + [("Accuracy (selection of the intended diagnosis)", "Frontier LLMs", None, None)]:
    outcome = "intended" if group.startswith("Accuracy") else "cue_associated"
    res = results[label]
    row = dict(group=group, scope=label, families=res[outcome, "cue_minus_no_cue"][3])
    for v in VERSIONS:
        row[f"{v}_pct"] = r1(res[outcome, f"level_{v}"][0])
    for name in CONTRASTS:
        point, lo, hi, _ = res[outcome, name]
        row.update({f"{name}_pp": r1(point), f"{name}_ci_low": r1(lo), f"{name}_ci_high": r1(hi)})
    rows20.append(row)
    key = "accuracy" if outcome == "intended" else "selection"
    for name in list(CONTRASTS) + [f"level_{v}" for v in VERSIONS]:
        point, lo, hi, n = res[outcome, name]
        est.append(dict(estimate=f"{label.lower().replace(' ', '_')}_{key}_{name}", value=r1(point),
                        ci_low=r1(lo), ci_high=r1(hi), n=n))
pd.DataFrame(rows20).to_csv(TABLES / "table_s20.csv", index=False)
point, lo, hi, n = results["Frontier LLMs"]["cue_associated", "model_cue_effect_minus_adjudicators_cue_effect"]
est.append(dict(estimate="cue_effect_minus_adjudicators_cue_effect_adjudicators_held_fixed_pp", value=r1(point),
                ci_low=r1(lo), ci_high=r1(hi), n=n))

# ---------------------------------------------------------------- Table S21: each family
text = items[items.answer_order == 1].pivot(index="family", columns="version", values="vignette")
fam = CELLS.pivot_table(index="family", columns="version", values="cue_associated", aggfunc="mean")
s21 = pd.DataFrame({"family": fam.index, "cue_type": [ctype[f].capitalize() for f in fam.index],
                    "cue_detail": [detail(text.loc[f, "no_cue"], text.loc[f, "cue"]) for f in fam.index],
                    "control_detail": [detail(text.loc[f, "no_cue"], text.loc[f, "control"]) for f in fam.index],
                    "no_cue_pct": fam.no_cue.round(1).to_numpy(), "cue_pct": fam.cue.round(1).to_numpy(),
                    "control_pct": fam.control.round(1).to_numpy()})
s21.to_csv(TABLES / "table_s21.csv", index=False)

effect = fam.control - fam.no_cue
for name, value in [("families_control_no_net_effect", int((effect == 0).sum())),
                    ("families_control_effect_within_10pp", int((effect.abs() <= 10).sum())),
                    ("families_control_effect_toward_at_least_25pp", int((effect >= 25).sum())),
                    ("families_control_effect_away_at_least_25pp", int((effect <= -25).sum())),
                    ("responses_with_valid_answer", int((resp.selected_role != "no_valid_answer").sum()))]:
    est.append(dict(estimate=name, value=value, ci_low=np.nan, ci_high=np.nan, n=len(resp) if "responses" in name else 100))
out = pd.DataFrame(est, dtype=object)
out.to_csv(ESTIMATES / "09_control_experiment.csv", index=False)
print(pd.DataFrame(rows20).to_string(index=False))
print("wrote table_s20.csv, table_s21.csv and estimates/09_control_experiment.csv")
