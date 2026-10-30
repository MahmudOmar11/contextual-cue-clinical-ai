"""Second arm of the irrelevant-control experiment, the clinical adjudicators' instruction: Extended Data Table 1.

The three frontier LLMs answered the 1,200 items of the irrelevant-control experiment with the adjudicators'
instruction and answer format: 100 probability points across the four diagnoses and the single most probable one
(data/control_responses_adjudicator_instruction.csv.gz). Two outcomes: the first choice is the cue-associated
diagnosis (percentage points), and the probability given to it (points). Responses are averaged over the four answer
orders within each model, family and version; families are weighted equally and the models equally.

The comparison with the adjudicators follows the main adjudication analysis (adjudication.py): case families and
adjudicators are reweighted with the draws of the frontier-LLM panel, and the models are held fixed. The prespecified
analysis also holds the adjudicators fixed at their observed family means and resamples case families only (10,000
multinomial draws).

Writes results/tables/extended_data_table_1.csv and results/estimates/13_adjudicator_instruction_arm.csv.
"""
import numpy as np
import pandas as pd

from adjudication import B, FAMILIES, HUMAN, P_CUE, RATINGS, SEED, SEL, adjudicator_draws, ci, reweighting
from common import DATA, ESTIMATES, FRONTIER, TABLES

VERSIONS = ["no_cue", "cue", "control"]
GEMINI = "google/gemini-3.1-pro-preview"
OUTCOMES = {"first_choice": SEL, "probability": P_CUE}  # column of the response file -> outcome index of the ratings

resp = pd.read_csv(DATA / "control_responses_adjudicator_instruction.csv.gz", keep_default_na=False)
assert len(resp) == 3600 and sorted(resp.model_id.unique()) == sorted(FRONTIER)
assert (resp[["p_A", "p_B", "p_C", "p_D"]].sum(axis=1) == 100).all()
resp["first_choice"] = 100.0 * (resp.selected_role == "cue_associated")
resp["probability"] = resp.p_cue_associated.astype(float)
cells = resp.groupby(["model_id", "family", "version"])[list(OUTCOMES)].mean()
MODELS = sorted(resp.model_id.unique())


def model_cells(outcome, models):
    """Families x versions, averaged over the models."""
    return np.array([[[cells.loc[(m, f, v), outcome] for v in VERSIONS] for f in FAMILIES] for m in models]).mean(0)


def r1(x):
    return round(float(x), 1)


# ---------------------------------------------------------------- main analysis: families and adjudicators reweighted
fw, _, aw = reweighting(len(MODELS))  # LLM weights are drawn but not used: the models are held fixed
fw = fw / fw.sum(1, keepdims=True)
hboot = adjudicator_draws(aw)
paper = {}
for outcome, k in OUTCOMES.items():
    scale = 100 if outcome == "first_choice" else 1  # first choice is a share of adjudicators; probability in points
    h_eff = (HUMAN[:, 1, k] - HUMAN[:, 0, k]) * scale
    h_draws = np.einsum("bf,bf->b", fw, (hboot[:, :, 1, k] - hboot[:, :, 0, k]) * scale)
    for scope, models in [("all", MODELS), ("without_gemini", [m for m in MODELS if m != GEMINI])]:
        mc = model_cells(outcome, models)
        m_eff, m_ctl = mc[:, 1] - mc[:, 0], mc[:, 2] - mc[:, 0]
        m_draws = np.einsum("bf,f->b", fw, m_eff)
        paper[outcome, scope] = dict(
            llms=(m_eff.mean(), *ci(m_draws)), adjudicators=(h_eff.mean(), *ci(h_draws)),
            extra=(m_eff.mean() - h_eff.mean(), *ci(m_draws - h_draws)),
            irrelevant=(m_ctl.mean(), *ci(np.einsum("bf,f->b", fw, m_ctl))))

# ---------------------------------------------------------------- prespecified: case families resampled only
rng = np.random.default_rng(SEED)
rng.multinomial(len(MODELS), np.full(len(MODELS), 1 / len(MODELS)), size=B)  # model weights: drawn, then held fixed
fm = rng.multinomial(len(FAMILIES), np.full(len(FAMILIES), 1 / len(FAMILIES)), size=B) / len(FAMILIES)
ratings = RATINGS.assign(first_choice=RATINGS.top_choice == "cue_associated")
by_version = ratings.groupby(["family", "version"]).agg(n=("first_choice", "size"), votes=("first_choice", "sum"),
                                                         probability=("p_cue_associated", "mean"))
adjudicator_mean = {"first_choice": 100.0 * by_version.votes / by_version.n, "probability": by_version.probability}
prespecified = {}
for outcome in OUTCOMES:
    v = adjudicator_mean[outcome].unstack("version").loc[FAMILIES]
    shift = (v.cue - v.no_cue).to_numpy()
    # per model and family: the model's cue effect minus the adjudicators' cue effect
    x = np.array([[cells.loc[(m, f, "cue"), outcome] - cells.loc[(m, f, "no_cue"), outcome] for f in FAMILIES]
                  for m in MODELS]) - shift
    point = float(np.mean(x.sum(axis=1) / len(FAMILIES)))
    per_model = (x @ fm.T) / (np.ones_like(x) @ fm.T)  # family-weighted mean per model under each draw
    draws = (per_model * (1 / len(MODELS))).sum(axis=0)  # the models held fixed, weighted equally
    prespecified[outcome] = (point, *ci(draws))

# ---------------------------------------------------------------- Extended Data Table 1
rows = []


def add(measure, llms, adjudicators=None, extra=None, prespecified_extra=None):
    row = dict(measure=measure)
    for name, values in [("frontier_llms", llms), ("clinical_adjudicators", adjudicators),
                         ("extra_llm_cue_effect", extra), ("prespecified_extra_llm_cue_effect", prespecified_extra)]:
        v = (np.nan, np.nan, np.nan) if values is None else values
        row.update({name: r1(v[0]), f"{name}_ci_low": r1(v[1]), f"{name}_ci_high": r1(v[2])})
    rows.append(row)


for label, outcome, scope in [("First choice, percentage points", "first_choice", "all"),
                              ("Probability, points", "probability", "all"),
                              ("First choice, without Gemini 3.1 Pro", "first_choice", "without_gemini"),
                              ("Probability, without Gemini 3.1 Pro", "probability", "without_gemini")]:
    r = paper[outcome, scope]
    add(label, r["llms"], r["adjudicators"], r["extra"], prespecified[outcome] if scope == "all" else None)
add("Irrelevant detail, first choice", paper["first_choice", "all"]["irrelevant"])
add("Irrelevant detail, probability", paper["probability", "all"]["irrelevant"])
pd.DataFrame(rows).to_csv(TABLES / "extended_data_table_1.csv", index=False)

valid = int((resp.selected_role != "no_valid_answer").sum())
est = pd.DataFrame([dict(estimate="responses_with_valid_answer", value=valid, ci_low=np.nan, ci_high=np.nan,
                         n=len(resp))])
est.to_csv(ESTIMATES / "13_adjudicator_instruction_arm.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
