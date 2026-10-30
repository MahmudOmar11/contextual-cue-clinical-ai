"""Analyses that count valid answers only, repeated with all attempts in the denominator: Table S24.

The LLM cue effects and prompt comparisons of 04_robustness_llm.py and the comparisons with the clinical adjudicators
(05_physician_validation.py, adjudication.py) use each LLM's valid answers only. Table S24 repeats them with every
response in the denominator, a response without a valid answer counting as neither diagnosis. The frontier LLMs gave
a valid answer in every response, so only results that include the core panel change. The last row sets the crossed
comparison with clinician participants on valid answers only against its main result, which counts every response;
it is read from the output of 04_robustness_matched.R, so run this script after it.

Writes results/tables/table_s24.csv.
"""
import importlib

import numpy as np
import pandas as pd

from adjudication import ADJUDICATORS, FAMILIES, Panel, llm_cells, pct
from common import ESTIMATES, TABLES, cases, model_responses

robustness = importlib.import_module("04_robustness_llm")  # a numbered script, imported for its functions

PROMPT_ROWS = {
    "llm_cue_effect_21_families_all_runs_pp": "LLM cue effect, 21 families, all runs",
    "llm_cue_effect_21_families_one_run_per_unique_llm_pp": "LLM cue effect, 21 families, one run per unique LLM",
    "llm_cue_effect_21_families_one_run_baseline_pp": "LLM cue effect, 21 families, one run, baseline prompt",
    "llm_cue_effect_21_families_one_run_base_rate_pp": "LLM cue effect, 21 families, one run, base-rate prompt",
    "llm_cue_effect_21_families_one_run_rare_or_serious_pp":
        "LLM cue effect, 21 families, one run, rare-or-serious prompt",
    "change_in_llm_cue_effect_21_families_base_rate_vs_baseline_pp":
        "Change from baseline, base-rate prompt, 21 families",
    "change_in_llm_cue_effect_21_families_rare_or_serious_vs_baseline_pp":
        "Change from baseline, rare-or-serious prompt, 21 families",
    "change_in_llm_cue_effect_100_families_base_rate_vs_baseline_pp":
        "Change from baseline, base-rate prompt, 100 families",
    "change_in_llm_cue_effect_100_families_rare_or_serious_vs_baseline_pp":
        "Change from baseline, rare-or-serious prompt, 100 families",
}
CUE_TYPES = ["travel or residence", "occupation or exposure", "family history", "lifestyle", "other"]
CUE_TYPE = cases().set_index("family").loc[FAMILIES, "cue_type"].to_numpy()
ALL = np.ones(len(FAMILIES), bool)
SIX = [c for c in ADJUDICATORS if c not in ("A", "B")]  # the six physicians (A and B were medical students)


def estimates(mr, all_attempts):
    """Label -> (value, ci_low, ci_high) under one denominator."""
    out = {PROMPT_ROWS[r["estimate"]]: (r["value"], r["ci_low"], r["ci_high"])
           for r in robustness.cue_effect_estimates(mr, all_attempts) if r["estimate"] in PROMPT_ROWS}
    cells = llm_cells(mr, all_attempts)
    LL = Panel(cells, "All 22 LLMs", fixed_llms=False)
    e = LL.effect(ALL)
    out["All 22 LLMs, cue effect, 100 families"] = (pct(e["llm"]), *map(pct, e["llm_ci"]))
    out["Extra LLM cue effect over the clinical adjudicators, all 22 LLMs"] = (pct(e["extra"]),
                                                                              *map(pct, e["extra_ci"]))
    for label, P in [("Same, against the six physician adjudicators", Panel(cells, "Six", fixed_llms=False, keep=SIX)),
                     ("Same, all 22 LLMs under the base-rate prompt",
                      Panel(cells, "Base-rate prompt", fixed_llms=False, prompt="base_rate"))]:
        e = P.effect(ALL)
        out[label] = (pct(e["extra"]), *map(pct, e["extra_ci"]))
    for t in CUE_TYPES:
        e = LL.effect(CUE_TYPE == t)
        out[f"Same, all 22 LLMs, {t} cues"] = (pct(e["extra"]), *map(pct, e["extra_ci"]))
    return out


mr = model_responses()
valid, attempts = estimates(mr, all_attempts=False), estimates(mr, all_attempts=True)
crossed = pd.read_csv(ESTIMATES / "04_robustness_matched.csv").set_index("estimate")
label = "Extra LLM cue effect over clinician participants, 21 families (main text: all attempts)"
cols = ["value", "ci_low", "ci_high"]
valid[label] = tuple(crossed.loc["extra_llm_cue_effect_selection_crossed_valid_answers_pp", cols])
attempts[label] = tuple(crossed.loc["extra_llm_cue_effect_selection_crossed_pp", cols])

out = pd.DataFrame([dict(analysis=k, valid_answers_only=v[0], valid_answers_only_ci_low=v[1],
                         valid_answers_only_ci_high=v[2], all_attempts=attempts[k][0],
                         all_attempts_ci_low=attempts[k][1], all_attempts_ci_high=attempts[k][2])
                    for k, v in valid.items()], dtype=object)
out.to_csv(TABLES / "table_s24.csv", index=False)
print(out.to_string(index=False))
