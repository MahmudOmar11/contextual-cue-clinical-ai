"""Comparisons restricted to identical case text: Table S23.

The core-panel baseline runs received an earlier wording of 25 families (data/items_earlier_wording.csv; the column
`wording` of data/model_responses.csv.gz); every other model run, the clinician participants and the clinical
adjudicators saw the final wording. Next to each main result, the table gives the same comparison on identical text:
clinician participants against the three frontier LLMs, and the clinical adjudicators against all 22 LLMs on the 75
families whose responses all received the final wording, and against the frontier LLMs (adjudication.py). The
comparisons with clinician participants are read from the output of 04_robustness_matched.R and
11_identical_text_matched.R, so run this script after them.

Writes results/tables/table_s23.csv.
"""
import numpy as np
import pandas as pd

from adjudication import FAMILIES, Panel, llm_cells, pct
from common import ESTIMATES, TABLES, form_families, model_responses

mr = model_responses()
earlier = sorted(mr.loc[mr.wording == "earlier", "family"].unique())
assert len(earlier) == 25 and set(mr.loc[mr.wording == "earlier", "panel"]) == {"core"}
assert set(mr.loc[mr.wording == "earlier", "prompt"]) == {"baseline"}
earlier_21 = len(set(earlier) & set(form_families()))
identical = ~np.isin(FAMILIES, earlier)
ALL = np.ones(len(FAMILIES), bool)

matched = pd.concat([pd.read_csv(ESTIMATES / "04_robustness_matched.csv"),
                     pd.read_csv(ESTIMATES / "11_identical_text_matched.csv")]).set_index("estimate")
LL = Panel(llm_cells(mr), "All 22 LLMs", fixed_llms=False)
FR = Panel(llm_cells(mr[mr.panel == "frontier"]), "Frontier LLMs", fixed_llms=True)

main_21 = f"Earlier wording for the Core-20 baseline runs in {earlier_21} of the 21 families"
main_100 = f"Earlier wording for the Core-20 baseline runs in {len(earlier)} of the 100 families"
rows = []


def add(comparison, case_text, families, value, lo, hi):
    rows.append(dict(comparison=comparison, case_text=case_text, families=families, extra_llm_cue_effect=value,
                     ci_low=lo, ci_high=hi))


def add_matched(comparison, case_text, estimate):
    r = matched.loc[estimate]
    add(comparison, case_text, int(r.n), r.value, r.ci_low, r.ci_high)


def add_adjudicated(comparison, case_text, panel, mask):
    e = panel.effect(mask)
    add(comparison, case_text, e["families"], pct(e["extra"]), pct(e["extra_ci"][0]), pct(e["extra_ci"][1]))


add_matched("Clinician participants vs all 22 LLMs, main result", main_21, "extra_llm_cue_effect_selection_crossed_pp")
add_matched("Clinician participants vs frontier LLMs", "Identical",
            "extra_llm_cue_effect_selection_crossed_frontier_llms_identical_text_pp")
add_matched("Clinician participants vs all 22 LLMs, accuracy, main result", main_21,
            "extra_llm_cue_effect_accuracy_crossed_pp")
add_matched("Clinician participants vs frontier LLMs, accuracy", "Identical",
            "extra_llm_cue_effect_accuracy_crossed_frontier_llms_identical_text_pp")
add_adjudicated("Clinical adjudicators vs all 22 LLMs, main result", main_100, LL, ALL)
add_adjudicated("Clinical adjudicators vs all 22 LLMs", f"Identical, the {identical.sum()} unedited families", LL,
                identical)
add_adjudicated("Clinical adjudicators vs frontier LLMs", "Identical", FR, ALL)

out = pd.DataFrame(rows)
out.to_csv(TABLES / "table_s23.csv", index=False)
print(out.to_string(index=False))
