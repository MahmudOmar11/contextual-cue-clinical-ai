"""The main comparisons without the case families flagged in the case-pair audit: Table S22, lower panel.

A word-by-word audit of the no-cue and cue versions of all 100 families flagged 17 (Table S22, upper panel): in six the
cue version also removed or added another phrase, in nine the cue came with a second clinical element, and two had
typing errors only. None was on the clinician form. The extra LLM cue effect over the clinical adjudicators
(adjudication.py) is repeated for all 100 families, without the six, and without all 17. For the irrelevant-control
experiment (first arm, baseline prompt), the frontier LLMs' cue effect minus the effect of the irrelevant detail uses
the method of Table S20 (09_control_experiment.py): case families resampled with multinomial weights (10,000 draws),
the three models held fixed, and the weights drawn afresh from the same seed for each set of families.

Writes results/tables/table_s22.csv.
"""
import importlib

import numpy as np
import pandas as pd

from adjudication import FAMILIES, Panel, llm_cells, pct
from common import TABLES, model_responses

control = importlib.import_module("09_control_experiment")  # a numbered script, imported for its functions

# Families flagged in the case-pair audit (Table S22, upper panel)
CHANGES_BEYOND_CUE = [43, 46, 47, 54, 81, 89]
SECOND_ELEMENT = [15, 44, 55, 56, 64, 66, 69, 71, 84]
TYPING_ONLY = [36, 65]
FAMILY_SETS = {"all_families": [], "without_6_families": CHANGES_BEYOND_CUE,
               "without_17_families": sorted(CHANGES_BEYOND_CUE + SECOND_ELEMENT + TYPING_ONLY)}

mr = model_responses()
cells = {"all": llm_cells(mr), "frontier": llm_cells(mr[mr.panel == "frontier"])}
panels = [("All 22 LLMs", Panel(cells["all"], "All 22 LLMs", fixed_llms=False)),
          ("Frontier LLMs", Panel(cells["frontier"], "Frontier LLMs", fixed_llms=True)),
          ("All 22 LLMs, base-rate prompt", Panel(cells["all"], "All 22 LLMs", fixed_llms=False, prompt="base_rate"))]

rows = []
adjudicators = dict(comparison="Clinical adjudicators' cue effect")
for label, P in panels:
    row = dict(comparison=label)
    for key, drop in FAMILY_SETS.items():
        e = P.effect(~np.isin(FAMILIES, drop))
        lo, hi = e["extra_ci"]
        row.update({key: pct(e["extra"]), f"{key}_ci_low": pct(lo), f"{key}_ci_high": pct(hi)})
        if label == "All 22 LLMs":
            adjudicators[key] = pct(e["adjudicators"])
    rows.append(row)

# Irrelevant-control experiment, as in Table S20
row = dict(comparison="Irrelevant-control experiment, frontier LLMs: "
                      "cue effect minus the effect of the irrelevant detail")
for key, drop in FAMILY_SETS.items():
    result = control.scope(control.CELLS[~control.CELLS.family.isin(drop)])
    point, lo, hi, _ = result["cue_associated", "cue_minus_control"]
    row.update({key: control.r1(point), f"{key}_ci_low": control.r1(lo), f"{key}_ci_high": control.r1(hi)})
rows += [row, adjudicators]

out = pd.DataFrame(rows)
out.to_csv(TABLES / "table_s22.csv", index=False)
print(out.to_string(index=False))
