"""LLM-only robustness: unique-LLM intervals, prompt-arm changes with one run per arm, and answer agreement.

Each unique LLM counts once. Within an LLM, rates use valid answers only: they are pooled over runs and answer
orders within a panel entry, then averaged over panel entries (Gemini 3.1 Pro appears in both panels).
Intervals resample case families and unique LLMs together (10,000 multinomial draws).

Writes results/estimates/04_robustness_llm.csv.
"""
import itertools

import numpy as np
import pandas as pd

from common import DATA, ESTIMATES, PROMPTS, VERSIONS, form_families, model_responses

B = 10000
SEED = 20260927
OUTCOMES = ["intended", "cue_associated"]


def llm_cells(d):
    """Valid-answer rates per unique LLM, prompt, family and version."""
    g = d.groupby(["model_id", "panel", "prompt", "family", "version"], as_index=False)[
        ["valid", "intended", "cue_associated"]].sum()
    for o in OUTCOMES:
        g[o] = g[o] / g.valid.replace(0, np.nan)
    return g.groupby(["model_id", "prompt", "family", "version"], as_index=False)[OUTCOMES].mean()


def cell_array(cells, families):
    """Array of LLMs x families x prompts x versions x outcomes (NaN where no valid answer)."""
    llms = sorted(cells.model_id.unique())
    idx = pd.MultiIndex.from_product([llms, families, PROMPTS, VERSIONS], names=["model_id", "family", "prompt", "version"])
    a = cells.set_index(["model_id", "family", "prompt", "version"])[OUTCOMES].reindex(idx).to_numpy()
    return a.reshape(len(llms), len(families), len(PROMPTS), len(VERSIONS), len(OUTCOMES))


def resampling_weights(n_llms, n_families):
    rng = np.random.default_rng(SEED + n_families)
    wl = rng.multinomial(n_llms, np.full(n_llms, 1 / n_llms), size=B) / n_llms
    wf = rng.multinomial(n_families, np.full(n_families, 1 / n_families), size=B) / n_families
    return wl, wf


def contrast(arr, weights, terms, outcome="cue_associated"):
    """Family- and LLM-weighted contrast over LLM-family pairs with valid answers in every term.

    terms: (prompt, version, sign) triples, for example a cue effect is (p, cue, +1), (p, no_cue, -1).
    """
    k = OUTCOMES.index(outcome)
    mask = np.ones(arr.shape[:2], bool)
    effect = np.zeros(arr.shape[:2])
    for prompt, version, sign in terms:
        x = arr[:, :, PROMPTS.index(prompt), VERSIONS.index(version), k]
        mask &= np.isfinite(x)
        effect += sign * np.nan_to_num(x)
    wl, wf = weights
    draws = (np.einsum("bm,bc,mc->b", wl, wf, effect * mask, optimize=True)
             / np.einsum("bm,bc,mc->b", wl, wf, mask, optimize=True))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return 100 * effect[mask].mean(), 100 * lo, 100 * hi


def cue_effect(prompt):
    return [(prompt, "cue", 1), (prompt, "no_cue", -1)]


def change_from_baseline(prompt):
    return cue_effect(prompt) + [("baseline", "cue", -1), ("baseline", "no_cue", 1)]


d = model_responses()
first_run = d[d.run == 1]
core_llms = set(first_run.loc[first_run.panel == "core", "model_id"])
variants = {
    "all_runs": d,
    "one_run": first_run,
    "one_run_per_unique_llm": first_run[(first_run.panel == "core") | ~first_run.model_id.isin(core_llms)],
}
F21 = form_families()
F100 = list(range(1, 101))

rows = []


def add(name, result, n):
    value, lo, hi = result
    rows.append(dict(estimate=name, value=round(value, 1), ci_low=round(lo, 1), ci_high=round(hi, 1), n=n))


arrays = {}
for variant, data in variants.items():
    cells = llm_cells(data)
    arrays[variant] = {nf: cell_array(cells, fams) for nf, fams in [(21, F21), (100, F100)]}
n_llms = arrays["all_runs"][21].shape[0]
assert n_llms == 22
W = {nf: resampling_weights(n_llms, nf) for nf in (21, 100)}

# LLM cue effect on the 21 clinician-form families: all runs, and one run per unique LLM
add("llm_cue_effect_21_families_all_runs_pp", contrast(arrays["all_runs"][21], W[21], cue_effect("baseline")), 21)
add("llm_cue_effect_21_families_one_run_per_unique_llm_pp",
    contrast(arrays["one_run_per_unique_llm"][21], W[21], cue_effect("baseline")), 21)

# Prompt arms with one run in every arm: cue effects on the 21 families, and changes from baseline
for prompt in PROMPTS:
    add(f"llm_cue_effect_21_families_one_run_{prompt}_pp", contrast(arrays["one_run"][21], W[21], cue_effect(prompt)), 21)
for nf in (21, 100):
    a = arrays["one_run"][nf]
    for prompt in PROMPTS[1:]:
        add(f"change_in_llm_cue_effect_{nf}_families_{prompt}_vs_baseline_pp",
            contrast(a, W[nf], change_from_baseline(prompt)), nf)

# Agreement of the selected diagnosis between repeated runs (baseline) and between answer orders (each prompt),
# over pairs of responses that both contain a valid answer
items = pd.read_csv(DATA / "items.csv", keep_default_na=False)
text = {(r.family, r.version, r.answer_order, x): " ".join(str(getattr(r, f"option_{x}")).lower().split())
        for r in items.itertuples() for x in "ABCD"}
d["diagnosis"] = [text.get((f, v, o, x), "") for f, v, o, x in zip(d.family, d.version, d.answer_order, d.selected_option)]


def agreement(data, fixed):
    same, n = 0, 0
    for _, g in data.groupby(fixed, sort=False):
        dx = g.loc[g.valid == 1, "diagnosis"].tolist()
        for a, b in itertools.combinations(dx, 2):
            same += a == b
            n += 1
    return 100 * same / n, n


keys = ["panel", "model_id", "prompt", "family", "version"]
value, n = agreement(d[d.prompt == "baseline"], keys + ["answer_order"])
rows.append(dict(estimate="agreement_between_repeated_runs_baseline_pct", value=round(value, 1), ci_low=np.nan,
                 ci_high=np.nan, n=n))
for prompt in PROMPTS:
    value, n = agreement(d[d.prompt == prompt], keys + ["run"])
    rows.append(dict(estimate=f"agreement_between_answer_orders_{prompt}_pct", value=round(value, 1), ci_low=np.nan,
                     ci_high=np.nan, n=n))

out = pd.DataFrame(rows)
out.to_csv(ESTIMATES / "04_robustness_llm.csv", index=False)
print(out.to_string(index=False))
