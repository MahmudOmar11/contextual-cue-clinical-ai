"""Independent physician validation: Table 3, Tables S15 to S18, and the numbers reported in the text.

Physician ratings are averaged within each case version, and case families are weighted equally. LLM rates use
the baseline benchmark responses (valid answers only), pooled over runs and answer orders within a panel entry and
averaged over panel entries, so that each unique LLM counts once. The frontier LLMs use their frontier-panel runs.

Intervals come from approximate crossed exponential reweighting (10,000 draws): family weights are shared between
LLMs and physicians, physicians are reweighted, and LLMs are reweighted for the 22-LLM panel; the three frontier
LLMs are held fixed. Within support subsets, which depend on the physicians' own ratings, only LLM effects are
reported.

Writes results/tables/table_3.csv, table_s15.csv to table_s18.csv, and results/estimates/05_physician_validation.csv.
"""
import numpy as np
import pandas as pd

from common import ESTIMATES, FRONTIER, TABLES, VERSIONS, cases, clinician_responses, form_families, model_responses, \
    physician_ratings

B = 10000
SEED = 20260928
AGREEMENT_SEED = 20260927
ROLES = ["intended", "cue_associated", "other_1", "other_2"]
PROBS = ["p_intended", "p_cue_associated", "p_other_1", "p_other_2"]
CUE_TYPES = ["travel or residence", "occupation or exposure", "family history", "lifestyle", "other"]
SEL, ACC = 0, 1  # outcome index: selection of the cue-associated diagnosis, accuracy (selection of the intended one)


def ci(v):
    return [float(x) for x in np.quantile(v, [0.025, 0.975])]


def pct(x):
    return round(100 * float(x), 1)


# ---------------------------------------------------------------- physician ratings
R = physician_ratings()
R["role"] = R.top_choice.map(ROLES.index)
PHYSICIANS = sorted(R.physician.unique())
FAMILIES = list(range(1, 101))
F = len(FAMILIES)
assert len(R) == 800 and len(PHYSICIANS) == 8
# physicians x families x versions x [cue-associated first, intended first, p cue-associated, p intended]
ARR = np.zeros((len(PHYSICIANS), F, 2, 4))
PRESENT = np.zeros((len(PHYSICIANS), F, 2))
for r in R.itertuples():
    i, j, k = PHYSICIANS.index(r.physician), FAMILIES.index(r.family), VERSIONS.index(r.version)
    PRESENT[i, j, k] = 1
    ARR[i, j, k] = [r.role == 1, r.role == 0, r.p_cue_associated, r.p_intended]
HUMAN = ARR.sum(0) / PRESENT.sum(0)[:, :, None]

# Per version: leading diagnosis by mean probability, majority first choice, share choosing the intended diagnosis
VERSION_CELLS = {}
for (fam, ver), g in R.groupby(["family", "version"]):
    votes = np.bincount(g.role, minlength=4)
    p = g[PROBS].mean().to_numpy()
    leaders = np.flatnonzero(np.isclose(p, p.max(), rtol=0, atol=1e-10))
    majority = np.flatnonzero(votes > len(g) / 2)
    VERSION_CELLS[fam, ver] = dict(
        n=len(g), leader=int(leaders[0]) if len(leaders) == 1 else -1,
        majority=int(majority[0]) if len(majority) == 1 else -1, intended_share=votes[0] / len(g),
        pair_agreement=float(sum(v * (v - 1) for v in votes) / (len(g) * (len(g) - 1))))


def both(fn):
    return np.array([fn(VERSION_CELLS[f, "no_cue"]) and fn(VERSION_CELLS[f, "cue"]) for f in FAMILIES])


SUPPORT = {  # subsets of families by the physicians' support for the intended diagnosis in both versions
    "Intended diagnosis led in both versions": both(lambda c: c["leader"] == 0),
    "More than half selected it in both": both(lambda c: c["majority"] == 0),
    "At least two thirds selected it in both": both(lambda c: c["intended_share"] >= 2 / 3 - 1e-12),
    "At least three quarters selected it in both": both(lambda c: c["intended_share"] >= 0.75),
    "All selected it in both": both(lambda c: c["intended_share"] == 1),
}
ALL = np.ones(F, bool)
FORM = np.isin(FAMILIES, form_families())
CUE_TYPE = cases().set_index("family").loc[FAMILIES, "cue_type"].to_numpy()


# ---------------------------------------------------------------- LLM rates
def llm_cells(d):
    """Valid-answer rates per unique LLM, prompt, family and version."""
    g = d.groupby(["model_id", "panel", "prompt", "family", "version"], as_index=False)[
        ["valid", "cue_associated", "intended"]].sum()
    for o in ("cue_associated", "intended"):
        g[o] = g[o] / g.valid.replace(0, np.nan)
    return g.groupby(["model_id", "prompt", "family", "version"], as_index=False)[["cue_associated", "intended"]].mean()


MR = model_responses()
CELLS = {"all": llm_cells(MR), "frontier": llm_cells(MR[MR.panel == "frontier"])}
assert sorted(CELLS["frontier"].model_id.unique()) == sorted(FRONTIER)


class Panel:
    """Crossed exponential reweighting of families, physicians and (unless held fixed) LLMs."""

    def __init__(self, cells, name, fixed_llms, prompt="baseline"):
        self.name, self.fixed = name, fixed_llms
        d = cells[cells.prompt == prompt]
        self.llms = sorted(d.model_id.unique())
        M = len(self.llms)
        ma = np.full((M, F, 2, 2), np.nan)
        for r in d.itertuples():
            ma[self.llms.index(r.model_id), FAMILIES.index(r.family), VERSIONS.index(r.version)] = \
                [r.cue_associated, r.intended]
        paired = np.isfinite(ma).all(axis=(2, 3))  # LLM-family pairs with both versions
        self.available = paired.sum(0) > 0
        ma = np.where(paired[:, :, None, None], ma, 0.0)
        self.cells = ma.sum(0) / np.maximum(paired.sum(0), 1)[:, None, None]
        rng = np.random.default_rng(SEED)
        self.fw = rng.exponential(1, (B, F))
        lw = rng.exponential(1, (B, M))
        pw = rng.exponential(1, (B, len(PHYSICIANS)))
        self.lboot = (np.einsum("bm,mfck->bfck", lw, ma)
                      / np.maximum(np.einsum("bm,mf->bf", lw, paired), 1e-200)[:, :, None, None])
        self.hboot = np.einsum("br,rfck->bfck", pw, ARR) / np.einsum("br,rfc->bfc", pw, PRESENT)[:, :, :, None]

    def w(self, mask):
        w = self.fw[:, mask]
        return w / w.sum(1, keepdims=True)

    def effect(self, mask, k=SEL):
        """LLM and physician cue effects and the extra LLM cue effect, with 95% intervals."""
        mask = mask & self.available
        w = self.w(mask)
        pm, ph = self.cells[mask][:, :, k].mean(0), HUMAN[mask][:, :, k].mean(0)
        if self.fixed:
            md = w @ (self.cells[mask, 1, k] - self.cells[mask, 0, k])
        else:
            mc = np.einsum("bf,bfc->bc", w, self.lboot[:, mask][..., k])
            md = mc[:, 1] - mc[:, 0]
        h = np.einsum("bf,bfc->bc", w, self.hboot[:, mask][..., k])
        hd = h[:, 1] - h[:, 0]
        return dict(families=int(mask.sum()), llm=pm[1] - pm[0], llm_ci=ci(md), physician=ph[1] - ph[0],
                    physician_ci=ci(hd), extra=(pm[1] - pm[0]) - (ph[1] - ph[0]), extra_ci=ci(md - hd))

    def llm_rates(self, mask, k):
        mask = mask & self.available
        w = self.w(mask)
        pm = self.cells[mask][:, :, k].mean(0)
        b = w @ self.cells[mask][:, :, k] if self.fixed else np.einsum("bf,bfc->bc", w, self.lboot[:, mask][..., k])
        return [(pm[c], *ci(b[:, c])) for c in (0, 1)]

    def physician_rates(self, mask, k):
        mask = mask & self.available
        w = self.w(mask)
        ph = HUMAN[mask][:, :, k].mean(0)
        b = np.einsum("bf,bfc->bc", w, self.hboot[:, mask][..., k])
        return [(ph[c], *ci(b[:, c])) for c in (0, 1)]

    def physician_effect(self, mask, k):
        mask = mask & self.available
        w = self.w(mask)
        ph = HUMAN[mask][:, :, k].mean(0)
        h = np.einsum("bf,bfc->bc", w, self.hboot[:, mask][..., k])
        return ph[0], ph[1], ph[1] - ph[0], ci(h[:, 1] - h[:, 0])


FR = Panel(CELLS["frontier"], "Frontier LLMs", fixed_llms=True)
LL = Panel(CELLS["all"], "All 22 LLMs", fixed_llms=False)
assert len(FR.llms) == 3 and len(LL.llms) == 22 and FR.available.all() and LL.available.all()
est = []


def add(name, value, lo=np.nan, hi=np.nan, n=np.nan):
    est.append(dict(estimate=name, value=value, ci_low=lo, ci_high=hi, n=n))


# ---------------------------------------------------------------- Table 3
e_fr, e_ll = FR.effect(ALL), LL.effect(ALL)
rows = []
for group, n, sel, acc, e in [("Independent physicians", 8, FR.physician_rates(ALL, SEL), FR.physician_rates(ALL, ACC), None),
                              ("Frontier LLMs", 3, FR.llm_rates(ALL, SEL), FR.llm_rates(ALL, ACC), e_fr),
                              ("All 22 LLMs", 22, LL.llm_rates(ALL, SEL), LL.llm_rates(ALL, ACC), e_ll)]:
    row = dict(group=group, n=n)
    for name, (v, lo, hi) in [("no_cue_accuracy", acc[0]), ("cue_accuracy", acc[1]),
                              ("no_cue_selection", sel[0]), ("cue_selection", sel[1])]:
        row.update({name: pct(v), f"{name}_ci_low": pct(lo), f"{name}_ci_high": pct(hi)})
    row.update(extra_llm_cue_effect=np.nan if e is None else pct(e["extra"]),
               extra_llm_cue_effect_ci_low=np.nan if e is None else pct(e["extra_ci"][0]),
               extra_llm_cue_effect_ci_high=np.nan if e is None else pct(e["extra_ci"][1]))
    rows.append(row)
pd.DataFrame(rows).to_csv(TABLES / "table_3.csv", index=False)

# Physicians' cue effect on selection and on the probability given to the cue-associated diagnosis
add("physician_cue_effect_selection_pp", pct(e_fr["physician"]), *map(pct, e_fr["physician_ci"]), F)
nc, cu, diff, (lo, hi) = FR.physician_effect(ALL, 2)
add("physician_probability_cue_associated_no_cue_pct", round(nc, 1), n=F)
add("physician_probability_cue_associated_cue_pct", round(cu, 1), n=F)
add("physician_probability_cue_associated_difference_pp", round(diff, 1), round(lo, 1), round(hi, 1), F)
for ver in VERSIONS:
    s = R[R.version == ver]
    add(f"physician_assessments_cue_associated_first_{ver}", int((s.role == 1).sum()), n=len(s))

# Leading diagnosis by mean probability
lead = {ver: np.array([VERSION_CELLS[f, ver]["leader"] for f in FAMILIES]) for ver in VERSIONS}
led_both = SUPPORT["Intended diagnosis led in both versions"]
add("families_intended_led_no_cue", int((lead["no_cue"] == 0).sum()), n=F)
add("families_intended_led_both_versions", int(led_both.sum()), n=F)
add("clinician_form_families_intended_led_both_versions", int((led_both & FORM).sum()), n=int(FORM.sum()))
add("versions_cue_associated_led_cue", int((lead["cue"] == 1).sum()), n=F)
add("versions_cue_associated_led_no_cue", int((lead["no_cue"] == 1).sum()), n=F)
per_version = PRESENT.sum(0).ravel()
add("ratings_per_version_median", float(np.median(per_version)), n=per_version.size)
add("ratings_per_version_min", int(per_version.min()), n=per_version.size)
add("ratings_per_version_max", int(per_version.max()), n=per_version.size)

# Base-rate prompt against the physicians
for panel, cells, fixed in [("all_llms", CELLS["all"], False), ("frontier_llms", CELLS["frontier"], True)]:
    e = Panel(cells, panel, fixed_llms=fixed, prompt="base_rate").effect(ALL)
    add(f"extra_llm_cue_effect_base_rate_prompt_{panel}_pp", pct(e["extra"]), *map(pct, e["extra_ci"]), F)

# ---------------------------------------------------------------- Table S15: participation and agreement
rng = np.random.default_rng(AGREEMENT_SEED)
rng.exponential(1, (B, len(PHYSICIANS)))  # the stream opens with physician weights, not used for agreement
fw = rng.exponential(1, (B, F))
fw = fw / fw.sum(1, keepdims=True)


def alpha(groups, binary=False):
    """Krippendorff's alpha (coincidence formulation); every version has at least two ratings."""
    n = sum(len(v) for v in groups)
    if binary:
        counts = np.array([[len(v) - sum(v), sum(v)] for v in groups], float)
        do = sum(2 * a * b / (a + b - 1) for a, b in counts) / n
        de = 2 * counts.sum(0).prod() / (n * (n - 1))
    else:
        do = sum(2 * len(v) * np.sum((np.asarray(v) - np.mean(v)) ** 2) / (len(v) - 1) for v in groups) / n
        flat = np.concatenate(groups)
        de = 2 * np.sum((flat - flat.mean()) ** 2) / (n - 1)
    return float(1 - do / de)


s15 = {}
for ver in VERSIONS:
    s = R[R.version == ver]
    groups = [g for _, g in s.groupby("family")]
    agree = np.array([VERSION_CELLS[f, ver]["pair_agreement"] for f in FAMILIES])
    res = {"agreement": (agree.mean(), *ci(fw @ agree))}
    intended = [(g.role == 0).to_numpy(int) for g in groups]
    point = {"alpha_intended": alpha(intended, binary=True),
             "alpha_p_intended": alpha([g.p_intended.to_numpy() for g in groups]),
             "alpha_p_cue_associated": alpha([g.p_cue_associated.to_numpy() for g in groups])}
    # family bootstrap of the alpha statistics
    cw = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=B)
    count = np.array([len(g) for g in groups], float)
    n = cw @ count
    pos = np.array([(g.role == 0).sum() for g in groups], float)
    neg = count - pos
    do = (cw @ (2 * pos * neg / (count - 1))) / n
    p, q = cw @ pos, cw @ neg
    res["alpha_intended"] = (point["alpha_intended"], *ci(1 - do / (2 * p * q / (n * (n - 1)))))
    for col, key in [("p_intended", "alpha_p_intended"), ("p_cue_associated", "alpha_p_cue_associated")]:
        sums = np.array([g[col].sum() for g in groups], float)
        sumsq = np.array([(g[col] ** 2).sum() for g in groups], float)
        do = (cw @ (2 * count * (sumsq - sums ** 2 / count) / (count - 1))) / n
        de = 2 * ((cw @ sumsq) - (cw @ sums) ** 2 / n) / (n - 1)
        res[key] = (point[key], *ci(1 - do / de))
    top_two_equal = s[PROBS].apply(lambda x: (x == x.max()).sum() > 1, axis=1)
    ratings = pd.Series(PRESENT[:, :, VERSIONS.index(ver)].sum(0)).value_counts()
    res.update(physicians=s.physician.nunique(), assessments=len(s),
               **{f"versions_{k}": int(ratings.get(k, 0)) for k in range(2, 7)},
               close=int(s.close_or_undetermined.sum()), tie=int(top_two_equal.sum()),
               flagged=int(s.problem_flagged.sum()))
    s15[ver] = res

rows = []
for key, label in [("physicians", "Physicians completing all 100 cases"), ("assessments", "Assessments")] + \
                  [(f"versions_{k}", f"Versions with {k} ratings") for k in range(2, 7)]:
    rows.append(dict(measure=label, no_cue=s15["no_cue"][key], cue=s15["cue"][key]))
for key, label, scale, digits in [
        ("agreement", "Exact pairwise agreement on the top diagnosis, %", 100, 1),
        ("alpha_intended", "Krippendorff's alpha, intended versus other diagnosis", 1, 2),
        ("alpha_p_intended", "Krippendorff's alpha, probability of the intended diagnosis", 1, 2),
        ("alpha_p_cue_associated", "Krippendorff's alpha, probability of the cue-associated diagnosis", 1, 2)]:
    row = dict(measure=label)
    for ver in VERSIONS:
        v, lo, hi = (round(scale * x, digits) for x in s15[ver][key])
        row.update({ver: v, f"{ver}_ci_low": lo, f"{ver}_ci_high": hi})
    rows.append(row)
for key, label in [("close", "Leading diagnoses judged approximately equal, or undecided"),
                   ("tie", "Top two probabilities numerically equal"), ("flagged", "Problem flagged with the case")]:
    rows.append(dict(measure=label, no_cue=s15["no_cue"][key], cue=s15["cue"][key]))
pd.DataFrame(rows, columns=["measure", "no_cue", "no_cue_ci_low", "no_cue_ci_high", "cue", "cue_ci_low", "cue_ci_high"],
             dtype=object).to_csv(TABLES / "table_s15.csv", index=False)
add("assessments_close_or_undetermined", int(R.close_or_undetermined.sum()), n=len(R))
add("assessments_problem_flagged", int(R.problem_flagged.sum()), n=len(R))

# ---------------------------------------------------------------- Table S16: LLM cue effect by physician support
rows = []
for label, mask in [("All families", ALL)] + list(SUPPORT.items()):
    ef, el = FR.effect(mask), LL.effect(mask)
    row = dict(subset=label, frontier_families=ef["families"], frontier_cue_effect=pct(ef["llm"]),
               frontier_ci_low=pct(ef["llm_ci"][0]), frontier_ci_high=pct(ef["llm_ci"][1]),
               all_llms_families=el["families"], all_llms_cue_effect=pct(el["llm"]),
               all_llms_ci_low=pct(el["llm_ci"][0]), all_llms_ci_high=pct(el["llm_ci"][1]))
    if label == "All families":  # support subsets are defined by the physicians, so no comparison with them
        row.update(frontier_extra_cue_effect=pct(ef["extra"]), frontier_extra_ci_low=pct(ef["extra_ci"][0]),
                   frontier_extra_ci_high=pct(ef["extra_ci"][1]), all_llms_extra_cue_effect=pct(el["extra"]),
                   all_llms_extra_ci_low=pct(el["extra_ci"][0]), all_llms_extra_ci_high=pct(el["extra_ci"][1]))
    rows.append(row)
pd.DataFrame(rows).to_csv(TABLES / "table_s16.csv", index=False)

# Against clinician participants on the clinician form, within the families the physicians supported:
# family-weighted cue effect of all 22 LLMs (all responses in the denominators) minus that of the participants
cl = clinician_responses()
clin = cl.groupby(["family", "version"]).cue_associated.mean().unstack()
base = MR[(MR.prompt == "baseline") & MR.family.isin(form_families())]
llm = (base.groupby(["model_id", "panel", "run", "family", "version"]).cue_associated.mean()
       .groupby(["model_id", "panel", "family", "version"]).mean()
       .groupby(["model_id", "family", "version"]).mean().unstack())
llm_shift = (llm.cue - llm.no_cue).groupby("family").mean()
clin_shift = clin.cue - clin.no_cue
for label, mask in [("All families", FORM)] + [(k, FORM & v) for k, v in SUPPORT.items()]:
    fams = np.array(FAMILIES)[mask]
    if len(fams):
        key = label.lower().replace(" ", "_")
        add(f"extra_llm_cue_effect_vs_clinician_participants_{key}_pp",
            pct(llm_shift.loc[fams].mean() - clin_shift.loc[fams].mean()), n=len(fams))

# ---------------------------------------------------------------- Table S17: each family
code = {0: "intended", 1: "cue_associated", 2: "other", 3: "other", -1: "tie"}
fam = pd.DataFrame({
    "family": FAMILIES,
    "ratings_no_cue": PRESENT[:, :, 0].sum(0).astype(int), "ratings_cue": PRESENT[:, :, 1].sum(0).astype(int),
    "cue_associated_first_choice_no_cue_pct": np.round(100 * HUMAN[:, 0, 0], 1),
    "cue_associated_first_choice_cue_pct": np.round(100 * HUMAN[:, 1, 0], 1),
    "cue_associated_probability_no_cue_pct": np.round(HUMAN[:, 0, 2], 1),
    "cue_associated_probability_cue_pct": np.round(HUMAN[:, 1, 2], 1),
    "intended_probability_no_cue_pct": np.round(HUMAN[:, 0, 3], 1),
    "intended_probability_cue_pct": np.round(HUMAN[:, 1, 3], 1),
    "leading_diagnosis_no_cue": [code[x] for x in lead["no_cue"]],
    "leading_diagnosis_cue": [code[x] for x in lead["cue"]],
    "frontier_llms_cue_effect_pp": np.round(100 * (FR.cells[:, 1, SEL] - FR.cells[:, 0, SEL]), 1),
    "all_llms_cue_effect_pp": np.round(100 * (LL.cells[:, 1, SEL] - LL.cells[:, 0, SEL]), 1)})
fam.to_csv(TABLES / "table_s17.csv", index=False)

# ---------------------------------------------------------------- Table S18: cue types (exploratory)
rows = []
for P, label in [(FR, "Frontier LLMs, 100 families"), (LL, "All 22 LLMs, 100 families")]:
    for t in CUE_TYPES:
        e = P.effect(CUE_TYPE == t)
        rows.append(dict(llm_panel=label, cue_type=t.capitalize(), families=e["families"],
                         physician_cue_effect=pct(e["physician"]), physician_ci_low=pct(e["physician_ci"][0]),
                         physician_ci_high=pct(e["physician_ci"][1]), llm_cue_effect=pct(e["llm"]),
                         llm_ci_low=pct(e["llm_ci"][0]), llm_ci_high=pct(e["llm_ci"][1]),
                         extra_llm_cue_effect=pct(e["extra"]), extra_ci_low=pct(e["extra_ci"][0]),
                         extra_ci_high=pct(e["extra_ci"][1])))
pd.DataFrame(rows).to_csv(TABLES / "table_s18.csv", index=False)

out = pd.DataFrame(est, dtype=object)
out.to_csv(ESTIMATES / "05_physician_validation.csv", index=False)
print(out.to_string(index=False))
