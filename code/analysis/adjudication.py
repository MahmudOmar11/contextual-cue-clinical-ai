"""The clinical adjudicators' ratings and the comparison of LLM cue effects with theirs.

Shared by 05_physician_validation.py and the scripts that repeat its comparisons (10 to 13). The eight clinical
adjudicators (codes A to H in data/physician_ratings.csv) were six physicians and two medical students (A and B).
Ratings are averaged within each case version, and case families are weighted equally. LLM rates use the benchmark
responses, pooled over runs and answer orders within a panel entry and averaged over panel entries, so that each
unique LLM counts once. The frontier LLMs use their frontier-panel runs.

Intervals come from approximate crossed exponential reweighting (10,000 draws): family weights are shared between
LLMs and adjudicators, adjudicators are reweighted, and LLMs are reweighted for the 22-LLM panel; the three frontier
LLMs are held fixed.
"""
import numpy as np

from common import VERSIONS, physician_ratings

B = 10000
SEED = 20260928
ROLES = ["intended", "cue_associated", "other_1", "other_2"]
SEL, ACC, P_CUE = 0, 1, 2  # outcome index: selection of the cue-associated diagnosis, accuracy, its probability


def ci(v):
    return [float(x) for x in np.quantile(v, [0.025, 0.975])]


def pct(x):
    return round(100 * float(x), 1)


RATINGS = physician_ratings()
RATINGS["role"] = RATINGS.top_choice.map(ROLES.index)
ADJUDICATORS = sorted(RATINGS.physician.unique())
FAMILIES = list(range(1, 101))
F = len(FAMILIES)
assert len(RATINGS) == 800 and len(ADJUDICATORS) == 8
# adjudicators x families x versions x [cue-associated first, intended first, p cue-associated, p intended]
ARR = np.zeros((len(ADJUDICATORS), F, 2, 4))
PRESENT = np.zeros((len(ADJUDICATORS), F, 2))
for r in RATINGS.itertuples():
    i, j, k = ADJUDICATORS.index(r.physician), FAMILIES.index(r.family), VERSIONS.index(r.version)
    PRESENT[i, j, k] = 1
    ARR[i, j, k] = [r.role == 1, r.role == 0, r.p_cue_associated, r.p_intended]
HUMAN = ARR.sum(0) / PRESENT.sum(0)[:, :, None]


def llm_cells(d, all_attempts=False):
    """Rates per unique LLM, prompt, family and version, among valid answers or, with all_attempts, among all
    responses (a response without a valid answer then counts as neither diagnosis)."""
    g = d.assign(responses=1).groupby(["model_id", "panel", "prompt", "family", "version"], as_index=False)[
        ["responses", "valid", "cue_associated", "intended"]].sum()
    denominator = g.responses if all_attempts else g.valid.replace(0, np.nan)
    for o in ("cue_associated", "intended"):
        g[o] = g[o] / denominator
    return g.groupby(["model_id", "prompt", "family", "version"], as_index=False)[["cue_associated", "intended"]].mean()


def reweighting(n_llms):
    """Exponential weights for case families, LLMs and adjudicators, drawn in that order from one stream."""
    rng = np.random.default_rng(SEED)
    fw = rng.exponential(1, (B, F))
    lw = rng.exponential(1, (B, n_llms))
    aw = rng.exponential(1, (B, len(ADJUDICATORS)))
    return fw, lw, aw


def adjudicator_draws(aw, arr=ARR, present=PRESENT):
    """Version means of the ratings under adjudicator weights: draws x families x versions x outcomes."""
    return np.einsum("br,rfck->bfck", aw, arr) / np.einsum("br,rfc->bfc", aw, present)[:, :, :, None]


class Panel:
    """Crossed exponential reweighting of families, adjudicators and (unless held fixed) LLMs."""

    def __init__(self, cells, name, fixed_llms, prompt="baseline", keep=None):
        self.name, self.fixed = name, fixed_llms
        idx = list(range(len(ADJUDICATORS))) if keep is None else [ADJUDICATORS.index(c) for c in keep]
        self.arr, self.present = ARR[idx], PRESENT[idx]
        assert self.present.sum(0).min() > 0  # every version keeps at least one rating
        self.human = self.arr.sum(0) / self.present.sum(0)[:, :, None]
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
        self.fw, lw, aw = reweighting(M)
        self.lboot = (np.einsum("bm,mfck->bfck", lw, ma)
                      / np.maximum(np.einsum("bm,mf->bf", lw, paired), 1e-200)[:, :, None, None])
        self.hboot = adjudicator_draws(aw[:, idx], self.arr, self.present)

    def w(self, mask):
        w = self.fw[:, mask]
        return w / w.sum(1, keepdims=True)

    def effect(self, mask, k=SEL):
        """LLM and adjudicator cue effects and the extra LLM cue effect, with 95% intervals."""
        mask = mask & self.available
        w = self.w(mask)
        pm, ph = self.cells[mask][:, :, k].mean(0), self.human[mask][:, :, k].mean(0)
        if self.fixed:
            md = w @ (self.cells[mask, 1, k] - self.cells[mask, 0, k])
        else:
            mc = np.einsum("bf,bfc->bc", w, self.lboot[:, mask][..., k])
            md = mc[:, 1] - mc[:, 0]
        h = np.einsum("bf,bfc->bc", w, self.hboot[:, mask][..., k])
        hd = h[:, 1] - h[:, 0]
        return dict(families=int(mask.sum()), llm=pm[1] - pm[0], llm_ci=ci(md), adjudicators=ph[1] - ph[0],
                    adjudicators_ci=ci(hd), extra=(pm[1] - pm[0]) - (ph[1] - ph[0]), extra_ci=ci(md - hd))

    def llm_rates(self, mask, k):
        mask = mask & self.available
        w = self.w(mask)
        pm = self.cells[mask][:, :, k].mean(0)
        b = w @ self.cells[mask][:, :, k] if self.fixed else np.einsum("bf,bfc->bc", w, self.lboot[:, mask][..., k])
        return [(pm[c], *ci(b[:, c])) for c in (0, 1)]

    def adjudicator_rates(self, mask, k):
        mask = mask & self.available
        w = self.w(mask)
        ph = self.human[mask][:, :, k].mean(0)
        b = np.einsum("bf,bfc->bc", w, self.hboot[:, mask][..., k])
        return [(ph[c], *ci(b[:, c])) for c in (0, 1)]

    def adjudicator_effect(self, mask, k):
        mask = mask & self.available
        w = self.w(mask)
        ph = self.human[mask][:, :, k].mean(0)
        h = np.einsum("bf,bfc->bc", w, self.hboot[:, mask][..., k])
        return ph[0], ph[1], ph[1] - ph[0], ci(h[:, 1] - h[:, 0])
