"""Clinician participants: confidence with participant-resampled intervals, and allocation of case versions.

Confidence intervals resample participants (10,000 draws) for each version-by-answer cell in turn, in sorted order.

Writes results/estimates/04_robustness_clinician.csv.
"""
import numpy as np
import pandas as pd

from common import ESTIMATES, cases, clinician_responses

B = 10000
SEED = 2026

d = clinician_responses().sort_values(["participant", "position"], ignore_index=True)
rows = []

# Mean confidence (1-5) by version and chosen answer
rng = np.random.default_rng(SEED)
participants = sorted(d.participant.unique())  # resampled in sorted code order
reported = {("no_cue", "intended"): "clinician_confidence_intended_diagnosis_no_cue",
            ("cue", "cue_associated"): "clinician_confidence_cue_associated_diagnosis_cue"}
for (version, answer), g in d.groupby(["version", "chosen"]):
    cell = g.groupby("participant").confidence.agg(["sum", "size"]).reindex(participants, fill_value=0)
    total, count = cell["sum"].to_numpy(float), cell["size"].to_numpy(float)
    boot = []
    for _ in range(B):
        idx = rng.choice(len(participants), len(participants))
        n = count[idx].sum()
        if n:
            boot.append(total[idx].sum() / n)
    if (version, answer) in reported:
        rows.append(dict(estimate=reported[(version, answer)], value=round(g.confidence.mean(), 2),
                         ci_low=round(np.percentile(boot, 2.5), 2), ci_high=round(np.percentile(boot, 97.5), 2),
                         n=len(g)))

# Responses per family and version
repeated = set(cases().query("repeated_on_clinician_form == 1").family)
per_version = d.groupby(["family", "version"]).size()
for label, sel in [("repeated_families", per_version.index.get_level_values(0).isin(repeated)),
                   ("other_families", ~per_version.index.get_level_values(0).isin(repeated))]:
    v = per_version[sel]
    rows.append(dict(estimate=f"responses_per_version_{label}_min", value=int(v.min()), n=int(sel.sum())))
    rows.append(dict(estimate=f"responses_per_version_{label}_max", value=int(v.max()), n=int(sel.sum())))

# Spacing of the two versions of each repeated family on a form
r = d[d.family.isin(repeated)].sort_values(["participant", "family", "position"])
pairs = r.groupby(["participant", "family"]).agg(first=("position", "min"), last=("position", "max"),
                                                 first_version=("version", "first"))
between = pairs["last"] - pairs["first"] - 1
rows += [dict(estimate="repeated_family_cases_between_versions_median", value=float(between.median()), n=len(pairs)),
         dict(estimate="repeated_family_cases_between_versions_min", value=int(between.min()), n=len(pairs)),
         dict(estimate="repeated_family_cases_between_versions_max", value=int(between.max()), n=len(pairs)),
         dict(estimate="repeated_family_pairs_cue_version_first", value=int((pairs.first_version == "cue").sum()),
              n=len(pairs))]

out = pd.DataFrame(rows, columns=["estimate", "value", "ci_low", "ci_high", "n"], dtype=object)
out.to_csv(ESTIMATES / "04_robustness_clinician.csv", index=False)
print(out.to_string(index=False))
