"""Descriptive tables for the clinician participants and the model benchmark.

Writes Table 1, Tables S2, S10, S12 and S13, and results/estimates/02_supplementary_tables.csv.
Region, specialty, practice setting and response times are withheld at the individual level;
their values are copied from data/clinician_aggregates.csv.
Run from the repository root: python3 code/analysis/02_supplementary_tables.py
"""
import pandas as pd

from common import DATA, ESTIMATES, TABLES, cases, clinician_responses, form_families, model_responses

clin = clinician_responses()
participants = pd.read_csv(DATA / "clinician_participants.csv")
aggregates = pd.read_csv(DATA / "clinician_aggregates.csv")
N_PARTICIPANTS = len(participants)

TRAINING_ORDER = ["attending_or_consultant", "resident_or_fellow", "medical_student", "other_physician_role"]
VERSION_LABEL = {"no_cue": "No cue", "cue": "Cue"}


def pct(k, n):
    return round(100 * k / n, 1)


def sentence_case(label):
    label = label.replace(" / ", " or ")
    return label[0] + label[1:].lower()


def aggregate(table, measure="participants"):
    a = aggregates[(aggregates["table"] == table) & (aggregates.measure == measure)]
    return a[["characteristic", "category", "value"]]


# ---------------------------------------------------------------- Table 1
TABLE1_TRAINING = {"attending_or_consultant": "Attending or consultant physician",
                   "resident_or_fellow": "Resident or fellow", "medical_student": "Medical student",
                   "other_physician_role": "Other physician role"}
rows = []
counts = participants.training_level.value_counts()
for level in TRAINING_ORDER:
    rows.append(("Training level", TABLE1_TRAINING[level], int(counts.get(level, 0))))
for characteristic in ["Region", "Specialty", "Practice setting"]:
    a = aggregate("table_1")
    a = a[a.characteristic == characteristic]
    for r in a.itertuples():
        label = r.category if characteristic == "Region" else sentence_case(r.category)
        rows.append((characteristic, label, int(r.value)))
table_1 = pd.DataFrame(rows, columns=["characteristic", "category", "n"])
table_1["percent"] = [pct(n, N_PARTICIPANTS) for n in table_1.n]
order = {c: i for i, c in enumerate(["Training level", "Region", "Specialty", "Practice setting"])}
table_1 = (table_1.assign(_c=table_1.characteristic.map(order), _n=-table_1.n)
           .sort_values(["_c", "_n", "category"], kind="stable").drop(columns=["_c", "_n"]))
table_1.to_csv(TABLES / "table_1.csv", index=False)

# ---------------------------------------------------------------- Table S2 (clinician data completeness)
per_participant = clin.groupby("participant").size()
per_version = clin.groupby(["participant", "version"]).size().unstack()
repeated = cases().query("repeated_on_clinician_form == 1").family.tolist()


def distinct(values):
    return ", ".join(str(v) for v in sorted(set(values)))


table_s2 = pd.DataFrame([
    ("Answer rows", len(clin)),
    ("Participants", clin.participant.nunique()),
    ("Rows per participant", distinct(per_participant)),
    ("Cue rows", int((clin.version == "cue").sum())),
    ("No-cue rows", int((clin.version == "no_cue").sum())),
    ("Cue rows per participant", distinct(per_version["cue"])),
    ("No-cue rows per participant", distinct(per_version["no_cue"])),
    ("Distinct case families", clin.family.nunique()),
    ("Repeated case families by design", distinct(repeated)),
], columns=["metric", "value"])
table_s2.to_csv(TABLES / "table_s2.csv", index=False)

# ---------------------------------------------------------------- Table S10 (confidence and response time)
OUTCOME_LABEL = {"intended": "Correct", "cue_associated": "Cue-associated diagnosis", "other": "Other incorrect"}
times = aggregates[aggregates["table"] == "table_s10"].pivot_table(
    index=["characteristic", "category"], columns="measure", values="value")
rows = []
for version in ["no_cue", "cue"]:
    for outcome in OUTCOME_LABEL:
        c = clin[(clin.version == version) & (clin.chosen == outcome)].confidence
        t = times.loc[(version, outcome)]
        rows.append((VERSION_LABEL[version], OUTCOME_LABEL[outcome], len(c), round(c.mean(), 2),
                     round(c.std(), 2), c.median(), t["mean_response_time_sec"], t["median_response_time_sec"]))
table_s10 = pd.DataFrame(rows, columns=["condition", "outcome", "n", "confidence_mean", "confidence_sd",
                                        "confidence_median", "response_time_mean_sec", "response_time_median_sec"])
table_s10.to_csv(TABLES / "table_s10.csv", index=False)

# ---------------------------------------------------------------- Table S12 (recorded rationale)
RATIONALE = {"specific_case_detail": "Anchored on a case detail", "most_probable": "Judged most probable",
             "rule_out": "Ruling out a serious diagnosis", "uncertain": "Uncertain"}
rows = []
for version in ["cue", "no_cue"]:
    chosen = clin[(clin.version == version) & (clin.chosen == "cue_associated")]
    counts = chosen.rationale.value_counts()
    ranked = sorted(counts.index, key=lambda r: (-counts[r], list(RATIONALE).index(r)))
    for r in ranked:
        rows.append((VERSION_LABEL[version], RATIONALE[r], int(counts[r]), pct(counts[r], len(chosen))))
table_s12 = pd.DataFrame(rows, columns=["condition", "rationale", "n", "percent"])
table_s12.to_csv(TABLES / "table_s12.csv", index=False)

# ---------------------------------------------------------------- Table S13 (clinician subgroups)
S13_TRAINING = {"attending_or_consultant": "Attending / consultant", "resident_or_fellow": "Resident / fellow",
                "medical_student": "Medical student", "other_physician_role": "Other clinician role"}
rows = []
for level in TRAINING_ORDER:
    g = clin[clin.training_level == level]
    no_cue, cue = g[g.version == "no_cue"], g[g.version == "cue"]
    rows.append(("Training level", S13_TRAINING[level], g.participant.nunique(),
                 pct(no_cue.cue_associated.sum(), len(no_cue)), pct(cue.cue_associated.sum(), len(cue)),
                 pct(no_cue.intended.sum(), len(no_cue)), pct(cue.intended.sum(), len(cue))))
wide = aggregates[aggregates["table"] == "table_s13"].pivot_table(
    index=["characteristic", "category"], columns="measure", values="value", sort=False)
for (characteristic, category), r in wide.iterrows():
    rows.append((characteristic, category, int(r["participants"]), r["no_cue_selection_pct"],
                 r["cue_selection_pct"], r["no_cue_accuracy_pct"], r["cue_accuracy_pct"]))
table_s13 = pd.DataFrame(rows, columns=["subgroup_type", "subgroup", "participants", "no_cue_selection",
                                        "cue_selection", "no_cue_accuracy", "cue_accuracy"])
type_order = {t: i for i, t in enumerate(["Training level", "Region", "Specialty"])}
table_s13 = table_s13.assign(_t=table_s13.subgroup_type.map(type_order)).sort_values(
    ["_t", "participants"], ascending=[True, False], kind="stable").drop(columns="_t")
table_s13.to_csv(TABLES / "table_s13.csv", index=False)

# ---------------------------------------------------------------- model benchmark counts
models = model_responses()
blocks = models.groupby(["panel", "prompt"], sort=False).size()
no_valid = models[models.valid == 0]
PANEL = {"core": "Core-20", "frontier": "Frontier-3"}
rows = [(f"Model responses, {PANEL[panel]}, {prompt.replace('_', '-')} prompt", n)
        for (panel, prompt), n in blocks.items()]
rows += [
    ("Model responses", len(models)),
    ("Model responses with a valid answer", int(models.valid.sum())),
    ("Model responses without a valid answer", len(no_valid)),
    ("Model-panel entries with responses without a valid answer", no_valid.groupby(["panel", "model_id"]).ngroups),
    ("Model-panel entries with responses without a valid answer, Frontier-3",
     no_valid[no_valid.panel == "frontier"].model_id.nunique()),
    ("Baseline model responses on the matched case families",
     int(((models.prompt == "baseline") & models.family.isin(form_families())).sum())),
    ("Clinician responses", len(clin)),
    ("Case families on the clinician form", len(form_families())),
]
estimates = pd.DataFrame(rows, columns=["estimate", "value"]).assign(ci_low=None, ci_high=None, n=None)
estimates.to_csv(ESTIMATES / "02_supplementary_tables.csv", index=False)
print(f"Wrote Table 1, Tables S2, S10, S12 and S13, and {len(estimates)} estimates.")
