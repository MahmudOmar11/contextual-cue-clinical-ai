# Contextual cue susceptibility in clinical AI

Code and data for **"Contextual Cue Susceptibility in Clinical AI: A Randomized Clinician-Comparator Study"**
(Omar, Agbareia, Charney, Abdulnour, Sakhuja, Klang, Nadkarni, Glicksberg).

The study presented 100 short clinical cases, each in two versions: without and with one added travel, exposure,
occupational, family-history or similar detail that points to a less likely diagnosis (the cue-associated
diagnosis). It compared how often 22 large language models (LLMs) and 47 clinician participants selected that
diagnosis, and eight clinical adjudicators (six physicians and two medical students) rated the cases, each rating one
version of every family. In an added experiment, the three frontier models also answered every case with a matched,
clinically irrelevant detail in place of the cue. This repository holds the data and the code that reproduce every
table, figure and reported estimate.

## Repository layout

```
data/            the released data (described below)
prompts/         the system prompts, the question template and the requested answer formats, verbatim
config/          the models queried and the request settings
code/analysis/   the analysis, in run order (01 to 08)
code/collect/    the script that queries the models, for anyone who wants to re-run them
code/run_all.sh  runs the full analysis
results/         the output of code/run_all.sh: tables, figures and estimates
docs/            the control records of the irrelevant-control experiment
```

## Data

All clinical cases are synthetic, and no patient data were used.

| File | Rows | Content |
|---|---|---|
| `cases.csv` | 100 | One row per case family: intended diagnosis, cue-associated diagnosis, the two other options, cue type, and whether the family was on the clinician form (21 families; families 1, 2 and 5 appeared there in both versions). |
| `items.csv` | 800 | The case set as questions: every case version in each of the four answer orders, with the vignette, options A to D in the order shown, and which letters hold the intended and the cue-associated diagnosis. |
| `model_responses.csv.gz` | 92,000 | Every model answer from the main runs: panel, model, the model version returned, prompt, run, family, version, answer order, the option selected, its role, and the model's raw output. |
| `explanation_responses.csv.gz` | 3,000 | The exploratory run in which five models also gave a short explanation and a confidence. |
| `clinician_responses.csv` | 1,128 | Every clinician answer: participant code, form (1 to 8), position on the form (1 to 24), family, version, answer order, the diagnosis chosen, confidence (1 to 5), and the rationale category recorded when the cue-associated diagnosis was chosen. |
| `clinician_participants.csv` | 47 | Participant code and training level. |
| `clinician_aggregates.csv` | 65 | Aggregate values for characteristics withheld at the individual level (see Privacy). |
| `physician_ratings.csv` | 800 | Every clinical adjudicator rating: adjudicator code in the column `physician` (A and B were medical students, C to H physicians), family, version, the probability given to each of the four diagnoses, the diagnosis chosen as most probable, whether the leading diagnoses were judged about equally likely or could not be separated, and whether a problem with the case was flagged. |
| `control_items.csv` | 1,200 | The irrelevant-control experiment as questions: every family in three versions (`no_cue`, `cue`, `control`) and four answer orders, with the same columns as `items.csv`. |
| `control_responses.csv.gz` | 3,600 | Every answer of the three frontier models in the irrelevant-control experiment: model, the model version returned, item, family, version, answer order, the option selected, its role, and the model's raw output. |

Column values:
- `version` is `no_cue` or `cue`, or, in the control files, also `control`. `answer_order` (1 to 4) is one of the four fixed orders of the options.
- `panel` is `core` (the 20-model Core-20 panel) or `frontier` (the 3-model Frontier-3 panel).
- `prompt` is `baseline`, `base_rate` or `rare_or_serious`.
- `selected_role` and `chosen` are `intended`, `cue_associated`, `other`, or, for models, `no_valid_answer`.
- `top_choice` is `intended`, `cue_associated`, `other_1` or `other_2`; the last two are `other_diagnosis_1` and `other_diagnosis_2` in `cases.csv`.
- `rationale` is `specific_case_detail`, `most_probable`, `rule_out` or `uncertain`, and is empty unless the cue-associated diagnosis was chosen.
- `training_level` is `attending_or_consultant`, `resident_or_fellow`, `medical_student` or `other_physician_role`.
- `raw_output` is the model's text exactly as returned. Most outputs are the requested JSON; some are truncated at the token limit, empty, or prose that contains the answer. `selected_option` is the answer as read when the runs were collected.
- A model panel entry is the pair of `panel` and `model_id`. Gemini 3.1 Pro was queried in both panels, giving 23 entries for 22 models. Responses without a valid answer stay in the denominators as neither diagnosis, as in the paper.

## Reproducing the results

Requirements:
- Python 3.9 or later with pandas, numpy and matplotlib (`pip install -r requirements.txt`).
- R 4.3 or later with lme4, dplyr, tidyr, readr and jsonlite (`install.packages(c("lme4", "dplyr", "tidyr", "readr", "jsonlite"))`).

The results were produced with Python 3.14.7 (pandas 3.0.3, numpy 2.4.6, matplotlib 3.11.1) and R 4.3.0 (lme4 1.1-34, dplyr 1.1.4, tidyr 1.3.1, readr 2.1.4, jsonlite 1.8.8). Mixed-model intervals can differ in the last digit with other lme4 versions.

```
bash code/run_all.sh
```

The full run takes about 20 seconds. All resampling uses fixed, arbitrary seeds, so repeated runs give identical output. Tables and estimates reproduce exactly; figure images can differ at the pixel level with other matplotlib versions or fonts. `06_figure_3.py` uses the output of `05_physician_validation.py`, and `07_explanation_screen.py` uses the parser in `code/collect/`, so run the scripts in order or use `run_all.sh`.

| Script | Produces |
|---|---|
| `01_matched_comparison.R` | Table 2; Tables S3 to S9 and S11: rates with Wilson intervals, the mixed-effects models and the rater-clustered bootstrap |
| `02_supplementary_tables.py` | Table 1; Tables S2, S10, S12, S13 |
| `03_figures_comparison.py` | Figures 1 and 2; Figures S1 to S3 |
| `04_robustness_matched.R` | Crossed resampling of case families, clinician participants and unique LLMs; clinician models with cue effects varying by family and participant |
| `04_robustness_llm.py` | One run per unique LLM; the prompt comparison with one run per prompt; agreement across runs and answer orders |
| `04_robustness_clinician.py` | Intervals for clinician confidence; allocation of case versions and spacing of repeated families |
| `05_physician_validation.py` | Table 3; Tables S15 to S18; the sensitivity analysis with the six physicians alone |
| `06_figure_3.py` | Figure 3 |
| `07_explanation_screen.py` | Tables S14 and S19; Figure S4 |
| `08_model_table.py` | Table S1 |
| `09_control_experiment.py` | Tables S20 and S21: the irrelevant-control experiment |

Tables are written to `results/tables/`, figures to `results/figures/`, and estimates reported in the text to `results/estimates/`.

## The irrelevant-control experiment

Each case family received a third version, in which a clinically irrelevant detail, matched to the cue in position
and form, takes the place of the cue. The original cases were written by physicians; the control details were drafted
by an LLM (Claude Opus 5.5) from written rules and physician-chosen examples of irrelevant details, and two physicians
from the study team judged all 100 clinically irrelevant to their case. The three frontier models answered all 1,200
items once with the baseline prompt (`prompts/system_baseline.txt`, `prompts/question_template.txt`) and the settings
in `config/control_request_settings.json`, which match the main runs except that up to 4,096 output tokens were
allowed.

`docs/control_experiment/` holds the automated rule checks of every control (`control_rule_checks.csv`) and the
drafts rewritten under the rules before the run (`control_revisions.csv`).

## Re-running the model queries

`code/collect/run_models.py` sends the case set to the models with the study's prompts, model identifiers and request settings; `--replicate` selects the models, case versions and answer orders of an original run. It needs an OpenRouter key in the environment variable `OPENROUTER_API_KEY` and incurs API costs.

Hosted models change over time, so new answers can differ from the released ones. The collector reads
`data/items.csv`; the requests of the irrelevant-control experiment can be rebuilt in the same way from
`data/control_items.csv` with the settings in `config/control_request_settings.json`.

```
python3 code/collect/run_models.py --replicate --panel core --prompt baseline --runs 1 --limit 5 --output requests.jsonl --dry-run
```

`--dry-run` writes the requests without sending them. Each output line carries the same columns as the released response files, together with the request sent. `python3 code/collect/run_models.py --help` lists all options.

## Privacy

Clinician participants:
- Identified only by random codes.
- The released files carry no names, contact details, timestamps or free text, and no individual response times.
- Training level is the only individual characteristic released.
- Region, specialty, practice setting and response times are available only as the aggregates in `clinician_aggregates.csv`, which the code uses for Table 1, Table S10 and Table S13.

Clinical adjudicators are identified only by the letters A to H.

Please do not attempt to re-identify any participant.

## License and citation

- Code: MIT (`LICENSE`).
- Data, prompts, tables and figures: CC BY 4.0 (`LICENSE-data`).
- To cite, see `CITATION.cff`.
