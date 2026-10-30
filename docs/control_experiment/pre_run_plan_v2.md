# Pre-run plan, version 2 (2 October 2026)

*The written plan of the irrelevant-control experiment, as released: personal names, internal file and package names and references to the review process are removed, and file references point to this repository; the design, hypotheses, settings and analysis plan are unchanged.*

**Title.** Do language models move toward a cue-associated diagnosis when an irrelevant detail is added? A matched irrelevant-control experiment.

**Status.** Version 2, written 2 October 2026 before any model call. It replaces version 1 of 28 September (`pre_run_plan_v1.md`). The changes from version 1:
- the controls are revised (41 of 100) under written rules;
- a baseline-prompt arm is added;
- intervals for a named model set hold the models fixed;
- the model set is declared below.

Nothing in this document is based on model output from this experiment. Before the first call, the overnight launcher writes a freeze record (UTC time and the SHA-256 of this document, the items, the prompts and the code).

**Choices declared on 2 October 2026:**
- the three frontier models;
- both arms;
- no blinded physician check of the controls before the run (it may be added later; see section 6).

**Context.** The question is whether the cue effect reflects clinical relevance (the added detail genuinely raises the probability of the cue-associated diagnosis) or salience (any added detail moves the models). This experiment adds a matched irrelevant control version to every case family.

## 1. Hypotheses

- **H1 (cue effect, replication).** Adding the original cue raises selection of the cue-associated diagnosis as the single most probable answer relative to the no-cue version (cue minus no cue greater than 0). Baseline arm.
- **H2 (relevance versus salience; primary).** The cue raises selection of the cue-associated diagnosis more than a matched irrelevant detail does (cue minus control greater than 0). Baseline arm. The control effect itself (control minus no cue) is reported with its interval:
  - a control effect close to zero means the models respond to the content of the cue;
  - a control effect comparable to the cue effect means any added detail moves them.
- **H3 (adjudicator arm).** The same contrasts on the physicians' 100-point probability scale, under the instruction the physicians received. The model cue shift exceeds the physicians' cue shift on the same families: the eight adjudicators moved 5.93 probability points and 5.30 first-choice points toward the cue-associated diagnosis. This is a directional expectation; the estimate is reported whatever its direction.
- **Secondary.** Selection of the intended diagnosis (accuracy) under each contrast, descriptive.

## 2. Design

- **Versions.** Within-family design: each of the 100 case families appears in three versions:
  - no cue: the frozen base text;
  - cue: the frozen original cue text;
  - control: the base text plus one matched irrelevant detail.

  Every version appears in the same four frozen option orders: 100 families x 3 versions x 4 orders = 1,200 items (`data/control_items.csv`). The no-cue and cue texts are the exact presentations the adjudicators saw, reproduced hash for hash (800 of 800). The control version uses the no-cue option order for each order seed.
- **Arms.** Each arm is a separate run into its own folder.
  - *Baseline arm* (`prompts/system_baseline.txt`): the June baseline prompt verbatim, read from the public repository (release v1.0.0). The model returns one selected option. This arm speaks to the cue effect reported in the paper.
  - *Adjudicator arm* (`prompts/system_adjudicator_instruction.txt`): the adjudicators' instruction and case screens, minimally adapted. The model distributes 100 probability points across the four diagnoses and names the single most likely one. The instruction to consider what is clinically common resembles the base-rate prompt, so cue effects are expected to be smaller in this arm.
  - **Arms run:** baseline, then adjudicator.
- **Repeats.** Each item is sent once to each model, with no repeats. Temperature 0 and seed 42 are sent on every request; some providers do not honour them (recorded per model).
- **Unit of analysis.** Model by family. Orders are averaged first, then families receive equal weight within a model, then models receive equal weight.

## 3. Models

**Declared set:** the three frontier models.

- **The three frontier models:** anthropic/claude-opus-4.8, openai/gpt-5.5 and google/gemini-3.1-pro-preview, as in the paper.
- **The 22 June models:** the 20 core models plus GPT-5.5 and Claude Opus 4.8; Gemini 3.1 Pro appears in both June panels and counts once.

Exact identifiers and the returned snapshot strings from June are in `config/models.json`. The returned model string of every response is stored, so version drift since June can be reported.

## 4. Prompt and decoding

**Baseline arm.** The June system prompt and question template, unchanged:

> "You are accurately and evidence-based answering a clinical question. Choose the single most probable diagnosis from the four options. Return structured JSON only as per instructions."

The output is `{"selected_option":"A|B|C|D","selected_diagnosis":"diagnosis text"}`. The selected option is read from the JSON object only. A letter is never taken from prose, so a reply without the JSON shape is re-asked.

**Adjudicator arm.** As in version 1. The output is `{"probabilities": {"A": int, ...}, "most_likely": "A"}`. Probabilities totalling 98 to 102 are rescaled to 100; other totals are unparsable.

**Decoding (both arms).**
- Temperature 0, seed 42, response_format json_object.
- max_tokens 4096. This is not the June caps of 512 and 1,024, so that models whose hidden reasoning cannot be switched off are not truncated.
- Hidden reasoning is turned off (effort none, excluded) wherever the provider accepts it. Gemini 3.1 Pro (and, in the 22-model set, gpt-oss-20b, gpt-oss-120b and gemini-3.5-flash) do not accept it and run at the provider default, as in June.
- If a provider rejects the reasoning field with HTTP 400, the request is retried once without it, and the fallback is recorded on the row.

## 5. Outcomes

**Per item.**
- Whether the cue-associated diagnosis is the selected (most likely) answer (0 or 100 percentage points).
- The same for the intended diagnosis.
- In the adjudicator arm, also the probability points on each.

**Per model and family.** Version means over the four orders, then the contrasts cue minus no cue, control minus no cue, and cue minus control.

## 6. The control texts

**How they are written.** One detail per family, placed where the cue sits and matched to it in form. The rules (2 October 2026):
- **R1 short:** the added detail is at most one word longer than the cue's.
- **R2 neutral:** no travel or residence, animal, environmental or occupational exposure, substance, diet or medication, immobility or prolonged sitting, posture, hand or voice use, or age cue.
- **R3 unrelated:** nothing linked to any of the four options. A family-history cue gets a benign, unrelated family fact (for example, a parent who wears hearing aids).
- **R4 same structure:** the control changes the case only where the cue does, or adds a detail. It never removes a word that the cue keeps.

**Revision and checks.** Version 1 drafted the controls on 28 September. On 2 October, 41 were rewritten so that all 100 meet the rules. Each change and its reason are in `docs/control_experiment/control_revisions.csv`. An automated check (`docs/control_experiment/control_rule_checks.csv`) tests every control for:
- length (R1);
- structure (R4);
- overlap with the words of any of the four options (R3);
- a screen of exposure, travel, animal, substance, sitting, hand-use, voice-use, age and disease terms (R2).

All 100 pass. A physician from the study team also checks a sample by hand.

**Optional blinded physician check (declared: not before the run).**
- **Raters:** two physicians who did not write the cases and have not seen model results.
- **What they rate:** the 100 controls and 100 original cues, shuffled and unlabelled, each beside its base case. The question is whether the variant changes how likely any of the four diagnoses is: 0 none, 1 slight, 2 substantial.
- **Acceptance:** a control is accepted only if both rate it 0.
- **Scope:** if the check is done, the accepted-controls scope is primary for H2 and the all-100 result is a sensitivity analysis. If it is not done, all 100 is primary. If it is done after the run, the accepted-controls results are recomputed with the same code and reported beside the all-100 results.
- **No edits after the run:** no control text is edited after the run; a rewritten control would need a new, separately declared run.

## 7. Analysis plan

- **Point estimates:** family-equal-weight means per model, then the mean over models.
- **Uncertainty:** 10,000 bootstrap draws, seed 20260928, percentile 95% intervals, paired across outcomes and contrasts within a scope.
  - For the three frontier models, a named set, the models are held fixed and families are resampled, as in the paper.
  - For the 22-model set, families and models are resampled together (crossed).

  Intervals are conditional on the prompt and settings.
- **Physician comparison:** each family's adjudicator cue shift is held at its observed value. The model cue shift minus the adjudicator cue shift is resampled over families. These intervals condition on the eight adjudicators and are narrower than the paper's, which also resample the adjudicators. The paper's physician estimate itself is quoted from the paper.
- **Inference:** a hypothesis is supported when the lower bound of its contrast is above 0. No p values and no multiplicity adjustment; every prespecified estimate is reported.
- **Sensitivity analyses:**
  - per-model results;
  - adjudicator arm only: first choice taken as the strict maximum of the stated probabilities;
  - with and without Gemini 3.1 Pro, which ran at provider-default reasoning;
  - with and without models whose returned snapshot string differs from June.
- **Exploratory, labelled as such:** results by cue type (the cue types of Supplementary Table S18).

## 8. Exclusions and missing data

- **Unparsable items:** an item is unparsable if the model gives no usable JSON after the first reply and two corrective re-asks. Unparsable items are excluded from the estimates and counted per model.
- **Models with many unparsable items:** a model with more than 10% unparsable items in any version is reported separately. In the 22-model set it is excluded from the pooled estimate. In the frontier set it is reported per model and the pooled estimate is shown with and without it.
- **Failed requests:** requests that fail after the retry policy (429 and 5xx with backoff, up to five retries) are resumed in a later pass. A row is final only when it is ok or unparsable.

## 9. Sample size

Fixed by design.
- **Frontier set:** 3,600 calls per arm.
- **22-model set:** 26,400 calls per arm.

**Pilot.** A pilot of two families (72 calls per arm for the frontier set) checks JSON validity and token usage. Pilot rows are part of the final data, because the design and prompt are unchanged.

**No design changes.** No look at the results will be used to change the design.

## 10. Deviations policy

Any change to the prompt, decoding settings, model list, items or analysis after this plan is written in a dated deviations section of the README before it is applied, with the reason. Changes forced by providers (a model withdrawn, a parameter rejected) are recorded as deviations and do not alter the estimands. The analysis code is frozen before the run; bug fixes are recorded with the original and corrected output.

## 11. Data and code availability

The following will be released:
- the items (with content hashes);
- the prompts of both arms;
- the model list;
- the collection and analysis code;
- the control rules, revision log and checks;
- every response row, with the raw provider response, usage and timestamps;
- the analysis outputs.
