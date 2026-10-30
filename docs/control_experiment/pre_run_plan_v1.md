# Pre-run plan, version 1 (28 September 2026)

*The written plan of the irrelevant-control experiment, as released: personal names, internal file and package names and references to the review process are removed, and file references point to this repository; the design, hypotheses, settings and analysis plan are unchanged.*

**Title.** Do language models move toward a cue-associated diagnosis when an irrelevant detail is added? A matched irrelevant-control experiment with model probability estimates on the physicians' scale.

**Status.** Written 28 September 2026 before any model call. Nothing in this document is based on model output from this experiment.

**Context.** The question is whether the cue effect reflects clinical relevance (the added detail genuinely raises the probability of the cue-associated diagnosis) or mere salience (any added detail moves the models). The physician adjudication used a 100-point probability allocation; the June model runs recorded only a single selected diagnosis. This experiment adds a matched irrelevant control version to every case family and asks the models for the same 100-point allocation the physicians gave.

## 1. Hypotheses

H1 (cue effect on the probability scale). Across models, adding the original cue raises the probability points the model assigns to the cue-associated diagnosis relative to the no-cue version (cue minus no cue greater than 0).

H2 (relevance versus salience). Adding a matched irrelevant detail raises the probability points on the cue-associated diagnosis much less than the cue does (cue minus control greater than 0). A control shift close to zero indicates that the models respond to the content of the cue. A control shift comparable to the cue shift indicates that any added detail moves the models.

H3 (scale comparison with physicians). The model cue shift exceeds the physicians' cue shift on the same 100-point scale. The eight adjudicators moved 5.93 points (probability) and 5.30 points (first choice) toward the cue-associated diagnosis. This is a directional expectation; the estimate and its interval will be reported whatever the direction.

Secondary. The same three questions for the first-choice outcome (whether the cue-associated diagnosis is named as the single most probable). Change in probability on the intended diagnosis is reported as a descriptive complement.

## 2. Design

- Within-family design: each of the 100 case families is presented in three versions: no cue (the frozen base text), cue (the frozen original cue text) and control (the base text plus one matched detail chosen to be clinically irrelevant to all four options). Every version is presented in the same four frozen option orders. 100 families x 3 versions x 4 orders = 1,200 items.
- Texts for the no-cue and cue versions are the exact frozen presentations the adjudicators saw, reproduced hash for hash. The control texts are those written on 28 September; the option order for the control version is the no-cue order for the same order seed.
- Each item is sent once to each of 26 models. No repeats. Temperature 0 and seed 42 are sent on every request; some providers do not honour them (recorded per model).
- Unit of analysis: model by family. Orders are averaged first, then families receive equal weight within a model, then models receive equal weight.

## 3. Models

The 22 unique models from the June runs (20 Core-20 models plus GPT-5.5 and Claude Opus 4.8; Gemini 3.1 Pro Preview appears in both June panels and counts once) and four current models chosen from the live OpenRouter listing on 28 September 2026: anthropic/claude-sonnet-5, openai/gpt-5.6-terra, google/gemini-3.8-flash and deepseek/deepseek-v4.1-flash. Exact identifiers and the returned snapshot strings from June are in `config/models.json`. The returned model string of every response is stored so version drift since June can be reported.

## 4. Prompt and decoding

The instructions are the adjudicators' instruction screen and case screen with minimal adaptation: the same standard (the most probable diagnosis in routine practice, not the most dangerous or the one most important to rule out), 100 points across the four options, and one single most likely diagnosis. Output is JSON only: {"probabilities": {"A": int, "B": int, "C": int, "D": int}, "most_likely": "A"}. The full text is in `prompts/system_adjudicator_instruction.txt` and `prompts/question_template_adjudicator_instruction.txt`.

Decoding: temperature 0, seed 42, max_tokens 4096, response_format json_object. Hidden reasoning is disabled (effort none, excluded from output) wherever the provider accepts it. Four June models do not accept it (gpt-oss-20b, gpt-oss-120b, gemini-3.5-flash, gemini-3.1-pro-preview) and run at the provider default, which is recorded. If a provider rejects the reasoning field with an HTTP 400, the request is retried once without it and the fallback is recorded on that row.

## 5. Outcomes

Per item: (a) probability points assigned to the cue-associated diagnosis, (b) whether the cue-associated diagnosis is named as the most likely (0 or 100 percentage points), and the same two quantities for the intended diagnosis. Probabilities whose four values total 98 to 102 are rescaled to 100; other totals are unparsable.

Per model and family: version means over the four orders, then the contrasts cue minus no cue, control minus no cue and cue minus control for each outcome.

## 6. Analysis plan

- Point estimates: family-equal-weight means per model, then the mean over models.
- Uncertainty: crossed family-by-model bootstrap, 10,000 draws, multinomial resampling weights, fixed seed 20260928, percentile 95 percent intervals. Draws are paired across outcomes and contrasts within a scope. Families and models are resampled as convenience collections, not as random samples from defined populations; intervals are conditional on the observed prompt and settings.
- Physician comparison: for each family the eight physicians' cue shift (mean probability on the cue-associated role, cue minus no cue; and the share naming it first) is taken from the adjudicators' ratings and held fixed. The model cue shift minus the physician cue shift, and the model control shift minus the physician cue shift, are bootstrapped over families and models with the physicians conditional on their observed means.
- Scopes: all 100 families (primary for H1 and H3). For H2 and every control contrast, if the blinded physician validation of the controls is performed, the primary scope is the families whose control was accepted by both physicians (rated 0 for relevance by both) and the all-100 result is a sensitivity. If the validation is not performed, the all-100 scope is primary for H2 and the controls stand on how they were written (one detail per family chosen to be unrelated to all four options); this choice is declared before the first model call. If the validation is performed after the run, the accepted-controls results are recomputed with the same code and reported beside the all-100 results.
- Inference: H1 and H2 are supported when the lower interval bound of the corresponding contrast is above 0. H3 is supported when the lower bound of the model-minus-physician difference is above 0. No p values, no multiplicity adjustment; every prespecified estimate is reported.
- Sensitivity: first choice defined by the strict maximum of the stated probabilities instead of the named most_likely; per-model results; results with and without models that ran at provider-default reasoning; results with and without models whose returned snapshot string differs from June.
- Exploratory, labelled as such: by cue type (the cue types of Supplementary Table S18), by model family and tier, and June-panel versus current models.

## 7. Exclusions and missing data

- An item is unparsable if the model gives no usable JSON after the first reply and two corrective re-asks. Unparsable items are excluded from the estimates and counted per model. A model with more than 10 percent unparsable items in any version is reported separately and excluded from the primary pooled estimate (its results remain in the per-model table).
- Requests that fail after the retry policy (429 and 5xx with exponential backoff, up to five retries) are resumed in a later pass; a row is only final when it is ok or unparsable.
- Controls later rated as relevant by either of the two blinded physicians are excluded from the accepted-controls scope. No control text is edited after the run; a rewritten control would require a new, separately declared run.
- most_likely inconsistent with the stated probabilities is kept and reported, not excluded.

## 8. Sample size

Fixed by design: 100 families, 26 models, 31,200 first-pass calls. No look at the results will be used to change the design. A pilot of two families (24 items x 26 models = 624 calls) checks JSON validity and token usage; pilot rows are part of the final data set because the design and prompt are unchanged.

## 9. Deviations policy

Any change to the prompt, decoding settings, model list, items or analysis after this plan is written down in a dated deviations section of the README before it is applied, with the reason. Changes forced by providers (a model withdrawn, a parameter rejected) are recorded as deviations and do not alter the estimands. The analysis code is frozen before the run; bug fixes are recorded with the original and corrected output.

## 10. Data and code availability

The items, the prompt, the model list, the code, every response row with the raw provider response, usage and timestamps, and the analysis outputs will be released.
