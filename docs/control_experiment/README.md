# Irrelevant-control experiment

- `pre_run_plan_v1.md` and `pre_run_plan_v2.md`: the written plan, version 1 (28 September 2026) and version 2
  (2 October 2026), both written before any model call. Version 2 was followed.
- `control_revisions.csv`: the control details rewritten under the rules before the run, with the reason for each.
- `control_rule_checks.csv`: the automated rule checks of every control detail.

The prompts of both arms are in `prompts/`, the request settings in `config/control_request_settings.json`, the
responses in `data/control_responses.csv.gz` (first arm) and `data/control_responses_adjudicator_instruction.csv.gz`
(second arm), and every response with its timing and token usage in `data/collection_records.csv.gz`.

Deviations from the plan:
- The blinded check of the controls by two physicians (plan, section 6) was replaced by a check in which two physicians
  from the study team each judged 50 of the controls; all 100 were judged clinically irrelevant.
- The second arm was begun on 2 October 2026 and completed on 28 October 2026 with the same code, items and prompt.
  Families 1 and 2 were asked again on 28 October and gave the same first choice as on 2 October in 96% to 100% of
  answers for each model.
