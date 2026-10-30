# Changes

## 1.2.0 (30 October 2026)

Answers and results
- Every selected option is now read from the raw output by `code/collect/read_answer.py`, one ordered rule: the final
  structured answer, else the last explicitly stated choice; a lone letter in prose never counts. This changed 133 of
  the 92,000 main-run labels (124 to another option, 7 to no valid answer, 2 from no valid answer), all in the core
  panel. The earlier labels of the core-panel runs came from the reader used during collection, which fell back to the
  first standalone letter whenever an output was not JSON alone. The explanation run and the irrelevant-control
  experiment are unchanged by the new reader.
- `code/collect/test_read_answer.py` checks the rule on worked cases and checks every released label against it.
- All tables, figures and estimates are regenerated from the new labels.
- The denominators of each analysis are stated in the README, and the analyses that count valid answers only are
  repeated with every response.
- New analyses: the comparisons without the audited case families, the comparisons on identical case text, the
  all-attempt versions, and the second arm of the irrelevant-control experiment.
- Figure panel letters are bold.

Collection
- `code/collect/run_models.py`: an output without a readable answer is requested again once, and the response with a
  readable answer is kept (the first, if both have one); `api_calls` counts every call of both requests.
  `code/collect/test_run_models.py` tests this offline.

New data and records
- `data/collection_records.csv.gz`: every response received in every run, with its time, the model version and hosting
  service returned, token usage, finish reason and status; `docs/run_manifest.csv` lists the runs.
- `code/records/build_response_files.py` rebuilds the released response files from the collection records, and
  `code/records/trace_records.py` traces every analysed response through them.
- `data/pilot_responses.csv.gz`: the outputs of the pilot used to choose the 21 clinician-form families.
- `data/items_earlier_wording.csv` and the `wording` column of `data/model_responses.csv.gz`: the case text each
  response received.
- `data/control_responses_adjudicator_instruction.csv.gz`, with its prompts in `prompts/`: the second arm of the
  irrelevant-control experiment.
- `docs/control_experiment/`: the pre-run plans of the irrelevant-control experiment.

## 1.1.0 (3 October 2026)

- The irrelevant-control experiment and the clinical adjudicators' ratings.

## 1.0.0 (30 September 2026)

- Code and data for the study.
