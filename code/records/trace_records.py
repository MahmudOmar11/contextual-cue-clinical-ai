"""Trace every analysed main-run response through the collection records, and compare output lengths with the limit.

For each of the 92,000 analysed records of the main runs: how many responses and API calls it took, whether the
analysed response was the last one received, and whether a response with a readable answer was ever passed over for
an analysed response without one. Then, for the analysed responses: how many exceeded the requested output limit,
by hosting service, and how many stopped at it. Writes results/estimates/trace_records.csv.
Run from the repository root: python3 code/records/trace_records.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "collect"))
from read_answer import read_answer  # noqa: E402

MAIN_RUNS = ["core_baseline", "core_base_rate", "core_rare_or_serious", "frontier_baseline", "frontier_base_rate",
             "frontier_rare_or_serious", "frontier_reasks"]
KEY = ["panel", "prompt", "run", "model_id", "family", "version", "answer_order"]


def main():
    rec = pd.read_csv(ROOT / "data/collection_records.csv.gz", keep_default_na=False, low_memory=False)
    rec = rec[rec.run_id.isin(MAIN_RUNS)].copy()
    items = pd.concat([pd.read_csv(ROOT / "data/items.csv"), pd.read_csv(ROOT / "data/items_earlier_wording.csv")])
    options = {(r.family, r.version, r.answer_order): {L: getattr(r, "option_" + L) for L in "ABCD"}
               for r in items.drop_duplicates(["family", "version", "answer_order"]).itertuples()}
    rec["readable"] = [r.status == "ok" and read_answer(r.raw_output, options[(r.family, r.version, r.answer_order)])[0] != ""
                       for r in rec.itertuples()]
    rec = rec.sort_values("timestamp_utc", kind="stable")
    g = rec.groupby(KEY, sort=False)
    per = g.agg(responses=("analysed", "size"), api_calls=("attempts", lambda s: pd.to_numeric(s).sum()),
                analysed_is_last=("analysed", lambda s: s.iloc[-1] == 1),
                analysed_readable=("readable", lambda s: bool(s[rec.loc[s.index, "analysed"] == 1].any())),
                other_readable=("readable", lambda s: bool(s[rec.loc[s.index, "analysed"] == 0].any())))
    assert len(per) == 92000
    kept = rec[rec.analysed == 1].copy()
    for c in ("max_tokens", "completion_tokens"):
        kept[c] = pd.to_numeric(kept[c], errors="coerce")
    over = kept[kept.completion_tokens > kept.max_tokens]
    at_limit = kept[kept.finish_reason == "length"]
    rows = [("records", len(per))]
    rows += [(f"records_with_{n}_responses", int(v)) for n, v in per.responses.value_counts().sort_index().items()]
    rows += [(f"records_with_{n}_api_calls", int(v)) for n, v in per.api_calls.value_counts().sort_index().items()]
    multi = per[per.responses > 1]
    rows += [("records_with_more_than_one_response_analysed_last", int(multi.analysed_is_last.sum())),
             ("readable_answer_passed_over", int((~per.analysed_readable & per.other_readable).sum())),
             ("outputs_over_requested_limit", len(over)),
             ("outputs_over_limit_with_valid_answer", int(over.readable.sum())),
             ("longest_output_characters", int(kept.raw_output.str.len().max())),
             ("outputs_stopped_at_limit", len(at_limit)),
             ("outputs_stopped_at_limit_with_valid_answer", int(at_limit.readable.sum()))]
    rows += [(f"outputs_over_limit_{provider}", int(n)) for provider, n in over.provider.value_counts().items()]
    out = pd.DataFrame(rows, columns=["estimate", "value"])
    path = ROOT / "results/estimates/trace_records.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
