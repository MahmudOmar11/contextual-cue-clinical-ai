"""Rebuild the released response files from the collection records.

Reads data/collection_records.csv.gz, keeps the responses marked as analysed, reads each answer from the raw output
with code/collect/read_answer.py, and writes the response files to an output folder (default rebuilt/):
  model_responses.csv.gz, explanation_responses.csv.gz, control_responses.csv.gz,
  control_responses_adjudicator_instruction.csv.gz
--check compares every rebuilt file with the released file in data/ and exits non-zero on any difference.
Run from the repository root: python3 code/records/build_response_files.py --check
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "code" / "collect"))
from read_answer import _json_objects, read_answer, selected_role  # noqa: E402

MAIN_RUNS = ["core_baseline", "core_base_rate", "core_rare_or_serious", "frontier_baseline", "frontier_base_rate",
             "frontier_rare_or_serious", "frontier_reasks"]
CONTROL_MODELS = ["anthropic/claude-opus-4.8", "openai/gpt-5.5", "google/gemini-3.1-pro-preview"]
GZ = {"method": "gzip", "mtime": 0}


def options_of(items):
    return {(r.family, r.version, r.answer_order): ({L: getattr(r, "option_" + L) for L in "ABCD"}, r.intended_option,
                                                    r.cue_associated_option) for r in items.itertuples()}


def with_answers(d, look):
    opts = [look[(r.family, r.version, r.answer_order)] for r in d.itertuples()]
    options = [o[0] for o in opts]
    d["selected_option"] = [read_answer(text, o)[0] for text, o in zip(d.raw_output, options)]
    d["selected_role"] = [selected_role(s, o[1], o[2]) for s, o in zip(d.selected_option, opts)]
    return d


def probabilities(text):
    """The four probabilities and the most likely letter of an adjudicator-instruction answer."""
    for obj in reversed(_json_objects(text)) if text else []:
        p = obj.get("probabilities")
        if isinstance(p, dict) and all(k in p for k in "ABCD"):
            return {k: int(round(float(p[k]))) for k in "ABCD"}, str(obj.get("most_likely", "")).strip().upper()
    raise ValueError("no probabilities in output")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=ROOT / "rebuilt")
    parser.add_argument("--check", action="store_true", help="compare with the released files in data/")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rec = pd.read_csv(DATA / "collection_records.csv.gz", keep_default_na=False, low_memory=False)
    rec = rec[rec.analysed == 1]
    items = pd.read_csv(DATA / "items.csv")
    look = options_of(items)
    look.update(options_of(pd.read_csv(DATA / "items_earlier_wording.csv")))  # same options and keys
    citems = pd.read_csv(DATA / "control_items.csv")
    clook = options_of(citems)
    cid = {(r.family, r.version, r.answer_order): r.item_id for r in citems.itertuples()}
    out = {}

    main_runs = rec[rec.run_id.isin(MAIN_RUNS)].copy()
    main_runs["release_row"] = main_runs.release_row.astype(int)
    m = with_answers(main_runs.sort_values("release_row"), look)
    out["model_responses.csv.gz"] = m[["panel", "model_id", "model_version_returned", "prompt", "run", "family", "version",
                                       "answer_order", "wording", "selected_option", "selected_role", "raw_output"]]

    expl = rec[rec.run_id == "explanation"].copy()
    expl["release_row"] = expl.release_row.astype(int)
    e = with_answers(expl.sort_values("release_row"), look)
    out["explanation_responses.csv.gz"] = e[["model_id", "model_version_returned", "prompt", "family", "version",
                                             "answer_order", "selected_option", "selected_role", "raw_output"]]

    def control(run_id):
        c = rec[rec.run_id == run_id].copy()
        c["item_id"] = [cid[(r.family, r.version, r.answer_order)] for r in c.itertuples()]
        c["model_rank"] = c.model_id.map({mid: i for i, mid in enumerate(CONTROL_MODELS)})
        return c.sort_values(["model_rank", "item_id"])

    c1 = with_answers(control("control_baseline_prompt"), clook)
    out["control_responses.csv.gz"] = c1[["model_id", "model_version_returned", "item_id", "family", "version",
                                          "answer_order", "selected_option", "selected_role", "raw_output"]]

    c2 = control("control_adjudicator_instruction")
    parsed = [probabilities(t) for t in c2.raw_output]
    for k in "ABCD":
        c2["p_" + k] = [p[k] for p, _ in parsed]
    c2["selected_option"] = [most for _, most in parsed]
    keys = [clook[(r.family, r.version, r.answer_order)] for r in c2.itertuples()]
    c2["selected_role"] = [selected_role(s, k[1], k[2]) for s, k in zip(c2.selected_option, keys)]
    c2["p_intended"] = [p[k[1]] for (p, _), k in zip(parsed, keys)]
    c2["p_cue_associated"] = [p[k[2]] for (p, _), k in zip(parsed, keys)]
    out["control_responses_adjudicator_instruction.csv.gz"] = c2[[
        "model_id", "model_version_returned", "item_id", "family", "version", "answer_order", "p_A", "p_B", "p_C", "p_D",
        "selected_option", "selected_role", "p_intended", "p_cue_associated", "raw_output"]]

    failures = 0
    for name, df in out.items():
        df = df.reset_index(drop=True)
        df.to_csv(args.output / name, index=False, compression=GZ)
        if args.check:
            rebuilt = pd.read_csv(args.output / name, keep_default_na=False)
            released = pd.read_csv(DATA / name, keep_default_na=False)
            same = rebuilt.shape == released.shape and (rebuilt.astype(str).values == released.astype(str).values).all()
            failures += not same
            print(f"{'same' if same else 'DIFFERENT'}: {name} ({len(rebuilt)} rows)")
    if args.check and failures:
        sys.exit(f"{failures} file(s) differ from the released data")


if __name__ == "__main__":
    main()
