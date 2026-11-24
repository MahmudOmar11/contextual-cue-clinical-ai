"""Agreement between the answer reader and a physician's blind reading of 229 model outputs.

data/parser_audit_labels.csv lists the outputs: all 133 whose label changed with the current reader and 96 unchanged
outputs drawn at random within the harder reading types. This script reads each output again with
code/collect/read_answer.py, checks that the result equals the reader's reading in the file and the released label,
and computes agreement with the physician's reading, overall and by sample.
Writes results/estimates/14_parser_audit.csv.
Run from the repository root: python3 code/analysis/14_parser_audit.py
"""
import sys

import pandas as pd

from common import DATA, ESTIMATES, ROOT

sys.path.insert(0, str(ROOT / "code" / "collect"))
from read_answer import read_answer  # noqa: E402

KEY = ["panel", "model_id", "prompt", "run", "family", "version", "answer_order"]
ITEM = ["family", "version", "answer_order"]

audit = pd.read_csv(DATA / "parser_audit_labels.csv", keep_default_na=False)
responses = pd.read_csv(DATA / "model_responses.csv.gz", keep_default_na=False, low_memory=False)
d = audit.merge(responses[KEY + ["wording", "selected_option", "raw_output"]], on=KEY, how="left", validate="one_to_one")
assert len(d) == len(audit) and d.raw_output.notna().all()
items = {w: pd.read_csv(DATA / f).set_index(ITEM) for w, f in [("final", "items.csv"),
                                                              ("earlier", "items_earlier_wording.csv")]}
read = []
for r in d.itertuples():
    row = items[r.wording].loc[(r.family, r.version, r.answer_order)]
    option, rule, _ = read_answer(r.raw_output, {L: row[f"option_{L}"] for L in "ABCD"})
    read.append((option or "none", rule))
assert [x[0] for x in read] == d.reader_reading.tolist(), "reader reading differs"
assert [x[1] for x in read] == d.reading_type.tolist(), "reading type differs"
assert (d.selected_option.replace("", "none") == d.reader_reading).all(), "released label differs"

rows = []
for name, g in [("all", d), ("changed", d[d["sample"] == "changed"]), ("unchanged", d[d["sample"] == "unchanged"])]:
    agree = int((g.reader_reading == g.physician_reading).sum())
    rows.append({"estimate": f"agreement_{name}_pct", "value": round(100 * agree / len(g), 1), "ci_low": None,
                 "ci_high": None, "n": len(g)})
    rows.append({"estimate": f"agreement_{name}_count", "value": agree, "ci_low": None, "ci_high": None, "n": len(g)})
pd.DataFrame(rows, dtype=object).to_csv(ESTIMATES / "14_parser_audit.csv", index=False)
print("wrote estimates/14_parser_audit.csv")
