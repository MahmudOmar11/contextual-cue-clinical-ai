"""Tests for read_answer.py. Run from the repository root: python3 code/collect/test_read_answer.py (or pytest)."""
from pathlib import Path

import pandas as pd

from read_answer import read_answer

ROOT = Path(__file__).resolve().parents[2]
OPTS = {"A": "Malaria", "B": "Upper respiratory tract infection", "C": "Hepatitis", "D": "Pneumonia"}


def check(text, options, want, rule=None):
    got, how, _ = read_answer(text, options)
    assert got == want, (text[:80], got, how, want)
    if rule:
        assert how == rule, (text[:80], how, rule)


def released(model_id, family, version, answer_order, prompt="baseline", run=1):
    """One released main-run response with the option texts of its question."""
    d = pd.read_csv(ROOT / "data/model_responses.csv.gz", keep_default_na=False)
    r = d[(d.model_id == model_id) & (d.prompt == prompt) & (d.run == run) & (d.family == family)
          & (d.version == version) & (d.answer_order == answer_order)].iloc[0]
    items = pd.read_csv(ROOT / "data/items.csv")
    it = items[(items.family == family) & (items.version == version) & (items.answer_order == answer_order)].iloc[0]
    return r, {L: it["option_" + L] for L in "ABCD"}


def test_final_json_answer_after_reasoning_text():
    r, options = released("anthropic/claude-sonnet-4.6", 14, "cue", 2)
    check(r.raw_output, options, "B", "json_embedded")
    assert r.selected_option == "B"


def test_stated_choice_is_read_over_a_letter_in_a_diagnosis_name():
    r, options = released("openai/gpt-oss-20b", 13, "no_cue", 2)
    assert "hepatitis B" in r.raw_output
    check(r.raw_output, options, "D", "prose_letter")
    assert r.selected_option == "D"


def test_released_labels_match_the_reader():
    d = pd.read_csv(ROOT / "data/model_responses.csv.gz", keep_default_na=False)
    items = pd.read_csv(ROOT / "data/items.csv")
    options = {(r.family, r.version, r.answer_order): {L: getattr(r, "option_" + L) for L in "ABCD"}
               for r in items.itertuples()}
    early = pd.read_csv(ROOT / "data/items_earlier_wording.csv")
    options.update({(r.family, r.version, r.answer_order): {L: getattr(r, "option_" + L) for L in "ABCD"}
                    for r in early.itertuples()})
    read = [read_answer(r.raw_output, options[(r.family, r.version, r.answer_order)])[0] for r in d.itertuples()]
    assert (d.selected_option == pd.Series(read)).all()


def test_plain_and_fenced_json():
    check('{"selected_option":"B","selected_diagnosis":"Upper respiratory tract infection"}', OPTS, "B", "json")
    check('Here:\n```json\n{"selected_option": "d"}\n```', OPTS, "D", "json")


def test_last_json_object_wins():
    check('Draft {"selected_option":"A"} then final {"selected_option":"C"}', OPTS, "C", "json_embedded")


def test_escaped_json_inside_json():
    check('{"final":"{\\"selected_option\\":\\"B\\",\\"selected_diagnosis\\":\\"Colon cancer\\"}"}', OPTS, "B")


def test_truncated_after_complete_field():
    check('{"selected_option": "B",', OPTS, "B", "json_field")
    check('{"selected_option":"C","selected_diagnosis', OPTS, "C", "json_field")


def test_cut_right_after_the_letter():
    check('{   "selected_option": "B', OPTS, "B", "json_field_cut")
    check('{"selected_option ": "C", "selected...', OPTS, "C", "json_field")


def test_prose_mentions_of_the_field_and_bare_option():
    check("So viral bronchitis is most likely. Return JSON with selected_option D and diagnosis text.", OPTS, "D")
    check('So option B. Return JSON with selected_option \\"B\\" and selected_diagnosis.', OPTS, "B")
    check("IPF is most likely. Option B.\nReturn JSON", OPTS, "B")
    check("Option A is less likely than option C.", OPTS, "")


def test_truncated_before_letter_has_no_answer():
    check('{\n  "selected_option": "', OPTS, "")
    check('{\n  "selected', OPTS, "")


def test_template_echo_is_not_an_answer():
    check('{"selected_option": "A|B|C|D", "selected_diagnosis": "Viral infection"}', OPTS, "")
    check('{"selected_option": "A|B|C|D", "selected_diagnosis": "Pneumonia"}', OPTS, "D", "json_diagnosis")


def test_lone_letters_in_prose_never_count():
    check("This is a viral illness in a young man with hepatitis B exposure risk.", OPTS, "")
    check("A 22-year-old man has fever.", OPTS, "")


def test_prose_choices():
    check("Malaria is unlikely. So answer B. Return JSON.", OPTS, "B", "prose_letter")
    check("So the answer is (D).", OPTS, "D", "prose_letter")
    check("Final answer: **C**", OPTS, "C", "prose_letter")
    check("I'd pick pneumonia. Return JSON.", OPTS, "D", "prose_name")
    check("So choose pneumonia.", OPTS, "D", "prose_name")


def test_bare_letter_after_so():
    check("So likely pneumonia. So D. Pneumonia. Return JSON.", OPTS, "D", "prose_letter")
    check("So A patient like this needs testing.", OPTS, "")


def test_last_statement_wins_and_negation_skipped():
    check("I might choose A. On reflection I choose D.", OPTS, "D")
    check("I would not choose A. Choose B.", OPTS, "B")
    check("We should not select A here.", OPTS, "")


def test_questions_are_not_answers():
    check("The most probable diagnosis: Pneumonia? But the question is vague.", OPTS, "")
    check("Should I choose A? Probably not.", OPTS, "")


def test_article_after_verb_is_not_a_letter():
    check("So choose a diagnosis that fits.", OPTS, "")
    check("Answer: A patient like this needs testing.", OPTS, "")


def test_letter_only_and_no_answer():
    check("B", OPTS, "B", "letter_only")
    check("(C).", OPTS, "C", "letter_only")
    check("[1]", OPTS, "")
    check('{\n  "selected_option":"  "\n}', OPTS, "")
    check("", OPTS, "", "empty")


if __name__ == "__main__":
    failures = 0
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print("FAIL", name, exc)
    print(f"{len(tests) - failures} of {len(tests)} tests passed")
    raise SystemExit(1 if failures else 0)
