"""Parse a model's raw text output into the selected option, explanation and confidence."""
import json
import re

OPTIONS = {"A", "B", "C", "D"}


def normalized_confidence(value):
    """Confidence as an integer from 0 to 100, or None if absent or not numeric."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        confidence = int(round(float(value)))
    elif isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if not match:
            return None
        confidence = int(round(float(match.group(0))))
    else:
        return None
    return max(0, min(100, confidence))


def decode_json(content):
    """Decode the whole text as JSON, or else the first JSON object inside it."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", content, 0)
    data, _ = json.JSONDecoder().raw_decode(content[start:])
    return data


def parse_response(content):
    """Return selected_option, selected_diagnosis and parse_status, plus
    short_explanation and confidence when the output provides them.

    Order of attempts: JSON (also inside a code fence), then the named fields
    by regular expression, then a lone option letter.
    """
    parsed = {"selected_option": "", "selected_diagnosis": "", "parse_status": "unparsed"}
    if not isinstance(content, str) or not content.strip():
        return parsed
    content = content.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        content = fence.group(1)
    try:
        data = decode_json(content)
        if isinstance(data, list):
            data = next((x for x in data if isinstance(x, dict) and x.get("selected_option")), {})
        if isinstance(data, dict):
            option = str(data.get("selected_option", "")).strip().upper()
            if option in OPTIONS:
                parsed.update({
                    "selected_option": option,
                    "selected_diagnosis": str(data.get("selected_diagnosis", "")).strip(),
                    "parse_status": "json",
                })
                explanation = str(
                    data.get("short_explanation") or data.get("brief_explanation") or data.get("explanation") or ""
                ).strip()
                if explanation:
                    parsed["short_explanation"] = explanation
                confidence = normalized_confidence(data.get("confidence"))
                if confidence is not None:
                    parsed["confidence"] = confidence
                return parsed
    except json.JSONDecodeError:
        pass

    option_field = re.search(r'"selected_option"\s*:\s*"([ABCD])', content, flags=re.IGNORECASE)
    diagnosis_field = re.search(r'"selected_diagnosis"\s*:\s*"([^"\r\n]+)', content, flags=re.IGNORECASE)
    if option_field:
        diagnosis = diagnosis_field.group(1).strip() if diagnosis_field else ""
        parsed.update({
            "selected_option": option_field.group(1).upper(),
            "selected_diagnosis": diagnosis,
            "parse_status": "regex_fields" if diagnosis else "regex_option",
        })
        return parsed

    letter = re.search(r"\b([ABCD])\b", content.upper())
    if letter:
        parsed.update({"selected_option": letter.group(1), "parse_status": "regex_option"})
    return parsed


def selected_role(option, intended_option, cue_associated_option):
    """Map a parsed option letter to intended, cue_associated, other or no_valid_answer."""
    if option not in OPTIONS:
        return "no_valid_answer"
    if option == intended_option:
        return "intended"
    if option == cue_associated_option:
        return "cue_associated"
    return "other"
