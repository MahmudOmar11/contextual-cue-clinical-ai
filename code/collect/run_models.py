"""Send the study questions to the models through the OpenRouter API.

Each request pairs one question from the case set in data/items.csv with one
model entry from config/models.json, under one prompt (the system prompts in
prompts/). The
request settings, including the output-token limits, come from
config/request_settings.json. Every response is written as one JSON line with
the request payload, the raw output and the parsed answer. Requests already in
the output file are skipped, so an interrupted run can be resumed with the same
command; requests that failed are recorded with their status and are not retried
on resume.

The questions must be named, in one of two ways:
  --replicate    send the questions of an original run: the models, case
                 versions and answer orders recorded in
                 data/model_responses.csv.gz for the chosen panel, prompt and
                 runs, or in data/explanation_responses.csv.gz with --explanation
  --items FILE   send the listed item_ids (one per line, or a CSV file with an
                 item_id column) to every model of the panel, or to the
                 explanation-run models with --explanation
--dry-run writes the payloads without contacting the API.

Examples (from the repository root):
  python3 code/collect/run_models.py --replicate --panel core --prompt baseline \\
      --runs 1 2 3 --dry-run --output requests_core_baseline.jsonl
  OPENROUTER_API_KEY=... python3 code/collect/run_models.py --replicate \\
      --panel frontier --prompt base_rate --models openai/gpt-5.5 \\
      --output responses_frontier_base_rate.jsonl
  OPENROUTER_API_KEY=... python3 code/collect/run_models.py --replicate \\
      --explanation --prompt rare_or_serious --output responses_explanation.jsonl

Standard library only.
"""
import argparse
import concurrent.futures
import csv
import gzip
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from parse_output import parse_response, selected_role

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPTS = ["baseline", "base_rate", "rare_or_serious"]

MAX_RETRIES = 2  # transient errors (HTTP 429 or 5xx, timeouts) are retried up to twice
RETRYABLE_HTTP = {429, 500, 502, 503, 504}

csv.field_size_limit(10**8)


def read_text(path):
    return path.read_text(encoding="utf-8").rstrip("\n")


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_item_ids(path):
    """item_ids from a text file (one per line) or a CSV file with an item_id column."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    col = 0
    if rows and "item_id" in rows[0]:
        col = rows[0].index("item_id")
        rows = rows[1:]
    return [r[col].strip() for r in rows if r[col].strip()]


# ---------------------------------------------------------------- requests

def build_payload(entry, item, system_prompt, template, json_shape, max_tokens, settings):
    """The request body, with the reasoning-off request only for models that accept it."""
    user_prompt = template.format(
        vignette=item["vignette"],
        option_A=item["option_A"],
        option_B=item["option_B"],
        option_C=item["option_C"],
        option_D=item["option_D"],
        json_shape=json_shape,
    )
    payload = {
        "model": entry["openrouter_id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": settings["temperature"],
        "max_tokens": max_tokens,
        "seed": settings["seed"],
        "response_format": settings["response_format"],
    }
    if entry["reasoning_off_requested"]:
        payload["reasoning"] = settings["reasoning_request_when_accepted"]
    return payload


def plan_requests(args, settings):
    """List of (model entry, item, run) in the order they are sent."""
    entries = json.loads(read_text(ROOT / "config" / "models.json"))["models"]
    items = {r["item_id"]: r for r in read_csv(ROOT / "data" / "items.csv")}
    item_of = {(r["family"], r["version"], r["answer_order"]): r["item_id"] for r in items.values()}
    key = lambda r: item_of[(r["family"], r["version"], r["answer_order"])]
    if args.explanation:
        explanation = settings["explanation_run"]
        if args.prompt not in explanation["prompts"]:
            sys.exit(f"The explanation run used the prompts {', '.join(explanation['prompts'])}.")
        entry_by_id = {}
        for m in entries:
            if m["openrouter_id"] in explanation["models"]:
                entry_by_id.setdefault(m["openrouter_id"], m)
        runs = list(range(1, explanation["runs"] + 1))
        scope = "the explanation run"
    else:
        entry_by_id = {m["openrouter_id"]: m for m in entries if m["panel"] == args.panel}
        runs = args.runs
        scope = f"the {args.panel} panel"

    if args.replicate:
        if args.explanation:
            source = [r for r in read_csv(ROOT / "data" / "explanation_responses.csv.gz") if r["prompt"] == args.prompt]
            pairs = [(r["model_id"], key(r), 1) for r in source]
        else:
            wanted_runs = {str(r) for r in runs}
            source = [
                r for r in read_csv(ROOT / "data" / "model_responses.csv.gz")
                if r["panel"] == args.panel and r["prompt"] == args.prompt and r["run"] in wanted_runs
            ]
            pairs = [(r["model_id"], key(r), int(r["run"])) for r in source]
    else:
        item_ids = read_item_ids(args.items)
        unknown = [i for i in item_ids if i not in items]
        if unknown:
            sys.exit(f"Not in data/items.csv: {', '.join(unknown[:10])}")
        pairs = [(m, i, run) for run in runs for m in entry_by_id for i in item_ids]

    if args.models:
        wanted = set(args.models)
        pairs = [p for p in pairs if p[0] in wanted]
        missing = sorted(wanted - set(entry_by_id))
        if missing:
            sys.exit(f"Not in {scope}: {', '.join(missing)}")
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        sys.exit("No requests match these options.")
    return [(entry_by_id[m], items[i], run) for m, i, run in pairs]


# ---------------------------------------------------------------- sending

def post(payload, api_key, timeout):
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def part_text(part):
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    for key in ("text", "content"):
        value = part.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return content_text(value)
    return ""


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(t for t in (part_text(p) for p in content) if t)
    if isinstance(content, dict):
        return part_text(content)
    return ""


def response_text(response):
    """The text of the first choice's message."""
    if isinstance(response, list):
        for x in response:
            text = response_text(x)
            if text:
                return text
        return content_text(response)
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message", {})
    if isinstance(message, dict):
        return content_text(message.get("content", ""))
    return content_text(message)


def is_timeout(exc):
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout))


def send(payload, api_key, timeout):
    """One request, retrying transient errors up to MAX_RETRIES times."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = post(payload, api_key, timeout)
            text = response_text(response)
            return {"status": "ok", "attempts": attempt + 1, "response": response,
                    "raw_output": text, "parsed": parse_response(text)}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if attempt < MAX_RETRIES and exc.code in RETRYABLE_HTTP:
                time.sleep(min(30.0, 2.0 * 2 ** attempt))
                continue
            return {"status": "http_error", "attempts": attempt + 1, "error_code": exc.code, "error": body}
        except Exception as exc:  # noqa: BLE001
            if attempt < MAX_RETRIES and is_timeout(exc):
                time.sleep(min(30.0, 2.0 * 2 ** attempt))
                continue
            return {"status": "error", "attempts": attempt + 1, "error": repr(exc)}


def parsed_as_json(result):
    return result["status"] == "ok" and result["parsed"]["parse_status"] == "json"


def send_with_parse_retry(payload, api_key, timeout):
    """Send; if the output did not parse as JSON, request it again once and keep the better result."""
    first = send(payload, api_key, timeout)
    if parsed_as_json(first):
        return {**first, "requested_again": False}
    second = send(payload, api_key, timeout)
    better = parsed_as_json(second) or (second["status"] == "ok" and first["status"] != "ok")
    return {**(second if better else first), "requested_again": True}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--panel", choices=["core", "frontier"], default="core",
                        help="model panel (not used with --explanation)")
    parser.add_argument("--prompt", choices=PROMPTS, required=True)
    parser.add_argument("--runs", type=int, nargs="+", default=[1], help="baseline runs 1-3; other prompts run 1")
    parser.add_argument("--explanation", action="store_true", help="the explanation run (core-panel models)")
    parser.add_argument("--replicate", action="store_true", help="re-send the questions of the original run")
    parser.add_argument("--items", type=Path, help="file of item_ids to send")
    parser.add_argument("--models", nargs="+", help="OpenRouter model ids to include (default: all)")
    parser.add_argument("--limit", type=int, help="send only the first N requests")
    parser.add_argument("--output", type=Path, required=True, help="JSON lines file to write")
    parser.add_argument("--dry-run", action="store_true", help="write the payloads without contacting the API")
    parser.add_argument("--workers", type=int, default=1, help="models processed in parallel")
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between requests of one worker")
    parser.add_argument("--timeout", type=int, default=90, help="seconds per request")
    args = parser.parse_args()
    if args.replicate and args.items:
        parser.error("use either --replicate or --items, not both")
    if not args.replicate and not args.items:
        parser.error("say what to run: --replicate (with --panel, --prompt and --runs, or --explanation) "
                     "or --items FILE")

    settings = json.loads(read_text(ROOT / "config" / "request_settings.json"))
    shapes = json.loads(read_text(ROOT / "prompts" / "json_shapes.json"))
    prefix = "system_explanation_" if args.explanation else "system_"
    system_prompt = read_text(ROOT / "prompts" / f"{prefix}{args.prompt}.txt")
    template = read_text(ROOT / "prompts" / "question_template.txt")
    json_shape = shapes["explanation_run" if args.explanation else "main_runs"]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not args.dry_run and not api_key:
        sys.exit("Set OPENROUTER_API_KEY, or use --dry-run.")

    requests = plan_requests(args, settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.dry_run:
        if args.output.exists() and args.output.stat().st_size and '"dry_run"' not in args.output.read_text(encoding="utf-8"):
            sys.exit(f"{args.output} already holds responses; choose another --output for a dry run.")
        args.output.write_text("", encoding="utf-8")
    elif args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            done = {json.loads(line)["request_key"] for line in f if line.strip()}

    lock = threading.Lock()
    written = [0]

    def process(batch):
        with args.output.open("a", encoding="utf-8") as out:
            for entry, item, run in batch:
                run_label = "explanation" if args.explanation else f"run{run}"
                key = f"{entry['panel']}|{args.prompt}|{run_label}|{entry['openrouter_id']}|{item['item_id']}"
                if key in done:
                    continue
                max_tokens = (settings["explanation_run"]["max_tokens"] if args.explanation
                              else settings["max_tokens"][entry["panel"]])
                payload = build_payload(entry, item, system_prompt, template, json_shape, max_tokens, settings)
                row = {
                    "request_key": key,
                    "panel": entry["panel"],
                    "prompt": args.prompt,
                    "run": run,
                    "explanation": args.explanation,
                    "model_id": entry["openrouter_id"],
                    "item_id": item["item_id"],
                    "family": int(item["family"]),
                    "version": item["version"],
                    "answer_order": int(item["answer_order"]),
                    "payload": payload,
                }
                if args.dry_run:
                    row["status"] = "dry_run"
                else:
                    result = send_with_parse_retry(payload, api_key, args.timeout)
                    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                    row.update(result)
                    if result["status"] == "ok":
                        option = result["parsed"]["selected_option"]
                        row["model_version_returned"] = result["response"].get("model", "")
                        row["selected_option"] = option
                        row["selected_role"] = selected_role(option, item["intended_option"], item["cue_associated_option"])
                with lock:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    written[0] += 1
                if not args.dry_run:
                    time.sleep(args.pause)

    if args.workers <= 1:
        process(requests)
    else:
        by_model = {}
        for r in requests:
            by_model.setdefault(r[0]["openrouter_id"], []).append(r)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in [pool.submit(process, batch) for batch in by_model.values()]:
                future.result()

    print(f"{len(requests)} requests planned; {written[0]} rows written to {args.output}")


if __name__ == "__main__":
    main()
