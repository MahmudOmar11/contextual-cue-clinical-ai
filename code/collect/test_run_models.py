"""Offline tests of the collector's re-ask rule, with a stand-in for the API. Run: python3 code/collect/test_run_models.py"""
import run_models

OPTIONS = {"A": "Malaria", "B": "Upper respiratory tract infection", "C": "Hepatitis", "D": "Pneumonia"}


def fake_send(results):
    calls = iter(results)
    return lambda payload, api_key, timeout: next(calls)


def ok(text, attempts=1):
    return {"status": "ok", "attempts": attempts, "response": {}, "raw_output": text}


def failed(attempts=3):
    return {"status": "http_error", "attempts": attempts, "error_code": 503, "error": "unavailable"}


def test_readable_first_response_is_kept_without_reask():
    run_models.send = fake_send([ok('{"selected_option":"B"}')])
    result = run_models.send_with_reask({}, "", 1, OPTIONS)
    assert result["raw_output"] == '{"selected_option":"B"}' and not result["requested_again"] and result["api_calls"] == 1


def test_readable_non_json_reask_replaces_an_unreadable_first_response():
    run_models.send = fake_send([ok("I am not sure."), ok("So the answer is D.")])
    result = run_models.send_with_reask({}, "", 1, OPTIONS)
    assert result["raw_output"] == "So the answer is D." and result["requested_again"] and result["api_calls"] == 2


def test_first_response_is_kept_when_neither_is_readable():
    run_models.send = fake_send([ok("Unclear."), ok("Still unclear.")])
    result = run_models.send_with_reask({}, "", 1, OPTIONS)
    assert result["raw_output"] == "Unclear." and result["api_calls"] == 2


def test_calls_are_counted_across_both_requests():
    run_models.send = fake_send([failed(3), failed(3)])
    result = run_models.send_with_reask({}, "", 1, OPTIONS)
    assert result["status"] == "http_error" and result["api_calls"] == 6


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print("FAIL", name, exc)
    print(f"{len(tests) - failures} of {len(tests)} tests passed")
    raise SystemExit(1 if failures else 0)
