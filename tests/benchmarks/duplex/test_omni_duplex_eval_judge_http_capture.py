# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Failed judge calls retain their original HTTP body without altering scores."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pybase64 as base64
import pytest
import requests

from vllm_omni.benchmarks.duplex import omni_duplex_eval_eval as evaluator
from vllm_omni.benchmarks.duplex import omni_duplex_eval_judge as judge_module
from vllm_omni.benchmarks.duplex import omni_duplex_eval_judge_capture as capture
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _response(content, *, finish_reason="stop", status=200):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(
        {
            "id": "chatcmpl-native-probe",
            "choices": [
                {
                    "message": {"content": content, "reasoning": "native reasoning field"},
                    "finish_reason": finish_reason,
                    "stop_reason": 151645,
                }
            ],
            "usage": {"prompt_tokens": 10564, "completion_tokens": 9, "total_tokens": 10573},
            "system_fingerprint": "native-fingerprint",
        },
        indent=2,
    ).encode()
    return response


def _failed_evaluation(tmp_path, monkeypatch, mocker, response):
    source = tmp_path / "response.json"
    source.write_text('[{"sentence":"The garlic is cut.","start":2,"end":3}]')
    source.with_name("response.meta.json").write_text(json.dumps({"clock": "media", "pace": "realtime"}))
    sample = DuplexSample.from_row(
        {"id": "362", "split": "RTD_world_knowledge", "video": "unused.mp4", "video_duration": 20}
    )
    monkeypatch.setattr(evaluator, "_extract_frames", lambda *args: [b"retained-frame"])
    post = mocker.Mock(return_value=response)
    monkeypatch.setattr(judge_module.requests, "post", post)
    score = tmp_path / "evaluation/RTD_world_knowledge/362.json"
    judge = judge_module.DuplexJudge("http://endpoint-secret.invalid", "judge", api_key="auth-secret")
    return lambda: evaluator.evaluate_sample(sample, source, score, judge), score, post


def _http_record(score):
    manifest = next((score.parent / "judge_requests").glob("request-*.judge.jsonl"))
    return json.loads(manifest.read_text())["http_response"]


def _body(record):
    body = base64.b64decode(record["body_base64"])
    assert len(body) == record["body_bytes"]
    assert hashlib.sha256(body).hexdigest() == record["body_sha256"]
    return body


def test_malformed_score_preserves_exact_http_body_and_original_error(tmp_path, monkeypatch, mocker):
    response = _response("C picks up the garlic")
    run, score, post = _failed_evaluation(tmp_path, monkeypatch, mocker, response)
    with pytest.raises(ValueError, match="^Judge response contains no score fields$"):
        run()
    record = _http_record(score)
    assert record["capture_status"] == "captured" and record["status_code"] == 200
    assert _body(record) == response.content
    payload = json.loads(_body(record))
    assert payload["choices"] == response.json()["choices"]
    assert payload["usage"] == response.json()["usage"]
    assert payload["system_fingerprint"] == "native-fingerprint"
    assert post.call_count == 1
    assert post.call_args.kwargs["timeout"] == 600.0
    assert post.call_args.kwargs["json"]["temperature"] == 0.1
    assert post.call_args.kwargs["json"]["max_tokens"] == 1200
    assert score.with_suffix(".judge.jsonl").read_text() == '{"raw_response": "C picks up the garlic"}\n'
    assert not score.exists() and evaluator.summarize_scores(score.parent.parent)["samples"] == 0
    manifest_text = next((score.parent / "judge_requests").glob("request-*.judge.jsonl")).read_text()
    assert "endpoint-secret" not in manifest_text and "auth-secret" not in manifest_text
    assert "Authorization" not in manifest_text and "headers" not in manifest_text


@pytest.mark.parametrize("failure", ["http", "incomplete", "invalid_json"])
def test_http_capture_preserves_original_judge_failure(tmp_path, monkeypatch, mocker, failure):
    response = _response("bare reply", status=503 if failure == "http" else 200)
    expected = requests.HTTPError if failure == "http" else judge_module.JudgeResponseError
    if failure == "incomplete":
        response = _response('{"temporal_score":3,"is_relevant":1}', finish_reason="length")
    elif failure == "invalid_json":
        response._content = b"not-json\xff"
        expected = requests.JSONDecodeError
    run, score, post = _failed_evaluation(tmp_path, monkeypatch, mocker, response)
    with pytest.raises(expected) as caught:
        run()
    if failure == "http":
        assert caught.value.response is response
        assert str(caught.value) == f"503 response from judge: {response.text[:500]}"
    elif failure == "incomplete":
        assert caught.value.raw_response == '{"temporal_score":3,"is_relevant":1}'
        assert caught.value.finish_reason == "length"
    assert _body(_http_record(score)) == response.content
    assert post.call_count == 1 and not score.exists()


def test_timeout_explicitly_records_absent_http_response(tmp_path, monkeypatch, mocker):
    run, score, post = _failed_evaluation(tmp_path, monkeypatch, mocker, _response("unused"))
    error = requests.Timeout("native timeout")
    post.side_effect = error
    with pytest.raises(requests.Timeout) as caught:
        run()
    assert caught.value is error
    assert _http_record(score) == {"capture_status": "unavailable"}
    assert post.call_count == 1 and not score.exists()


def test_response_byte_limit_is_explicit_and_does_not_store_a_prefix(tmp_path, monkeypatch, mocker):
    response = _response("bare reply")
    run, score, _ = _failed_evaluation(tmp_path, monkeypatch, mocker, response)
    monkeypatch.setattr(capture, "_MAX_RESPONSE_BYTES", len(response.content) - 1)
    with pytest.raises(ValueError, match="no score fields"):
        run()
    record = _http_record(score)
    assert record["capture_status"] == "response_byte_limit"
    assert record["status_code"] == 200
    assert record["body_bytes"] == len(response.content)
    assert record["body_sha256"] == hashlib.sha256(response.content).hexdigest()
    assert "body_base64" not in record


def test_response_serialization_failure_preserves_original_parse_error(tmp_path, monkeypatch, mocker, caplog):
    run, score, post = _failed_evaluation(tmp_path, monkeypatch, mocker, _response("bare reply"))

    def fail_encoding(*args, **kwargs):
        raise OSError("diagnostic-secret")

    monkeypatch.setattr(capture, "_failed_response", fail_encoding)
    with pytest.raises(ValueError, match="^Judge response contains no score fields$"):
        run()
    assert "OSError" in caplog.text and "diagnostic-secret" not in caplog.text
    assert post.call_count == 1 and not score.exists()


def test_success_returns_original_content_without_serializing_response(tmp_path, monkeypatch, mocker):
    post = mocker.Mock(return_value=_response('{"success_score":1}'))
    monkeypatch.setattr(judge_module.requests, "post", post)

    def unexpected_serialization(*args, **kwargs):
        raise AssertionError("successful calls must not serialize capture data")

    monkeypatch.setattr(capture, "_failed_response", unexpected_serialization)
    monkeypatch.setattr(capture, "_json_bytes", unexpected_serialization)
    with capture.capture_judge_request():
        result = judge_module.DuplexJudge("http://unused", "judge").chat("original prompt")
    assert result == '{"success_score":1}' and post.call_count == 1
    assert not list(tmp_path.iterdir())


def test_threaded_capture_keeps_each_request_with_its_response(tmp_path, monkeypatch, mocker):
    def post(*args, **kwargs):
        return _response(kwargs["json"]["messages"][0]["content"])

    monkeypatch.setattr(judge_module.requests, "post", post)
    judge = judge_module.DuplexJudge("http://unused", "judge")

    def worker(name):
        trace = tmp_path / "split" / f"{name}.judge.jsonl"
        with capture.capture_judge_request() as captured:
            assert judge.chat(name) == name
            capture.save_failed_judge_request(captured, trace, 0)
        manifest = next((trace.parent / "judge_requests").glob(f"request-{name}.judge-*.judge.jsonl"))
        record = json.loads(manifest.read_text())
        assert record["request_body"]["messages"][0]["content"] == name
        assert json.loads(_body(record["http_response"]))["choices"][0]["message"]["content"] == name

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, ["first", "second"]))
