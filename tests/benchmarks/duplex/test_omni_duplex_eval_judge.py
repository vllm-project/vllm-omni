# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Judge failures must not become model scores or leave stale scores active."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
import requests

from vllm_omni.benchmarks.duplex import omni_duplex_eval_eval as evaluator
from vllm_omni.benchmarks.duplex import omni_duplex_eval_judge as judge_module
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_metrics import parse_judge_json

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _reply(content: str, finish_reason: str | None = "stop", *, status: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(
        {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    ).encode()
    return response


@pytest.fixture
def judge_http(monkeypatch):
    post = Mock()
    monkeypatch.setattr(judge_module.requests, "post", post)
    return post


@pytest.fixture
def sample_paths(tmp_path):
    response = tmp_path / "response.json"
    response.write_text('[{"sentence": "The shirt is blue.", "start": 1, "end": 2}]')
    score = tmp_path / "scores" / "sample.json"
    sample = DuplexSample.from_row({"id": "sample", "split": "PR_correction", "question_text": "The shirt is red."})
    return sample, response, score


def _evaluate(sample_paths):
    sample, response, score = sample_paths
    return evaluator.evaluate_sample(sample, response, score, judge_module.DuplexJudge("http://unused", "judge"))


def _trace(score):
    return [json.loads(line) for line in score.with_suffix(".judge.jsonl").read_text().splitlines()]


@pytest.mark.parametrize(
    "reply",
    [
        "I cannot evaluate this.",
        '{"reasoning": "unknown"}',
        '{"success_score": 99}',
        '{"success_score": -1}',
        '{"success_score": 0.5}',
        '{"success_score": true}',
        '{"success_score": null}',
        '{"success_score": "bad"}',
        '{"temporal_score": 4, "is_relevant": 1}',
        '{"temporal_score": 1.5, "is_relevant": 1}',
        '{"temporal_score": 1, "is_relevant": 2}',
        '{"content_score": NaN}',
        '{"content_score": Infinity}',
        '{"content_score": 3.001}',
        '{"content_score": "NaN"}',
        '{"content_score": "Infinity"}',
        '{"content_score": ""}',
        'noise "success_score": 10',
        'noise "content_score": -1',
    ],
)
def test_invalid_scores_are_rejected(reply):
    with pytest.raises(ValueError):
        parse_judge_json(reply)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"success_score": "1"}', {"success_score": 1}),
        ('{"temporal_score": "2.0", "is_relevant": "1"}', {"temporal_score": 2, "is_relevant": 1}),
        ('{"content_score": "2.50"}', {"content_score": 2.5}),
        ('```json\n{"success_score": 0,}\n```', {"success_score": 0}),
        ('noise "success_score": 1', {"success_score": 1}),
    ],
)
def test_valid_official_formats_are_preserved(reply, expected):
    assert parse_judge_json(reply) == expected


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
def test_incomplete_reply_preserves_diagnostics_without_a_score(sample_paths, judge_http, finish_reason):
    raw = '{"success_score": 1, "reasoning": "Unfinished'
    judge_http.return_value = _reply(raw, finish_reason)
    with pytest.raises(ValueError, match="finish_reason"):
        _evaluate(sample_paths)
    score = sample_paths[2]
    assert not score.exists()
    trace = _trace(score)
    assert len(trace) == 1
    assert trace[0]["raw_response"] == raw
    assert trace[0]["finish_reason"] == finish_reason
    assert evaluator.summarize_scores(score.parent)["samples"] == 0


@pytest.mark.parametrize(
    "completion",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"finish_reason": None}, id="null"),
        pytest.param({"finish_reason": "length"}, id="length"),
        pytest.param({"finish_reason": "content_filter"}, id="content_filter"),
        pytest.param({"finish_reason": "tool_calls"}, id="tool_calls"),
        pytest.param({"finish_reason": "function_call"}, id="function_call"),
        pytest.param({"finish_reason": "unknown"}, id="unknown"),
    ],
)
def test_valid_score_requires_explicit_completion(sample_paths, judge_http, completion):
    raw = '{"success_score": 0}'
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({"choices": [{"message": {"content": raw}, **completion}]}).encode()
    judge_http.return_value = response
    with pytest.raises(judge_module.JudgeResponseError, match="finish_reason"):
        _evaluate(sample_paths)
    score = sample_paths[2]
    assert not score.exists()
    assert evaluator.summarize_scores(score.parent)["samples"] == 0
    trace = _trace(score)
    assert len(trace) == 1
    assert trace[0]["raw_response"] == raw
    assert trace[0]["finish_reason"] == completion.get("finish_reason")


@pytest.mark.parametrize("raw", ['{"reasoning": "unknown"}', '{"success_score": "bad"}', '{"content_score": 1}'])
def test_invalid_pr_reply_is_not_published(sample_paths, judge_http, raw):
    judge_http.return_value = _reply(raw)
    with pytest.raises(ValueError):
        _evaluate(sample_paths)
    score = sample_paths[2]
    assert not score.exists()
    assert _trace(score)[0]["raw_response"] == raw


@pytest.mark.parametrize("failure", ["invalid_score", "http_error", "incomplete_reply"])
def test_failed_retry_archives_old_score_and_excludes_it_from_summary(sample_paths, judge_http, failure):
    judge_http.return_value = _reply('{"success_score": 1}')
    _evaluate(sample_paths)
    score = sample_paths[2]
    previous = score.read_bytes()
    # An unrelated sample must remain counted through another sample's retries.
    sample, response, _ = sample_paths
    neighbor = score.with_name("neighbor.json")
    judge_http.return_value = _reply('{"success_score": 0}')
    _evaluate((replace(sample, id="neighbor"), response, neighbor))
    neighbor_bytes = neighbor.read_bytes()
    if failure == "http_error":
        judge_http.return_value = _reply("Judge unavailable", status=503)
        error: type[Exception] = requests.HTTPError
    else:
        judge_http.return_value = _reply(
            '{"success_score": "bad"}' if failure == "invalid_score" else '{"success_score": 1}',
            "stop" if failure == "invalid_score" else "length",
        )
        error = ValueError
    with pytest.raises(error):
        _evaluate(sample_paths)
    assert not score.exists()
    archives = list(score.parent.glob("sample.json.*.previous"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == previous
    summary = evaluator.summarize_scores(score.parent)
    assert summary["samples"] == 1
    assert summary["pr"]["mean_all_success"] == 0
    assert neighbor.read_bytes() == neighbor_bytes

    # Recovery publishes only the new result, never the archived successful one.
    judge_http.return_value = _reply('{"success_score": 0}')
    _evaluate(sample_paths)
    summary = evaluator.summarize_scores(score.parent)
    assert summary["samples"] == 2
    assert summary["pr"]["mean_all_success"] == 0
    assert list(score.parent.glob("sample.json.*.previous")) == archives
    assert archives[0].read_bytes() == previous
    assert neighbor.read_bytes() == neighbor_bytes


def test_successful_retry_counts_valid_zero_and_refreshes_trace(sample_paths, judge_http):
    judge_http.return_value = _reply('{"success_score": 1}')
    _evaluate(sample_paths)
    judge_http.return_value = _reply('{"success_score": 0}')
    result = _evaluate(sample_paths)
    score = sample_paths[2]
    assert result["pr"]["total_score"] == 0
    assert json.loads(score.read_text()) == result
    assert evaluator.summarize_scores(score.parent)["samples"] == 1
    assert len(list(score.parent.glob("sample.json.*.previous"))) == 1
    trace = _trace(score)
    assert len(trace) == 1
    assert trace[0]["raw_response"] == '{"success_score": 0}'


@pytest.mark.parametrize(
    ("replies", "missing_field"),
    [
        ([{"temporal_score": 2}], "is_relevant"),
        ([{"is_relevant": 1}], "temporal_score"),
        ([{"temporal_score": 2, "is_relevant": 1}, {"success_score": 1}], "content_score"),
        ([{"temporal_score": 0, "is_relevant": 0}, {"content_score": 0}], None),
    ],
)
def test_rtd_requires_complete_scores_before_publishing(sample_paths, judge_http, monkeypatch, replies, missing_field):
    _, response, score = sample_paths
    sample = DuplexSample.from_row({"id": "sample", "split": "RTD_OCR", "video": "unused.mp4", "video_duration": 10})
    monkeypatch.setattr(evaluator, "materialize_media", lambda *args: "unused.mp4")
    monkeypatch.setattr(evaluator, "_extract_frames", lambda *args: [b"jpeg"])
    judge_http.side_effect = [_reply(json.dumps(reply)) for reply in replies]
    if missing_field:
        with pytest.raises(ValueError, match=missing_field):
            _evaluate((sample, response, score))
        assert not score.exists()
        assert evaluator.summarize_scores(score.parent)["samples"] == 0
    else:
        result = _evaluate((sample, response, score))
        assert result["temporal"]["summary"]["avg_temporal_score"] == 0
        assert result["content"]["content_score"] == 0
        assert evaluator.summarize_scores(score.parent)["samples"] == 1
    assert judge_http.call_count == len(replies)
    assert [entry["raw_response"] for entry in _trace(score)] == [json.dumps(reply) for reply in replies]
