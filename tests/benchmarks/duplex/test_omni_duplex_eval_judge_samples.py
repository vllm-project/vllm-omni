# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-DuplexEval ``judge_samples``: N judge calls per measurement, aggregated by the lower median."""

import itertools
import json
from dataclasses import dataclass

import pytest
import requests

from vllm_omni.benchmarks.duplex import omni_duplex_eval_eval as evaluator
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_judge import JudgeResponseError
from vllm_omni.benchmarks.duplex.omni_duplex_eval_metrics import aggregate_judge_votes

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

_PROSE_REPLY = "C cuts the pineapple"


@dataclass(frozen=True)
class _Call:
    method: str
    followup: tuple[str, str] | None
    mode: str | None = None


class _QueuedJudge:
    """Returns queued replies per judge method, raising queued exceptions, and records every call."""

    model = "fake-judge"

    def __init__(self, *, chat=(), temporal=(), content=()):
        self.replies = {"chat": list(chat), "temporal": list(temporal), "content": list(content)}
        self.calls: list[_Call] = []

    def _next(self, call: _Call) -> str:
        self.calls.append(call)
        reply = self.replies[call.method].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def chat(self, content, *, followup=None):
        return self._next(_Call("chat", followup))

    def temporal(self, prompt, frames, *, followup=None):
        return self._next(_Call("temporal", followup))

    def content(self, prompt, video=None, frames=None, *, mode="video_url", followup=None):
        return self._next(_Call("content", followup, mode))


def _success(score: int) -> str:
    return json.dumps({"success_score": score, "reasoning": f"scored {score}"})


def _temporal(score: int, relevant: int) -> str:
    return json.dumps({"temporal_score": score, "temporal_reasoning": f"scored {score}", "is_relevant": relevant})


def _content(score: float) -> str:
    return json.dumps({"content_score": score, "content_reasoning": f"scored {score}"})


def _trace(score_path):
    trace_path = score_path.with_suffix(".judge.jsonl")
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]


def _pr_response(tmp_path, chunks):
    response = tmp_path / "sample.json"
    response.write_text(json.dumps({"sentences": [], "chunks": chunks}), encoding="utf-8")
    response.with_name("sample.meta.json").write_text(json.dumps({"clock": "media"}), encoding="utf-8")
    return response


def _correction(tmp_path):
    sample = DuplexSample(
        id="sample",
        split="PR_correction",
        family="pr",
        task_type="correction",
        video=None,
        question_text="The shirt is red.",
        answer1="The shirt is blue.",
    )
    return sample, _pr_response(tmp_path, [{"text": "The shirt is blue.", "current_time": 1.0}])


def _reminder(tmp_path, chunks):
    sample = DuplexSample(
        id="sample",
        split="PR_proactive_reminder",
        family="pr",
        task_type="proactive_reminder",
        video=None,
        question_text="Remind me when the man comes back.",
        reminder1="10.0",
        reminder2="30.0",
    )
    return sample, _pr_response(tmp_path, chunks)


def _rtd(tmp_path, monkeypatch, sentences):
    response = tmp_path / "sample.json"
    response.write_text(json.dumps(sentences), encoding="utf-8")
    response.with_name("sample.meta.json").write_text(json.dumps({"clock": "media"}), encoding="utf-8")
    monkeypatch.setattr(evaluator, "materialize_media", lambda *args: tmp_path / "video.mp4")
    monkeypatch.setattr(evaluator, "_extract_frames", lambda *args: [b"\xff\xd8jpeg"])
    sample = DuplexSample(
        id="sample", split="RTD_OCR", family="rtd", task_type=None, video="video.mp4", video_duration=20
    )
    return sample, response


@pytest.mark.parametrize(
    ("field", "values", "expected"),
    [
        ("success_score", [1, 0, 1], 1),
        ("success_score", [0, 1, 0], 0),
        ("success_score", [0, 1], 0),
        ("temporal_score", [3, 1, 2], 2),
        ("temporal_score", [3, 0, 0], 0),
        ("temporal_score", [1, 3, 1, 3], 1),
        ("content_score", [2.15, 2.4, 2.15], 2.15),
        ("content_score", [2.0, 1.0, 0.0], 1.0),
        ("content_score", [2.0, 2.5], 2.0),
        ("content_score", [2.5], 2.5),
    ],
    ids=[
        "binary-majority-1",
        "binary-majority-0",
        "binary-tie-resolves-low",
        "no-majority-middle-draw",
        "majority-low",
        "even-tie-resolves-low",
        "content-not-rounded",
        "content-no-majority",
        "content-tie-resolves-low",
        "single-draw",
    ],
)
def test_lower_median_aggregation(field, values, expected):
    votes = [{field: value, "reasoning": f"draw {i}"} for i, value in enumerate(values)]
    assert aggregate_judge_votes(votes, (field,)) == {field: expected}


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_lower_median_equals_any_strict_majority(count):
    for values in itertools.product(range(4), repeat=count):
        aggregated = aggregate_judge_votes([{"temporal_score": value} for value in values], ("temporal_score",))
        assert aggregated["temporal_score"] in values
        for value in set(values):
            if values.count(value) * 2 > count:
                assert aggregated["temporal_score"] == value


def test_fields_are_aggregated_independently():
    votes = [
        {"temporal_score": 3, "is_relevant": 0},
        {"temporal_score": 1, "is_relevant": 1},
        {"temporal_score": 2, "is_relevant": 1},
    ]
    assert aggregate_judge_votes(votes, ("temporal_score", "is_relevant")) == {"temporal_score": 2, "is_relevant": 1}


def test_aggregation_requires_a_vote():
    with pytest.raises(ValueError):
        aggregate_judge_votes([], ("success_score",))


@pytest.mark.parametrize("kwargs", [{}, {"judge_samples": 1}], ids=["default", "explicit-1"])
def test_single_call_keeps_the_existing_output(tmp_path, kwargs):
    sample, response = _correction(tmp_path)
    judge = _QueuedJudge(chat=[_success(1)])
    score = tmp_path / "scores" / "sample.json"
    result = evaluator.evaluate_sample(sample, response, score, judge, **kwargs)
    assert "judge_samples" not in result
    assert result["pr"]["events"] == [
        {"response_segment": "The shirt is blue.", "success_score": 1, "reasoning": "scored 1"}
    ]
    assert len(judge.calls) == 1
    assert _trace(score) == [{"raw_response": _success(1), "error": None}]


def test_reminder_events_vote_per_event_and_keep_the_all_success_reduction(tmp_path):
    chunks = [{"text": "He is back.", "current_time": 12.0}, {"text": "There he is again.", "current_time": 31.0}]
    sample, response = _reminder(tmp_path, chunks)
    judge = _QueuedJudge(chat=[_success(1), _success(0), _success(1), _success(0), _success(0), _success(1)])
    score = tmp_path / "scores" / "sample.json"
    result = evaluator.evaluate_sample(sample, response, score, judge, judge_samples=3)
    first, second = result["pr"]["events"]
    assert (first["success_score"], second["success_score"]) == (1, 0)
    assert [vote["success_score"] for vote in first["judge_votes"]] == [1, 0, 1]
    assert [vote["success_score"] for vote in second["judge_votes"]] == [0, 0, 1]
    assert "reasoning" not in first and first["judge_votes"][0]["reasoning"] == "scored 1"
    assert result["pr"]["all_success"] is False and result["pr"]["total_score"] == 0
    assert result["judge_samples"] == 3
    assert [line["judge_draw"] for line in _trace(score)] == [0, 1, 2, 0, 1, 2]
    assert json.loads(score.read_text(encoding="utf-8")) == result


def test_empty_reminder_window_is_not_judged(tmp_path):
    sample, response = _reminder(tmp_path, [{"text": "He is back.", "current_time": 12.0}])
    judge = _QueuedJudge(chat=[_success(1)] * 3)
    result = evaluator.evaluate_sample(sample, response, tmp_path / "scores" / "sample.json", judge, judge_samples=3)
    first, second = result["pr"]["events"]
    assert first["success_score"] == 1 and len(first["judge_votes"]) == 3
    assert second == {"start_time": 30.0, "response_segment": "", "success_score": 0}
    assert len(judge.calls) == 3
    assert result["pr"]["all_success"] is False


def test_rtd_votes_each_sentence_and_the_content_judgment(tmp_path, monkeypatch):
    sentences = [
        {"sentence": "A sign reads OPEN.", "start": 3, "end": 6},
        {"sentence": "Thanks for watching.", "start": 10, "end": 12},
    ]
    sample, response = _rtd(tmp_path, monkeypatch, sentences)
    judge = _QueuedJudge(
        temporal=[_temporal(3, 1), _temporal(1, 1), _temporal(2, 0), _temporal(0, 0), _temporal(2, 0), _temporal(1, 1)],
        content=[_content(2.0), _content(1.0), _content(0.0)],
    )
    score = tmp_path / "scores" / "sample.json"
    result = evaluator.evaluate_sample(sample, response, score, judge, judge_samples=3)
    first, second = result["temporal"]["sentences"]
    assert (first["temporal_score"], first["is_relevant"]) == (2, 1)
    assert (second["temporal_score"], second["is_relevant"]) == (1, 0)
    assert len(first["judge_votes"]) == len(second["judge_votes"]) == 3
    summary = result["temporal"]["summary"]
    assert summary["avg_temporal_score"] == 2
    assert (summary["relevant_sentences_count"], summary["irrelevant_sentences_count"]) == (1, 1)
    assert result["content"]["content_score"] == 1.0
    assert [vote["content_score"] for vote in result["content"]["judge_votes"]] == [2.0, 1.0, 0.0]
    assert evaluator.summarize_scores(score.parent)["rtd"]["mean_content_score"] == 1.0


def test_format_retry_stays_inside_its_draw(tmp_path):
    sample, response = _correction(tmp_path)
    judge = _QueuedJudge(chat=[_success(1), _PROSE_REPLY, _success(0), _success(1)])
    score = tmp_path / "scores" / "sample.json"
    result = evaluator.evaluate_sample(sample, response, score, judge, judge_samples=3)
    (event,) = result["pr"]["events"]
    assert [vote["success_score"] for vote in event["judge_votes"]] == [1, 0, 1]
    assert event["success_score"] == 1
    assert [(line["judge_draw"], line.get("judge_retry")) for line in _trace(score)] == [
        (0, None),
        (1, None),
        (1, True),
        (2, None),
    ]
    retry = judge.calls[2].followup
    assert retry is not None and retry[0] == _PROSE_REPLY


@pytest.mark.parametrize(
    ("failure", "traced_draws"),
    [
        (JudgeResponseError('{"success_score": 1', "length"), [0, 1]),
        (requests.HTTPError("503 response from judge: unavailable"), [0]),
    ],
    ids=["incomplete-reply", "http-error"],
)
def test_a_failed_draw_fails_the_sample(tmp_path, failure, traced_draws):
    sample, response = _correction(tmp_path)
    judge = _QueuedJudge(chat=[_success(1), failure])
    score = tmp_path / "scores" / "sample.json"
    with pytest.raises(type(failure)):
        evaluator.evaluate_sample(sample, response, score, judge, judge_samples=3)
    assert not score.exists()
    assert len(judge.calls) == 2
    assert [line["judge_draw"] for line in _trace(score)] == traced_draws
    assert evaluator.summarize_scores(score.parent)["samples"] == 0


def test_context_overflow_moves_every_content_draw_to_frame_sample(tmp_path, monkeypatch):
    sample, response = _rtd(tmp_path, monkeypatch, [{"sentence": "A sign reads OPEN.", "start": 3, "end": 6}])
    rejected = requests.Response()
    rejected.status_code = 400
    rejected.encoding = "utf-8"
    rejected._content = b"Input length (83158) exceeds model's maximum context length (65536)."
    overflow = requests.HTTPError("400 response from judge", response=rejected)
    judge = _QueuedJudge(
        temporal=[_temporal(3, 1)] * 3,
        content=[overflow, _content(2.5), _content(3.0), _content(2.5)],
    )
    score = tmp_path / "scores" / "sample.json"
    result = evaluator.evaluate_sample(sample, response, score, judge, judge_samples=3, judge_video_mode="video_url")
    modes = [call.mode for call in judge.calls if call.method == "content"]
    assert modes == ["video_url", "frame-sample", "frame-sample", "frame-sample"]
    content = result["content"]
    assert content["judge_input_fallback"] == "frame-sample"
    assert content["content_score"] == 2.5
    assert len(content["judge_votes"]) == 3
    assert content["frame_count"] == 1


@pytest.mark.parametrize("judge_samples", [0, -1])
def test_invalid_judge_samples_is_rejected_before_touching_scores(tmp_path, judge_samples):
    sample, response = _correction(tmp_path)
    score = tmp_path / "scores" / "sample.json"
    score.parent.mkdir()
    score.write_text('{"previous": true}', encoding="utf-8")
    judge = _QueuedJudge()
    with pytest.raises(ValueError, match="judge_samples"):
        evaluator.evaluate_sample(sample, response, score, judge, judge_samples=judge_samples)
    assert list(score.parent.iterdir()) == [score]
    assert score.read_text(encoding="utf-8") == '{"previous": true}'
    assert judge.calls == []
