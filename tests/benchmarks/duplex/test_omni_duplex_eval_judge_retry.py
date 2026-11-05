# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-DuplexEval: a judge reply without score fields must not fail the sample.

The judge model can answer with prose (e.g. a bare VQA-style option such as
"C cuts the pineapple" seen on RTD_interaction_relation/512) instead of the
required score JSON. ``evaluate_sample`` echoes the offending reply back once
and re-asks for the JSON, so the format — not the pinned rubric — is corrected.
These tests pin the single retry, its trace record, and the failure path.
"""

import json

import pytest

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import evaluate_sample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_judge import JudgeResponseError

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

_PROSE_REPLY = "C cuts the pineapple"
_SCORE_JSON = '{"success_score": 1}'


class _FlakyJudge:
    """Returns queued replies in order and records every chat call."""

    model = "fake-judge"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[object, tuple[str, str] | None]] = []

    def chat(self, content, *, followup=None):
        self.calls.append((content, followup))
        return self.replies.pop(0)


class _TruncatedRetryJudge(_FlakyJudge):
    """The format correction itself comes back cut off."""

    def chat(self, content, *, followup=None):
        self.calls.append((content, followup))
        if followup is not None:
            raise JudgeResponseError(_PROSE_REPLY, "length")
        return self.replies.pop(0)


def _pr_sample() -> DuplexSample:
    return DuplexSample(
        id="512",
        split="RTD_interaction_relation",
        family="pr",
        task_type="correction",
        video=None,
        question_text="Please correct my summary.",
        answer1="the corrected summary",
    )


def _write_response(tmp_path):
    response_path = tmp_path / "512.response.json"
    response_path.write_text(
        json.dumps({"chunks": [{"text": "Here is the corrected summary.", "current_time": 1.0}]}),
        encoding="utf-8",
    )
    return response_path


def _trace(tmp_path) -> list[dict]:
    trace_path = tmp_path / "512.judge.jsonl"
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]


def test_prose_reply_is_retried_and_scored(tmp_path):
    judge = _FlakyJudge([_PROSE_REPLY, _SCORE_JSON])
    result = evaluate_sample(_pr_sample(), _write_response(tmp_path), tmp_path / "512.json", judge)
    assert result["pr"]["all_success"] is True
    # Two judge calls: the original one and exactly one format correction.
    assert len(judge.calls) == 2
    first_content, first_followup = judge.calls[0]
    assert first_followup is None
    _, retried_followup = judge.calls[1]
    assert retried_followup == (_PROSE_REPLY, retried_followup[1])
    assert "success_score" in retried_followup[1]
    assert first_content == judge.calls[1][0]  # same scoring prompt, echoed format fix only
    lines = _trace(tmp_path)
    assert [line.get("judge_retry") for line in lines] == [None, True]


def test_second_prose_reply_raises(tmp_path):
    judge = _FlakyJudge([_PROSE_REPLY, _PROSE_REPLY])
    with pytest.raises(ValueError):
        evaluate_sample(_pr_sample(), _write_response(tmp_path), tmp_path / "512.json", judge)
    assert len(judge.calls) == 2
    lines = _trace(tmp_path)
    # One record per request: both replies are kept, and the retry's failure
    # names its reason on the same record.
    assert [line.get("judge_retry") for line in lines] == [None, True]
    assert [line["raw_response"] for line in lines] == [_PROSE_REPLY, _PROSE_REPLY]
    assert "error" in lines[-1]


def test_truncated_retry_is_traced(tmp_path):
    judge = _TruncatedRetryJudge([_PROSE_REPLY])
    with pytest.raises(JudgeResponseError):
        evaluate_sample(_pr_sample(), _write_response(tmp_path), tmp_path / "512.json", judge)
    lines = _trace(tmp_path)
    assert lines[-1]["judge_retry"] is True
    assert lines[-1]["finish_reason"] == "length"
    assert "error" in lines[-1]


def test_immediate_score_json_skips_retry(tmp_path):
    judge = _FlakyJudge([_SCORE_JSON])
    result = evaluate_sample(_pr_sample(), _write_response(tmp_path), tmp_path / "512.json", judge)
    assert result["pr"]["all_success"] is True
    assert len(judge.calls) == 1
    assert len(_trace(tmp_path)) == 1


def test_parse_error_reaches_the_retry_and_the_trace(tmp_path):
    """The format retry carries the parse error itself, and the trace records it.

    A judge reply can fail to parse without being prose — e.g. a malformed
    value in a field that is present — so the retry has to say what failed,
    and the first trace line has to show which reply failed to parse.
    """
    judge = _FlakyJudge([_PROSE_REPLY, _SCORE_JSON])
    result = evaluate_sample(_pr_sample(), _write_response(tmp_path), tmp_path / "512.json", judge)
    assert result["pr"]["all_success"] is True

    _, retried_followup = judge.calls[1]
    assert "no score fields" in retried_followup[1]

    first, _retried = _trace(tmp_path)
    assert first["raw_response"] == _PROSE_REPLY
    assert "no score fields" in first["error"]
