# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Missing proactive responses cannot earn credit from the reference answer."""

import json

import pytest

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import evaluate_sample

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


@pytest.mark.parametrize("task", ["correction", "proactive_reminder", "post_event_reminder"])
@pytest.mark.parametrize("reminder", [None, 1.0])
@pytest.mark.parametrize(
    "response",
    [
        {"chunks": []},
        {"chunks": [{"text": " \n\t", "current_time": 1.0}]},
        {"sentences": []},
        {"sentences": [{"sentence": " \n\t", "start": 1.0, "end": 2.0}]},
    ],
)
def test_empty_proactive_response_fails_without_judge(tmp_path, task, reminder, response):
    class Judge:
        model = "false-positive-judge"
        calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            return '{"success_score": 1}'

    sample = DuplexSample(
        id="0",
        split="PR_correction",
        family="pr",
        task_type=task,
        video=None,
        question_text="The girl wears black trousers.",
        answer1="She wears an orange dress.",
        reminder1=reminder,
    )
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(response))
    score_path = tmp_path / "score.json"
    judge = Judge()
    result = evaluate_sample(sample, response_path, score_path, judge)

    assert result["pr"]["all_success"] is False
    assert result["pr"]["total_score"] == 0
    assert all(event["success_score"] == 0 for event in result["pr"]["events"])
    assert judge.calls == 0
    assert json.loads(score_path.read_text())["pr"] == result["pr"]
