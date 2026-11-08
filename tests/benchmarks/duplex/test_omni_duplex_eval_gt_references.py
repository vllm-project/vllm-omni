# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression tests for the Omni-DuplexEval content-prompt GT references."""

from __future__ import annotations

import json

import pytest

from vllm_omni.benchmarks.duplex import omni_duplex_eval_eval as eval_module
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("answer1", "answer2", "expected_block"),
    [
        ("A cat sits.", "", "Reference annotations, for reference only:\n1. A cat sits.\n"),
        (
            "A cat sits.",
            "A cat sits.",
            "Reference annotations, for reference only:\n1. A cat sits.\n",
        ),
        ("  A cat sits.  ", "", "Reference annotations, for reference only:\n1. A cat sits.\n"),
    ],
)
def test_content_prompt_uses_official_gt_reference_semantics(tmp_path, monkeypatch, answer1, answer2, expected_block):
    # Issue #9054: the official evaluator drops empty answers and deduplicates
    # references (first occurrence) before building the pinned content prompt.
    response = tmp_path / "response.json"
    response.write_text(json.dumps([{"sentence": "A person moves.", "start": 1, "end": 2}]), encoding="utf-8")
    response.with_name("response.meta.json").write_text(json.dumps({"clock": "media"}), encoding="utf-8")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(eval_module, "extract_jpeg", lambda *args, **kwargs: b"\xff\xd8jpeg")
    prompts = []

    class Judge:
        def temporal(self, *args, **kwargs):
            return '{"temporal_score": 3, "is_relevant": 1}'

        def content(self, prompt, video, frames=None, *, mode="video_url"):
            prompts.append(prompt)
            return '{"content_score": 3}'

    sample = DuplexSample(
        id="rtd",
        split="RTD_OCR",
        family="rtd",
        task_type=None,
        video=video,
        video_duration=4,
        answer1=answer1,
        answer2=answer2,
    )
    eval_module.evaluate_sample(sample, response, tmp_path / "score.json", Judge())

    assert expected_block in prompts[0]
    assert "\n2. " not in prompts[0]
    assert prompts[0].count("A cat sits.") == 1
