# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.duplex import omni_duplex_eval_eval as eval_module
from vllm_omni.benchmarks.duplex.omni_duplex_eval_clock import (
    extract_timed_sentences,
    normalize_response_items,
    split_text,
    validate_clock,
)
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import (
    DuplexSample,
    canonical_task_type,
    family_for_split,
    load_samples,
    task_type_for_split,
)
from vllm_omni.benchmarks.duplex.omni_duplex_eval_metrics import (
    PROTOCOL_PIN,
    build_content_prompt,
    build_reminder_prompt,
    build_temporal_prompt,
    parse_judge_json,
    reminder_window,
    summarize_pr_results,
    temporal_window,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_dataset_mapping_and_manifest(tmp_path):
    manifest = tmp_path / "samples.json"
    manifest.write_text(json.dumps([{"id": "a", "split": "RTD_OCR", "video": "clip.mp4"}]), encoding="utf-8")
    sample = load_samples(manifest, media_root=tmp_path)[0]
    assert sample.family == "rtd"
    assert sample.video == str(tmp_path / "clip.mp4")
    assert family_for_split("PR_event_reminder") == "pr"
    assert canonical_task_type("post-event-reminder") == "post_event_reminder"
    assert task_type_for_split("PR_correction") == "correction"
    assert load_samples([{"id": "pr", "split": "PR_correction"}])[0].task_type == "correction"


def test_hugging_face_subset_names_are_splits(monkeypatch):
    calls = []

    def fake_load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        return [{"id": "pr", "question_text": "Correct this."}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    sample = load_samples("Hothan/Omni-DuplexEval", split="PR_correction")[0]
    assert calls == [("Hothan/Omni-DuplexEval", {"split": "PR_correction"})]
    assert sample.split == "PR_correction"
    assert sample.task_type == "correction"


def test_response_aliases_and_clock_guard():
    assert split_text("One. Two!") == ["One.", "Two!"]
    assert normalize_response_items({"chunks": [{"text": "x", "current_time": 800}]}) == [
        {"sentence": "x", "start": 800.0, "end": 800.0}
    ]
    with pytest.raises(ValueError, match="clock=invalid"):
        validate_clock({"clock": "invalid"})
    validate_clock({"clock": "invalid"}, allow_invalid=True)
    timed = extract_timed_sentences([{"type": "response.output_text.delta", "delta": "Done.", "_media_clock_ms": 800}])
    assert [item.as_dict() for item in timed] == [{"sentence": "Done.", "start": 0.8, "end": 0.8}]
    with pytest.raises(ValueError, match="only clock=media"):
        extract_timed_sentences([], clock="wall")


def test_protocol_windows_and_parsing():
    assert PROTOCOL_PIN == "ca3c122b4d4bf67afd6b18ea5e724b4561bdde48"
    assert temporal_window(4, 5, 10) == (2.0, 3.0)
    assert temporal_window(1, 1.2, 10) == (0.0, 0.5)
    assert reminder_window(5) == (5.0, 15.0)
    assert parse_judge_json('noise {"content_score": 2.345, "is_relevant": 1}') == {
        "content_score": 2.35,
        "is_relevant": 1,
    }
    assert parse_judge_json('noise "success_score": 1') == {"success_score": 1}
    assert summarize_pr_results([{"task_type": "correction", "all_success": 1}])["mean_all_success"] == 1.0


def test_protocol_prompt_wording_is_pinned():
    prompts = [
        build_temporal_prompt(1, 2, "response", "question"),
        build_content_prompt("response", "question", ["one", "two"]),
        build_reminder_prompt("instruction", "response", "proactive_reminder"),
        build_reminder_prompt("instruction", "response", "post_event_reminder"),
        build_reminder_prompt("instruction", "response", "correction", "answer"),
    ]
    assert [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts] == [
        "5f7f5fcb4a37e4d2eea8ba23beabf74630e6fc2fb5b2661ceb94403254bb04d5",
        "1fccbed6c1ed8ea1c76d5564d7289ed6adca3678f6a0da2cc735942123d64457",
        "d0a0e3a9d873d2f368c5c99b9fe46681f683cc36a9fe0e09b99d6cbca1d6dd0c",
        "08f9951243c859d271f82ee53da22cd0bdab1fe93c6ca04f4870d3b959c0803a",
        "3de131a7e36cb4b8ac369bb05e87ede518be20221e214df3644211c179131a73",
    ]


def test_frame_sample_content_passes_frames(tmp_path, monkeypatch):
    response = tmp_path / "response.json"
    response.write_text(json.dumps([{"sentence": "A person moves.", "start": 1, "end": 2}]), encoding="utf-8")
    response.with_name("response.meta.json").write_text(json.dumps({"clock": "media"}), encoding="utf-8")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(eval_module, "extract_jpeg", lambda *args, **kwargs: b"\xff\xd8jpeg")

    class Judge:
        def temporal(self, *args, **kwargs):
            return '{"temporal_score": 3, "is_relevant": 1}'

        def content(self, prompt, video, frames=None, *, mode="video_url"):
            assert mode == "frame-sample"
            assert frames == [b"\xff\xd8jpeg", b"\xff\xd8jpeg"]
            return '{"content_score": 3}'

    sample = DuplexSample(
        id="rtd",
        split="RTD_OCR",
        family="rtd",
        task_type=None,
        video=video,
        video_duration=4,
    )
    score = eval_module.evaluate_sample(
        sample,
        response,
        tmp_path / "score.json",
        Judge(),
        judge_video_mode="frame-sample",
    )
    assert score["content"]["frame_count"] == 2
