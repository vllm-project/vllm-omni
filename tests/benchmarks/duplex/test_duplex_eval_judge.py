# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Offline unit tests for the Qwen3-Omni judge adaptation (M3).

Covers the three-level ``select_judge_text`` strategy, the ``modalities``
request-body opt-in, the parameterised ``content_frame_limit`` frame budget
(including byte-for-byte equivalence with the legacy 16-frame implementation)
and the unchanged judge protocol / ``timeout=600``.

Fully offline: the HTTP transport is replaced by a stub, no server is needed.

See ``plans/omni-duplex-eval-backend-design.md`` §5.6a/b, §9 rows T7-T9.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    from vllm_omni.benchmarks.duplex import omni_duplex_eval_eval as eval_mod
    from vllm_omni.benchmarks.duplex import omni_duplex_eval_judge as judge_mod
except ImportError:  # pragma: no cover - offline stub environment
    pytest.skip("judge/eval module is not importable", allow_module_level=True)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _old_content_frame_times(duration: float) -> list[float]:
    """The pre-M3 16-frame implementation, reproduced verbatim for regression."""
    if duration <= 0:
        return []
    times = []
    timestamp = 0.0
    while timestamp < duration:
        times.append(min(timestamp, max(0.0, duration - 0.01)))
        timestamp += 3.0
    if len(times) > 16:
        stride = (len(times) - 1) / 15
        times = [times[round(i * stride)] for i in range(16)]
    return times


# ================================================================== #
# ``select_judge_text`` three-level strategy (T8)
# ================================================================== #


class TestSelectJudgeText:
    """Level 1: text modality; level 2: first non-empty; level 3: choices[0]."""

    def test_prefers_text_modality_choice(self) -> None:
        payload = {
            "choices": [
                {"modality": "audio", "message": {"content": "AUDIO"}},
                {"modality": "text", "message": {"content": "TEXT"}},
            ]
        }
        assert judge_mod.select_judge_text(payload) == "TEXT"

    def test_modality_under_message_is_honoured(self) -> None:
        payload = {
            "choices": [
                {"message": {"modality": "audio", "content": "AUDIO"}},
                {"message": {"modality": "text", "content": "TEXT"}},
            ]
        }
        assert judge_mod.select_judge_text(payload) == "TEXT"

    def test_text_modality_with_blank_text_falls_back(self) -> None:
        payload = {
            "choices": [
                {"modality": "text", "message": {"content": "   "}},
                {"message": {"content": "SECOND"}},
            ]
        }
        assert judge_mod.select_judge_text(payload) == "SECOND"

    def test_first_non_empty_message_without_modality(self) -> None:
        payload = {
            "choices": [
                {"message": {"content": ""}},
                {"message": {"content": "SECOND"}},
            ]
        }
        assert judge_mod.select_judge_text(payload) == "SECOND"

    def test_single_choice_is_returned(self) -> None:
        payload = {"choices": [{"message": {"content": "ONLY"}}]}
        assert judge_mod.select_judge_text(payload) == "ONLY"

    def test_blank_choices_fall_back_to_first(self) -> None:
        payload = {"choices": [{"message": {"content": ""}}, {"message": {"content": "  "}}]}
        assert judge_mod.select_judge_text(payload) == ""

    def test_list_content_flattens_text_parts(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "A"},
                            {"type": "image_url", "image_url": {"url": "x"}},
                            {"type": "text", "text": "B"},
                        ]
                    }
                }
            ]
        }
        assert judge_mod.select_judge_text(payload) == "AB"

    @pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": "nope"}, {"choices": None}])
    def test_no_choices_raises(self, payload) -> None:
        with pytest.raises(ValueError, match="no choices"):
            judge_mod.select_judge_text(payload)


# ================================================================== #
# ``DuplexJudge`` request-body invariance (T9)
# ================================================================== #


class _FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "OK"}}]}


class TestDuplexJudgeRequestBody:
    """``modalities=None`` keeps the body unchanged; an explicit list opts in."""

    def _patch_transport(self, monkeypatch) -> dict:
        captured: dict = {}

        def _post(url, *, json=None, headers=None, timeout=None, **kwargs):
            captured.update(url=url, body=json, headers=headers, timeout=timeout)
            return _FakeResponse()

        monkeypatch.setattr(judge_mod.requests, "post", _post)
        return captured

    def test_defaults(self) -> None:
        parameters = inspect.signature(judge_mod.DuplexJudge.__init__).parameters
        assert parameters["timeout"].default == 600
        assert parameters["modalities"].default is None
        assert parameters["api_key"].default == "EMPTY"

    def test_body_has_no_extra_keys_without_modalities(self, monkeypatch) -> None:
        captured = self._patch_transport(monkeypatch)
        judge = judge_mod.DuplexJudge("http://judge:8000", "test-judge")
        assert judge.modalities is None

        assert judge.chat("prompt") == "OK"

        assert set(captured["body"]) == {"model", "messages", "temperature", "max_tokens"}
        assert captured["body"]["model"] == "test-judge"
        assert captured["body"]["temperature"] == 0.1
        assert captured["body"]["max_tokens"] == 1200
        assert captured["timeout"] == 600
        assert captured["url"].endswith("/v1/chat/completions")

    def test_modalities_are_written_when_requested(self, monkeypatch) -> None:
        captured = self._patch_transport(monkeypatch)
        judge = judge_mod.DuplexJudge("http://judge:8000", "test-judge", modalities=["text"])
        assert judge.modalities == ("text",)

        judge.chat("prompt")

        assert captured["body"]["modalities"] == ["text"]

    def test_empty_modalities_is_not_written(self, monkeypatch) -> None:
        captured = self._patch_transport(monkeypatch)
        judge = judge_mod.DuplexJudge("http://judge:8000", "test-judge", modalities=[])
        assert judge.modalities is None
        judge.chat("prompt")
        assert "modalities" not in captured["body"]

    def test_system_message_is_prepended(self, monkeypatch) -> None:
        captured = self._patch_transport(monkeypatch)
        judge_mod.DuplexJudge("http://judge:8000", "test-judge").chat("hi", system="be careful")
        messages = captured["body"]["messages"]
        assert messages[0] == {"role": "system", "content": "be careful"}
        assert messages[1]["role"] == "user"

    def test_url_builder_is_unchanged(self) -> None:
        assert judge_mod._build_openai_url("http://h:1", "/chat/completions") == "http://h:1/v1/chat/completions"
        assert judge_mod._build_openai_url("http://h:1/v1", "/chat/completions") == "http://h:1/v1/chat/completions"
        assert (
            judge_mod._build_openai_url("http://h:1/v1/chat/completions", "/chat/completions")
            == "http://h:1/v1/chat/completions"
        )


# ================================================================== #
# ``_content_frame_times`` frame budget (T7)
# ================================================================== #


class TestContentFrameTimes:
    """``max_frames=16`` reproduces the legacy sequence; smaller caps truncate."""

    @pytest.mark.parametrize("duration", [0.0, 1.0, 3.0, 10.0, 47.9, 48.0, 100.0, 300.0, 1000.0])
    def test_default_matches_legacy(self, duration: float) -> None:
        assert eval_mod._content_frame_times(duration) == _old_content_frame_times(duration)
        assert eval_mod._content_frame_times(duration, max_frames=16) == _old_content_frame_times(duration)

    def test_constant_is_sixteen(self) -> None:
        assert eval_mod.CONTENT_FRAME_LIMIT == 16

    def test_smaller_budget_truncates(self) -> None:
        legacy = _old_content_frame_times(300.0)
        capped = eval_mod._content_frame_times(300.0, max_frames=8)
        assert len(capped) == 8
        assert capped[0] == legacy[0]
        assert capped[-1] == legacy[-1]

    def test_short_clip_is_not_padded(self) -> None:
        times = eval_mod._content_frame_times(5.0, max_frames=8)
        assert 0 < len(times) <= 8

    @pytest.mark.parametrize("duration", [0.0, -5.0])
    def test_non_positive_duration_is_empty(self, duration: float) -> None:
        assert eval_mod._content_frame_times(duration) == []


# ================================================================== #
# ``evaluate_sample(..., content_frame_limit=...)`` wiring
# ================================================================== #


class _StubJudge:
    model = "stub-judge"

    def temporal(self, prompt, frames):
        return "{}"

    def content(self, prompt, video=None, frames=None, *, mode="video_url"):
        return "{}"


class TestEvaluateSampleFrameLimit:
    """``content_frame_limit`` reaches ``_content_frame_times(max_frames=...)``."""

    def _prepare(self, tmp_path, monkeypatch) -> Path:
        response_path = tmp_path / "s1.json"
        response_path.write_text(json.dumps([]), encoding="utf-8")
        monkeypatch.setattr(eval_mod, "validate_clock", lambda meta, allow_invalid=False: None)
        monkeypatch.setattr(eval_mod, "materialize_media", lambda *a, **k: tmp_path / "video.mp4")
        monkeypatch.setattr(eval_mod, "video_duration", lambda path: 30.0)
        monkeypatch.setattr(eval_mod, "_extract_frames", lambda path, times: [b"\xff\xd8frame"])
        monkeypatch.setattr(eval_mod, "build_content_prompt", lambda *a, **k: "content-prompt")
        monkeypatch.setattr(eval_mod, "parse_judge_json", lambda text: {})
        monkeypatch.setattr(eval_mod, "summarize_temporal_results", lambda rows: {})
        return response_path

    def _sample(self) -> SimpleNamespace:
        return SimpleNamespace(
            id="s1",
            split="RTD_OCR",
            family="rtd",
            task_type=None,
            video="",
            video_duration=30.0,
            question_text="q",
            answer1="a1",
            answer2="a2",
        )

    def test_explicit_limit_is_forwarded(self, tmp_path, monkeypatch) -> None:
        response_path = self._prepare(tmp_path, monkeypatch)
        recorded: dict = {}

        def _record(duration, *, max_frames=16):
            recorded["max_frames"] = max_frames
            return [0.0, 1.0]

        monkeypatch.setattr(eval_mod, "_content_frame_times", _record)

        eval_mod.evaluate_sample(
            self._sample(),
            response_path,
            tmp_path / "score.json",
            _StubJudge(),
            judge_video_mode="frame-sample",
            content_frame_limit=7,
        )

        assert recorded["max_frames"] == 7

    def test_default_limit_is_constant(self, tmp_path, monkeypatch) -> None:
        response_path = self._prepare(tmp_path, monkeypatch)
        recorded: dict = {}

        def _record(duration, *, max_frames=16):
            recorded["max_frames"] = max_frames
            return [0.0, 1.0]

        monkeypatch.setattr(eval_mod, "_content_frame_times", _record)

        eval_mod.evaluate_sample(
            self._sample(),
            response_path,
            tmp_path / "score.json",
            _StubJudge(),
            judge_video_mode="frame-sample",
        )

        assert recorded["max_frames"] == eval_mod.CONTENT_FRAME_LIMIT

    def test_signature_default_is_constant(self) -> None:
        parameters = inspect.signature(eval_mod.evaluate_sample).parameters
        assert parameters["content_frame_limit"].default == eval_mod.CONTENT_FRAME_LIMIT
