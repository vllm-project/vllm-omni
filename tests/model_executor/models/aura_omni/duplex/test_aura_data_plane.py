# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for AuraDataPlaneSession (unwrap runner envelope; do not drop audio)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vllm_omni.model_executor.models.aura_omni.duplex.data_plane import AuraDataPlaneSession


def _encode_audio(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str | None:
    del sample_rate, fmt, speed
    if audio is None:
        return None
    return "encoded-audio"


def test_project_unwraps_data_plane_outputs_and_emits_audio_then_done() -> None:
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage0-turn1"
    plane.begin_request(request_id)
    audio = np.zeros(16, dtype=np.float32)
    output = SimpleNamespace(
        request_id=request_id,
        finished=True,
        stage_id=3,
        outputs=[
            SimpleNamespace(
                text="",
                cumulative_text="",
                finished=True,
                multimodal_output={"audio": audio, "sr": 24000},
            )
        ],
        multimodal_output={"audio": audio, "sr": 24000},
    )
    events = list(plane.project({"data_plane_outputs": [output]}))
    assert events, "project must not silently drop the runner envelope"
    assert any(event.get("audio") == "encoded-audio" for event in events)
    assert events[-1].get("end_of_turn") is True
    assert events[0].get("data_plane_request_id") == request_id


def test_project_holds_silent_token_prefix() -> None:
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1-turn1"
    plane.begin_request(request_id)
    partial = SimpleNamespace(
        request_id=request_id,
        finished=False,
        stage_id=1,
        outputs=[SimpleNamespace(text="<|sil", cumulative_text="<|sil", finished=False)],
    )
    assert list(plane.project({"data_plane_outputs": [partial]})) == []


def test_project_emits_chinese_brackets_immediately() -> None:
    """[沉默] is ordinary text — do not hold as a <|silent|> prefix."""
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1_t2"
    plane.begin_request(request_id)
    partial = SimpleNamespace(
        request_id=request_id,
        finished=False,
        stage_id=1,
        outputs=[SimpleNamespace(text="[", cumulative_text="[", finished=False)],
    )
    events = list(plane.project({"data_plane_outputs": [partial]}))
    assert len(events) == 1
    assert events[0].get("text") == "["


def test_project_flushes_held_silent_prefix_when_text_becomes_speech() -> None:
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1-turn3"
    plane.begin_request(request_id)
    partial = SimpleNamespace(
        request_id=request_id,
        finished=False,
        stage_id=1,
        outputs=[SimpleNamespace(text="<|", cumulative_text="<|", finished=False)],
    )
    assert list(plane.project({"data_plane_outputs": [partial]})) == []
    spoken = SimpleNamespace(
        request_id=request_id,
        finished=False,
        stage_id=1,
        outputs=[SimpleNamespace(text="<|hello", cumulative_text="<|hello", finished=False)],
    )
    events = list(plane.project({"data_plane_outputs": [spoken]}))
    assert len(events) == 1
    assert events[0].get("text") == "<|hello"


def test_project_marks_finished_silent_text_as_listen() -> None:
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1_t4"
    plane.begin_request(request_id)
    final = SimpleNamespace(
        request_id=request_id,
        finished=True,
        stage_id=1,
        outputs=[SimpleNamespace(text="<|silent|>", cumulative_text="<|silent|>", finished=True)],
    )
    events = list(plane.project({"data_plane_outputs": [final]}))
    assert any(e.get("silent") and e.get("is_listen") for e in events)
