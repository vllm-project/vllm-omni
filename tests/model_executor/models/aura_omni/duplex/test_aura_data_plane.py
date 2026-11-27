# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for AuraDataPlaneSession (unwrap runner envelope; do not drop audio)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.model_executor.models.aura_omni.duplex.data_plane import AuraDataPlaneSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
    assert events[-1].get("abort_data_plane_request") is True
    assert events[0].get("data_plane_request_id") == request_id


def test_project_sends_only_the_new_tail_of_a_growing_waveform() -> None:
    seen: list[int] = []

    def encode(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str:
        del sample_rate, fmt, speed
        count = int(np.asarray(audio).reshape(-1).size)
        seen.append(count)
        return f"pcm-{count}"

    plane = AuraDataPlaneSession(encode)
    request_id = "duplex-s.abc.e.0.r.stage3-turn1"
    plane.begin_request(request_id)

    def chunk(n: int, *, finished: bool) -> SimpleNamespace:
        audio = np.arange(n, dtype=np.float32)
        return SimpleNamespace(
            request_id=request_id,
            finished=finished,
            stage_id=3,
            outputs=[
                SimpleNamespace(
                    text="",
                    cumulative_text="",
                    finished=False,
                    multimodal_output={"audio": audio, "sr": 24000},
                )
            ],
        )

    first = list(plane.project_output(chunk(8, finished=False)))
    second = list(plane.project_output(chunk(12, finished=True)))
    assert seen == [8, 4]
    assert first[0]["audio_duration_ms"] == round(8 * 1000 / 24000)
    assert second[0]["audio_duration_ms"] == round(4 * 1000 / 24000)
    assert second[-1].get("end_of_turn") is True
    # A parked sentence-TTS Talker request must be aborted with the turn, not
    # only forgotten; mid-turn chunks must not abort it.
    assert not first[0].get("abort_data_plane_request")
    assert second[-1].get("abort_data_plane_request") is True


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
    listen = [e for e in events if e.get("silent") and e.get("is_listen")]
    assert listen
    assert any(e.get("abort_data_plane_request") is True for e in events)
    assert listen[-1].get("listen_source") == "aura_silent"
    assert listen[-1].get("reason") == "aura_silent"
    assert listen[-1].get("model_listen") is True


def test_project_finished_spoken_turn_carries_model_context() -> None:
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1-turn5"
    plane.begin_request(request_id)
    final = SimpleNamespace(
        request_id=request_id,
        finished=True,
        stage_id=1,
        outputs=[SimpleNamespace(text="你好。", cumulative_text="你好。", finished=True)],
    )
    events = list(plane.project({"data_plane_outputs": [final]}))
    context = [event for event in events if event.get("model_context_text")]
    assert context and context[-1]["model_context_text"] == "你好。"
    assert context[-1].get("is_listen") is False
    assert context[-1].get("text") == ""
    assert context[-1].get("listen_source") is None
    assert context[-1].get("reason") is None


def test_project_box_only_turn_keeps_the_tokens_and_ends_without_audio() -> None:
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1-turn6"
    plane.begin_request(request_id)
    box = "<|box_start|>(712.0,531.0),(888.0,662.0)<|box_end|>"
    final = SimpleNamespace(
        request_id=request_id,
        finished=True,
        stage_id=1,
        outputs=[SimpleNamespace(text=box, cumulative_text=box, finished=True)],
    )
    events = list(plane.project({"data_plane_outputs": [final]}))
    assert any(event.get("text") == box for event in events)
    context = [event for event in events if event.get("model_context_text")]
    assert context and context[-1]["model_context_text"] == box
    assert events[-1].get("end_of_turn") is True
    assert events[-1].get("abort_data_plane_request") is True
    assert all(event.get("is_listen") is not True for event in events)
    assert all(not event.get("audio") for event in events)


def test_project_box_plus_tool_keeps_the_box_and_ends_the_turn() -> None:
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1-turn9"
    plane.begin_request(request_id)
    box = "<|box_start|>(712.0,531.0),(888.0,662.0)<|box_end|>"
    text = f"<|respond|>{box}<tool_call>\n<function=get_city_weather>\n</function>\n</tool_call>"
    final = SimpleNamespace(
        request_id=request_id,
        finished=True,
        stage_id=1,
        outputs=[SimpleNamespace(text=text, cumulative_text=text, finished=True)],
    )
    events = list(plane.project({"data_plane_outputs": [final]}))
    assert any(event.get("text") == box for event in events)
    calls = [event for event in events if event.get("aura_function_call")]
    assert len(calls) == 1
    assert calls[0]["aura_tool_call_ends_turn"] is True
    assert calls[0]["aura_function_call"]["name"] == "get_city_weather"


def test_project_prose_with_a_box_and_a_tool_call_keeps_the_turn_open() -> None:
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1-turn10"
    plane.begin_request(request_id)
    box = "<|box_start|>(1,2),(3,4)<|box_end|>"
    text = f"<|respond|>在這裡。{box}<tool_call>\n<function=get_city_weather>\n</function>\n</tool_call>"
    final = SimpleNamespace(
        request_id=request_id,
        finished=True,
        stage_id=1,
        outputs=[SimpleNamespace(text=text, cumulative_text=text, finished=True)],
    )
    events = list(plane.project({"data_plane_outputs": [final]}))
    assert any("在這裡。" in str(event.get("text") or "") and box in str(event.get("text") or "") for event in events)
    calls = [event for event in events if event.get("aura_function_call")]
    assert [event["aura_tool_call_ends_turn"] for event in calls] == [False]


def test_project_emits_every_tool_call_and_only_the_last_releases_the_turn() -> None:
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1-turn7"
    plane.begin_request(request_id)
    text = (
        "<|respond|><tool_call>\n<function=get_current_datetime>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_current_location>\n</function>\n</tool_call>"
    )
    final = SimpleNamespace(
        request_id=request_id,
        finished=True,
        stage_id=1,
        outputs=[SimpleNamespace(text=text, cumulative_text=text, finished=True)],
    )
    calls = [event for event in plane.project({"data_plane_outputs": [final]}) if event.get("aura_function_call")]
    assert [event["aura_function_call"]["name"] for event in calls] == [
        "get_current_datetime",
        "get_current_location",
    ]
    assert [event["aura_tool_call_ends_turn"] for event in calls] == [False, True]


def test_project_does_not_republish_a_tool_call_when_stop_text_is_appended() -> None:
    plane = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1-turn8"
    plane.begin_request(request_id)
    text = "<|respond|>先查一下。<tool_call>\n<function=calculator>\n<parameter=expression>\n1\n</parameter>\n</function>\n</tool_call>"
    first = SimpleNamespace(
        request_id=request_id,
        finished=True,
        stage_id=1,
        outputs=[SimpleNamespace(text=text, cumulative_text=text, finished=True)],
    )
    second = SimpleNamespace(
        request_id=request_id,
        finished=True,
        stage_id=1,
        outputs=[SimpleNamespace(text=text + "<|im_end|>", cumulative_text=text + "<|im_end|>", finished=True)],
    )
    first_calls = [
        event["aura_function_call"]["call_id"]
        for event in plane.project({"data_plane_outputs": [first]})
        if event.get("aura_function_call")
    ]
    second_calls = [
        event["aura_function_call"]["call_id"]
        for event in plane.project({"data_plane_outputs": [second]})
        if event.get("aura_function_call")
    ]
    assert len(first_calls) == 1
    assert second_calls == []
