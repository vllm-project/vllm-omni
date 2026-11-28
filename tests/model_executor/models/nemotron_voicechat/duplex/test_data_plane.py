# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.data_plane import (
    NemotronVoiceChatDataPlaneContext,
    NemotronVoiceChatDataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_RUNTIME = {
    "nvc_text_bos_id": 0,
    "nvc_text_eos_id": 1,
    "nvc_text_pad_id": 12,
    "nvc_function_sotc_id": 20,
    "nvc_function_eotc_id": 21,
    "nvc_tokenizer_ref": "test-tokenizer",
}


def _projector(encode_audio=lambda *_: None) -> NemotronVoiceChatDataPlaneSession:
    projector = NemotronVoiceChatDataPlaneSession(encode_audio)
    projector.configure_runtime(_RUNTIME, tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: "text"))
    return projector


def _stage0_output(
    request_id: str = "req-0",
    *,
    text_token: int = 12,
    function_token: int | None = None,
    fast_path: bool | None = None,
) -> SimpleNamespace:
    metadata: dict[str, object] = {"nvc_text_token_ids": [text_token]}
    if function_token is not None:
        metadata["nvc_function_token"] = [function_token]
    if fast_path is not None:
        metadata["nvc_function_fast_path"] = int(fast_path)
    completion = SimpleNamespace(multimodal_output=metadata)
    return SimpleNamespace(
        stage_id=0,
        request_output=SimpleNamespace(request_id=request_id, outputs=[completion]),
    )


def test_function_channel_projects_completed_call_without_ending_speech() -> None:
    projector = _projector()
    projector._decode = lambda token_ids: (
        '[{"name":"weather","arguments":{"city":"Shanghai"}}]' if token_ids == [99] else ""
    )
    context = NemotronVoiceChatDataPlaneContext(epoch=0)

    events = []
    for function_token in (20, 99, 21):
        events.extend(projector.project_output(_stage0_output(function_token=function_token), context=context))

    listen = [event for event in events if event.get("is_listen") is True]
    function = [event for event in events if event.get("function_call") is True]
    assert len(listen) == 3
    assert len(function) == 1
    assert function[0]["name"] == "weather"
    assert function[0]["arguments"] == '{"city":"Shanghai"}'

    projector._decode = lambda _token_ids: "not-json"
    bad_events = []
    for function_token in (20, 98, 21):
        bad_events.extend(projector.project_output(_stage0_output(function_token=function_token)))
    assert any(event.get("error_code") == "nemotron_function_call_parse_error" for event in bad_events)


def test_new_epoch_request_does_not_inherit_partial_function_or_frame_state() -> None:
    projector = _projector(lambda *_: "audio")
    old_request = duplex_resource_request_id(DuplexFence("sid", epoch=1), "stage0")
    new_request = duplex_resource_request_id(DuplexFence("sid", epoch=2), "stage0")

    projector.begin_request(old_request)
    list(projector.project_output(_stage0_output(old_request, text_token=42, function_token=20)))
    projector.begin_request(new_request)

    # EOTC in the new epoch must not close the old epoch's partial function.
    new_events = list(projector.project_output(_stage0_output(new_request, text_token=1, function_token=21)))
    assert not [event for event in new_events if event.get("function_call") is True]
    assert not [event for event in new_events if event.get("end_of_turn") is True]

    audio = SimpleNamespace(
        stage_id=2,
        request_output=SimpleNamespace(
            request_id=new_request,
            outputs=[
                SimpleNamespace(multimodal_output={"model_outputs": [np.ones(1764, dtype=np.float32)], "sr": [22050]})
            ],
        ),
    )
    audio_events = list(projector.project_output(audio))
    assert sum(event.get("end_of_turn") is True for event in audio_events) == 1


def test_delayed_tool_audio_counts_coverage_without_swallowing_speech_across_responses():
    encoded = []

    def encode(audio, *_):
        encoded.append(np.asarray(audio).copy())
        return "audio"

    projector = _projector(encode)
    projector._decode = lambda ids: '{"name":"lookup","arguments":{}}' if ids == [99] else "text"

    def text(token, fast=False, function=None):
        output = _stage0_output(text_token=token, function_token=function, fast_path=fast)
        return list(projector.project_output(output))

    def audio(values):
        output = SimpleNamespace(
            stage_id=2,
            request_id="req-0",
            outputs=[
                SimpleNamespace(
                    multimodal_output={
                        "audio": np.repeat(np.asarray(values, dtype=np.float32), 1764),
                        "sr": 22050,
                    }
                )
            ],
        )
        return list(projector.project_output(output))

    for _ in range(3):
        text(42)
        text(12, fast=True, function=20)
        # Speech from before SOTC arrives while the call is active.
        assert audio([1])[0]["audio_data"] == "audio"
        text(12, fast=True, function=99)
        calls = text(12, function=21)  # EOTC: the current fast flag is already clear.
        assert sum(event.get("function_call") is True for event in calls) == 1
        # Delayed tool audio is still filtered and accounts for both frames.
        assert audio([2, 3]) == []
        text(1)
        events = audio([4, 5])
        assert events[-1]["end_of_turn"] is True
        assert encoded[-1].tolist() == [4] * 1764 + [5] * 1764
        state = projector._requests["req-0"]
        assert state.audio_frames == state.text_frames
        assert state.pending_speech_end_frames == []
        assert state.suppressed_audio_frames == set()
    # A coalesced chunk spanning speech, a call and EOS filters only the call.
    text(42)
    text(12, fast=True)
    text(1)
    assert audio([6, 7, 8])[-1]["end_of_turn"] is True
    assert encoded[-1].tolist() == [6] * 1764 + [8] * 1764
