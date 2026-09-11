# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.messages import DuplexFence
from vllm_omni.engine.duplex.runtime import duplex_resource_request_id
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


def _stage0_output(request_id: str = "req-0", *, text_token: int = 12, function_token: int | None = None) -> object:
    metadata: dict[str, object] = {"nvc_text_token_ids": [text_token]}
    if function_token is not None:
        metadata["nvc_function_token"] = [function_token]
    completion = SimpleNamespace(multimodal_output=metadata)
    return SimpleNamespace(
        stage_id=0,
        request_output=SimpleNamespace(request_id=request_id, outputs=[completion]),
    )


def _audio_output(frames: int, *, request_id: str = "req-0"):
    return SimpleNamespace(
        stage_id=2,
        request_output=SimpleNamespace(
            request_id=request_id,
            outputs=[SimpleNamespace(multimodal_output={"audio": np.ones(1764 * frames), "sr": 22050})],
        ),
    )


def test_utterance_eos_preserves_continuous_audio_including_silence():
    projector = _projector(lambda *_: "audio")
    text_events = []
    for token in (42, 1, 12, 12):
        text_events.extend(projector.project_output(_stage0_output(text_token=token)))

    events = list(projector.project_output(_audio_output(4)))

    assert len(events) == 4
    assert [event["audio_duration_ms"] for event in events] == [80] * 4
    assert not any(event["end_of_turn"] for event in events)
    assert [event["transcript"] for event in text_events if event.get("transcript_done")] == ["text"]
    assert projector._requests["req-0"].audio_frames == 4

    # A later utterance must still emit audio; the retained stream is not
    # terminal simply because its previous response reached model EOS.
    list(projector.project_output(_stage0_output(text_token=43)))
    next_events = list(projector.project_output(_audio_output(1)))
    assert len(next_events) == 1 and next_events[0]["audio_data"] == "audio"


def test_delayed_eos_is_transcript_only_not_stream_completion():
    projector = _projector(lambda *_: "audio")
    list(projector.project_output(_stage0_output(text_token=42)))
    list(projector.project_output(_audio_output(2)))
    end = list(projector.project_output(_stage0_output(text_token=1)))
    assert len(end) == 1 and end[0]["transcript_done"] is True
    list(projector.project_output(_stage0_output(text_token=12)))
    following = list(projector.project_output(_audio_output(1)))
    assert len(following) == 1 and following[0]["end_of_turn"] is False


def test_continuous_drain_uses_delivered_frames_and_deduplicated_engine_receipts():
    projector = _projector(lambda *_: "audio")
    for seq in (1, 2, 2):
        projector.note_accepted_input("req-0", seq)
    for token in (42, 12):
        list(projector.project_output(_stage0_output(text_token=token)))
    list(projector.project_output(_audio_output(1)))
    projector.mark_outputs_delivered("req-0")
    assert projector.drain_status("req-0")["drained"] is False
    list(projector.project_output(_audio_output(1)))
    assert projector.drain_status("req-0")["drained"] is False
    projector.mark_outputs_delivered("req-0")
    assert projector.drain_status("req-0") == {
        "accepted_frames": 2,
        "text_frames": 2,
        "audio_frames": 2,
        "drained": True,
    }


def test_continuous_drain_rejects_duplicate_output_frames():
    projector = _projector(lambda *_: "audio")
    projector.note_accepted_input("req-0", 1)
    list(projector.project_output(_audio_output(2)))
    projector.mark_outputs_delivered("req-0")
    with pytest.raises(RuntimeError, match="more frames"):
        projector.drain_status("req-0")


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
    assert any(event.get("transcript_done") is True for event in new_events)

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
    assert not any(event.get("end_of_turn") is True for event in audio_events)
