# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Aura duplex data-plane must surface Code2Wav MultimodalPayload audio."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.aura_omni.duplex.data_plane import (
    AuraDataPlaneSession,
    _audio_value,
    _multimodal,
)
from vllm_omni.outputs.mm_outputs import MultimodalPayload


def _encode(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str | None:
    del sample_rate, fmt, speed
    if audio is None:
        return None
    return "encoded-audio"


def test_audio_value_accepts_from_raw_audio_key() -> None:
    raw = {"model_outputs": [torch.zeros(8)], "sr": [torch.tensor(24000)]}
    mm = MultimodalPayload.from_raw(raw, "audio")
    assert mm is not None
    assert "audio" in mm
    audio = _audio_value(dict(mm))
    assert isinstance(audio, torch.Tensor)
    assert int(audio.numel()) == 8


def test_audio_value_accepts_wrongly_tagged_text_key() -> None:
    # If stage output_modality is wrong, from_raw remaps model_outputs → "text".
    raw = {"model_outputs": torch.zeros(4), "sr": torch.tensor(24000)}
    mm = MultimodalPayload.from_raw(raw, "text")
    assert mm is not None
    audio = _audio_value(dict(mm))
    assert isinstance(audio, torch.Tensor)


def test_multimodal_skips_empty_outer_dict_and_reads_completion() -> None:
    mm = MultimodalPayload.from_dict({"audio": torch.zeros(3), "sr": torch.tensor(24000)})
    completion = SimpleNamespace(multimodal_output=mm)
    # Empty Mapping on the outer object must not hide completion payload.
    outer = SimpleNamespace(multimodal_output={})
    got = _multimodal(outer, completion)
    assert "audio" in got


def test_project_emits_audio_from_multimodal_completion() -> None:
    plane = AuraDataPlaneSession(_encode)
    request_id = "duplex-s.test.e.0.r.stage0-turn0"
    plane.begin_request(request_id)
    mm = MultimodalPayload.from_raw(
        {"model_outputs": [torch.zeros(16)], "sr": [torch.tensor(24000)]},
        "audio",
    )
    completion = SimpleNamespace(
        text="",
        cumulative_text="",
        finished=False,
        multimodal_output=mm,
    )
    output = SimpleNamespace(
        request_id=request_id,
        finished=False,
        stage_id=3,
        outputs=[completion],
        multimodal_output={},
    )
    events = list(plane.project({"data_plane_outputs": [output]}))
    assert events, "project must emit TTS audio for MultimodalPayload"
    assert any(event.get("audio") == "encoded-audio" for event in events)


def test_streaming_chunk_completion_finished_is_not_turn_eos() -> None:
    """Code2Wav chunks set CompletionOutput.finished; that must not EOT.

    Runner._send_one_model_output_event drops is_terminal request ids before
    emit. Marking terminal (or end_of_turn) on the first 8-frame chunk made
    WS response.* never leave the engine.
    """
    plane = AuraDataPlaneSession(_encode)
    request_id = "duplex-s.test.e.0.r.stage0-turn0"
    plane.begin_request(request_id)
    mm = MultimodalPayload.from_raw(
        {"model_outputs": [torch.zeros(16)], "sr": [torch.tensor(24000)]},
        "audio",
    )
    completion = SimpleNamespace(
        text="",
        cumulative_text="",
        finished=True,
        multimodal_output=mm,
    )
    output = SimpleNamespace(
        request_id=request_id,
        finished=False,
        stage_id=3,
        outputs=[completion],
        multimodal_output={},
    )
    events: list[dict[str, object]] = []
    for event in plane.project({"data_plane_outputs": [output]}):
        assert plane.is_terminal(request_id) is False
        events.append(event)
    assert events
    assert events[0].get("audio") == "encoded-audio"
    assert events[0].get("end_of_turn") is False
    assert plane.is_terminal(request_id) is False
