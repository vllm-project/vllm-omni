# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.model_executor.models.qwen3_omni.duplex.data_plane import (
    Qwen3OmniDataPlaneContext,
    Qwen3OmniDataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int, str, float | None]] = []

    def __call__(self, audio: object, sample_rate: int, fmt: str, speed: float | None) -> str | None:
        self.calls.append((audio, sample_rate, fmt, speed))
        samples = int(np.asarray(audio, dtype=np.float32).size)
        return f"enc-{samples}-{fmt}"


def _wav(*, request_id: str, audio: object, finished: bool = False, stage_id: int = 2) -> SimpleNamespace:
    mm = {"audio": audio, "sr": 24000}
    return SimpleNamespace(
        request_id=request_id,
        stage_id=stage_id,
        finished=finished,
        outputs=[SimpleNamespace(text="", multimodal_output=mm)],
        multimodal_output=mm,
    )


def test_data_plane_projects_thinker_text_then_audio() -> None:
    encoder = RecordingEncoder()
    plane = Qwen3OmniDataPlaneSession(encoder)
    request_id = "sess.e0.r.stage0_t0"
    plane.begin_request(request_id)

    thinker = SimpleNamespace(
        request_id=request_id,
        stage_id=0,
        finished=False,
        outputs=[SimpleNamespace(text="hello", cumulative_text="hello", multimodal_output={})],
        multimodal_output={},
    )
    text_events = list(plane.project_output(thinker))
    assert text_events == [
        {
            "stage_role": "thinker",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": "hello",
            "end_of_turn": False,
        }
    ]

    audio = np.zeros(8, dtype=np.float32)
    audio_events = list(plane.project_output(_wav(request_id=request_id, audio=audio, finished=True)))
    assert audio_events[0]["stage_role"] == "tts"
    assert audio_events[0]["audio"] == "enc-8-wav"
    assert audio_events[0]["end_of_turn"] is True
    assert plane.is_terminal(request_id) is True


def test_data_plane_encodes_only_new_cumulative_samples_with_session_format() -> None:
    encoder = RecordingEncoder()
    plane = Qwen3OmniDataPlaneSession(encoder)
    request_id = "sess.e0.r.stage0_t0"
    plane.begin_request(request_id)
    context = Qwen3OmniDataPlaneContext(response_format="pcm", speed=1.25)

    first = np.arange(4, dtype=np.float32)
    second = np.arange(10, dtype=np.float32)
    events1 = list(plane.project_output(_wav(request_id=request_id, audio=first), context=context))
    events2 = list(plane.project_output(_wav(request_id=request_id, audio=second, finished=True), context=context))

    assert [call[2] for call in encoder.calls] == ["pcm", "pcm"]
    assert [call[3] for call in encoder.calls] == [1.25, 1.25]
    assert np.array_equal(np.asarray(encoder.calls[0][0], dtype=np.float32), first)
    assert np.array_equal(np.asarray(encoder.calls[1][0], dtype=np.float32), second[4:])
    assert events1[0]["audio_format"] == "pcm"
    assert events1[0]["audio_duration_ms"] == int(4 * 1000 / 24000)
    assert events2[0]["audio"] == "enc-6-pcm"
    assert events2[0]["end_of_turn"] is True


def test_data_plane_drains_new_list_chunks_only() -> None:
    encoder = RecordingEncoder()
    plane = Qwen3OmniDataPlaneSession(encoder)
    request_id = "sess.e0.r.stage0_t0"
    plane.begin_request(request_id)
    context = Qwen3OmniDataPlaneContext(response_format="pcm")

    chunk0 = np.ones(3, dtype=np.float32)
    chunk1 = np.full(5, 2.0, dtype=np.float32)
    events1 = list(plane.project_output(_wav(request_id=request_id, audio=[chunk0]), context=context))
    events2 = list(
        plane.project_output(_wav(request_id=request_id, audio=[chunk0, chunk1], finished=True), context=context)
    )

    assert len(encoder.calls) == 2
    assert np.array_equal(np.asarray(encoder.calls[0][0], dtype=np.float32), chunk0)
    assert np.array_equal(np.asarray(encoder.calls[1][0], dtype=np.float32), chunk1)
    assert events1[0]["audio"] == "enc-3-pcm"
    assert events1[0]["end_of_turn"] is False
    assert events2[0]["audio"] == "enc-5-pcm"
    assert events2[0]["audio_format"] == "pcm"
    assert events2[0]["end_of_turn"] is True


def test_data_plane_ignores_talker_latent() -> None:
    encoder = RecordingEncoder()
    plane = Qwen3OmniDataPlaneSession(encoder)
    request_id = "sess.e0.r.stage0_t0"
    plane.begin_request(request_id)
    latent = np.ones(8, dtype=np.float32)
    talker = SimpleNamespace(
        request_id=request_id,
        stage_id=1,
        finished=False,
        outputs=[SimpleNamespace(text="", multimodal_output={"latent": latent})],
        multimodal_output={"latent": latent},
    )
    assert list(plane.project_output(talker)) == []
    assert encoder.calls == []
