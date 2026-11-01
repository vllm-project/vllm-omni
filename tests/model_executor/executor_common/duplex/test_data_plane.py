# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, DuplexDataPlaneContext
from vllm_omni.model_executor.common.duplex.data_plane import CumulativeAudioTextDataPlane

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode_audio(audio, _sample_rate, _response_format, _speed):
    if audio is None:
        return None
    size = int(np.asarray(audio, dtype=np.float32).size)
    return f"audio-{size}" if size else None


def _output(request_id: str, *, text: str, samples: int, sr: int = 24000) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        outputs=[SimpleNamespace(text=text, multimodal_output={})],
        multimodal_output={"model_outputs": np.arange(samples, dtype=np.float32), "sr": sr},
        finished=False,
    )


class _TestDataPlane(CumulativeAudioTextDataPlane):
    stage_role = "codec"
    default_sample_rate_hz = 16000


def test_is_a_duplex_data_plane_and_projects_cumulative_output_as_deltas() -> None:
    plane = CumulativeAudioTextDataPlane(_encode_audio)
    plane.begin_request("req")
    context = DuplexDataPlaneContext(response_format="wav", modalities=("audio", "text"))

    projected = list(
        plane.project(
            {"data_plane_outputs": [_output("req", text="he", samples=4), _output("req", text="hello", samples=6)]},
            context=context,
        )
    )

    assert isinstance(plane, DuplexDataPlane)
    assert [item["text"] for item in projected] == ["he", "llo"]
    assert [item["audio_data"] for item in projected] == ["audio-4", "audio-2"]
    assert [item["audio_duration_ms"] for item in projected] == [0, 0]
    assert all(item["sample_rate_hz"] == 24000 for item in projected)
    assert all(item["data_plane_request_id"] == "req" for item in projected)
    assert all(item["is_listen"] is False and item["end_of_turn"] is False for item in projected)
    assert projected[0]["stage_role"] == "tts"
    assert projected[0]["runtime_impl"] == "scheduler_data_plane"
    assert projected[0]["audio_format"] == "wav"


def test_subclass_constants_and_default_sample_rate_reach_the_projection() -> None:
    plane = _TestDataPlane(_encode_audio)
    output = SimpleNamespace(
        request_id="req",
        outputs=[SimpleNamespace(text="", multimodal_output={})],
        multimodal_output={"audio": np.zeros(160, dtype=np.float32)},
        finished=False,
    )

    [item] = list(plane.project({"data_plane_outputs": [output]}))

    assert item["stage_role"] == "codec"
    assert item["sample_rate_hz"] == 16000
    assert item["audio_duration_ms"] == 10


def test_nothing_new_yields_nothing_and_non_results_are_ignored() -> None:
    plane = CumulativeAudioTextDataPlane(_encode_audio)
    same = _output("req", text="hi", samples=4)

    assert len(list(plane.project({"data_plane_outputs": [same, same]}))) == 1
    assert list(plane.project("not a dict")) == []
    assert list(plane.project({"data_plane_outputs": None})) == []


def test_terminal_and_close_bookkeeping() -> None:
    plane = CumulativeAudioTextDataPlane(_encode_audio)
    session_request = duplex_resource_request_id(DuplexFence("sid", epoch=0), "stage0")
    plane.begin_request(session_request)
    plane.begin_request("other-req")

    assert plane.is_terminal(None) is False
    assert plane.is_terminal(session_request) is False
    plane.mark_terminal(session_request)
    assert plane.is_terminal(session_request) is True
    plane.begin_request(session_request)
    assert plane.is_terminal(session_request) is False

    plane.close_session("sid", active_request_id="other-req")
    assert plane._requests == {}

    plane.mark_terminal("late")
    plane.close_stream("late")
    assert plane.is_terminal("late") is False
