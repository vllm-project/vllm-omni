# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Composition of PCM deltas with accepted-input/transport-delivery totals."""

from dataclasses import dataclass

import numpy as np
import pytest

from vllm_omni.model_executor.models.personaplex.duplex.data_plane import PersonaPlexDataPlaneSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Output:
    request_id: str
    multimodal_output: dict[str, object]


def _output(request_id: str, samples: int, text: str = "") -> _Output:
    return _Output(request_id, {"audio": np.ones(samples, dtype=np.float32), "sr": 24000, "text": text})


@pytest.mark.parametrize("sizes", [(1920, 1920), (3840, 1920), (0, 1920, 0, 1920)])
@pytest.mark.parametrize("sessions", [1, 2])
def test_delta_totals_are_distinct_from_delivered_watermark(sizes, sessions):
    """Projection accumulates deltas; only the ordered send hook acknowledges them."""

    def encode(audio, _rate, _format, _speed):
        return "encoded" if audio is not None and np.asarray(audio).size else None

    projector = PersonaPlexDataPlaneSession(encode)
    request_ids = [f"request-{i}" for i in range(sessions)]
    for request_id in request_ids:
        projector.note_accepted_input(request_id, sum(sizes) // 1920 + 1)
    total = 0
    for size in sizes:
        for request_id in request_ids:
            events = list(projector.project({"data_plane_outputs": [_output(request_id, size)]}))
            assert len(events) == int(size > 0)
            assert projector.drain_status(request_id)["audio_frames"] == total // 1920
            assert not projector.drain_status(request_id)["drained"]
            projector.mark_outputs_delivered(request_id)
            assert projector.drain_status(request_id)["audio_frames"] == (total + size) // 1920
        total += size
    for request_id in request_ids:
        assert projector.drain_status(request_id) == {
            "accepted_frames": total // 1920 + 1,
            "expected_audio_frames": total // 1920,
            "model_delay_frames": 1,
            "audio_frames": total // 1920,
            "drained": True,
        }


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_failed_encode_cannot_advance_delivery_or_transcript(failure):
    """Neither a rejected nor a raising encoder may consume pending output."""
    fail = False

    def encode(audio, _rate, _format, _speed):
        if fail:
            if failure == "exception":
                raise ValueError("encoder failure")
            return None
        return "encoded"

    projector = PersonaPlexDataPlaneSession(encode)
    projector.note_accepted_input("request", 3)
    list(projector.project({"data_plane_outputs": [_output("request", 1920, "first")]}))
    projector.mark_outputs_delivered("request")
    fail = True
    pending = _output("request", 1920, "first second")
    with pytest.raises((RuntimeError, ValueError)):
        list(projector.project({"data_plane_outputs": [pending]}))
    projector.mark_outputs_delivered("request")
    assert projector.drain_status("request")["audio_frames"] == 1
    assert not projector.drain_status("request")["drained"]
    fail = False
    retried = list(projector.project({"data_plane_outputs": [pending]}))
    assert len(retried) == 1 and retried[0]["text"] == " second"
    assert not projector.drain_status("request")["drained"]
    projector.mark_outputs_delivered("request")
    assert projector.drain_status("request")["drained"]


def test_close_resets_totals_without_touching_another_stream():
    """Reusing one request ID must not retain its counters or reset another's."""
    projector = PersonaPlexDataPlaneSession(lambda *_: "encoded")
    for request_id in ("first", "second"):
        projector.note_accepted_input(request_id, 2)
        list(projector.project({"data_plane_outputs": [_output(request_id, 1920)]}))
        projector.mark_outputs_delivered(request_id)
    projector.mark_terminal("first")
    projector.close_stream("first")
    assert projector.drain_status("second")["drained"]
    projector.begin_request("first")
    projector.note_accepted_input("first", 2)
    assert projector.drain_status("first")["audio_frames"] == 0
    assert not projector.drain_status("first")["drained"]
