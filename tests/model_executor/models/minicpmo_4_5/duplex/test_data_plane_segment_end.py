# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deduplicate output boundaries independently of input submission timing."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import (
    MiniCPMO45DataPlaneContext,
    MiniCPMO45DataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _boundary(seq: int, *, samples: int = 0, cache_epoch: int = 0):
    return SimpleNamespace(
        request_id="request",
        finished=False,
        outputs=[SimpleNamespace(text="hello" if samples else "", token_ids=[], multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(samples, dtype=np.float32),
            "sr": 24000,
            "meta.duplex_turn_id": 0,
            "meta.duplex_epoch": 0,
            "meta.tts_is_last_chunk": True,
            "meta.duplex_audio_chunk_seq": seq,
            "meta.duplex_audio_cache_epoch": cache_epoch,
        },
    )


def _project(plane, output):
    context = MiniCPMO45DataPlaneContext(auto_responds=True, active_response_id="response", active_response_turn_id=0)
    return list(plane.project_output(output, context=context))


@pytest.mark.parametrize("samples", [0, 24000])
def test_queued_inputs_do_not_suppress_later_segment_boundaries(samples):
    plane = MiniCPMO45DataPlaneSession(
        lambda audio, *args: "audio" if audio is not None and np.asarray(audio).size else ""
    )
    # Two inputs can already be in flight before the first TTS segment ends.
    plane.begin_request("request")
    plane.begin_request("request")
    first = _project(plane, _boundary(0, samples=samples))
    second = _project(plane, _boundary(1, samples=2 * samples))
    assert sum(bool(item.get("abort_data_plane_request")) for item in first) == 1
    assert sum(bool(item.get("abort_data_plane_request")) for item in second) == 1
    assert all(not item["end_of_turn"] for item in first + second)


def test_new_input_does_not_rearm_an_already_delivered_boundary():
    plane = MiniCPMO45DataPlaneSession(
        lambda audio, *args: "audio" if audio is not None and np.asarray(audio).size else ""
    )
    plane.begin_request("request")
    assert _project(plane, _boundary(3))
    plane.begin_request("request")
    assert _project(plane, _boundary(3)) == []
    assert _project(plane, _boundary(2)) == []
    assert _project(plane, _boundary(4))


def test_chunk_sequence_can_restart_in_a_new_audio_cache_epoch():
    plane = MiniCPMO45DataPlaneSession(
        lambda audio, *args: "audio" if audio is not None and np.asarray(audio).size else ""
    )
    plane.begin_request("request")
    assert _project(plane, _boundary(8))
    assert _project(plane, _boundary(0, cache_epoch=1))
    assert _project(plane, _boundary(9, cache_epoch=0)) == []
