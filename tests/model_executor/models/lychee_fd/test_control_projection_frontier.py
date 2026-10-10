# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.lychee_fd.duplex.codec import CodecStreamState
from vllm_omni.model_executor.models.lychee_fd.duplex.data_plane import LycheeDataPlaneSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
REQ = "duplex-s.cHJvYmU.e.0.r.stage0"


def project(plane, ticks, controls):
    payload = {
        "lychee_tick": torch.tensor(ticks),
        "lychee_control_token_ids": torch.tensor(controls),
        "lychee_text_token_ids": torch.tensor([158358] * len(ticks)),
        "lychee_speech_token_ids": torch.tensor([158359] * len(ticks)),
    }
    completion = SimpleNamespace(multimodal_output=payload, finish_reason=None)
    output = SimpleNamespace(request_id=REQ, outputs=[completion])
    return list(plane.project({"data_plane_outputs": [output]}))


@pytest.mark.parametrize("eos_tick", [10, 19])
def test_cumulative_eos_cannot_rewind_published_control_frontier(eos_tick):
    plane = LycheeDataPlaneSession()
    plane.codec_streams.states[REQ] = CodecStreamState()
    ticks = [9, eos_tick, 29]
    controls = [158357, 158353, 158357]
    assert [event["lychee_tick"] for event in project(plane, ticks, controls)] == [9, 29]
    assert project(plane, ticks, controls) == []
    following = project(plane, ticks + [39], controls + [158357])
    assert [event["lychee_tick"] for event in following] == [39]
    assert plane._last_projected_tick[REQ] == 39


def test_new_eos_advances_frontier_and_stream_close_resets_request_cursor():
    plane = LycheeDataPlaneSession()
    plane.codec_streams.states[REQ] = CodecStreamState()
    assert [event["lychee_tick"] for event in project(plane, [9], [158357])] == [9]
    assert project(plane, [9, 10], [158357, 158353]) == []
    assert plane._last_projected_tick[REQ] == 10
    plane.close_stream(REQ)
    assert [event["lychee_tick"] for event in project(plane, [9], [158357])] == [9]
