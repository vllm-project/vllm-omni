# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deferred codec output must survive reuse of captured graph storage."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import _MossCodecStreamSession

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


class _Codec(nn.Module):
    downsample_rate = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((), device="cuda"))

    def initialize_decoder_state_pool(self, capacity, scratch):
        self.state = torch.zeros(capacity + scratch, device="cuda")

    def reset_decoder_state_slots(self, slots):
        self.state[slots] = 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("device_output", [False, True])
@torch.inference_mode()
def test_output_survives_graph_replay_and_terminal_slot_reuse(device_output):
    codec = _Codec()
    session = _MossCodecStreamSession(
        codec, state_capacity=2, n_vq=2, vllm_config=None, return_device_audio=device_output
    )
    assert [session.acquire(), session.acquire()] == [0, 1]
    static_codes = torch.zeros((2, 2, 3), device="cuda", dtype=torch.long)
    static_audio = torch.zeros((2, 2, 6), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            static_audio.copy_(static_codes.sum(0)[:, None, :].repeat_interleave(2, -1))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_audio.copy_(static_codes.sum(0)[:, None, :].repeat_interleave(2, -1))

    def decode(codes, slots, **kwargs):
        static_codes.copy_(codes)
        graph.replay()
        return static_audio, None, 2

    session._cudagraph_wrapper = SimpleNamespace(decode=decode)
    plan = {0: torch.ones((2, 1), device="cuda"), 1: torch.full((2, 3), 2, device="cuda")}
    first = session.step(plan, terminal_slots={0})
    assert first[0].device.type == ("cuda" if device_output else "cpu")
    assert first[0].shape == (2, 2)  # Stereo tail trimmed to one frame.
    assert first[1].shape == (2, 6)
    session.release(0, state_already_reset=True)
    assert session.acquire() == 0
    session.step({0: torch.full((2, 3), 7, device="cuda"), 1: torch.full((2, 3), 9, device="cuda")})
    # Check only after the next replay, as a delayed output consumer would.
    torch.testing.assert_close(first[0].cpu(), torch.full((2, 2), 2.0))
    torch.testing.assert_close(first[1].cpu(), torch.full((2, 6), 4.0))
