# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dropping the upstream chunk-streaming attention buffers of the flow."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
    StepAudio2Token2WavCore,
    _StreamState,
    drop_upstream_chunk_att_buffers,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("att_cache_buffer", torch.zeros(2, 3, 4), persistent=False)
        self.register_buffer("cnn_cache_buffer", torch.zeros(5), persistent=False)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = _Estimator()
        self.register_buffer("rand_noise", torch.randn(1, 2, 8), persistent=False)
        self.register_buffer("att_cache_buffer", torch.zeros(4, 4), persistent=False)


class _Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()


def test_drop_removes_only_the_attention_buffers():
    flow = _Flow()
    noise = flow.decoder.rand_noise.clone()
    released = drop_upstream_chunk_att_buffers(flow)
    assert released == (2 * 3 * 4 + 4 * 4) * 4
    assert not hasattr(flow.decoder, "att_cache_buffer")
    assert not hasattr(flow.decoder.estimator, "att_cache_buffer")
    # The noise the batched backend draws from and the CNN buffers stay.
    assert torch.equal(flow.decoder.rand_noise, noise)
    assert flow.decoder.estimator.cnn_cache_buffer.shape == (5,)
    assert drop_upstream_chunk_att_buffers(flow) == 0


def test_drop_tolerates_a_flow_without_them():
    assert drop_upstream_chunk_att_buffers(nn.Module()) == 0


@pytest.mark.parametrize("dropped", [False, True])
def test_chunk_streaming_refuses_dropped_buffers(dropped):
    core = StepAudio2Token2WavCore("/nonexistent", device="cpu", drop_upstream_chunk_att_buffers=dropped)
    assert core._chunk_att_buffers_dropped is False
    core._chunk_att_buffers_dropped = dropped
    state = _StreamState()
    state.stream_cache = object()
    if dropped:
        with pytest.raises(RuntimeError, match="drop_upstream_chunk_att_buffers"):
            core.setup_stream_for("prompt.wav", state)
        with pytest.raises(RuntimeError, match="drop_upstream_chunk_att_buffers"):
            core.stream_chunk_for([1], "prompt.wav", False, state)
    else:
        # Without the switch the guard passes and the call reaches the prompt.
        core.cache["prompt.wav"] = None
        with pytest.raises(TypeError):
            core.setup_stream_for("prompt.wav", state)
