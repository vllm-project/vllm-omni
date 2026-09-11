# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
    MossTTSCodecDecoder,
    _MossCodecStreamSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class CausalCodec(nn.Module):
    downsample_rate = 1

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def initialize_decoder_state_pool(self, capacity, scratch):
        self.state = torch.zeros(capacity + scratch)

    def reset_decoder_state_slots(self, slots):
        self.state[slots] = 0

    def decode_streaming_batch(self, codes, lengths, slots, valid_rows):
        audio = codes.sum(0).float().cumsum(-1) + self.state[slots, None]
        self.state[slots] = audio[:, -1]
        return SimpleNamespace(audio=audio[:, None], audio_lengths=lengths)


def session():
    return _MossCodecStreamSession(CausalCodec(), state_capacity=8, n_vq=2, vllm_config=None)


def test_terminal_padding_preserves_audio_live_state_and_reused_slots():
    s = session()
    assert [s.acquire(), s.acquire()] == [0, 1]
    plan = {0: torch.ones(2, 3), 1: torch.ones(2, 15)}
    with pytest.raises(ValueError, match="Only terminal"):
        s.step(plan)
    assert s._codec.state.count_nonzero() == 0
    s.step({0: torch.ones(2, 1), 1: torch.full((2, 1), 2)})
    actual = s.step(plan, terminal_slots={0})
    torch.testing.assert_close(actual[0], torch.tensor([[4.0, 6.0, 8.0]]))
    torch.testing.assert_close(actual[1], torch.arange(6.0, 36.0, 2.0)[None])
    s.release(0, state_already_reset=True)
    assert s.acquire() == 0
    resumed = s.step({0: torch.ones(2, 1), 1: torch.ones(2, 1)})
    assert [resumed[i].item() for i in (0, 1)] == [2, 36]


@pytest.mark.parametrize("graph_capacity, expected_calls", [(None, 5), (4, 5), (8, 3)])
def test_only_existing_compatible_graph_buckets_are_coalesced(mocker, graph_capacity, expected_calls):
    s = session()
    s._cudagraph_wrapper = (
        None
        if graph_capacity is None
        else SimpleNamespace(
            batch_sizes=[1, graph_capacity],
            _select_frame_size=lambda size, allow_padding: next((b for b in [1, 8, 15] if b >= size), None),
            decode=lambda *args, **kwargs: None,
        )
    )
    step = mocker.spy(s, "step")
    decoder = object.__new__(MossTTSCodecDecoder)
    nn.Module.__init__(decoder)
    decoder._stream_max_step_frames = 15
    decoder._stream_state_capacity = 8
    decoder._stream_req_slots = {}
    decoder._ensure_stream_session = lambda: s
    lengths = [3, 7, 8, 1, 15]
    finished = [True, True, False, True, False]
    result = decoder._decode_streaming_batch(
        [
            (i, str(i), torch.ones(2, length), done)
            for i, (length, done) in enumerate(zip(lengths, finished, strict=True))
        ]
    )
    assert step.call_count == expected_calls
    assert [result[i].shape[-1] for i in range(5)] == lengths
    if graph_capacity == 8:
        assert list(step.call_args_list[0].args[0]) == [0, 1, 2]
        assert step.call_args_list[0].kwargs["terminal_slots"] == {0, 1}
        assert list(step.call_args_list[1].args[0]) == [3]  # Preserve the exact one-frame graph.
