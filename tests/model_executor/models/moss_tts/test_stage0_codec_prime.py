# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY
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
        self.config = SimpleNamespace(codebook_size=16)

    def initialize_decoder_state_pool(self, capacity, scratch):
        self.state = torch.zeros(capacity + scratch)

    def reset_decoder_state_slots(self, slots):
        self.state[slots] = 0

    def decode_streaming_batch(self, codes, lengths, slots, valid_rows):
        audio = codes.sum(0).float().cumsum(-1) + self.state[slots, None]
        self.state[slots] = audio[:, -1]
        return SimpleNamespace(audio=audio[:, None].repeat(1, 2, 1), audio_lengths=lengths)


def decoder():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(n_vq=2),
            async_chunk=True,
            stage_connector_config={},
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    d = MossTTSCodecDecoder(vllm_config=config)
    d._codec = CausalCodec()
    d._sr_tensor = torch.tensor(48000)
    d._n_channels, d._n_vq = 2, 2
    d._stream_max_step_frames = 15
    d._stream_session = _MossCodecStreamSession(d._codec, state_capacity=8, n_vq=2, vllm_config=None)
    return d


def call(d, request_id, codes, first=False, finished=False):
    return d.forward(
        input_ids=torch.tensor(codes, dtype=torch.long),
        seq_token_counts=[len(codes)],
        runtime_additional_information=[
            {
                "request_id": request_id,
                "meta": {
                    "req_id": [request_id],
                    "first_audio": first,
                    "finished": finished,
                },
            }
        ],
    ).multimodal_outputs


@pytest.mark.parametrize("first", [False, True])
def test_trim_only_delivered_frame_without_losing_causal_history(first):
    d = decoder()
    first_output = call(d, "a", [1, 2], first=first)
    expected = torch.tensor([[3.0], [3.0]])[:, 1:] if first else torch.tensor([[3.0], [3.0]])
    torch.testing.assert_close(first_output["model_outputs"][0], expected)
    rest = call(d, "a", [2, 3], finished=True)
    torch.testing.assert_close(rest["model_outputs"][0], torch.tensor([[8.0], [8.0]]))
    if first:
        assert bool(rest[FIRST_AUDIO_REQUIRED_KEY][0])
    assert not d._stream_req_slots and not d._stream_first_audio_requests
    reused = call(d, "new", [1, 2], finished=True)
    torch.testing.assert_close(reused["model_outputs"][0], torch.tensor([[3.0], [3.0]]))


def test_empty_terminal_keeps_ordering_promise_until_cleanup():
    d = decoder()
    call(d, "a", [1, 2], first=True)
    end = call(d, "a", [], finished=True)
    assert bool(end[FIRST_AUDIO_REQUIRED_KEY][0])
    d.on_requests_finished({"a"})
    assert not d._stream_first_audio_requests
