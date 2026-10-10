# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A second streaming-decoder graph set can replay concurrently with the first."""

import threading

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.moss_tts.cuda_graph_streaming_decoder_wrapper import (
    CUDAGraphStreamingDecoderWrapper,
)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]

CAP, DIM = 8, 64


class _Codec(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(DIM, DIM, device=device) / DIM)
        self.state = torch.zeros(CAP + 16, DIM, device=device)

    def decode_streaming_tensors(self, codes, lengths, slots, valid_rows):
        x = codes.float().sum(0)[..., None].expand(-1, -1, DIM) + self.state[slots][:, None]
        y = x @ self.weight
        self.state.index_copy_(0, slots, y[:, -1])
        return y.transpose(1, 2).contiguous(), lengths

    def reset_decoder_state_slots(self, slots):
        self.state.index_fill_(0, slots.to(self.state.device), 0)


def _wrapper(codec, **kwargs):
    return CUDAGraphStreamingDecoderWrapper(
        codec,
        state_capacity=CAP,
        num_quantizers=2,
        vllm_config=None,
        compiled_decode=codec.decode_streaming_tensors,
        **kwargs,
    )


def test_private_wrapper_captures_on_its_own_stream_and_replays_concurrently():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    codec = _Codec(device)
    main = _wrapper(codec, batch_sizes=[4], frame_sizes=[3])
    fast = _wrapper(codec, batch_sizes=[1, 2], frame_sizes=[1], scratch_base=CAP + 4, private_pool=True)
    main.warmup(device)
    fast.warmup(device)
    assert main.is_ready and fast.is_ready
    assert main._capture_stream is None and fast._capture_stream is not None

    codes_main = torch.randint(0, 5, (2, 4, 3), device=device)
    codes_fast = torch.randint(0, 5, (2, 2, 1), device=device)
    errors = []

    def run(wrapper, codes, slots, iters):
        stream = torch.cuda.Stream(device=device)
        slot_ids = torch.tensor(slots, device=device)
        try:
            with torch.cuda.stream(stream):
                for _ in range(iters):
                    codec.reset_decoder_state_slots(slot_ids)
                    audio = wrapper.decode(codes, slot_ids)[0][: len(slots)].clone()
                    expected = (codes.float().sum(0)[..., None].expand(-1, -1, DIM) @ codec.weight).transpose(1, 2)
                    stream.synchronize()
                    torch.testing.assert_close(audio, expected, rtol=1e-3, atol=1e-3)
        except Exception as error:  # surfaced below
            errors.append(error)

    threads = [
        threading.Thread(target=run, args=(main, codes_main, [0, 1, 2, 3], 200)),
        threading.Thread(target=run, args=(fast, codes_fast, [4, 5], 200)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "concurrent graph replay did not finish"
    assert not errors, errors
