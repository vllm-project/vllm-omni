# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming col2im preserves PCM inputs and ping-pong slot state."""

import pytest
import torch
from vllm.triton_utils import triton

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import _col2im_kernel

pytestmark = [pytest.mark.core_model, pytest.mark.tts]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("batch,frames,channels,rate", [(1, 1, 5, 2), (3, 3, 129, 3), (9, 1, 64, 4)])
def test_col2im_partial_tiles_and_reordered_slot_state(batch: int, frames: int, channels: int, rate: int):
    torch.manual_seed(23)
    capacity = batch + 2
    z = torch.randn(batch, frames, 2 * rate, channels).to(torch.bfloat16)
    bias = torch.randn(channels).to(torch.bfloat16)
    state = torch.randn(2, capacity, rate, channels).to(torch.bfloat16)
    expected_state = state.clone()
    slots = torch.arange(batch - 1, -1, -1, dtype=torch.int32)
    positions = torch.arange(batch, dtype=torch.int32)
    parity = torch.arange(capacity, dtype=torch.int32) % 2
    expected = torch.empty(batch, frames, rate, channels, dtype=torch.bfloat16)
    for b in range(batch):
        slot = int(slots[b])
        side = int(parity[slot])
        for t in range(frames):
            previous = z[b, t - 1, rate:] if t else state[side, slot] if positions[b] else torch.zeros(rate, channels)
            expected[b, t] = (z[b, t, :rate].float() + previous.float() + bias.float()).to(torch.bfloat16)
        expected_state[1 - side, slot] = z[b, -1, rate:]

    z_gpu, bias_gpu, state_gpu = z.cuda(), bias.cuda(), state.cuda()
    slots_gpu, positions_gpu, parity_gpu = slots.cuda(), positions.cuda(), parity.cuda()
    actual = torch.empty_like(expected, device="cuda")
    width = min(128, triton.next_power_of_2(channels))
    rows = batch * frames
    block_rows = 2  # several rows per program, crossing request boundaries
    _col2im_kernel[(triton.cdiv(rows, block_rows), rate, triton.cdiv(channels, width))](
        z_gpu,
        bias_gpu,
        state_gpu,
        slots_gpu,
        positions_gpu,
        parity_gpu,
        actual,
        frames,
        capacity,
        rows,
        R=rate,
        C=channels,
        BR=block_rows,
        BC=width,
        num_warps=4,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(state_gpu.cpu(), expected_state, rtol=0, atol=0)
