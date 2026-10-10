# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Settled-frame and input-record kernels equal the torch indexing they replace."""

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = pytest.mark.core_model


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("step_dtype", [torch.float32, torch.bfloat16])
def test_settled_frames_match_torch(step_dtype):
    from vllm_omni.worker_v2.model_states.eager_mtp_kernels import settled_frames

    device = torch.device("cuda", 0)
    H = 2048
    eager = torch.randn(16, H, device=device).to(torch.bfloat16)
    step = torch.randn(1, H, device=device, dtype=step_dtype)
    embeds = torch.randn(40, H, device=device, dtype=torch.bfloat16)
    slots = torch.tensor([3, 0, 15, 7], device=device)
    rows = torch.tensor([5, 9, 0, 39], device=device)
    expected = embeds.clone()
    expected.index_copy_(0, rows, (eager.index_select(0, slots) + step.to(eager.dtype)).to(expected.dtype))
    settled_frames(torch.stack((slots, rows)).contiguous(), 4, eager, step, embeds)
    torch.testing.assert_close(embeds, expected, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_record_rows_match_torch():
    from vllm_omni.worker_v2.model_states.eager_mtp_kernels import record_rows

    device = torch.device("cuda", 0)
    H = 2048
    slab = torch.zeros(8, 32, H, device=device, dtype=torch.bfloat16)
    embeds = torch.randn(12, H, device=device, dtype=torch.bfloat16)
    index = torch.tensor([[1, 1, 6, 0], [0, 1, 31, 5], [3, 4, 11, 0]], device=device)
    expected = slab.clone()
    expected[index[0], index[1]] = embeds.index_select(0, index[2])
    record_rows(index, 4, embeds, slab)
    torch.testing.assert_close(slab, expected, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("n,size,frames", [(3, 8, 1), (8, 8, 1), (2, 4, 25)])
def test_codec_graph_inputs_match_the_copies(n, size, frames):
    from vllm.triton_utils import triton

    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import _graph_inputs_kernel

    device = torch.device("cuda", 0)
    nq, scratch = 16, 99
    codes = torch.randint(0, 2048, (n, frames, nq), device=device, dtype=torch.int64)
    meta = torch.randint(0, 64, (5 * n,), device=device, dtype=torch.int64)
    slots, pos = meta[n : 2 * n], meta[4 * n :]
    statics = [
        torch.full((size, frames, nq), -1, device=device, dtype=torch.int32),
        torch.full((size,), -1, device=device, dtype=torch.int32),
        torch.full((size,), -1, device=device, dtype=torch.int32),
    ]
    expected = [t.clone() for t in statics]
    expected[0][:n].copy_(codes)
    expected[1][:n].copy_(slots)
    expected[2][:n].copy_(pos)
    expected[1][n:].fill_(scratch)
    expected[2][n:].zero_()
    width = frames * nq
    _graph_inputs_kernel[(size,)](
        codes, slots, pos, *statics, n, scratch, W=width, WP=triton.next_power_of_2(width)
    )  # fmt: skip
    for got, want in zip(statics, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
