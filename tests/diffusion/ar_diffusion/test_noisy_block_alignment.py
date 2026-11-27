# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA regression for the real CausVid 480x832 logical KV block."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import ar_diffusion_paged_attention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.parametrize("used_blocks", [1, 2])
def test_unaligned_noisy_block_matches_dense_cuda(monkeypatch, used_blocks):
    monkeypatch.delenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", raising=False)
    # Actual CausVid latent (1,16,3,60,104), patch=(1,2,2): 4680 tokens.
    block_size = 4680
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(17)
    key = torch.randn((3, block_size, 2, 64), device=device, dtype=torch.float16, generator=generator)
    value = torch.randn(key.shape, device=device, dtype=torch.float16, generator=generator)
    query = torch.randn((8, 2, 64), device=device, dtype=torch.float16, generator=generator)
    # Physical ordering is deliberately different from logical ordering.
    order = [2, 0][:used_blocks]
    table = torch.tensor([order + [1] * (3 - len(order))], dtype=torch.int32, device=device)
    length = used_blocks * block_size
    actual = ar_diffusion_paged_attention(
        query,
        key,
        value,
        block_table=table,
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([length], dtype=torch.int32, device=device),
        max_query_len=8,
        max_seq_len=3 * block_size,
        softmax_scale=64**-0.5,
    )
    dense_k = torch.cat([key[index] for index in order])
    dense_v = torch.cat([value[index] for index in order])
    expected = (
        F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0).float(),
            dense_k.transpose(0, 1).unsqueeze(0).float(),
            dense_v.transpose(0, 1).unsqueeze(0).float(),
        )
        .squeeze(0)
        .transpose(0, 1)
        .to(actual.dtype)
    )
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
