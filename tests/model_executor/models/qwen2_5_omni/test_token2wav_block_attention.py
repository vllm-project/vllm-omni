# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import block_sparse_attention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK_SIZE = 24


def _dense_masked_reference(query, key, value, look_backward_block, look_ahead_block):
    """The original Token2Wav DiT attention: dense SDPA with a block-window mask."""
    seq_len = query.shape[2]
    block_indices = torch.arange(seq_len) // BLOCK_SIZE
    block_diff = block_indices[None, :] - block_indices[:, None]
    mask = (block_diff >= -look_backward_block) & (block_diff <= look_ahead_block)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


# seq_len covers: shorter than one block, a multiple of the block size, and a
# partial last block.
@pytest.mark.parametrize("seq_len", [10, 96, 101])
@pytest.mark.parametrize(("look_backward_block", "look_ahead_block"), [(0, 0), (1, 0), (0, 1)])
def test_block_sparse_attention_matches_dense_masked_sdpa(seq_len, look_backward_block, look_ahead_block):
    torch.manual_seed(0)
    query, key, value = (torch.randn(2, 4, seq_len, 16) for _ in range(3))

    out = block_sparse_attention(
        query,
        key,
        value,
        block_size=BLOCK_SIZE,
        look_backward_block=look_backward_block,
        look_ahead_block=look_ahead_block,
    )
    ref = _dense_masked_reference(query, key, value, look_backward_block, look_ahead_block)

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
