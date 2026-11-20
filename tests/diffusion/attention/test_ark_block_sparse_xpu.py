# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.ops.ark_block_sparse import (
    ark_block_sparse_attention,
    supports_ark_k_block_valid,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU is unavailable", allow_module_level=True)

pytest.importorskip("auto_round_kernel", reason="auto-round-lib is unavailable")
try:
    _K_BLOCK_VALID = supports_ark_k_block_valid()
except (ImportError, RuntimeError) as exc:
    pytest.skip(str(exc), allow_module_level=True)
if not _K_BLOCK_VALID:
    pytest.skip("auto-round-lib>=0.17.0 with k_block_valid is required", allow_module_level=True)


def _expected_output(query, key, value, block_map, block_sizes, block_size, scale):
    """Dense reference over the exact token range each routed block owns."""
    batch, seq_len_q, heads, head_dim = query.shape
    seq_len_kv = key.shape[1]
    q_blocks = seq_len_q // block_size
    keep = torch.zeros((batch, heads, seq_len_q, seq_len_kv), dtype=torch.bool, device=query.device)
    for q_index in range(q_blocks):
        q_start, q_end = q_index * block_size, (q_index + 1) * block_size
        for k_index in range(block_sizes.numel()):
            k_start = k_index * block_size
            k_end = min(k_start + int(block_sizes[k_index]), seq_len_kv)
            if k_end <= k_start:
                continue
            selected = block_map[:, :, q_index : q_index + 1, k_index : k_index + 1]
            keep[:, :, q_start:q_end, k_start:k_end] = selected.expand(batch, heads, q_end - q_start, k_end - k_start)

    q = query.float().transpose(1, 2)
    k = key.float().transpose(1, 2)
    v = value.float().transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~keep, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v).transpose(1, 2).to(query.dtype)


@hardware_test(res={"xpu": "B70"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    ("seq_len_q", "seq_len_kv", "block_size", "block_sizes", "selection"),
    [
        # Partial 64-token K blocks: edge tiles keep 64/32/16 real tokens.
        (192, 192, 64, [64, 32, 16], [[0, 1, 2], [1, 2], [0, 2]]),
        # Logical tile128 K split into two 64-token compute blocks, tail half-full.
        (256, 256, 128, [128, 96], [[0, 1], [0, 1]]),
    ],
)
def test_partial_k_blocks_match_dense_reference(seq_len_q, seq_len_kv, block_size, block_sizes, selection, dtype):
    torch.manual_seed(20261120)
    torch.xpu.manual_seed(20261120)
    heads, head_dim = 8, 128
    scale = head_dim**-0.5
    query = torch.randn(1, seq_len_q, heads, head_dim, device="xpu", dtype=dtype)
    key = torch.randn(1, seq_len_kv, heads, head_dim, device="xpu", dtype=dtype)
    value = torch.randn(1, seq_len_kv, heads, head_dim, device="xpu", dtype=dtype)

    q_blocks = seq_len_q // block_size
    k_blocks = len(block_sizes)
    block_map = torch.zeros((1, heads, q_blocks, k_blocks), dtype=torch.bool, device="xpu")
    for q_block, selected in enumerate(selection):
        block_map[:, :, q_block, list(selected)] = True

    sizes = torch.tensor(block_sizes, dtype=torch.int32, device="xpu")
    output = ark_block_sparse_attention(query, key, value, block_map, sizes, scale, block_size=block_size)
    expected = _expected_output(query, key, value, block_map, sizes, block_size, scale)
    torch.xpu.synchronize()

    torch.testing.assert_close(output.float(), expected.float(), atol=2e-2, rtol=0)


@hardware_test(res={"xpu": "B70"}, num_cards=1)
def test_zero_length_block_is_never_routed():
    """A transport-only block must be dropped from the map instead of entering softmax."""
    torch.manual_seed(7)
    torch.xpu.manual_seed(7)
    seq_len_q = seq_len_kv = 256
    heads, head_dim, block_size = 4, 128, 128
    scale = head_dim**-0.5
    dtype = torch.bfloat16
    query = torch.randn(1, seq_len_q, heads, head_dim, device="xpu", dtype=dtype)
    key = torch.randn(1, seq_len_kv, heads, head_dim, device="xpu", dtype=dtype)
    value = torch.randn(1, seq_len_kv, heads, head_dim, device="xpu", dtype=dtype)

    block_map = torch.ones((1, heads, seq_len_q // block_size, seq_len_kv // block_size), dtype=torch.bool)
    block_map = block_map.to("xpu")
    # Second logical tile is empty: it must be dropped after the tile128 split.
    sizes = torch.tensor([128, 0], dtype=torch.int32, device="xpu")

    output = ark_block_sparse_attention(query, key, value, block_map, sizes, scale, block_size=block_size)
    expected = _expected_output(query, key, value, block_map, sizes, block_size, scale)
    torch.xpu.synchronize()

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected.float(), atol=2e-2, rtol=0)
