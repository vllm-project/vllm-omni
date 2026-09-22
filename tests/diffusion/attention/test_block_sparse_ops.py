# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.attention.ops.block_sparse import (
    build_prefix_dense_block_map,
    fastvideo_block_sparse_attn_bshd,
    mean_pool_tiles,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("prefix_blocks,topk", [(0, 1), (2, 0), (2, 1), (2, 3), (2, 5)])
def test_block_map_keeps_prefix_dense_and_selects_non_prefix_keys(prefix_blocks, topk):
    sparse_blocks = 3
    blocks = prefix_blocks + sparse_blocks
    scores = torch.arange(2 * blocks * blocks, dtype=torch.float32).reshape(1, 2, blocks, blocks)
    block_map = build_prefix_dense_block_map(scores, prefix_blocks, sparse_blocks, topk)
    expected = torch.zeros_like(scores, dtype=torch.bool)
    # Scores increase with key index, so top-k always selects the last keys.
    for query_block in range(blocks):
        for key_block in range(blocks):
            if (
                query_block < prefix_blocks
                or key_block < prefix_blocks
                or key_block >= blocks - min(topk, sparse_blocks)
            ):
                expected[..., query_block, key_block] = True
    assert torch.equal(block_map, expected)


@pytest.mark.parametrize("block_size", [4, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pooling_uses_valid_tile_sizes_and_fp32_accumulation(block_size, dtype):
    sizes = torch.tensor([block_size, block_size - 1, 0], dtype=torch.int32)
    tiles = torch.arange(2 * 3 * block_size * 2 * 3).reshape(2, 3, block_size, 2, 3).to(dtype)
    tiles[:, 1, -1] = 0
    tiles[:, 2] = 0
    expected = torch.stack(
        [tiles[:, 0].float().mean(dim=1), tiles[:, 1, :-1].float().mean(dim=1), torch.zeros(2, 2, 3)], dim=2
    )
    result = mean_pool_tiles(tiles.flatten(1, 2), sizes, block_size)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_provider_wrapper_restores_transport_rows_and_bshd_layout(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_VSA_SM100A", raising=False)
    calls = []

    def block_sparse_attn(q, k, v, block_map, sizes):
        calls.append((q.shape, block_map.shape, sizes.tolist()))
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert sizes.dtype == torch.int32
        return q + k + v, None

    module = types.ModuleType("fastvideo_kernel.block_sparse_attn")
    setattr(module, "block_sparse_attn", block_sparse_attn)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", types.ModuleType("fastvideo_kernel"))
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.block_sparse_attn", module)
    query = torch.randn(1, 4 * 64, 2, 8)
    block_map = torch.ones(1, 2, 4, 4, dtype=torch.bool)
    sizes = torch.tensor([64, 17, 64, 0])
    output = fastvideo_block_sparse_attn_bshd(query, query, query, block_map, sizes, 3)
    expected = torch.zeros_like(query)
    expected[:, : 3 * 64] = query[:, : 3 * 64] * 3
    torch.testing.assert_close(output, expected)
    assert calls == [(torch.Size([1, 2, 3 * 64, 8]), torch.Size([1, 2, 3, 3]), [64, 17, 64])]


def test_custom_op_fake_preserves_metadata_without_loading_provider(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", None)
    with FakeTensorMode():
        query = torch.empty(1, 4 * 64, 2, 8, dtype=torch.bfloat16)
        block_map = torch.empty(1, 2, 4, 4, dtype=torch.bool)
        sizes = torch.empty(4, dtype=torch.int32)
        output = fastvideo_block_sparse_attn_bshd(query, query, query, block_map, sizes, 3)
        assert output.shape == query.shape
        assert output.dtype == query.dtype
        assert output.device == query.device


@pytest.mark.parametrize("error_type", [ImportError, RuntimeError, torch.AcceleratorError])
def test_native_provider_failure_only_recovers_without_accelerator_fault(monkeypatch, mocker, error_type):
    monkeypatch.setenv("FASTVIDEO_VSA_SM100A", "1")
    error = error_type("native provider failure")
    native = mocker.Mock()
    native.is_supported.return_value = True
    native.block_sparse_attn_sm100a.side_effect = error
    provider = types.ModuleType("fastvideo_kernel")
    setattr(provider, "block_sparse_attn_sm100a", native)
    index = types.ModuleType("fastvideo_kernel.triton_kernels.index")
    setattr(
        index,
        "map_to_index",
        mocker.Mock(return_value=(torch.zeros(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32))),
    )
    triton = types.ModuleType("fastvideo_kernel.block_sparse_attn")
    fallback = mocker.Mock(side_effect=lambda q, *args: (q.clone(), None))
    setattr(triton, "block_sparse_attn", fallback)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", provider)
    monkeypatch.setitem(
        sys.modules, "fastvideo_kernel.triton_kernels", types.ModuleType("fastvideo_kernel.triton_kernels")
    )
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.triton_kernels.index", index)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.block_sparse_attn", triton)
    query = torch.arange(128 * 2 * 8, dtype=torch.float32).reshape(1, 128, 2, 8)
    block_map = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    sizes = torch.tensor([64, 64], dtype=torch.int32)

    if error_type is torch.AcceleratorError:
        with pytest.raises(torch.AcceleratorError, match="native provider failure") as raised:
            fastvideo_block_sparse_attn_bshd(query, query, query, block_map, sizes, 2)
        assert raised.value is error
        fallback.assert_not_called()
    else:
        output = fastvideo_block_sparse_attn_bshd(query, query, query, block_map, sizes, 2)
        torch.testing.assert_close(output, query, rtol=0, atol=0)
        fallback.assert_called_once()
