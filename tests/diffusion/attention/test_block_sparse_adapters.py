# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sparse adapter contract tests, independent of the block-selection strategy."""

import math

import pytest
import torch

from tests.helpers.block_sparse import make_attention_inputs, selected_attention_reference
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.block_selection.abstract import BlockSelection
from vllm_omni.diffusion.attention.capabilities import CompilationMode

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cuda]


@pytest.fixture(params=["FLASH_ATTN"])
def sparse_adapter(request):
    # Add a provider here and its dependency/device setup below to run the same contract.
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    if request.param == "FLASH_ATTN":
        if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)):
            pytest.skip("Requires Hopper or datacenter Blackwell")
        pytest.importorskip("flash_attn.cute")
    adapter_cls = DiffusionAttentionBackendEnum[request.param].get_class().get_block_sparse_adapter()
    assert adapter_cls is not None
    torch.compiler.reset()
    try:
        yield adapter_cls()
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize(
    "kv_heads,dtype,pattern,compile_case",
    [
        (1, torch.float16, "per_row", False),
        (2, torch.bfloat16, "per_row", True),
        (4, torch.float16, "per_row", False),
        (4, torch.bfloat16, "full", False),
    ],
)
@pytest.mark.parametrize("value_size", [64, 128])
@torch.inference_mode()
def test_selected_attention_contract(sparse_adapter, kv_heads, dtype, pattern, compile_case, value_size):
    # FA4 b33 adapts tiles to 64x64 on Hopper. Blackwell defaults to
    # tile_n=128 and two 128-row Q stages for these sequence lengths.
    blackwell = torch.cuda.get_device_capability()[0] == 10
    block_size = (256, 128) if blackwell else (64, 64)
    q, k, v = make_attention_inputs(
        kv_heads=kv_heads,
        q_len=513 if blackwell else 129,
        kv_len=577 if blackwell else 193,
        value_size=value_size,
        dtype=dtype,
    )
    q, k, v = (torch.stack((t, t), dim=-1)[..., 0] for t in (q, k, v))
    scale = q.shape[-1] ** -0.5
    rows = (q.shape[0], q.shape[2], math.ceil(q.shape[1] / block_size[0]))
    key_blocks = math.ceil(k.shape[1] / block_size[1])
    if pattern == "full":
        indices = torch.arange(key_blocks, dtype=torch.int32, device=q.device).expand(*rows, key_blocks).contiguous()
        counts = torch.full(rows, key_blocks, dtype=torch.int32, device=q.device)
    else:
        # Distinct query-head patterns, variable counts and a partially filled final KV block.
        first = (torch.arange(math.prod(rows), device=q.device).reshape(rows) % 2).int()
        indices = torch.stack((first, torch.full_like(first, key_blocks - 1)), dim=-1)
        counts = first + 1
    selection = BlockSelection(indices, counts)
    tensors = (q, k, v, indices, counts)
    snapshots = tuple(t.clone() for t in tensors)
    sparse_adapter.prepare("auto", q.shape[-1], q.shape[2], k.shape[2], q.device, block_size)
    expected = selected_attention_reference(q, k, v, selection, scale, block_size)

    # Warm the actual inputs outside compilation, without a synthetic support probe.
    result = sparse_adapter.execute(q, k, v, selection, scale, block_size)
    assert result.shape == (*q.shape[:-1], v.shape[-1])
    assert result.dtype == q.dtype and result.device == q.device and result.is_contiguous()
    torch.testing.assert_close(result.float(), expected, atol=0.004, rtol=0.02)
    run = sparse_adapter.execute
    if compile_case and sparse_adapter.compilation_mode is not CompilationMode.EAGER_ONLY:
        run = torch.compile(run, fullgraph=True)
        torch.testing.assert_close(run(q, k, v, selection, scale, block_size), result, atol=0, rtol=0)

    saved_result = result.clone()
    torch.testing.assert_close(run(q, k, -v, selection, scale, block_size), -saved_result, atol=0, rtol=0)
    # Change active counts without changing storage capacity: adapters must read the new pattern.
    updated = BlockSelection(indices, torch.full_like(counts, indices.shape[-1]))
    updated_expected = selected_attention_reference(q, k, v, updated, scale, block_size)
    torch.testing.assert_close(
        run(q, k, v, updated, scale, block_size).float(), updated_expected, atol=0.004, rtol=0.02
    )
    torch.testing.assert_close(result, saved_result, atol=0, rtol=0)
    for original, snapshot in zip(tensors, snapshots):
        torch.testing.assert_close(original, snapshot, atol=0, rtol=0)
