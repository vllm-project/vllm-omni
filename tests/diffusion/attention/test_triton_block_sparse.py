# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.ops import triton_block_sparse
from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd, resolve_block_sparse_provider
from vllm_omni.diffusion.attention.ops.triton_block_sparse import require_triton_sparse
from vllm_omni.diffusion.data import AttentionSpec
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def clear_capability_cache():
    for fn in (require_triton_sparse, resolve_block_sparse_provider):
        fn.cache_clear()
    yield
    for fn in (require_triton_sparse, resolve_block_sparse_provider):
        fn.cache_clear()


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 0), (10, 3), (12, 0), (12, 1)])
def test_explicit_triton_uses_the_operand_device(monkeypatch, capability):
    cap = Mock(return_value=capability)
    monkeypatch.setattr(current_omni_platform, "get_device_capability", cap)
    module = ModuleType("fastvideo_kernel.block_sparse_attn")
    kernel = Mock()
    setattr(module, "block_sparse_attn_triton", kernel)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.block_sparse_attn", module)
    assert resolve_block_sparse_provider("triton", "bf16", torch.device("cuda:3"), torch.bfloat16, 128, 128) == "triton"
    cap.assert_called_once_with(3)
    kernel.assert_not_called()


@pytest.mark.parametrize("capability", [(8, 9), (10, 3)])
def test_auto_tile128_uses_portable_kernel(monkeypatch, capability):
    monkeypatch.setattr(current_omni_platform, "get_device_capability", lambda _: capability)
    module = ModuleType("fastvideo_kernel.block_sparse_attn")
    setattr(module, "block_sparse_attn_triton", Mock())
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.block_sparse_attn", module)
    assert resolve_block_sparse_provider("auto", "bf16", torch.device("cuda:0"), torch.bfloat16, 128, 128) == "triton"


def test_triton_missing_dependency_fails_before_generation(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "get_device_capability", lambda _: (8, 9))
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.block_sparse_attn", None)
    with pytest.raises(ImportError, match="vllm-omni\\[vsa\\]"):
        resolve_block_sparse_provider("triton", "bf16", torch.device("cuda:0"), torch.bfloat16, 128, 128)


def test_triton_config_rejects_sage_and_other_backends():
    assert (
        AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_provider="triton").backend_kwargs()["provider"] == "triton"
    )
    with pytest.raises(ValueError, match="Sage"):
        AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_provider="triton", fastvideo_vsa_precision="sage")
    with pytest.raises(ValueError, match="FASTVIDEO_VSA"):
        AttentionSpec(backend="FLASH_ATTN", fastvideo_vsa_provider="triton")


@pytest.mark.parametrize("block_size", [64, 128])
def test_triton_normalizes_base2_lse_and_empty_rows(monkeypatch, block_size):
    query = torch.ones(1, block_size * 2, 2, 128, dtype=torch.bfloat16)
    block_map = torch.tensor([[[[True, True], [False, True]]]]).expand(1, 2, 2, 2)
    sizes = torch.tensor([17, 0], dtype=torch.int32)

    def kernel(query, key, value, indices, counts, block_sizes):
        output = torch.ones_like(query)
        lse = torch.full(query.shape[:3], math.log2(17), dtype=torch.float32)
        output[:, :, block_size:] = torch.nan
        lse[:, :, block_size:] = torch.nan
        assert counts[:, :, block_size // 64 :].count_nonzero() == 0
        return output, lse

    monkeypatch.setattr(triton_block_sparse, "require_triton_sparse", lambda device: kernel)
    output, lse = block_sparse_attn_bshd(
        query,
        query,
        query,
        block_map,
        sizes,
        128**-0.5,
        provider="triton",
        block_size=block_size,
        return_lse=True,
    )
    torch.testing.assert_close(output[:, :block_size], query[:, :block_size])
    torch.testing.assert_close(lse[:, :, :block_size], torch.full((1, 2, block_size), math.log(17)))
    assert output[:, block_size:].count_nonzero() == 0
    assert torch.isneginf(lse[:, :, block_size:]).all()
    assert lse.is_contiguous() and lse.dtype == torch.float32
    default_output = block_sparse_attn_bshd(
        query,
        query,
        query,
        block_map,
        sizes,
        128**-0.5,
        provider="triton",
        block_size=block_size,
    )
    torch.testing.assert_close(default_output, output, rtol=0, atol=0)


@pytest.mark.parametrize("bad_lse", [torch.zeros(1, 64, 2), torch.zeros(1, 2, 64, dtype=torch.bfloat16)])
def test_triton_rejects_incompatible_lse_abi(monkeypatch, bad_lse):
    query = torch.ones(1, 64, 2, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(
        triton_block_sparse,
        "require_triton_sparse",
        lambda device: lambda query, *args: (query.clone(), bad_lse),
    )
    with pytest.raises(ValueError, match="FP32 base-2 LSE"):
        block_sparse_attn_bshd(
            query,
            query,
            query,
            torch.ones(1, 2, 1, 1, dtype=torch.bool),
            torch.tensor([64], dtype=torch.int32),
            128**-0.5,
            provider="triton",
            return_lse=True,
        )
