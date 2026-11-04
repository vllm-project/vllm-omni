# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.ops.block_sparse import resolve_block_sparse_provider
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
