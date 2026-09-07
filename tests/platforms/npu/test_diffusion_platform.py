# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("available", [False, True])
def test_dense_flash_capability_tracks_mindiesd(monkeypatch, available):
    pytest.importorskip("vllm_ascend")
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "mindiesd" and available else None,
    )

    assert NPUOmniPlatform.supports_diffusion_dense_flash_attention() is available


def test_paged_config_uses_ascend_kernel_block_size() -> None:
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

    from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    NPUOmniPlatform.configure_diffusion_vllm_config(
        vllm_config,
        SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER),
    )

    assert vllm_config.cache_config.block_size == AscendAttentionBackend.get_supported_kernel_block_sizes()[0]


def test_strict_ulysses_paged_backend_bypasses_ascend_pcp() -> None:
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend,
        AscendAttentionBackendImpl,
        AscendAttentionMetadataBuilder,
    )

    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    assert (
        NPUOmniPlatform.get_diffusion_paged_kv_attn_backend(
            AscendAttentionBackend,
            ulysses_degree=1,
        )
        is AscendAttentionBackend
    )
    backend = NPUOmniPlatform.get_diffusion_paged_kv_attn_backend(
        AscendAttentionBackend,
        ulysses_degree=2,
    )
    assert backend.get_impl_cls() is AscendAttentionBackendImpl
    assert backend.get_builder_cls() is AscendAttentionMetadataBuilder
