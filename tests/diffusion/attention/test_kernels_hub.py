# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.platforms import current_omni_platform


@pytest.mark.core_model
@pytest.mark.cpu
def test_kernels_hub_platform_fallback(monkeypatch: pytest.MonkeyPatch):
    """Test that when kernels package is not available, the platform fallback logic

    routes to local FLASH_ATTN.
    """
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    # Temporarily hide/remove kernels module if present
    monkeypatch.setitem(sys.modules, "kernels", None)

    # Use monkeypatch to mock CudaOmniPlatform capability and package checker to allow FLASH_ATTN
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.envs import PACKAGES_CHECKER

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(8, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": True})

    # Test FLASH_ATTN_HUB falls back to FLASH_ATTN when kernels is unavailable
    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_HUB", head_size=64)
    assert backend_path == DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

    # Test FLASH_ATTN_3_HUB falls back to FLASH_ATTN when kernels is unavailable
    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_3_HUB", head_size=64)
    assert backend_path == DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

    # Test FLASH_ATTN_3_HUB falls back to FLASH_ATTN_HUB on pre-Hopper GPUs
    kernels_module = types.ModuleType("kernels")
    monkeypatch.setitem(sys.modules, "kernels", kernels_module)
    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_3_HUB", head_size=64)
    assert backend_path == DiffusionAttentionBackendEnum.FLASH_ATTN_HUB.get_path()


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_kernels_hub_execution():
    """Verify basic forward of flash_attn_hub and flash_attn_3_hub, comparing with SDPA reference."""
    device = torch.device(current_omni_platform.device_type)
    dtype = torch.bfloat16

    num_heads = 8
    head_dim = 64
    seq_len = 32
    batch_size = 1

    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = q.clone()
    v = q.clone()

    # 1. Test PyTorch SDPA reference
    sdpa_impl = SDPAImpl(num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False)
    attn_metadata_sdpa = AttentionMetadata(attn_mask=None)
    output_ref = sdpa_impl.forward(q.clone(), k.clone(), v.clone(), attn_metadata_sdpa)

    # 2. Test FlashAttentionHubBackend (FlashAttention 2)
    from vllm_omni.diffusion.attention.backends.flash_attn_hub import FlashAttentionHubImpl

    fa_hub_impl = FlashAttentionHubImpl(
        num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False
    )
    output_fa_hub = fa_hub_impl.forward(q.clone(), k.clone(), v.clone(), attn_metadata_sdpa)
    assert output_fa_hub.shape == q.shape
    assert not torch.isnan(output_fa_hub).any()
    max_diff = torch.max(torch.abs(output_ref - output_fa_hub)).item()
    assert max_diff < 1e-2, f"FlashAttentionHub output differs too much from SDPA reference: {max_diff}"

    # 3. Test FlashAttention3HubBackend (FlashAttention 3, Hopper+ only)
    major, _minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("FLASH_ATTN_3_HUB execution requires Hopper-class GPU (compute capability >= 9.0)")

    from vllm_omni.diffusion.attention.backends.flash_attn_hub import FlashAttention3HubImpl

    fa3_hub_impl = FlashAttention3HubImpl(
        num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False
    )
    output_fa3_hub = fa3_hub_impl.forward(q.clone(), k.clone(), v.clone(), attn_metadata_sdpa)
    assert output_fa3_hub.shape == q.shape
    assert not torch.isnan(output_fa3_hub).any()
    max_diff = torch.max(torch.abs(output_ref - output_fa3_hub)).item()
    assert max_diff < 1e-2, f"FlashAttention3Hub output differs too much from SDPA reference: {max_diff}"
