# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.platforms import current_omni_platform


def _clear_hub_module_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.attention.backends import flash_attn_hub

    monkeypatch.setattr(flash_attn_hub, "_hub_modules", {})


@pytest.mark.core_model
@pytest.mark.cpu
def test_explicit_kernels_hub_selection_does_not_fallback(monkeypatch: pytest.MonkeyPatch):
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

    with pytest.raises(ImportError, match="explicitly selected"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_HUB", head_size=64)
    with pytest.raises(ImportError, match="explicitly selected"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_3_HUB", head_size=64)

    kernels_module = types.ModuleType("kernels")
    monkeypatch.setitem(sys.modules, "kernels", kernels_module)
    with pytest.raises(ValueError, match="require.*Hopper GPU"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_3_HUB", head_size=64)

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 3)),
    )
    with pytest.raises(ValueError, match="current FA2 kernels require"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_HUB", head_size=64)
    with pytest.raises(ValueError, match="require.*Hopper GPU"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_3_HUB", head_size=64)


@pytest.mark.core_model
@pytest.mark.cpu
def test_explicit_kernels_hub_variant_miss_does_not_fallback(monkeypatch: pytest.MonkeyPatch):
    """Explicit Hub backends must fail loudly when no build variant resolves (#6971)."""
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": True})
    _clear_hub_module_cache(monkeypatch)

    kernels_module = types.ModuleType("kernels")

    def _missing_variant(repo_id, version=None, **kwargs):
        raise FileNotFoundError(f"Cannot find a build variant for this system in {repo_id}")

    kernels_module.get_kernel = _missing_variant  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kernels", kernels_module)

    with pytest.raises(RuntimeError, match="no compatible build"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_3_HUB", head_size=64)
    with pytest.raises(RuntimeError, match="no compatible build"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_HUB", head_size=64)


@pytest.mark.core_model
@pytest.mark.cpu
def test_explicit_kernels_hub_preflight_uses_v2_when_v1_misses(monkeypatch: pytest.MonkeyPatch):
    """Preflight must share _load_hub_module's (1, 2) policy, not pin version=1."""
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": True})
    _clear_hub_module_cache(monkeypatch)

    kernels_module = types.ModuleType("kernels")
    versions_tried: list[int | None] = []

    def _v1_miss_v2_hit(repo_id, version=None, **kwargs):
        versions_tried.append(version)
        if version == 1:
            raise FileNotFoundError(f"Cannot find a build variant for this system in {repo_id}")
        return types.SimpleNamespace(name=f"{repo_id}@v{version}")

    kernels_module.get_kernel = _v1_miss_v2_hit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kernels", kernels_module)

    fa3_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_3_HUB", head_size=64)
    assert fa3_path == DiffusionAttentionBackendEnum.FLASH_ATTN_3_HUB.get_path()
    fa2_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_HUB", head_size=64)
    assert fa2_path == DiffusionAttentionBackendEnum.FLASH_ATTN_HUB.get_path()
    assert versions_tried == [1, 2, 1, 2]


@pytest.mark.core_model
@pytest.mark.cpu
def test_explicit_flash_attention_unavailable_does_not_fallback(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(8, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})

    with pytest.raises(ValueError, match="explicitly selected but is unsupported"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN", head_size=64)

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 3)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": True})
    monkeypatch.setattr(CudaOmniPlatform, "has_flash_attn_4", classmethod(lambda cls: False))
    with pytest.raises(ValueError, match="requires CuTe FlashAttention-4"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN", head_size=64)


@pytest.mark.core_model
@pytest.mark.cpu
def test_explicit_sage_attention_unsupported_arch_does_not_reach_kernel(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 3)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})

    with pytest.raises(ValueError, match="does not provide a kernel for sm_103"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN", head_size=64)


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
