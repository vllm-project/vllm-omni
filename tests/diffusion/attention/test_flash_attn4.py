# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch
import torch.nn.functional as F
from vllm.platforms import current_platform

FLASH_ATTN4_MODULE = "vllm_omni.diffusion.attention.backends.flash_attn4"


def _fa4_installed() -> bool:
    try:
        from flash_attn.cute import flash_attn_func  # noqa: F401

        return True
    except Exception:
        return False


requires_fa4 = pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.cuda.is_available() and _fa4_installed()),
    reason="requires CUDA and flash-attn-4 (pip install --pre flash-attn-4)",
)


def _sdpa_reference(query, key, value, softmax_scale, causal):
    # (B, S, H, D) -> (B, H, S, D)
    q, k, v = (t.transpose(1, 2).float() for t in (query, key, value))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2)


@requires_fa4
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_flash_attn4_matches_sdpa(causal: bool, head_dim: int):
    backend_module = importlib.import_module(FLASH_ATTN4_MODULE)
    torch.manual_seed(0)

    batch, seq_len, num_heads = 2, 512, 4
    softmax_scale = head_dim**-0.5
    impl = backend_module.FlashAttention4Impl(
        num_heads=num_heads,
        head_size=head_dim,
        softmax_scale=softmax_scale,
        causal=causal,
    )

    query = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    out = impl.forward_cuda(query, key, value)
    ref = _sdpa_reference(query, key, value, softmax_scale, causal)

    assert out.shape == query.shape
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)


@requires_fa4
def test_flash_attn4_cross_attention_shape():
    """Cosmos3-style cross attention: kv longer than q, non-causal."""
    backend_module = importlib.import_module(FLASH_ATTN4_MODULE)
    torch.manual_seed(0)

    batch, q_len, kv_len, num_heads, head_dim = 1, 256, 768, 8, 128
    softmax_scale = head_dim**-0.5
    impl = backend_module.FlashAttention4Impl(
        num_heads=num_heads,
        head_size=head_dim,
        softmax_scale=softmax_scale,
        causal=False,
    )

    query = torch.randn(batch, q_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(batch, kv_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)

    out = impl.forward_cuda(query, key, value)
    ref = _sdpa_reference(query, key, value, softmax_scale, causal=False)

    assert out.shape == query.shape
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)


@requires_fa4
def test_flash_attn4_masked_varlen_path():
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

    backend_module = importlib.import_module(FLASH_ATTN4_MODULE)
    torch.manual_seed(0)

    batch, seq_len, num_heads, head_dim = 2, 128, 4, 64
    softmax_scale = head_dim**-0.5
    impl = backend_module.FlashAttention4Impl(
        num_heads=num_heads,
        head_size=head_dim,
        softmax_scale=softmax_scale,
        causal=False,
    )

    query = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    # Second sample only attends over its first 64 tokens.
    attn_mask = torch.ones(batch, seq_len, dtype=torch.bool, device="cuda")
    attn_mask[1, 64:] = False

    out = impl.forward_cuda(query, key, value, AttentionMetadata(attn_mask=attn_mask))

    ref0 = _sdpa_reference(query[:1], key[:1], value[:1], softmax_scale, causal=False)
    ref1 = _sdpa_reference(
        query[1:, :64],
        key[1:, :64],
        value[1:, :64],
        softmax_scale,
        causal=False,
    )
    assert torch.allclose(out[:1].float(), ref0, atol=2e-2, rtol=2e-2)
    assert torch.allclose(out[1:, :64].float(), ref1, atol=2e-2, rtol=2e-2)
    # Padded (masked-out) query rows are zero-filled by _pad_input.
    assert torch.all(out[1:, 64:] == 0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="platform selection tests require CUDA platform")
def test_cuda_platform_selects_flash_attn4(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 3)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})
    from vllm_omni.diffusion.attention.backends.utils import fa as fa_utils

    monkeypatch.setattr(fa_utils, "is_flash_attn_4_installed", lambda: True)

    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_4", head_size=128)

    assert backend_path == DiffusionAttentionBackendEnum.FLASH_ATTN_4.get_path()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="platform selection tests require CUDA platform")
def test_cuda_platform_falls_back_when_flash_attn4_missing(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.attention.backends.utils import fa as fa_utils
    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 3)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})
    monkeypatch.setattr(fa_utils, "is_flash_attn_4_installed", lambda: False)

    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("FLASH_ATTN_4", head_size=128)

    assert backend_path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
