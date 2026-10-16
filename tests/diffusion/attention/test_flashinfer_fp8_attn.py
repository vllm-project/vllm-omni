# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.flashinfer_attn import FlashInferAttentionImpl

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.cuda]

_FP8_KWARGS = {"backend_kwargs": {"quant": {"dtype_qk": "fp8_e4m3"}}}


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from vllm.utils.flashinfer import is_flashinfer_cudnn_fp8_prefill_attn_supported
    except Exception:
        return False
    return is_flashinfer_cudnn_fp8_prefill_attn_supported()


requires_fp8_cudnn = pytest.mark.skipif(
    not _supported(),
    reason="requires SM100+ GPU with FlashInfer cuDNN FP8 prefill attention support",
)


def _impl(**kwargs) -> FlashInferAttentionImpl:
    kw = {**_FP8_KWARGS, **kwargs}
    return FlashInferAttentionImpl(8, 128, 1.0 / math.sqrt(128), causal=False, num_kv_heads=8, **kw)


def _sdpa_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    qp, kp, vp = (x.permute(0, 2, 1, 3) for x in (q, k, v))
    return F.scaled_dot_product_attention(qp, kp, vp, is_causal=False, scale=scale).permute(0, 2, 1, 3).float()


def test_fp8_requires_supported_platform():
    """Selecting fp8 on an unsupported platform raises instead of silently degrading."""
    if _supported():
        pytest.skip("unsupported-platform check only runs off SM100")
    with pytest.raises(ValueError, match="fp8_e4m3"):
        _impl()


@requires_fp8_cudnn
@pytest.mark.parametrize("sq, skv", [(256, 256), (256, 128), (128, 256)])
def test_fp8_matches_sdpa_loose(sq, skv):
    """FP8 attention output tracks the bf16 SDPA reference (per-tensor quant is lossy)."""
    torch.manual_seed(0)
    B, H, D = 2, 8, 128
    scale = 1.0 / math.sqrt(D)
    q, k, v = (torch.randn(B, s, H, D, device="cuda", dtype=torch.bfloat16) for s in (sq, skv, skv))

    out = _impl().forward_cuda(q, k, v, None).float()
    assert out.shape == (B, sq, H, D)
    assert torch.isfinite(out).all()

    ref = _sdpa_ref(q, k, v, scale)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.1, f"FP8 vs SDPA rel err {rel:.4f} too high — scale/wiring bug suspected"


@requires_fp8_cudnn
def test_fp8_custom_softmax_scale():
    """A non-default softmax scale must reach the kernel (fresh shape signature,
    so the FlashInfer plan cache does not mask it)."""
    torch.manual_seed(0)
    B, S, H, D = 1, 128, 8, 64
    scale = 2.0 / math.sqrt(D)
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    impl = FlashInferAttentionImpl(H, D, scale, causal=False, num_kv_heads=H, **_FP8_KWARGS)
    out = impl.forward_cuda(q, k, v, None).float()
    ref = _sdpa_ref(q, k, v, scale)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.1, f"custom-scale rel err {rel:.4f} too high"


@requires_fp8_cudnn
def test_fp8_all_zero_input_finite():
    """All-zero q/k/v (e.g. zero-initialized model weights) must not NaN."""
    B, S, H, D = 2, 32, 8, 128
    z = torch.zeros(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    out = _impl().forward_cuda(z, z, z, None)
    assert torch.isfinite(out.float()).all()


@requires_fp8_cudnn
def test_workspace_shared_across_impls():
    """All impl instances share one cuDNN workspace per device."""
    from vllm_omni.diffusion.attention.backends import flashinfer_attn as mod

    q = torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16)
    _impl().forward_cuda(q, q, q, None)
    _impl().forward_cuda(q, q, q, None)

    workspaces = [ws for dev, ws in mod._workspace_by_device.items() if dev.type == "cuda"]
    assert len(workspaces) == 1, f"expected 1 shared workspace, got {len(workspaces)}"
    assert workspaces[0].numel() == mod._WORKSPACE_SIZE_BYTES
