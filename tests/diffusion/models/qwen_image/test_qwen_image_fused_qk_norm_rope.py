# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer as qwen_mod
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    _FUSED_MIN_TOKENS,
    _qwen_image_qk_norm_rope,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

BATCH = 2
Q_HEADS = 24
K_HEADS = 24
HEAD_DIM = 128
EPS = 1e-6
SEED = 2026


@dataclass(frozen=True)
class QwenImageQKInput:
    q: torch.Tensor
    k: torch.Tensor
    norm_q: nn.RMSNorm
    norm_k: nn.RMSNorm
    freqs: torch.Tensor


def _make_input(
    *,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    packed_qkv_view: bool,
) -> QwenImageQKInput:
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + seq_len)

    if packed_qkv_view:
        qkv_dim = (Q_HEADS + K_HEADS + K_HEADS) * HEAD_DIM
        qkv = torch.randn(BATCH, seq_len, qkv_dim, device=device, dtype=dtype, generator=gen)
        q, k, _ = qkv.split(
            [
                Q_HEADS * HEAD_DIM,
                K_HEADS * HEAD_DIM,
                K_HEADS * HEAD_DIM,
            ],
            dim=-1,
        )
        q = q.unflatten(-1, (Q_HEADS, HEAD_DIM))
        k = k.unflatten(-1, (K_HEADS, HEAD_DIM))
    else:
        q = torch.randn(BATCH, seq_len, Q_HEADS, HEAD_DIM, device=device, dtype=dtype, generator=gen)
        k = torch.randn(BATCH, seq_len, K_HEADS, HEAD_DIM, device=device, dtype=dtype, generator=gen)

    norm_q = nn.RMSNorm(HEAD_DIM, eps=EPS, device=device, dtype=dtype)
    norm_k = nn.RMSNorm(HEAD_DIM, eps=EPS, device=device, dtype=dtype)
    norm_q.requires_grad_(False)
    norm_k.requires_grad_(False)
    with torch.no_grad():
        norm_q.weight.copy_(torch.randn(HEAD_DIM, device=device, dtype=dtype, generator=gen))
        norm_k.weight.copy_(torch.randn(HEAD_DIM, device=device, dtype=dtype, generator=gen))
    angles = torch.randn(seq_len, HEAD_DIM // 2, device=device, dtype=torch.float32, generator=gen)
    freqs = torch.polar(torch.ones_like(angles), angles)
    return QwenImageQKInput(q=q, k=k, norm_q=norm_q, norm_k=norm_k, freqs=freqs)


def _eager_rotary(data: QwenImageQKInput) -> tuple[torch.Tensor, torch.Tensor]:
    q = data.norm_q(data.q)
    k = data.norm_k(data.k)
    rope = RotaryEmbedding(is_neox_style=False)
    cos = data.freqs.real.to(data.q.dtype)
    sin = data.freqs.imag.to(data.q.dtype)
    return rope(q, cos, sin), rope(k, cos, sin)


def _fp32_complex_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Local copy of the old CUDA helper; fused kernel still tracks this math."""
    paired = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(paired * freqs.unsqueeze(1)).flatten(3).to(x.dtype)


def _fused_kernel_reference(data: QwenImageQKInput) -> tuple[torch.Tensor, torch.Tensor]:
    q = data.norm_q(data.q)
    k = data.norm_k(data.k)
    return _fp32_complex_rotary(q, data.freqs), _fp32_complex_rotary(k, data.freqs)


def _run(data: QwenImageQKInput, *, use_fused: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    return _qwen_image_qk_norm_rope(
        data.q,
        data.k,
        data.norm_q,
        data.norm_k,
        data.freqs,
        RotaryEmbedding(is_neox_style=False),
        EPS,
        use_fused=use_fused,
    )


@pytest.fixture
def force_always_fuse(monkeypatch):
    """Disable the short-seq host gate so fused-kernel correctness tests keep
    exercising the Triton path on tiny shapes."""
    monkeypatch.setattr(qwen_mod, "fused_qk_norm_rope_min_tokens", lambda _default: 0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_qk_norm_rope_cuda_fp32_fallback_matches_reference():
    data = _make_input(
        seq_len=7,
        dtype=torch.float32,
        device=torch.device("cuda:0"),
        packed_qkv_view=True,
    )

    actual_q, actual_k = _run(data, use_fused=False)
    expected_q, expected_k = _eager_rotary(data)

    torch.testing.assert_close(actual_q, expected_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_qk_norm_rope_cuda_fp16_fallback_matches_reference():
    data = _make_input(
        seq_len=7,
        dtype=torch.float16,
        device=torch.device("cuda:0"),
        packed_qkv_view=True,
    )

    actual_q, actual_k = _run(data, use_fused=False)
    expected_q, expected_k = _eager_rotary(data)

    torch.testing.assert_close(actual_q, expected_q, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-3, rtol=1e-3)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [1, 7, 257, 1024])
@pytest.mark.parametrize("packed_qkv_view", [False, True])
def test_qwen_image_fused_qk_norm_rope_cuda_matches_fp32_rope_reference(
    seq_len: int,
    packed_qkv_view: bool,
    force_always_fuse,
):
    data = _make_input(
        seq_len=seq_len,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        packed_qkv_view=packed_qkv_view,
    )

    actual_q, actual_k = _run(data)
    expected_q, expected_k = _fused_kernel_reference(data)

    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_fused_qk_norm_rope_torch_compile_fullgraph_capture(force_always_fuse):
    data = _make_input(
        seq_len=257,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        packed_qkv_view=True,
    )
    rope = RotaryEmbedding(is_neox_style=False)

    def fn(
        q: torch.Tensor,
        k: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _qwen_image_qk_norm_rope(
            q,
            k,
            data.norm_q,
            data.norm_k,
            freqs,
            rope,
            EPS,
        )

    compiled_fn = torch.compile(fn, dynamic=True, fullgraph=True)
    expected_q, expected_k = fn(data.q, data.k, data.freqs)
    actual_q, actual_k = compiled_fn(data.q, data.k, data.freqs)

    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_short_seq_default_gate_uses_eager_path(monkeypatch):
    """Below _FUSED_MIN_TOKENS the helper must stay on the eager chain (#7780)."""
    monkeypatch.delenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", raising=False)
    # BATCH*seq_len = 2*512 = 1024 < 2048
    data = _make_input(
        seq_len=512,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        packed_qkv_view=False,
    )
    assert BATCH * 512 < _FUSED_MIN_TOKENS

    actual_q, actual_k = _run(data, use_fused=True)
    expected_q, expected_k = _eager_rotary(data)

    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_above_gate_uses_fused_path(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", raising=False)
    # BATCH*seq_len = 2*1024 = 2048 == _FUSED_MIN_TOKENS → fuse
    data = _make_input(
        seq_len=1024,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        packed_qkv_view=False,
    )
    assert BATCH * 1024 >= _FUSED_MIN_TOKENS

    actual_q, actual_k = _run(data, use_fused=True)
    expected_q, expected_k = _fused_kernel_reference(data)

    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)
