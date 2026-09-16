# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    _apply_qwen_image_rotary_emb,
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


def _reference(data: QwenImageQKInput) -> tuple[torch.Tensor, torch.Tensor]:
    q = data.norm_q(data.q)
    k = data.norm_k(data.k)
    if data.q.device.type == "cuda":
        return (
            _apply_qwen_image_rotary_emb(q, data.freqs),
            _apply_qwen_image_rotary_emb(k, data.freqs),
        )

    rope = RotaryEmbedding(is_neox_style=False)
    cos = data.freqs.real.to(data.q.dtype)
    sin = data.freqs.imag.to(data.q.dtype)
    return rope(q, cos, sin), rope(k, cos, sin)


def _run(data: QwenImageQKInput) -> tuple[torch.Tensor, torch.Tensor]:
    return _qwen_image_qk_norm_rope(
        data.q,
        data.k,
        data.norm_q,
        data.norm_k,
        data.freqs,
        RotaryEmbedding(is_neox_style=False),
        EPS,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_qk_norm_rope_cuda_fp32_fallback_matches_reference():
    data = _make_input(
        seq_len=7,
        dtype=torch.float32,
        device=torch.device("cuda:0"),
        packed_qkv_view=True,
    )

    actual_q, actual_k = _run(data)
    expected_q, expected_k = _reference(data)

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

    actual_q, actual_k = _run(data)
    expected_q, expected_k = _reference(data)

    torch.testing.assert_close(actual_q, expected_q, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-3, rtol=1e-3)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [1, 7, 257, 1024])
@pytest.mark.parametrize("packed_qkv_view", [False, True])
def test_qwen_image_fused_qk_norm_rope_cuda_matches_fp32_rope_reference(
    seq_len: int,
    packed_qkv_view: bool,
):
    data = _make_input(
        seq_len=seq_len,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        packed_qkv_view=packed_qkv_view,
    )

    actual_q, actual_k = _run(data)
    expected_q, expected_k = _reference(data)

    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_fused_qk_norm_rope_torch_compile_fullgraph_capture():
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
