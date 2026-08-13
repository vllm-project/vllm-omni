# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.qwen_image.fused_qk_norm_rope import (
    qwen_image_fused_qk_norm_rope,
    qwen_image_fused_qk_norm_rope_fast_path,
    qwen_image_qk_norm_rope_fast_path_supported,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]

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
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


def _reference_qwen_image_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope = RotaryEmbedding(is_neox_style=False)
    q = F.rms_norm(q, (q.shape[-1],), q_weight, EPS)
    k = F.rms_norm(k, (k.shape[-1],), k_weight, EPS)
    return (
        rope.forward_native(q, cos, sin),
        rope.forward_native(k, cos, sin),
    )


def _assert_close_with_error_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    atol: float,
    rtol: float,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        abs_diff = (actual.float() - expected.float()).abs().flatten()
        tolerance = atol + rtol * expected.float().abs().flatten()
        mismatch = abs_diff > tolerance
        mismatch_count = int(mismatch.sum().item())
        total = abs_diff.numel()
        p99 = torch.quantile(abs_diff, 0.99).item() if total else 0.0
        stats = (
            f"{name} error stats: "
            f"max={abs_diff.max().item() if total else 0.0:.6g}, "
            f"p99={p99:.6g}, "
            f"mean={abs_diff.mean().item() if total else 0.0:.6g}, "
            f"mismatch={mismatch_count}/{total} ({mismatch_count / total:.4%}), "
            f"atol={atol}, rtol={rtol}"
        )
        raise AssertionError(f"{stats}\n{exc}") from exc


def _make_input(
    *,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    packed_qkv_view: bool,
    batch: int = BATCH,
) -> QwenImageQKInput:
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + seq_len)

    if packed_qkv_view:
        qkv_dim = (Q_HEADS + K_HEADS + K_HEADS) * HEAD_DIM
        qkv = torch.randn(batch, seq_len, qkv_dim, device=device, dtype=dtype, generator=gen)
        q, k, _v = qkv.split(
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
        q = torch.randn(batch, seq_len, Q_HEADS, HEAD_DIM, device=device, dtype=dtype, generator=gen)
        k = torch.randn(batch, seq_len, K_HEADS, HEAD_DIM, device=device, dtype=dtype, generator=gen)

    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.float32, generator=gen)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.float32, generator=gen)
    freqs = torch.randn(seq_len, HEAD_DIM // 2, device=device, dtype=torch.float32, generator=gen)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return QwenImageQKInput(q=q, k=k, q_weight=q_weight, k_weight=k_weight, cos=cos, sin=sin)


def test_qwen_image_fused_qk_norm_rope_cpu_fp32_fallback_matches_reference():
    data = _make_input(seq_len=7, dtype=torch.float32, device=torch.device("cpu"), packed_qkv_view=True)

    ref_q, ref_k = _reference_qwen_image_qk_norm_rope(data.q, data.k, data.q_weight, data.k_weight, data.cos, data.sin)
    out_q, out_k = qwen_image_fused_qk_norm_rope(data.q, data.k, data.q_weight, data.k_weight, data.cos, data.sin, EPS)

    torch.testing.assert_close(out_q, ref_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_k, ref_k, atol=1e-5, rtol=1e-5)


def test_qwen_image_qk_norm_rope_fast_path_support_rejects_cpu():
    q = torch.empty(1, 1024, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    cos = torch.empty(1024, HEAD_DIM // 2, dtype=torch.bfloat16)

    assert not qwen_image_qk_norm_rope_fast_path_supported(q, cos)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_qk_norm_rope_fast_path_support_rejects_empty_sequence():
    empty_q = torch.empty(1, 0, Q_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    empty_cos = torch.empty(0, HEAD_DIM // 2, device="cuda:0", dtype=torch.bfloat16)
    min_q = torch.empty(1, 1, Q_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    min_cos = torch.empty(1, HEAD_DIM // 2, device="cuda:0", dtype=torch.bfloat16)

    assert not qwen_image_qk_norm_rope_fast_path_supported(empty_q, empty_cos)
    assert qwen_image_qk_norm_rope_fast_path_supported(min_q, min_cos)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_fused_qk_norm_rope_cuda_fp32_fallback_matches_reference():
    seq_len = 7
    head_dim = HEAD_DIM
    dtype = torch.float32
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + head_dim)
    q = torch.randn(BATCH, seq_len, Q_HEADS, head_dim, device=device, dtype=dtype, generator=gen)
    k = torch.randn(BATCH, seq_len, K_HEADS, head_dim, device=device, dtype=dtype, generator=gen)
    q_weight = torch.randn(head_dim, device=device, dtype=torch.float32, generator=gen)
    k_weight = torch.randn(head_dim, device=device, dtype=torch.float32, generator=gen)
    freqs = torch.randn(seq_len, head_dim // 2, device=device, dtype=torch.float32, generator=gen)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)

    assert not qwen_image_qk_norm_rope_fast_path_supported(q, cos)
    ref_q, ref_k = _reference_qwen_image_qk_norm_rope(q, k, q_weight, k_weight, cos, sin)
    out_q, out_k = qwen_image_fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, EPS)

    _assert_close_with_error_stats(out_q, ref_q, name="query", atol=1e-5, rtol=1e-5)
    _assert_close_with_error_stats(out_k, ref_k, name="key", atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_fused_qk_norm_rope_cuda_non_default_head_dim_matches_reference():
    seq_len = 7
    head_dim = 32
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + head_dim)
    q = torch.randn(BATCH, seq_len, Q_HEADS, head_dim, device=device, dtype=dtype, generator=gen)
    k = torch.randn(BATCH, seq_len, K_HEADS, head_dim, device=device, dtype=dtype, generator=gen)
    q_weight = torch.randn(head_dim, device=device, dtype=torch.float32, generator=gen)
    k_weight = torch.randn(head_dim, device=device, dtype=torch.float32, generator=gen)
    freqs = torch.randn(seq_len, head_dim // 2, device=device, dtype=torch.float32, generator=gen)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)

    assert qwen_image_qk_norm_rope_fast_path_supported(q, cos)
    ref_q, ref_k = _reference_qwen_image_qk_norm_rope(q, k, q_weight, k_weight, cos, sin)
    out_q, out_k = qwen_image_fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, EPS)

    _assert_close_with_error_stats(out_q, ref_q, name="query", atol=1e-2, rtol=1e-2)
    _assert_close_with_error_stats(out_k, ref_k, name="key", atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [1, 2, 7, 64, 257, 1024])
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize("packed_qkv_view", [False, True])
def test_qwen_image_fused_qk_norm_rope_cuda_matches_reference(
    seq_len: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    packed_qkv_view: bool,
):
    data = _make_input(
        seq_len=seq_len,
        dtype=dtype,
        device=torch.device("cuda:0"),
        packed_qkv_view=packed_qkv_view,
    )

    ref_q, ref_k = _reference_qwen_image_qk_norm_rope(data.q, data.k, data.q_weight, data.k_weight, data.cos, data.sin)
    out_q, out_k = qwen_image_fused_qk_norm_rope(data.q, data.k, data.q_weight, data.k_weight, data.cos, data.sin, EPS)

    _assert_close_with_error_stats(out_q, ref_q, name="query", atol=atol, rtol=rtol)
    _assert_close_with_error_stats(out_k, ref_k, name="key", atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_fused_qk_norm_rope_fast_path_matches_reference():
    data = _make_input(
        seq_len=257,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        packed_qkv_view=True,
    )

    ref_q, ref_k = _reference_qwen_image_qk_norm_rope(data.q, data.k, data.q_weight, data.k_weight, data.cos, data.sin)
    out_q, out_k = qwen_image_fused_qk_norm_rope_fast_path(
        data.q,
        data.k,
        data.q_weight,
        data.k_weight,
        data.cos,
        data.sin,
        EPS,
    )

    _assert_close_with_error_stats(out_q, ref_q, name="query", atol=1e-2, rtol=1e-2)
    _assert_close_with_error_stats(out_k, ref_k, name="key", atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_fused_qk_norm_rope_4096_image_seq_matches_reference():
    data = _make_input(
        seq_len=4096,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        packed_qkv_view=True,
        batch=1,
    )

    assert qwen_image_qk_norm_rope_fast_path_supported(data.q, data.cos)
    ref_q, ref_k = _reference_qwen_image_qk_norm_rope(data.q, data.k, data.q_weight, data.k_weight, data.cos, data.sin)
    out_q, out_k = qwen_image_fused_qk_norm_rope_fast_path(
        data.q,
        data.k,
        data.q_weight,
        data.k_weight,
        data.cos,
        data.sin,
        EPS,
    )

    _assert_close_with_error_stats(out_q, ref_q, name="query", atol=1e-2, rtol=1e-2)
    _assert_close_with_error_stats(out_k, ref_k, name="key", atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_image_fused_qk_norm_rope_torch_compile_fullgraph_capture():
    data = _make_input(
        seq_len=257,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        packed_qkv_view=True,
    )

    def fn(
        q: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return qwen_image_fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, EPS)

    compiled_fn = torch.compile(fn, dynamic=True, fullgraph=True)
    ref_q, ref_k = fn(data.q, data.k, data.q_weight, data.k_weight, data.cos, data.sin)
    out_q, out_k = compiled_fn(data.q, data.k, data.q_weight, data.k_weight, data.cos, data.sin)

    torch.testing.assert_close(out_q, ref_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out_k, ref_k, atol=1e-2, rtol=1e-2)
