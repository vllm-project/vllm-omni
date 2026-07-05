# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness tests for Sensenova-U1 Triton kernels.

Verifies that fused-RMSNorm+3D Rope Triton kernels
produce results numerically equivalent to the PyTorch reference.
All computations follow the kernel's internal float32 promotion rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

SEED = 42
B = 1
H_Q = 32
H_K = 8
HEAD_DIM = 128
T_DIM = 64
HW_DIM = 64
HW_ROPE_DIM = 32
QKV_DIM = (H_Q + H_K + H_K) * HEAD_DIM
EPS = 1e-6
DEVICE = torch.device("cuda:0")


@dataclass(frozen=True)
class KernelInput:
    seq_len: int
    q: torch.Tensor
    k: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    q_norm_hw_weight: torch.Tensor
    k_norm_hw_weight: torch.Tensor
    cos_t: torch.Tensor
    sin_t: torch.Tensor
    cos_h: torch.Tensor
    sin_h: torch.Tensor
    cos_w: torch.Tensor
    sin_w: torch.Tensor

    def args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.q,
            self.k,
            self.q_norm_weight,
            self.k_norm_weight,
            self.q_norm_hw_weight,
            self.k_norm_hw_weight,
            self.cos_t,
            self.sin_t,
            self.cos_h,
            self.sin_h,
            self.cos_w,
            self.sin_w,
        )


try:
    from vllm_omni.diffusion.models.sensenova_u1.fused_rmsnorm_rope import triton_qk_norm_rope

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

triton_available = pytest.mark.skipif(not _TRITON_AVAILABLE, reason="Triton not available on this platform")


# ---------------------------------------------------------------------------
# PyTorch reference implementations (mirror the kernel's internal arithmetic)
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    input_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return xf.to(input_dtype) * weight


def assert_close_with_error_stats(
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


def reference_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_hw_weight: torch.Tensor,
    k_norm_hw_weight: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference matching the model's current math."""
    q_t, q_hw = q.chunk(2, dim=-1)
    k_t, k_hw = k.chunk(2, dim=-1)

    q_t = rmsnorm(q_t, q_norm_weight).transpose(1, 2)
    k_t = rmsnorm(k_t, k_norm_weight).transpose(1, 2)
    q_hw = rmsnorm(q_hw, q_norm_hw_weight).transpose(1, 2)
    k_hw = rmsnorm(k_hw, k_norm_hw_weight).transpose(1, 2)

    q_h, q_w = q_hw.chunk(2, dim=-1)
    k_h, k_w = k_hw.chunk(2, dim=-1)

    q_t, k_t = apply_rope(q_t, k_t, cos_t, sin_t)
    q_h, k_h = apply_rope(q_h, k_h, cos_h, sin_h)
    q_w, k_w = apply_rope(q_w, k_w, cos_w, sin_w)

    query = torch.cat([q_t, q_h, q_w], dim=-1)
    key = torch.cat([k_t, k_h, k_w], dim=-1)
    return query, key


def make_input(seq_len: int, dtype: torch.dtype, device: torch.device, seed: int) -> KernelInput:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + seq_len)

    # Match real model layout: q/k/v are views split from fused qkv.
    qkv = torch.randn(B, seq_len, QKV_DIM, device=device, dtype=dtype, generator=gen)
    q, k, _v = qkv.split([H_Q * HEAD_DIM, H_K * HEAD_DIM, H_K * HEAD_DIM], dim=-1)
    q = q.view(B, seq_len, H_Q, HEAD_DIM)
    k = k.view(B, seq_len, H_K, HEAD_DIM)

    q_norm_weight = torch.randn(T_DIM, device=device, dtype=dtype, generator=gen)
    k_norm_weight = torch.randn(T_DIM, device=device, dtype=dtype, generator=gen)
    q_norm_hw_weight = torch.randn(HW_DIM, device=device, dtype=dtype, generator=gen)
    k_norm_hw_weight = torch.randn(HW_DIM, device=device, dtype=dtype, generator=gen)

    # Match Qwen3RotaryEmbedding: emb = torch.cat((freqs, freqs), dim=-1).
    def make_rope_pair(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos_half = torch.randn(B, seq_len, dim // 2, device=device, dtype=dtype, generator=gen)
        sin_half = torch.randn(B, seq_len, dim // 2, device=device, dtype=dtype, generator=gen)
        return torch.cat((cos_half, cos_half), dim=-1), torch.cat((sin_half, sin_half), dim=-1)

    cos_t, sin_t = make_rope_pair(T_DIM)
    cos_h, sin_h = make_rope_pair(HW_ROPE_DIM)
    cos_w, sin_w = make_rope_pair(HW_ROPE_DIM)

    return KernelInput(
        seq_len=seq_len,
        q=q,
        k=k,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        q_norm_hw_weight=q_norm_hw_weight,
        k_norm_hw_weight=k_norm_hw_weight,
        cos_t=cos_t,
        sin_t=sin_t,
        cos_h=cos_h,
        sin_h=sin_h,
        cos_w=cos_w,
        sin_w=sin_w,
    )


# ---------------------------------------------------------------------------
# Fused RMSNorm + 3D Rope
# ---------------------------------------------------------------------------


@triton_available
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 9, 16, 32, 64, 128, 256, 260, 271, 512, 1024, 2048, 4096])
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
def test_fused_qk_norm_rope_matches_reference(seq_len: int, dtype: torch.dtype, atol: float, rtol: float):
    data = make_input(seq_len, dtype=dtype, device=DEVICE, seed=SEED)
    call_args = data.args()
    ref_q, ref_k = reference_kernel(*call_args)
    out_q, out_k = triton_qk_norm_rope(*call_args, EPS)

    assert_close_with_error_stats(out_q, ref_q, name="query", atol=atol, rtol=rtol)
    assert_close_with_error_stats(out_k, ref_k, name="key", atol=atol, rtol=rtol)
