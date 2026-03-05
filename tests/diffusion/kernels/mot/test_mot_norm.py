# ruff: noqa: N803, E741
"""Layer-level correctness & performance test for MoTRMSNorm.

Compares two equivalent computation paths:
  - Reference: 2x vLLM RMSNorm CUDA kernel + PyTorch index scatter/gather
    (rms_norm(x[text_idx], text_w) + rms_norm(x[vae_idx], vae_w))
  - Target:    1x MoTRMSNorm fused Triton kernel
    (mot_norm(x, text_indices, vae_indices))

Usage::
    pytest tests/diffusion/kernels/mot/test_mot_norm.py -v -s
"""

from __future__ import annotations

import time

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import rms_norm

from vllm_omni.diffusion.layers.mot.mot_layernorm import MoTRMSNorm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_indices(
    M: int,
    text_ratio: float,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    M_text = max(1, int(M * text_ratio))
    perm = torch.randperm(M, device=device)
    text_indices = perm[:M_text].sort().values
    vae_indices = perm[M_text:].sort().values
    return text_indices, vae_indices


def _benchmark(fn, warmup: int = 20, iters: int = 200) -> float:
    """Return mean latency in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _reference_forward(
    x: torch.Tensor,
    text_indices: torch.Tensor,
    vae_indices: torch.Tensor,
    text_weight: torch.Tensor,
    vae_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference: index-gather → 2x vLLM rms_norm CUDA kernel → scatter."""
    output = torch.empty_like(x)
    output[text_indices] = rms_norm(x[text_indices], text_weight, eps)
    output[vae_indices] = rms_norm(x[vae_indices], vae_weight, eps)
    return output


def _reference_forward_head_norm(
    x: torch.Tensor,
    text_indices: torch.Tensor,
    vae_indices: torch.Tensor,
    text_weight: torch.Tensor,
    vae_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference for head_norm=True path.

    The MoT routing happens on token dimension while RMSNorm is applied
    independently on each head's last dimension.
    """
    output = torch.empty_like(x)
    hidden_size = x.shape[-1]
    output[text_indices] = rms_norm(
        x[text_indices].reshape(-1, hidden_size),
        text_weight,
        eps,
    ).reshape_as(x[text_indices])
    output[vae_indices] = rms_norm(
        x[vae_indices].reshape(-1, hidden_size),
        vae_weight,
        eps,
    ).reshape_as(x[vae_indices])
    return output


def _check_and_report(ref: torch.Tensor, mot: torch.Tensor, tag: str):
    """Compare outputs, print metrics, and assert correctness.

    Both ``ref`` and ``mot`` are in the original compute dtype (e.g. bf16).
    We upcast to fp32 solely for computing error metrics with higher
    arithmetic precision — the actual layer outputs remain bf16.
    """
    ref_hp = ref.float()
    mot_hp = mot.float()

    abs_err = (ref_hp - mot_hp).abs()
    max_abs = abs_err.max().item()

    denom = ref_hp.abs().clamp(min=1.0)
    max_rel = (abs_err / denom).max().item()

    cos_sim = (
        torch.nn.functional.cosine_similarity(
            ref_hp,
            mot_hp,
            dim=-1,
        )
        .min()
        .item()
    )

    print(f"\n  [{tag}]  max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  min_cos_sim={cos_sim:.6f}")

    # RMSNorm is element-wise (no cross-element accumulation like GEMM),
    # so the error between two fp32-accumulating implementations should
    # be very small — well within bf16 rounding.
    assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim:.6f}"
    assert max_rel < 0.05, f"Max relative error too large: {max_rel:.4e}"


def _run_timing(
    ref_fn,
    mot_fn,
    tag: str,
    warmup: int = 20,
    iters: int = 200,
):
    """Benchmark both paths and print timing comparison."""
    ref_ms = _benchmark(ref_fn, warmup=warmup, iters=iters)
    mot_ms = _benchmark(mot_fn, warmup=warmup, iters=iters)
    speedup = ref_ms / mot_ms if mot_ms > 0 else float("inf")
    print(f"  [{tag}]  Ref(2x rms_norm): {ref_ms:.3f} ms  |  MoT(fused): {mot_ms:.3f} ms  |  Speedup: {speedup:.2f}x")


# =========================================================================
#  Test: MoTRMSNorm vs 2x vLLM rms_norm
# =========================================================================


@pytest.mark.parametrize(
    "M, hidden_size, text_ratio",
    [
        (2048, 3584, 0.01),
        (8192, 3584, 0.01),
        (2048, 128, 0.01),
        (8192, 128, 0.01),
    ],
    ids=[
        "M2048_H3584_layernorm",
        "M8192_H3584_layernorm",
        "M2048_H128_qknorm",
        "M8192_H128_qknorm",
    ],
)
def test_mot_rms_norm(M: int, hidden_size: int, text_ratio: float):
    torch.manual_seed(42)
    dtype = torch.bfloat16

    # --- Build MoT layer ---
    mot_norm = MoTRMSNorm(hidden_size, eps=_EPS).cuda()
    with torch.no_grad():
        W_text = torch.randn(hidden_size, dtype=dtype, device="cuda")
        W_vae = torch.randn(hidden_size, dtype=dtype, device="cuda")
        mot_norm.weight.data.copy_(W_text)
        mot_norm.gen_weight.data.copy_(W_vae)

    # --- Build reference weights (same data) ---
    ref_text_weight = W_text.clone()
    ref_vae_weight = W_vae.clone()

    # --- Inputs ---
    x = torch.randn(M, hidden_size, dtype=dtype, device="cuda")
    text_idx, vae_idx = _make_indices(M, text_ratio)

    tag = f"RMSNorm M={M} H={hidden_size}"

    # vLLM's rms_norm() uses vllm._custom_ops which may inspect global
    # config; wrap in VllmConfig context for safety.
    with set_current_vllm_config(VllmConfig()):
        # --- Correctness (also warms up Triton JIT) ---
        with torch.no_grad():
            ref = _reference_forward(
                x,
                text_idx,
                vae_idx,
                ref_text_weight,
                ref_vae_weight,
                _EPS,
            )
            mot_out = mot_norm(x, text_idx, vae_idx)

        _check_and_report(ref, mot_out, tag)

        # --- Performance ---
        with torch.no_grad():
            _run_timing(
                lambda: _reference_forward(
                    x,
                    text_idx,
                    vae_idx,
                    ref_text_weight,
                    ref_vae_weight,
                    _EPS,
                ),
                lambda: mot_norm(x, text_idx, vae_idx),
                tag,
            )


@pytest.mark.parametrize(
    "M, num_heads, head_dim, text_ratio",
    [
        (2048, 28, 128, 0.01),
    ],
    ids=[
        "M2048_NH28_HD128_qknorm_head_norm",
    ],
)
def test_mot_rms_norm_head_norm(
    M: int,
    num_heads: int,
    head_dim: int,
    text_ratio: float,
):
    torch.manual_seed(42)
    dtype = torch.bfloat16

    # --- Build MoT layer (head_norm path) ---
    mot_norm = MoTRMSNorm(head_dim, head_norm=True, eps=_EPS).cuda()
    with torch.no_grad():
        W_text = torch.randn(head_dim, dtype=dtype, device="cuda")
        W_vae = torch.randn(head_dim, dtype=dtype, device="cuda")
        mot_norm.weight.data.copy_(W_text)
        mot_norm.gen_weight.data.copy_(W_vae)

    # --- Build reference weights (same data) ---
    ref_text_weight = W_text.clone()
    ref_vae_weight = W_vae.clone()

    # --- Inputs ---
    x = torch.randn(M, num_heads, head_dim, dtype=dtype, device="cuda")
    text_idx, vae_idx = _make_indices(M, text_ratio)

    tag = f"RMSNorm(head_norm=True) M={M} NH={num_heads} HD={head_dim}"

    with set_current_vllm_config(VllmConfig()):
        # --- Correctness (also warms up Triton JIT) ---
        with torch.no_grad():
            ref = _reference_forward_head_norm(
                x,
                text_idx,
                vae_idx,
                ref_text_weight,
                ref_vae_weight,
                _EPS,
            )
            mot_out = mot_norm(x, text_idx, vae_idx)

        _check_and_report(ref, mot_out, tag)
