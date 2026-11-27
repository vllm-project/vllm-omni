# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Triton kernels for Ming's CFM DiT flow head.

Both kernels keep batch size, sequence length, and row strides as runtime
arguments (never ``tl.constexpr``), so each kernel compiles once across all
CUDA-graph batch buckets. Dispatch is gated on NVIDIA CUDA (``current_omni_platform.is_cuda()``),
``HAS_TRITON``, and ``not torch.compiler.is_compiling()`` so ROCm/CPU/NPU/XPU
and ``torch.compile`` callers keep their existing paths untouched.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.platforms import current_omni_platform

# Maximum sequence length handled by the single-program fused QKV+RoPE+attention
# kernel. Covers both single-frame Ming (S=3), Aggregator (S=2..5), and
# 4-frame Ming-omni (S=37) in one SRAM tile per (batch, head).
MAX_FUSED_ATTN_SEQ_LEN = 64
_SUPPORTED_HEAD_DIMS = (32, 64, 128)
_SHORT_ATTN_DOT_THRESHOLD = 8
_SUPPORTED_ATTN_DTYPES = (torch.bfloat16, torch.float16)
_SUPPORTED_STEP_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def cfm_triton_supported(tensor: torch.Tensor) -> bool:
    """Return True when the NVIDIA CUDA Triton fast path can run safely."""
    return HAS_TRITON and current_omni_platform.is_cuda() and tensor.is_cuda and not torch.compiler.is_compiling()


def cfm_qkv_rope_attn_supported(
    x: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    seq_len: int,
) -> bool:
    """Return True when ``fused_qkv_rope_attn`` supports this attention shape."""
    return (
        cfm_triton_supported(x)
        and x.dtype in _SUPPORTED_ATTN_DTYPES
        and heads > 0
        and head_dim in _SUPPORTED_HEAD_DIMS
        and 0 < seq_len <= MAX_FUSED_ATTN_SEQ_LEN
    )


if HAS_TRITON:

    @triton.jit(do_not_specialize=["seq_len"])
    def _qkv_rope_attn_kernel(
        qkv_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        stride_qkv_b,
        stride_qkv_s,
        stride_cos_s,
        stride_out_b,
        stride_out_s,
        seq_len,
        inner_dim: tl.constexpr,
        head_dim: tl.constexpr,
        half_d: tl.constexpr,
        block_s: tl.constexpr,
        use_dot: tl.constexpr,
    ):
        batch = tl.program_id(0)
        head = tl.program_id(1)
        s = tl.arange(0, block_s)
        k = tl.arange(0, half_d)
        mask_s = s < seq_len

        base_q = qkv_ptr + batch * stride_qkv_b + s[:, None] * stride_qkv_s + head * head_dim
        base_k = base_q + inner_dim
        base_v = base_k + inner_dim

        c_e = tl.load(cos_ptr + s[:, None] * stride_cos_s + (2 * k)[None, :], mask=mask_s[:, None], other=0.0).to(
            tl.float32
        )
        s_e = tl.load(sin_ptr + s[:, None] * stride_cos_s + (2 * k)[None, :], mask=mask_s[:, None], other=0.0).to(
            tl.float32
        )
        c_o = tl.load(cos_ptr + s[:, None] * stride_cos_s + (2 * k + 1)[None, :], mask=mask_s[:, None], other=0.0).to(
            tl.float32
        )
        s_o = tl.load(sin_ptr + s[:, None] * stride_cos_s + (2 * k + 1)[None, :], mask=mask_s[:, None], other=0.0).to(
            tl.float32
        )

        dtype: tl.constexpr = qkv_ptr.dtype.element_ty

        qe = tl.load(base_q + (2 * k)[None, :], mask=mask_s[:, None], other=0.0).to(tl.float32)
        qo = tl.load(base_q + (2 * k + 1)[None, :], mask=mask_s[:, None], other=0.0).to(tl.float32)
        q_rot_e = (qe * c_e - qo * s_e).to(dtype)
        q_rot_o = (qo * c_o + qe * s_o).to(dtype)

        ke = tl.load(base_k + (2 * k)[None, :], mask=mask_s[:, None], other=0.0).to(tl.float32)
        ko = tl.load(base_k + (2 * k + 1)[None, :], mask=mask_s[:, None], other=0.0).to(tl.float32)
        k_rot_e = (ke * c_e - ko * s_e).to(dtype)
        k_rot_o = (ko * c_o + ke * s_o).to(dtype)

        scale: tl.constexpr = head_dim**-0.5
        if use_dot:
            scores = (
                tl.dot(q_rot_e, tl.trans(k_rot_e), out_dtype=tl.float32)
                + tl.dot(q_rot_o, tl.trans(k_rot_o), out_dtype=tl.float32)
            ) * scale
        else:
            qef = q_rot_e.to(tl.float32)
            qof = q_rot_o.to(tl.float32)
            kef = k_rot_e.to(tl.float32)
            kof = k_rot_o.to(tl.float32)
            scores = (
                tl.sum(qef[:, None, :] * kef[None, :, :], axis=2) + tl.sum(qof[:, None, :] * kof[None, :, :], axis=2)
            ) * scale

        scores = tl.where(mask_s[None, :], scores, -float("inf"))
        scores = scores - tl.max(scores, axis=1)[:, None]
        probs = tl.exp(scores)
        probs = probs / tl.sum(probs, axis=1)[:, None]

        ve = tl.load(base_v + (2 * k)[None, :], mask=mask_s[:, None], other=0.0)
        vo = tl.load(base_v + (2 * k + 1)[None, :], mask=mask_s[:, None], other=0.0)
        if use_dot:
            p_cast = probs.to(dtype)
            oe = tl.dot(p_cast, ve, out_dtype=tl.float32)
            oo = tl.dot(p_cast, vo, out_dtype=tl.float32)
        else:
            oe = tl.sum(probs[:, :, None] * ve.to(tl.float32)[None, :, :], axis=1)
            oo = tl.sum(probs[:, :, None] * vo.to(tl.float32)[None, :, :], axis=1)

        base_out = out_ptr + batch * stride_out_b + s[:, None] * stride_out_s + head * head_dim
        tl.store(base_out + (2 * k)[None, :], oe.to(dtype), mask=mask_s[:, None])
        tl.store(base_out + (2 * k + 1)[None, :], oo.to(dtype), mask=mask_s[:, None])

    @triton.jit(do_not_specialize=["pred_half_stride", "step"])
    def _cfm_euler_sde_step_kernel(
        pred_ptr,
        state_ptr,
        noise_ptr,
        t_ptr,
        sde_args_ptr,
        out_ptr,
        out_cast_ptr,
        pred_stride_row,
        pred_half_stride,
        state_stride_row,
        step,
        width: tl.constexpr,
        block: tl.constexpr,
        add_noise: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, block)
        mask = col < width
        pred_dtype: tl.constexpr = pred_ptr.dtype.element_ty
        state_dtype: tl.constexpr = state_ptr.dtype.element_ty
        cast_dtype: tl.constexpr = out_cast_ptr.dtype.element_ty

        cond = tl.load(pred_ptr + row * pred_stride_row + col, mask=mask, other=0.0).to(tl.float32)
        null = tl.load(pred_ptr + pred_half_stride + row * pred_stride_row + col, mask=mask, other=0.0).to(tl.float32)
        cfg = tl.load(sde_args_ptr).to(pred_dtype).to(tl.float32)
        t0 = tl.load(t_ptr + step).to(tl.float32)
        t1 = tl.load(t_ptr + step + 1).to(tl.float32)
        dt = (t1 - t0).to(state_dtype).to(tl.float32)

        diff = (cond - null).to(pred_dtype).to(tl.float32)
        scaled = (diff * cfg).to(pred_dtype).to(tl.float32)
        flow = (cond + scaled).to(pred_dtype).to(state_dtype).to(tl.float32)
        y = tl.load(state_ptr + row * state_stride_row + col, mask=mask, other=0.0).to(tl.float32)
        y = (y + (flow * dt).to(state_dtype).to(tl.float32)).to(state_dtype).to(tl.float32)
        if add_noise:
            sigma = tl.load(sde_args_ptr + 1).to(tl.float32)
            temp = tl.load(sde_args_ptr + 2).to(tl.float32)
            z = tl.load(noise_ptr + row * state_stride_row + col, mask=mask, other=0.0).to(tl.float32)
            noise_scale = (sigma * tl.sqrt(temp) * tl.sqrt(tl.abs(dt))).to(state_dtype).to(tl.float32)
            y = (y + (noise_scale * z).to(state_dtype).to(tl.float32)).to(state_dtype).to(tl.float32)
        y_out = y.to(state_dtype)
        tl.store(out_ptr + row * state_stride_row + col, y_out, mask=mask)
        tl.store(out_cast_ptr + row * state_stride_row + col, y_out.to(cast_dtype), mask=mask)


def fused_qkv_rope_attn(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
) -> torch.Tensor:
    """Apply interleaved-pair RoPE to Q/K and unmasked self-attention directly on fused QKV.

    Args:
        qkv: ``[Batch, SeqLen, 3 * InnerDim]`` contiguous QKV projection output.
        cos: Precomputed RoPE cosine table whose last two dims cover ``[SeqLen, HeadDim]``.
        sin: Precomputed RoPE sine table matching ``cos``.
        heads: Number of attention heads (``InnerDim // HeadDim``).
    """
    if not HAS_TRITON:
        raise RuntimeError("fused_qkv_rope_attn requires Triton")
    if qkv.ndim != 3:
        raise ValueError(f"Expected rank-3 qkv [Batch, SeqLen, 3*InnerDim], got {tuple(qkv.shape)}")
    batch, seq_len, three_inner = qkv.shape
    if three_inner % 3 != 0:
        raise ValueError(f"Expected qkv last dim divisible by 3, got {three_inner}")
    inner_dim = three_inner // 3
    if heads <= 0 or inner_dim % heads != 0:
        raise ValueError(f"Expected inner_dim={inner_dim} divisible by heads={heads}")
    head_dim = inner_dim // heads
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={head_dim}; supported={_SUPPORTED_HEAD_DIMS}")
    if seq_len <= 0 or seq_len > MAX_FUSED_ATTN_SEQ_LEN:
        raise ValueError(f"Unsupported seq_len={seq_len}; expected 1..{MAX_FUSED_ATTN_SEQ_LEN}")

    qkv_contig = qkv.contiguous()
    cos_2d = cos.reshape(-1, head_dim).contiguous()
    sin_2d = sin.reshape(-1, head_dim).contiguous()
    if cos_2d.shape[0] < seq_len or sin_2d.shape[0] < seq_len:
        raise ValueError(f"RoPE table too short for seq_len={seq_len}: cos={tuple(cos_2d.shape)}")

    out = torch.empty((batch, seq_len, inner_dim), device=qkv_contig.device, dtype=qkv_contig.dtype)
    use_dot = seq_len > _SHORT_ATTN_DOT_THRESHOLD
    block_s = max(16 if use_dot else 4, triton.next_power_of_2(seq_len))
    _qkv_rope_attn_kernel[(batch, heads)](
        qkv_contig,
        cos_2d,
        sin_2d,
        out,
        qkv_contig.stride(0),
        qkv_contig.stride(1),
        cos_2d.stride(0),
        out.stride(0),
        out.stride(1),
        seq_len,
        inner_dim=inner_dim,
        head_dim=head_dim,
        half_d=head_dim // 2,
        block_s=block_s,
        use_dot=use_dot,
        num_warps=4 if use_dot else 1,
    )
    return out


def fused_cfm_euler_sde_step(
    pred_cfg: torch.Tensor,
    y0: torch.Tensor,
    sde_rnd_step: torch.Tensor,
    t: torch.Tensor,
    sde_args: torch.Tensor,
    step: int,
    *,
    add_noise: bool,
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine CFG velocity, apply one Euler (+ optional SDE) step, and emit model-dtype cast."""
    if not HAS_TRITON:
        raise RuntimeError("fused_cfm_euler_sde_step requires Triton")
    if pred_cfg.ndim != 3 or y0.ndim != 3:
        raise ValueError(f"Expected rank-3 pred_cfg and y0, got {tuple(pred_cfg.shape)} and {tuple(y0.shape)}")
    batch, patch_size, latent_dim = y0.shape
    if pred_cfg.shape != (2 * batch, patch_size, latent_dim):
        raise ValueError(f"pred_cfg shape {tuple(pred_cfg.shape)} does not match 2x y0 shape {tuple(y0.shape)}")
    if sde_rnd_step.shape != y0.shape:
        raise ValueError(f"sde_rnd_step shape {tuple(sde_rnd_step.shape)} != y0 shape {tuple(y0.shape)}")
    if y0.dtype not in _SUPPORTED_STEP_DTYPES:
        raise ValueError(f"Unsupported solver state dtype {y0.dtype}")
    if t.ndim != 1 or step < 0 or step + 1 >= t.shape[0]:
        raise ValueError(f"Invalid step={step} for timestep schedule of shape {tuple(t.shape)}")
    if sde_args.numel() < 3:
        raise ValueError(f"Expected at least 3 sde_args entries, got {sde_args.numel()}")

    pred_c = pred_cfg.contiguous()
    y0_c = y0.contiguous()
    noise_c = sde_rnd_step.contiguous()
    t_c = t.contiguous()
    args_c = sde_args.contiguous()

    n_rows = batch * patch_size
    out = torch.empty_like(y0_c)
    out_cast = torch.empty_like(y0_c, dtype=model_dtype)
    _cfm_euler_sde_step_kernel[(n_rows,)](
        pred_c,
        y0_c,
        noise_c,
        t_c,
        args_c,
        out,
        out_cast,
        latent_dim,
        n_rows * latent_dim,
        latent_dim,
        step,
        width=latent_dim,
        block=triton.next_power_of_2(latent_dim),
        add_noise=add_noise,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out, out_cast


__all__ = [
    "MAX_FUSED_ATTN_SEQ_LEN",
    "cfm_qkv_rope_attn_supported",
    "cfm_triton_supported",
    "fused_cfm_euler_sde_step",
    "fused_qkv_rope_attn",
]
