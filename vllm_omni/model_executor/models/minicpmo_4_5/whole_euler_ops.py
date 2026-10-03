# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# ruff: noqa: N803

"""Fused Euler/CFG step of the Whole-Euler CFM solve (Stage 2).

Every Euler step of ``WholeEulerCFMGraphWrapper._run_euler_loop`` stages the
estimator input as ``cat(cat(x, x), mu, speakers, cond)`` on the channel axis
(two copies), the estimator's ``in_proj`` reads it transposed (a third copy,
plus a separate bias add because the transposed input cannot fold into
``addmm``), and the step ends with five elementwise passes for
``x + dt * ((1 + r) * cond - r * uncond)``.

Only ``x`` changes between steps, so ``stage_estimator_input`` writes the
whole input once per solve, frames-major ``(2B, T, C_in)`` (the layout
``in_proj`` reads), and ``euler_cfg_step`` does the CFG combination and the
update in one pass and writes the new ``x`` into both CFG halves of that
buffer for the next step. The arithmetic is PyTorch's, op by op in fp32
without FMA contraction, so ``x`` is bitwise identical to the eager step;
only ``in_proj`` changes kernel (bias in the GEMM epilogue), which is fp32
rounding-level.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


def fused_euler_supported(tensor: torch.Tensor) -> bool:
    return HAS_TRITON and tensor.is_cuda and tensor.dtype == torch.float32


@triton.jit
def _stage_kernel(
    out_ptr,
    x_ptr,
    mu_ptr,
    spk_ptr,
    cond_ptr,
    batch,
    frames,
    stride_xn,
    stride_xc,
    stride_xt,
    stride_mn,
    stride_mc,
    stride_mt,
    stride_sn,
    stride_sc,
    stride_cn,
    stride_cc,
    stride_ct,
    C_X: tl.constexpr,
    C_MU: tl.constexpr,
    C_SPK: tl.constexpr,
    C_COND: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    c = tl.arange(0, BLOCK_C)
    width: tl.constexpr = C_X + C_MU + C_SPK + C_COND
    frame = (t < frames)[:, None]
    is_x = (c < C_X)[None, :]
    is_mu = ((c >= C_X) & (c < C_X + C_MU))[None, :]
    is_spk = ((c >= C_X + C_MU) & (c < C_X + C_MU + C_SPK))[None, :]
    is_cond = ((c >= C_X + C_MU + C_SPK) & (c < width))[None, :]
    xc = c[None, :]
    mc = (c - C_X)[None, :]
    sc = c - C_X - C_MU
    cc = (c - C_X - C_MU - C_SPK)[None, :]
    tt = t[:, None]
    # Each channel is a plain copy of one source (selected, not summed, so -0.0 stays -0.0).
    x = tl.load(x_ptr + (n % batch) * stride_xn + xc * stride_xc + tt * stride_xt, mask=frame & is_x, other=0.0)
    mu = tl.load(mu_ptr + n * stride_mn + mc * stride_mc + tt * stride_mt, mask=frame & is_mu, other=0.0)
    speaker = tl.load(spk_ptr + n * stride_sn + sc * stride_sc, mask=(sc >= 0) & (sc < C_SPK), other=0.0)
    cond = tl.load(cond_ptr + n * stride_cn + cc * stride_cc + tt * stride_ct, mask=frame & is_cond, other=0.0)
    value = tl.where(is_x, x, tl.where(is_mu, mu, tl.where(is_spk, speaker[None, :], cond)))
    tl.store(out_ptr + (n * frames + tt) * width + xc, value, mask=frame & (c < width)[None, :])


@triton.jit
def _euler_kernel(
    x_out_ptr,
    x_ptr,
    estimate_ptr,
    staged_ptr,
    batch,
    frames,
    stride_xn,
    stride_xc,
    stride_on,
    stride_oc,
    stride_en,
    stride_ec,
    stride_et,
    cond_scale,
    uncond_scale,
    dt,
    CHANNELS: tl.constexpr,
    STAGED_WIDTH: tl.constexpr,
    WRITE_STAGED: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    c = tl.arange(0, BLOCK_C)
    mask = (t < frames)[:, None] & (c < CHANNELS)[None, :]
    tt = t[:, None]
    cc = c[None, :]
    conditional = tl.load(estimate_ptr + b * stride_en + cc * stride_ec + tt * stride_et, mask=mask, other=0.0)
    unconditional = tl.load(
        estimate_ptr + (b + batch) * stride_en + cc * stride_ec + tt * stride_et, mask=mask, other=0.0
    )
    x = tl.load(x_ptr + b * stride_xn + cc * stride_xc + tt, mask=mask, other=0.0)
    # ``(1 + r) * cond - r * uncond``, then ``x + dt * v``: the eager op order (no FMA, see the launch).
    velocity = conditional * cond_scale - unconditional * uncond_scale
    x = x + velocity * dt
    tl.store(x_out_ptr + b * stride_on + cc * stride_oc + tt, x, mask=mask)
    if WRITE_STAGED:
        staged = staged_ptr + tt * STAGED_WIDTH + cc
        tl.store(staged + b * frames * STAGED_WIDTH, x, mask=mask)
        tl.store(staged + (b + batch) * frames * STAGED_WIDTH, x, mask=mask)


_BLOCK_T = 32
_STAGE_BLOCK_T = 8


def stage_estimator_input(
    out: torch.Tensor,
    x: torch.Tensor,
    mu_cfg: torch.Tensor,
    speakers_cfg: torch.Tensor,
    cond_cfg: torch.Tensor,
) -> torch.Tensor:
    """``cat((cat((x, x)), mu, speakers, cond), dim=1).transpose(1, 2)`` into contiguous ``out`` (2B, T, C_in)."""
    batch, c_x, frames = (int(d) for d in x.shape)
    c_mu, c_spk, c_cond = int(mu_cfg.shape[1]), int(speakers_cfg.shape[1]), int(cond_cfg.shape[1])
    width = c_x + c_mu + c_spk + c_cond
    if tuple(out.shape) != (2 * batch, frames, width) or not out.is_contiguous():
        raise ValueError(f"stage_estimator_input: out must be contiguous {(2 * batch, frames, width)}")
    if not fused_euler_supported(out):
        speakers = speakers_cfg.unsqueeze(-1).expand(-1, -1, frames)
        out.copy_(torch.cat((torch.cat((x, x), dim=0), mu_cfg, speakers, cond_cfg), dim=1).transpose(1, 2))
        return out
    _stage_kernel[(2 * batch, triton.cdiv(frames, _STAGE_BLOCK_T))](
        out,
        x,
        mu_cfg,
        speakers_cfg,
        cond_cfg,
        batch,
        frames,
        *x.stride(),
        *mu_cfg.stride(),
        *speakers_cfg.stride(),
        *cond_cfg.stride(),
        C_X=c_x,
        C_MU=c_mu,
        C_SPK=c_spk,
        C_COND=c_cond,
        BLOCK_T=_STAGE_BLOCK_T,
        BLOCK_C=triton.next_power_of_2(width),
        num_warps=4,
    )
    return out


def euler_cfg_step(
    x_out: torch.Tensor,
    x: torch.Tensor,
    estimate: torch.Tensor,
    dt: float,
    inference_cfg_rate: float,
    staged: torch.Tensor | None = None,
) -> torch.Tensor:
    """``_euler_step(x, estimate, dt, r, B)`` into ``x_out`` (may be ``x``), and into ``staged``'s x channels.

    ``x`` / ``x_out`` are ``(B, C, T)`` with contiguous frames, ``estimate``
    is ``(2B, C, T)`` ``[cond | uncond]`` (any strides), ``staged`` the
    ``(2B, T, C_in)`` buffer of ``stage_estimator_input`` whose first ``C``
    channels of both CFG halves receive the new ``x``.
    """
    batch, channels, frames = (int(d) for d in x.shape)
    if tuple(estimate.shape) != (2 * batch, channels, frames) or tuple(x_out.shape) != tuple(x.shape):
        raise ValueError("euler_cfg_step: shape mismatch")
    if x.stride(2) != 1 or x_out.stride(2) != 1:
        raise ValueError("euler_cfg_step: x needs contiguous frames")
    if staged is not None and (
        not staged.is_contiguous() or int(staged.shape[0]) != 2 * batch or int(staged.shape[1]) != frames
    ):
        raise ValueError("euler_cfg_step: staged must be a contiguous (2B, T, C_in) buffer")
    if not fused_euler_supported(x):
        conditional, unconditional = estimate.split(batch, dim=0)
        velocity = (1.0 + inference_cfg_rate) * conditional - inference_cfg_rate * unconditional
        x_out.copy_(x + dt * velocity)
        if staged is not None:
            staged[:, :, :channels].copy_(torch.cat((x_out, x_out), dim=0).transpose(1, 2))
        return x_out
    _euler_kernel[(batch, triton.cdiv(frames, _BLOCK_T))](
        x_out,
        x,
        estimate,
        staged if staged is not None else x_out,
        batch,
        frames,
        x.stride(0),
        x.stride(1),
        x_out.stride(0),
        x_out.stride(1),
        *estimate.stride(),
        1.0 + float(inference_cfg_rate),
        float(inference_cfg_rate),
        float(dt),
        CHANNELS=channels,
        STAGED_WIDTH=int(staged.shape[2]) if staged is not None else 0,
        WRITE_STAGED=staged is not None,
        BLOCK_T=_BLOCK_T,
        BLOCK_C=triton.next_power_of_2(channels),
        num_warps=4,
        # The eager step rounds every product and sum; FMA contraction would not.
        enable_fp_fusion=False,
    )
    return x_out
