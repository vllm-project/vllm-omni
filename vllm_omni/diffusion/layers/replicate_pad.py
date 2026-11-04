# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Bit-exact ``F.pad(x, pad, mode="replicate")`` for 5D tensors.

ATen's ``replication_pad3d`` runs far below memory bandwidth (~20% of the
HunyuanVideo-1.5 VAE decode). Replicate padding only copies values, so a
gather at clamped input indices reproduces it exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.diffusion.layers.vae_rms_norm import kernels_allowed

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK = 4096
_MAX_GRID_Y = 65535

if HAS_TRITON:

    @triton.jit
    def _replicate_pad_kernel(
        X,
        OUT,
        channels,
        frames,
        height,
        out_frames,
        out_plane,
        pad_t,
        pad_h,
        pad_w,
        stride_n,
        stride_c,
        stride_t,
        stride_h,
        stride_w,
        WIDTH: tl.constexpr,
        OUT_WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One program row per output (n, c, t) plane; the constexpr width keeps the per-element divide cheap.
        plane = tl.program_id(0)
        nc = plane // out_frames
        t = tl.minimum(tl.maximum(plane % out_frames - pad_t, 0), frames - 1)
        base = (nc // channels).to(tl.int64) * stride_n + (nc % channels).to(tl.int64) * stride_c
        base += t.to(tl.int64) * stride_t
        idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < out_plane
        h = tl.minimum(tl.maximum(idx // OUT_WIDTH - pad_h, 0), height - 1)
        w = tl.minimum(tl.maximum(idx % OUT_WIDTH - pad_w, 0), WIDTH - 1)
        vals = tl.load(X + base + h.to(tl.int64) * stride_h + w.to(tl.int64) * stride_w, mask=mask)
        tl.store(OUT + plane.to(tl.int64) * out_plane + idx, vals, mask=mask)


def _kernel_eligible(x: torch.Tensor, pad: tuple[int, ...]) -> bool:
    if not (HAS_TRITON and x.is_cuda and kernels_allowed(x)):
        return False
    if x.dim() != 5 or x.dtype not in _DTYPES or x.numel() == 0 or len(pad) != 6 or min(pad) < 0:
        return False
    # Layouts for which F.pad returns a contiguous tensor (contiguous, or the frame-major view).
    return x.is_contiguous() or x.transpose(1, 2).is_contiguous()


def replicate_pad(x: torch.Tensor, pad: tuple[int, ...]) -> torch.Tensor:
    """``F.pad(x, pad, mode="replicate")`` with identical values and strides."""
    pad = tuple(pad)
    if not _kernel_eligible(x, pad):
        return F.pad(x, pad, mode="replicate")
    pad_wl, pad_wr, pad_ht, pad_hb, pad_tf, pad_tb = pad
    n, c, t, h, w = x.shape
    out_t, out_h, out_w = t + pad_tf + pad_tb, h + pad_ht + pad_hb, w + pad_wl + pad_wr
    out_plane = out_h * out_w
    grid = (n * c * out_t, triton.cdiv(out_plane, _BLOCK))
    if out_plane >= 2**31 or grid[1] > _MAX_GRID_Y or grid[0] >= 2**31:
        return F.pad(x, pad, mode="replicate")
    out = torch.empty((n, c, out_t, out_h, out_w), dtype=x.dtype, device=x.device)
    with torch.get_device_module().device(x.device):
        _replicate_pad_kernel[grid](
            x,
            out,
            c,
            t,
            h,
            out_t,
            out_plane,
            pad_tf,
            pad_ht,
            pad_wl,
            *x.stride(),
            WIDTH=w,
            OUT_WIDTH=out_w,
            BLOCK=_BLOCK,
            num_warps=8,
        )
    return out


__all__ = ["replicate_pad"]
