# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Triton kernels for the AuK codec's alias-free SnakeBeta activation.

The eager module oversamples with a grouped transposed convolution, applies
SnakeBeta, and decimates with a grouped strided convolution. Both grouped
convolutions carry one 12-tap filter shared by every channel, and cuDNN and
ATen serve them with generic depthwise kernels that dominate the decode (more
than half of a 12 s decode on H200). Here the oversampling is written as its
polyphase form fused with the activation, and the decimation as a direct
strided FIR, each one pass over memory per row. The arithmetic is the same
as the eager module in fp32; only the summation order of the taps differs.

The ops are registered with ``torch.library.triton_op`` so the compiled
decode buckets trace them instead of breaking the graph.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

__all__ = ["alias_free_snake"]

_BLOCK = 1024

if HAS_TRITON:

    @triton.jit
    def _upsample_snake_kernel(
        x_ptr,
        out_ptr,
        taps_ptr,
        exp_alpha_ptr,
        inv_beta_ptr,
        channels,
        in_len,
        out_len,
        ratio: tl.constexpr,
        taps: tl.constexpr,
        pad: tl.constexpr,
        crop: tl.constexpr,
        block: tl.constexpr,
    ):
        # out[m] = snake(ratio * sum_k taps[k] * xpad[(m + crop - k) / ratio]) over the k
        # with (m + crop - k) divisible by ratio, where xpad is x replicate-padded by pad
        # on both sides. That is conv_transpose1d(stride=ratio) followed by the crop.
        row = tl.program_id(0)
        offs = tl.program_id(1) * block + tl.arange(0, block)
        mask = offs < out_len
        x_row = x_ptr + row.to(tl.int64) * in_len
        acc = tl.zeros((block,), dtype=tl.float32)
        for k in tl.static_range(taps):
            pos = offs + (crop - k)
            hit = (pos % ratio) == 0
            src = pos // ratio - pad
            src = tl.minimum(tl.maximum(src, 0), in_len - 1)
            w = tl.load(taps_ptr + k)
            val = tl.load(x_row + src, mask=mask & hit, other=0.0)
            acc += tl.where(hit, w * val, 0.0)
        u = acc * ratio
        channel = row % channels
        alpha = tl.load(exp_alpha_ptr + channel)
        inv_beta = tl.load(inv_beta_ptr + channel)
        s = tl.sin(u * alpha)
        u = u + inv_beta * (s * s)
        tl.store(out_ptr + row.to(tl.int64) * out_len + offs, u, mask=mask)

    @triton.jit
    def _lowpass_kernel(
        x_ptr,
        out_ptr,
        taps_ptr,
        in_len,
        out_len,
        stride: tl.constexpr,
        taps: tl.constexpr,
        pad_left: tl.constexpr,
        block: tl.constexpr,
    ):
        # out[n] = sum_k taps[k] * x[clamp(stride * n + k - pad_left)], i.e. replicate
        # padding followed by a strided conv1d.
        row = tl.program_id(0)
        offs = tl.program_id(1) * block + tl.arange(0, block)
        mask = offs < out_len
        x_row = x_ptr + row.to(tl.int64) * in_len
        acc = tl.zeros((block,), dtype=tl.float32)
        for k in tl.static_range(taps):
            src = offs * stride + (k - pad_left)
            src = tl.minimum(tl.maximum(src, 0), in_len - 1)
            acc += tl.load(taps_ptr + k) * tl.load(x_row + src, mask=mask, other=0.0)
        tl.store(out_ptr + row.to(tl.int64) * out_len + offs, acc, mask=mask)

    @torch.library.triton_op("vllm_omni::auk_alias_free_snake", mutates_args=())
    def alias_free_snake(
        x: torch.Tensor,
        up_taps: torch.Tensor,
        down_taps: torch.Tensor,
        exp_alpha: torch.Tensor,
        inv_beta: torch.Tensor,
        up_ratio: int,
        up_pad: int,
        up_crop: int,
        down_stride: int,
        down_pad_left: int,
        down_pad_right: int,
    ) -> torch.Tensor:
        """Upsample by ``up_ratio``, SnakeBeta, low-pass and decimate: ``[B, C, T] -> [B, C, T_out]``."""
        batch, channels, length = x.shape
        x = x.contiguous()
        up_len = length * up_ratio
        oversampled = torch.empty((batch, channels, up_len), device=x.device, dtype=torch.float32)
        rows = batch * channels
        torch.library.wrap_triton(_upsample_snake_kernel)[(rows, triton.cdiv(up_len, _BLOCK))](
            x,
            oversampled,
            up_taps,
            exp_alpha,
            inv_beta,
            channels,
            length,
            up_len,
            ratio=up_ratio,
            taps=up_taps.numel(),
            pad=up_pad,
            crop=up_crop,
            block=_BLOCK,
        )
        taps = down_taps.numel()
        out_len = (up_len + down_pad_left + down_pad_right - taps) // down_stride + 1
        out = torch.empty((batch, channels, out_len), device=x.device, dtype=torch.float32)
        torch.library.wrap_triton(_lowpass_kernel)[(rows, triton.cdiv(out_len, _BLOCK))](
            oversampled,
            out,
            down_taps,
            up_len,
            out_len,
            stride=down_stride,
            taps=taps,
            pad_left=down_pad_left,
            block=_BLOCK,
        )
        return out

else:  # pragma: no cover

    def alias_free_snake(*args, **kwargs):
        raise RuntimeError("Triton is not available")
