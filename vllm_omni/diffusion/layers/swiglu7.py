# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Interleaved clamped SwiGLU7 with a native compile path.

The Triton expression is adapted from SGLang revision
25536524af6701518d0b8ec0efca8109c23b3706 (Apache-2.0).
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_omni.diffusion.layers.custom_op import CustomOp


@triton.jit
def _swiglu7_kernel(x_ptr, out_ptr, numel, block_size: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    mask = offsets < numel
    gate = tl.load(x_ptr + 2 * offsets, mask=mask, other=0).to(tl.float32)
    linear = tl.load(x_ptr + 2 * offsets + 1, mask=mask, other=0).to(tl.float32)
    # Comparisons preserve NaNs; gate is only upper-clamped, unlike linear.
    gate = tl.where(gate > 7.0, 7.0, gate)
    linear = tl.where(linear < -7.0, -7.0, linear)
    linear = tl.where(linear > 7.0, 7.0, linear)
    result = gate * tl.sigmoid(1.702 * gate) * (linear + 1.0)
    tl.store(out_ptr + offsets, result, mask=mask)


class SwiGLU7(CustomOp):
    """SwiGLU7 for packed ``[..., gate_0, up_0, gate_1, up_1, ...]`` inputs.

    Arithmetic is evaluated in FP32 and cast once to the requested output
    dtype. Eager inference with contiguous inputs and the released constants
    uses Triton; compilation, autograd and other layouts retain the native
    expression so this pointwise operation does not become a fusion barrier.
    """

    @staticmethod
    def forward_native(
        x: torch.Tensor,
        alpha: float = 1.702,
        limit: float = 7.0,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype if out_dtype is None else out_dtype
        x = x.to(torch.float32)
        gate, linear = x[..., ::2], x[..., 1::2]
        gate = gate.clamp(max=limit)
        linear = linear.clamp(min=-limit, max=limit)
        return (gate * torch.sigmoid(alpha * gate) * (linear + 1.0)).to(out_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        alpha: float = 1.702,
        limit: float = 7.0,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling() or (torch.is_grad_enabled() and x.requires_grad):
            return self.forward_native(x, alpha, limit, out_dtype)
        supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
        target_dtype = x.dtype if out_dtype is None else out_dtype
        if (
            x.device.type in ("cpu", "meta")
            or x.ndim == 0
            or x.shape[-1] % 2
            or not x.is_contiguous()
            or x.dtype not in supported_dtypes
            or target_dtype not in supported_dtypes
            or alpha != 1.702
            or limit != 7.0
        ):
            return self.forward_native(x, alpha, limit, out_dtype)
        output = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=target_dtype)
        if output.numel():
            _swiglu7_kernel[(triton.cdiv(output.numel(), 256),)](
                x, output, output.numel(), block_size=256, num_warps=4, enable_fp_fusion=False
            )
        return output

    # Platforms without a Triton implementation still have the native formula.
    forward_npu = forward_native
