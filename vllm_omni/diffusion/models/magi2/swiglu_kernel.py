# SPDX-License-Identifier: Apache-2.0

"""Fused BF16 SwiGLU7 kernel for MAGI-2 MUSA execution.

The elementwise kernel is adapted from SGLang revision
``25536524af6701518d0b8ec0efca8109c23b3706``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _swiglu7_kernel(
    x_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    gate = tl.load(x_ptr + offsets * 2, mask=mask, other=0.0).to(tl.float32)
    linear = tl.load(x_ptr + offsets * 2 + 1, mask=mask, other=0.0).to(
        tl.float32
    )
    # Preserve NaNs exactly like the reference ``torch.minimum``/clamp path;
    # plain Triton minimum/max lowering is allowed to select the finite bound
    # for a NaN on some MUSA compiler revisions.
    gate = tl.where(gate != gate, gate, tl.minimum(gate, 7.0))
    linear = tl.where(
        linear != linear,
        linear,
        tl.maximum(tl.minimum(linear, 7.0), -7.0),
    )
    activated = gate * tl.sigmoid(1.702 * gate) * (linear + 1.0)
    tl.store(out_ptr + offsets, activated, mask=mask)


def _swiglu7(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError("SwiGLU7 expects an even final dimension")
    if not x.is_contiguous():
        raise ValueError("fused SwiGLU7 requires contiguous input")
    out = torch.empty(
        (*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype
    )
    numel = out.numel()
    if numel == 0:
        return out
    _swiglu7_kernel[(triton.cdiv(numel, 256),)](
        x,
        out,
        numel,
        BLOCK=256,
        num_warps=4,
    )
    return out


def _swiglu7_fake(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError(f"expected an even last dimension, got {x.shape[-1]}")
    return torch.empty(
        (*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype
    )


direct_register_custom_op(
    "magi2_swiglu7",
    _swiglu7,
    mutates_args=[],
    fake_impl=_swiglu7_fake,
)
magi2_swiglu7 = torch.ops.vllm.magi2_swiglu7


__all__ = ["magi2_swiglu7"]
