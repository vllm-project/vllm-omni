# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fused two-input SwiGLU for the Qwen-Image 2.1 MLP.

``QwenImage21SwiGLUFeedForward`` keeps its two independent column-parallel projections
(``gate_layer`` and ``proj``), so the activation sees two separate tensors. The eager
expression ``silu(gate) * up`` therefore costs two full-width elementwise kernels and one
BF16 intermediate the size of the MLP hidden state. This module fuses the pair into a
single launch that reads both projections and writes the product once.

The ABI deliberately stays two-input rather than adopting vLLM's packed ``SiluAndMul``:
packing would need a per-forward ``torch.cat([gate, up])`` that copies both tensors, which
costs more than the activation kernels it would replace.

Rounding contract, matching the eager expression::

    silu(gate)  -> round to BF16            # nn.SiLU() on a BF16 tensor
    result      -> BF16 mul with `up`       # elementwise multiply in BF16

``torch.nn.functional.silu`` on a BF16 tensor promotes to FP32, computes ``x / (1 + expf(-x))``
and rounds back, so the kernel divides rather than multiplying by a sigmoid.
"""

from __future__ import annotations

import torch
from torch.library import Library
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion.layers.numerics import mul_rn_f32, round_bf16_to_fp32, silu_f32
from vllm_omni.platforms import current_omni_platform

_BLOCK = 4096
_SUPPORTED_DTYPE = torch.bfloat16
_OP_NAME = "qwen_image_21_silu_mul"


@triton.jit
def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, elements, block: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    live = offsets < elements
    gate = tl.load(gate_ptr + offsets, mask=live, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=live, other=0.0).to(tl.float32)
    activated = round_bf16_to_fp32(silu_f32(gate))
    tl.store(out_ptr + offsets, mul_rn_f32(activated, up), mask=live)


def _launch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(gate)
    elements = out.numel()
    if elements:
        _silu_mul_kernel[(triton.cdiv(elements, _BLOCK),)](gate, up, out, elements, block=_BLOCK)
    return out


def _reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


def _supported(gate: torch.Tensor, up: torch.Tensor) -> bool:
    return (
        HAS_TRITON
        and current_omni_platform.is_cuda()
        and gate.is_cuda
        and gate.dtype is _SUPPORTED_DTYPE
        and up.dtype is _SUPPORTED_DTYPE
        and gate.shape == up.shape
        and gate.is_contiguous()
        and up.is_contiguous()
    )


def _fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _launch(gate, up)


def _fused_silu_mul_fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    del up
    return torch.empty_like(gate)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, _OP_NAME):
    direct_register_custom_op(
        op_name=_OP_NAME,
        op_func=_fused_silu_mul,
        fake_impl=_fused_silu_mul_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` for two independently projected tensors of equal shape."""
    if gate.shape != up.shape:
        raise ValueError(f"gate and up must match, got {tuple(gate.shape)} and {tuple(up.shape)}")
    if not _supported(gate, up):
        # CPU and non-contiguous inputs stay on plain torch, so the op never needs a
        # non-CUDA kernel registration.
        return _reference(gate, up)
    return torch.ops.vllm_omni.qwen_image_21_silu_mul(gate, up)


__all__ = ["fused_silu_mul"]
