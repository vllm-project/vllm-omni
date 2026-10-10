# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""BF16 pointwise fusion preserving eager intermediate rounding."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton


def eligible(*values: torch.Tensor) -> bool:
    first = values[0]
    return (
        HAS_TRITON
        and torch.version.hip is None
        and first.device.type == "cuda"
        and first.dtype == torch.bfloat16
        and first.numel() > 0
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and all(
            t.shape == first.shape and t.is_contiguous() and t.dtype == first.dtype and t.device == first.device
            for t in values
        )
    )


if HAS_TRITON:

    @triton.jit
    def _residual_kernel(
        input_ptr,
        update_ptr,
        gate_ptr,
        output_ptr,
        elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        index = tl.program_id(0) * block_size + tl.arange(0, block_size)
        x = tl.load(input_ptr + index, index < elements, 0).to(tl.float32)
        y = tl.load(update_ptr + index, index < elements, 0).to(tl.float32)
        gate = tl.load(gate_ptr + index, index < elements, 0).to(tl.float32)
        product = (gate * y).to(tl.bfloat16).to(tl.float32)
        tl.store(output_ptr + index, (x + product).to(tl.bfloat16), index < elements)


if HAS_TRITON:

    @triton.jit
    def _silu_mul_kernel(gate_ptr, up_ptr, output_ptr, elements: tl.constexpr, block_size: tl.constexpr):
        index = tl.program_id(0) * block_size + tl.arange(0, block_size)
        gate = tl.load(gate_ptr + index, index < elements, 0).to(tl.float32)
        up = tl.load(up_ptr + index, index < elements, 0).to(tl.float32)
        activated = tl.div_rn(gate, 1.0 + tldevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
        tl.store(output_ptr + index, (activated * up).to(tl.bfloat16), index < elements)


def residual(hidden: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if not eligible(hidden, update, gate):
        return hidden + gate * update
    output = torch.empty_like(hidden)
    _residual_kernel[(triton.cdiv(hidden.numel(), 1024),)](
        hidden, update, gate, output, hidden.numel(), 1024, enable_fp_fusion=False
    )
    return output


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if not eligible(gate, up):
        return torch.nn.functional.silu(gate) * up
    output = torch.empty_like(gate)
    _silu_mul_kernel[(triton.cdiv(gate.numel(), 1024),)](gate, up, output, gate.numel(), 1024, enable_fp_fusion=False)
    return output


if HAS_TRITON:

    @triton.jit
    def _rope_kernel(
        input_ptr,
        freq_ptr,
        output_ptr,
        elements: tl.constexpr,
        heads: tl.constexpr,
        sequence: tl.constexpr,
        x_stride0: tl.constexpr,
        x_stride1: tl.constexpr,
        x_stride2: tl.constexpr,
        freq_stride0: tl.constexpr,
        freq_stride1: tl.constexpr,
        block_size: tl.constexpr,
    ):
        index = tl.program_id(0) * block_size + tl.arange(0, block_size)
        pair = index % 64
        head = (index // 64) % heads
        token = (index // (64 * heads)) % sequence
        batch = index // (64 * heads * sequence)
        base = batch * x_stride0 + token * x_stride1 + head * x_stride2 + pair * 2
        real = tl.load(input_ptr + base, index < elements, 0).to(tl.float32)
        imag = tl.load(input_ptr + base + 1, index < elements, 0).to(tl.float32)
        cosine = tl.load(freq_ptr + token * freq_stride0 + pair * freq_stride1, index < elements, 0)
        sine = tl.load(freq_ptr + token * freq_stride0 + pair * freq_stride1 + 1, index < elements, 0)
        # Match CUDA's complex64 multiply: the real leading product and the
        # imaginary trailing product use FMA. The other products round first.
        rotated_real = tl.fma(real, cosine, -imag * sine)
        rotated_imag = tl.fma(imag, cosine, real * sine)
        tl.store(output_ptr + index * 2, rotated_real.to(tl.bfloat16), index < elements)
        tl.store(output_ptr + index * 2 + 1, rotated_imag.to(tl.bfloat16), index < elements)


def rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor | None:
    if (
        not HAS_TRITON
        or torch.version.hip is not None
        or x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.ndim != 4
        or x.shape[-1] != 128
        or x.stride(-1) != 1
        or freqs.dtype != torch.complex64
        or freqs.device != x.device
        or freqs.shape != (x.shape[1], 64)
        or torch.is_grad_enabled()
        or x.numel() == 0
        or torch.compiler.is_compiling()
    ):
        return None
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    real_freqs = torch.view_as_real(freqs)
    _rope_kernel[(triton.cdiv(x.numel() // 2, 256),)](
        x,
        real_freqs,
        output,
        x.numel() // 2,
        x.shape[2],
        x.shape[1],
        *x.stride()[:3],
        *real_freqs.stride()[:2],
        256,
        enable_fp_fusion=False,
    )
    return output
