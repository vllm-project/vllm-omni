# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Numerical primitives shared by exact Triton kernels.

Eager PyTorch dispatches each elementwise step as its own kernel, so it never contracts a
multiply and an add into an FMA and it rounds every op to the declared dtype before the
next one. Triton is free to do both. The wrappers here pin a single IEEE operation per call
so a fused kernel can reproduce the eager rounding points exactly.

The transcendental wrappers are chosen to match ATen's CUDA kernels bit for bit, which is
not the same thing as matching ``tl.exp``/``tl.tanh``: the Triton builtins lower to the
``ex2.approx``/``tanh.approx`` hardware instructions, while ATen calls ``expf``/``tanhf``
from libdevice. Measured on SM120, ``tl.tanh`` differs from ``torch.tanh`` by up to 133 ULP.
"""

from vllm.triton_utils import tl, triton

_libdevice = tl.extra.cuda.libdevice


@triton.jit
def round_bf16_to_fp32(value):
    """RNE-round FP32 to BF16 precision while retaining an FP32 register."""

    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def add_rn_f32(x, y):
    return tl.inline_asm_elementwise(
        asm="add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sub_rn_f32(x, y):
    return tl.inline_asm_elementwise(
        asm="sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def mul_rn_f32(x, y):
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def fma_rn_f32(x, y, accumulator):
    return tl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[x, y, accumulator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rn_f32(x, y):
    return tl.inline_asm_elementwise(
        asm="div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rsqrt_approx_f32(x):
    """`rsqrt.approx.f32`, which is exactly what `torch.rsqrt` lowers to on CUDA."""

    return tl.inline_asm_elementwise(
        asm="rsqrt.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def exp_f32(x):
    """Libdevice `expf`, matching `torch.exp` rather than Triton's `ex2.approx` lowering."""

    return _libdevice.exp(x)


@triton.jit
def tanh_f32(x):
    """Libdevice `tanhf`, matching `torch.tanh` (up to 133 ULP away from `tanh.approx`)."""

    return _libdevice.tanh(x)


@triton.jit
def shfl_down_f32(value, delta: tl.constexpr):
    """Warp shuffle-down, used to reproduce ATen's warp-tree reduction order."""

    return tl.inline_asm_elementwise(
        asm="shfl.sync.down.b32 $0, $1, $2, 0x1f, 0xffffffff;",
        constraints="=f,f,n",
        args=[value, delta],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sigmoid_f32(x):
    """`1 / (1 + expf(-x))` with the division pinned; matches `torch.sigmoid` exactly."""

    return div_rn_f32(1.0, add_rn_f32(1.0, exp_f32(-x)))


@triton.jit
def silu_f32(x):
    """`x / (1 + expf(-x))`; ATen divides instead of multiplying by the sigmoid."""

    return div_rn_f32(x, add_rn_f32(1.0, exp_f32(-x)))


__all__ = [
    "add_rn_f32",
    "div_rn_f32",
    "exp_f32",
    "fma_rn_f32",
    "mul_rn_f32",
    "round_bf16_to_fp32",
    "rsqrt_approx_f32",
    "shfl_down_f32",
    "sigmoid_f32",
    "silu_f32",
    "sub_rn_f32",
    "tanh_f32",
]
