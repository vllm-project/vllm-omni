# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Numerical primitives shared by exact LTX-2 Triton kernels."""

from vllm.triton_utils import tl, triton


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
def rsqrt_approx_f32(x):
    return tl.inline_asm_elementwise(
        asm="rsqrt.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def shfl_down_f32(value, delta: tl.constexpr):
    return tl.inline_asm_elementwise(
        asm="shfl.sync.down.b32 $0, $1, $2, 0x1f, 0xffffffff;",
        constraints="=f,f,n",
        args=[value, delta],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def four_warp_sum(values):
    """Match the four-warp reduction tree used by PyTorch RMSNorm."""

    values = add_rn_f32(values, shfl_down_f32(values, 16))
    values = add_rn_f32(values, shfl_down_f32(values, 8))
    values = add_rn_f32(values, shfl_down_f32(values, 4))
    values = add_rn_f32(values, shfl_down_f32(values, 2))
    values = add_rn_f32(values, shfl_down_f32(values, 1))
    threads = tl.arange(0, 128)
    warp_0 = tl.sum(tl.where(threads == 0, values, 0.0))
    warp_1 = tl.sum(tl.where(threads == 32, values, 0.0))
    warp_2 = tl.sum(tl.where(threads == 64, values, 0.0))
    warp_3 = tl.sum(tl.where(threads == 96, values, 0.0))
    return add_rn_f32(
        add_rn_f32(warp_0, warp_2),
        add_rn_f32(warp_1, warp_3),
    )


@triton.jit
def rms_reciprocal_fma(
    input_ptr,
    row_base,
    eps,
    hidden_size: tl.constexpr,
):
    """Four-warp vec4 RMS reduction with the CUDA/PyTorch accumulation tree."""

    threads = tl.arange(0, 128)
    accumulator = tl.zeros((128,), dtype=tl.float32)
    for vector_block in tl.static_range(hidden_size // (128 * 4)):
        vector = threads + vector_block * 128
        base = row_base + vector * 4
        for lane in tl.static_range(4):
            value = tl.load(input_ptr + base + lane).to(tl.float32)
            accumulator = fma_rn_f32(value, value, accumulator)
    total = four_warp_sum(accumulator)
    return rsqrt_approx_f32(total / hidden_size + eps)


__all__ = [
    "add_rn_f32",
    "fma_rn_f32",
    "four_warp_sum",
    "mul_rn_f32",
    "rms_reciprocal_fma",
    "round_bf16_to_fp32",
    "rsqrt_approx_f32",
]
