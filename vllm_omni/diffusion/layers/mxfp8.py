# SPDX-License-Identifier: Apache-2.0
"""MXFP8 producers: e4m3 payload plus one E8M0 scale per 32 elements along K,
the scales in the cuBLASLt ``SWIZZLE_32_4_4`` layout that
``torch.nn.functional.scaled_mm(..., BlockWise1x32)`` consumes on Blackwell GPUs.

Scale ``(r, c)`` of the ``[rows, K/32]`` scale matrix lives at byte
``((r // 128) * ceil(K/32 / 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4``,
rows padded to 128 and scale columns to 4, padding zero. Exponent
``e = ceil(log2(amax / 448))`` exactly from the float bits; ``q = e4m3(x * 2**-e)``;
scale byte ``e + 127``. Every producer quantizes the bf16-rounded value the
unfused bf16 kernel stores, so each is byte-exact against that kernel followed
by ``mxfp8_quantize_swizzled`` (itself byte-exact vs ``flashinfer.mxfp8_quantize``).
"""

# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice

_E4M3 = torch.float8_e4m3fn


def _scale_numel(rows: int, k: int) -> int:
    n_groups = k // 32
    return -(-rows // 128) * 128 * (-(-n_groups // 4) * 4)


@triton.jit
def _mx_e8m0_from_amax(amax):
    bits = amax.to(tl.int32, bitcast=True)
    e0 = ((bits >> 23) & 0xFF) - 135
    # amax / 2**e0 lies in [256, 512); bump e0 when it exceeds 448 = 1.75 * 2**8
    the = (((e0 + 135) << 23) | 0x600000).to(tl.float32, bitcast=True)
    e = e0 + (amax > the).to(tl.int32)
    e = tl.maximum(e, -127)
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    return e + 127, inv


@triton.jit
def _mx_scale_offsets(r, c, n_col_blocks):
    tile = (r // 128) * n_col_blocks + (c // 4)
    return tile * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)


@triton.jit
def _mxfp8_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    rows,
    k,
    n_groups,
    n_col_blocks,
    stride_x,
    BLOCK_R: tl.constexpr,  # noqa: N803
    G: tl.constexpr,  # noqa: N803
):
    pid_r = tl.program_id(0)
    pid_g = tl.program_id(1)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    g = pid_g * G + tl.arange(0, G)
    c = pid_g * (G * 32) + tl.arange(0, G * 32)
    rmask = r < rows
    mask = rmask[:, None] & (c < k)[None, :]
    x = tl.load(x_ptr + r[:, None].to(tl.int64) * stride_x + c[None, :], mask=mask, other=0.0).to(tl.float32)
    x3 = tl.reshape(x, [BLOCK_R, G, 32])
    amax = tl.max(tl.abs(x3), axis=2)
    sbyte, inv = _mx_e8m0_from_amax(amax)
    q = tl.reshape(x3 * inv[:, :, None], [BLOCK_R, G * 32])
    tl.store(
        q_ptr + r[:, None].to(tl.int64) * k + c[None, :],
        q.to(tl.float8e4nv),
        mask=mask,
    )
    smask = rmask[:, None] & (g < n_groups)[None, :]
    tl.store(
        s_ptr + _mx_scale_offsets(r[:, None], g[None, :], n_col_blocks),
        sbyte.to(tl.uint8),
        mask=smask,
    )


def can_use_mxfp8_swizzled(x: torch.Tensor) -> bool:
    """Row-major floating CUDA 2D tensor with K % 32 == 0, outside torch.compile."""
    return (
        x.is_cuda
        and x.ndim == 2
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.stride(-1) == 1
        and x.shape[-1] % 32 == 0
        and not torch.compiler.is_compiling()
    )


def _alloc(rows: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty(rows, k, dtype=_E4M3, device=device)
    s = torch.zeros(_scale_numel(rows, k), dtype=torch.uint8, device=device)
    return q, s


def mxfp8_quantize_swizzled(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``[rows, k]`` -> ``(fp8 [rows, k], swizzled e8m0 scale bytes)``."""
    if not can_use_mxfp8_swizzled(x):
        raise ValueError("expected a row-major floating CUDA [rows, k] tensor, k % 32 == 0")
    rows, k = x.shape
    q, s = _alloc(rows, k, x.device)
    if rows == 0:
        return q, s
    n_groups = k // 32
    n_col_blocks = -(-n_groups // 4)
    block_r, g = 32, 8
    grid = (triton.cdiv(rows, block_r), triton.cdiv(n_groups, g))
    with torch.get_device_module().device(x.device):
        _mxfp8_quant_kernel[grid](
            x,
            q,
            s,
            rows,
            k,
            n_groups,
            n_col_blocks,
            x.stride(0),
            BLOCK_R=block_r,
            G=g,
            num_warps=4,
        )
    return q, s


@triton.jit
def _bf16_round(x):
    # Explicit conversion pair preserves the CUDA kernel's two BF16 roundings.
    # Unlike the integer rounding shortcut this also preserves NaN conversion.
    return tl.inline_asm_elementwise(
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cuda_silu_product(gate, up):
    # The installed vLLM SM120 SASS uses expf range reduction and corrected
    # division gate / (1 + expf(-gate)), then BF16 round, FMUL, BF16 round.
    # tl.sigmoid uses a different approximation: never assume byte identity.
    act = _bf16_round(libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate)))
    # Current generic CUDA act_and_mul keeps its (up + beta), beta=+0, add.
    # Preserve its signed-zero behavior too; an ordinary +0 can be optimized out.
    up_zero = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, 0f00000000;",
        constraints="=f,f",
        args=[up],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return _bf16_round(libdevice.mul_rn(act, up_zero))


@triton.jit
def _silu_mxfp8_cuda_kernel(
    X,  # noqa: N803
    Q,  # noqa: N803
    S,  # noqa: N803
    Y,  # noqa: N803
    rows,
    hidden,
    groups,
    col_blocks,
    stride_row,
    STORE_BF16: tl.constexpr,  # noqa: N803
    BLOCK_R: tl.constexpr,  # noqa: N803
    G: tl.constexpr,  # noqa: N803
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    g = tl.program_id(1) * G + tl.arange(0, G)
    c = tl.program_id(1) * (G * 32) + tl.arange(0, G * 32)
    mask = (r < rows)[:, None] & (c < hidden)[None, :]
    base = X + r[:, None].to(tl.int64) * stride_row
    gate = tl.load(base + c[None, :], mask, other=0).to(tl.float32)
    up = tl.load(base + hidden + c[None, :], mask, other=0).to(tl.float32)
    prod = _cuda_silu_product(gate, up)
    if STORE_BF16:
        tl.store(Y + r[:, None].to(tl.int64) * hidden + c[None, :], prod, mask)
    p3 = tl.reshape(prod, [BLOCK_R, G, 32])
    amax = tl.max(tl.abs(p3), axis=2)
    scale_byte, inv = _mx_e8m0_from_amax(amax)
    quant = tl.reshape(p3 * inv[:, :, None], [BLOCK_R, G * 32])
    tl.store(Q + r[:, None].to(tl.int64) * hidden + c[None, :], quant.to(tl.float8e4nv), mask)
    tl.store(
        S + _mx_scale_offsets(r[:, None], g[None, :], col_blocks),
        scale_byte.to(tl.uint8),
        (r < rows)[:, None] & (g < groups)[None, :],
    )


def silu_mul_mxfp8_cuda(x: torch.Tensor, *, return_bf16: bool = False):
    """BF16 [M, 2K] gate|up -> E4M3 [M,K], padded swizzled E8M0 bytes.

    return_bf16 additionally returns the rounded activation for validation.
    """
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.stride(1) != 1
        or x.shape[1] % 64
        or torch.compiler.is_compiling()
    ):
        raise ValueError("Expected resident row-major BF16 [M,2K], K multiple of 32, outside compile")
    rows, twice = x.shape
    hidden = twice // 2
    q, s = _alloc(rows, hidden, x.device)
    y = torch.empty((rows, hidden), dtype=torch.bfloat16, device=x.device) if return_bf16 else q
    if rows:
        groups = hidden // 32
        with torch.get_device_module().device(x.device):
            _silu_mxfp8_cuda_kernel[(triton.cdiv(rows, 16), triton.cdiv(groups, 8))](
                x,
                q,
                s,
                y,
                rows,
                hidden,
                groups,
                triton.cdiv(groups, 4),
                x.stride(0),
                STORE_BF16=return_bf16,
                BLOCK_R=16,
                G=8,
                num_warps=4,
                enable_fp_fusion=True,
                # vLLM's installed CUDA SASS uses non-FTZ exp/div arithmetic.
                # Triton's libdevice reflection otherwise defaults to FTZ.
                enable_reflect_ftz=False,
            )
    return (q, s, y) if return_bf16 else (q, s)


def mxfp8_scaled_mm(
    quantized: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    bias: torch.Tensor | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """Multiply E4M3 operands with swizzled E8M0 scales, one scale per 32 values.

    Weight is stored as [N,K]. Keep this import lazy: the block-scaled public
    PyTorch API is required only by callers explicitly selecting MXFP8.
    """
    from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

    return scaled_mm(
        quantized,
        weight.t(),
        scale_a=activation_scale.view(torch.float8_e8m0fnu),
        scale_b=weight_scale.view(torch.float8_e8m0fnu),
        scale_recipe_a=ScalingType.BlockWise1x32,
        scale_recipe_b=ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        bias=bias,
        output_dtype=output_dtype,
        use_fast_accum=use_fast_accum,
    )


@torch.library.custom_op("vllm_omni::mxfp8_linear", mutates_args=())
def mxfp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Opaque quantize/GEMM boundary for regional compilation."""
    q, activation_scale = mxfp8_quantize_swizzled(x.reshape(-1, x.shape[-1]).contiguous())
    out = mxfp8_scaled_mm(q, weight, activation_scale, scale)
    return out.reshape(*x.shape[:-1], weight.shape[0])


@mxfp8_linear.register_fake
def _mxfp8_linear_fake(x, weight, scale):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@torch.library.custom_op("vllm_omni::silu_mxfp8_linear", mutates_args=())
def silu_mxfp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fuse CUDA-compatible BF16 SwiGLU rounding and MXFP8 quantization."""
    q, activation_scale = silu_mul_mxfp8_cuda(x.reshape(-1, x.shape[-1]).contiguous())
    out = mxfp8_scaled_mm(q, weight, activation_scale, scale)
    return out.reshape(*x.shape[:-1], weight.shape[0])


@silu_mxfp8_linear.register_fake
def _silu_mxfp8_linear_fake(x, weight, scale):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))
