# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen-Image's attention residual followed by non-affine AdaLayerNorm.

Both the updated residual and the MLP input are live outputs. Two pointwise
kernels fuse the residual/FP32 cast and the norm-output cast/modulation around
PyTorch's native FP32 LayerNorm. Keeping the native reduction matters: even a
single FP32 ULP can change BF16 rounding and accumulate across denoising steps.
Every low precision operation retains its eager rounding boundary.
"""

import os
from functools import cache

import torch
import torch.nn.functional as F
from torch.library import Library
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


def _native_layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=eps)


def _native_layer_norm_fake(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


# A compile boundary prevents Inductor from replacing the native reduction with
# a different summation order. The pointwise Triton launches remain traceable.
_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "gated_residual_native_layer_norm"):
    direct_register_custom_op(
        op_name="gated_residual_native_layer_norm",
        op_func=_native_layer_norm,
        fake_impl=_native_layer_norm_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


if HAS_TRITON:
    # Explicit PTX keeps both rounding and subnormals. CUDA libdevice rn
    # intrinsics inherit Triton's FTZ mode, including when traced by Inductor.
    @triton.jit
    def _add_rn(x, y):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn(x, y):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _gated_residual_cast_kernel(
        x_ptr,
        branch_ptr,
        gate_ptr,
        residual_out_ptr,
        norm_input_ptr,
        seq_len: tl.constexpr,
        hidden_size: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_s: tl.constexpr,
        branch_stride_b: tl.constexpr,
        branch_stride_s: tl.constexpr,
        gate_stride_b: tl.constexpr,
        block_size: tl.constexpr,
        cast_output: tl.constexpr,
        tiles_per_row: tl.constexpr = 1,
        program_offset: tl.constexpr = 0,
    ):
        pid = tl.program_id(0).to(tl.int64) - program_offset
        row = pid // tiles_per_row
        batch, token = row // seq_len, row % seq_len
        d = (pid % tiles_per_row) * block_size + tl.arange(0, block_size)
        mask = d < hidden_size
        dtype = x_ptr.dtype.element_ty
        x = tl.load(x_ptr + batch * x_stride_b + token * x_stride_s + d, mask, other=0).to(tl.float32)
        branch = tl.load(branch_ptr + batch * branch_stride_b + token * branch_stride_s + d, mask, other=0).to(
            tl.float32
        )
        gate = tl.load(gate_ptr + batch * gate_stride_b + d, mask, other=0).to(tl.float32)
        # Keep separate IEEE operations even when Inductor rebuilds launch
        # options and drops enable_fp_fusion=False.
        product = _mul_rn(gate, branch).to(dtype).to(tl.float32)
        residual = _add_rn(x, product).to(dtype)
        tl.store(residual_out_ptr + row * hidden_size + d, residual, mask)
        if cast_output:
            tl.store(norm_input_ptr + row * hidden_size + d, residual.to(tl.float32), mask)

    @triton.jit
    def _cast_modulate_kernel(
        normalized_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        hidden_size: tl.constexpr,
        scale_stride_b: tl.constexpr,
        shift_stride_b: tl.constexpr,
        block_size: tl.constexpr,
        tiles_per_row: tl.constexpr = 1,
        program_offset: tl.constexpr = 0,
    ):
        pid = tl.program_id(0).to(tl.int64) - program_offset
        row = pid // tiles_per_row
        batch = row // seq_len
        d = (pid % tiles_per_row) * block_size + tl.arange(0, block_size)
        mask = d < hidden_size
        dtype = out_ptr.dtype.element_ty
        normalized = tl.load(normalized_ptr + row * hidden_size + d, mask, other=0).to(dtype).to(tl.float32)
        scale = tl.load(scale_ptr + batch * scale_stride_b + d, mask, other=0).to(tl.float32)
        shift = tl.load(shift_ptr + batch * shift_stride_b + d, mask, other=0).to(tl.float32)
        factor = _add_rn(tl.full((), 1.0, tl.float32), scale).to(dtype).to(tl.float32)
        modulated = _mul_rn(normalized, factor).to(dtype).to(tl.float32)
        y = _add_rn(modulated, shift).to(dtype)
        tl.store(out_ptr + row * hidden_size + d, y, mask)


def _v2_enabled() -> bool:
    # "norm2" isolates launch tiling from the additional caller sites.
    # Neither mode is a measured default until the GPU acceptance matrix.
    return os.environ.get("VLLM_OMNI_QWEN_ADALN_V2", "0") in ("1", "norm2")


def _v2_sites_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_QWEN_ADALN_V2", "0") == "1"


def _pointwise_config(b: int, s: int, d: int) -> tuple[int, int, int]:
    # Pointwise work has no cross-column reduction. Splitting short text rows
    # gives more CTAs than the former one-CTA-per-row / 8-warp launch.
    # Values are candidates, not a hardware-independent speedup assertion.
    block = 256 if b * s < 64 else 1024
    return block, 4, (d + block - 1) // block


def _supports_pointwise(x: torch.Tensor, branch: torch.Tensor | None, modulation: tuple[torch.Tensor, ...]) -> bool:
    if (
        not HAS_TRITON
        or not current_platform.is_cuda()
        or not x.is_cuda
        or x.ndim != 3
        or x.dtype not in (torch.float32, torch.bfloat16)
        or x.numel() == 0
        or not 0 < x.shape[-1] <= 8192
    ):
        return False
    tensors = (x, *modulation) if branch is None else (x, branch, *modulation)
    if any(t.device != x.device or t.dtype != x.dtype or t.layout != torch.strided for t in tensors):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return False
    if x.stride(-1) != 1:
        return False
    if branch is not None and (branch.shape != x.shape or branch.stride(-1) != 1):
        return False
    b, _, d = x.shape
    return all(t.shape == (b, 1, d) and t.stride(-1) == 1 for t in modulation)


def _native_norm_boundary(x: torch.Tensor, eps: float) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm_omni.gated_residual_native_layer_norm(x, eps)
    return _native_layer_norm(x, eps)


def try_fused_native_adaln(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor | None:
    """Experimental non-affine norm1: native FP32 norm, fused cast/modulate."""
    if not _v2_sites_enabled() or not _supports_pointwise(x, None, (scale, shift)):
        return None
    b, s, d = x.shape
    normalized = _native_norm_boundary(x.float(), eps)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    block, warps, tiles = _pointwise_config(b, s, d)
    _cast_modulate_kernel[(b * s * tiles,)](
        normalized,
        scale,
        shift,
        out,
        s,
        d,
        scale.stride(0),
        shift.stride(0),
        block,
        tiles,
        program_offset=0,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out


def try_fused_gated_residual(x: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor | None:
    """Experimental final MLP residual; uses the same eager-rounding kernel."""
    if not _v2_sites_enabled() or not _supports_pointwise(x, branch, (gate,)):
        return None
    b, s, d = x.shape
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    block, warps, tiles = _pointwise_config(b, s, d)
    _gated_residual_cast_kernel[(b * s * tiles,)](
        x,
        branch,
        gate,
        out,
        out,
        s,
        d,
        x.stride(0),
        x.stride(1),
        branch.stride(0),
        branch.stride(1),
        gate.stride(0),
        block,
        False,
        tiles,
        program_offset=0,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out


@cache
def _is_benchmarked_device(device_index: int) -> bool:
    return current_platform.get_device_name(device_index) == "NVIDIA A100-SXM4-40GB"


def _has_measured_speedup(residual: torch.Tensor, branch: torch.Tensor) -> bool:
    # The local probe used sliced attention output; the E2E run used contiguous
    # output at the same three sequence lengths.
    # Keep other layouts and compiled execution on the caller's original layers.
    if os.environ.get("VLLM_OMNI_QWEN_ADALN_V2") == "off" or torch.compiler.is_compiling():
        return False
    b, s, d = residual.shape
    return (
        residual.dtype == torch.bfloat16
        and b == 1
        and s in (12, 29, 4096)
        and d == 3072
        and residual.is_contiguous()
        and branch.stride(1) in (d, 2 * d)
        and _is_benchmarked_device(residual.device.index)
    )


def try_fused_gated_residual_adaln(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return two new outputs, or None so the caller uses its original layers.

    By default enables measured A100 BF16 [1,S,3072] layouts at S=12/29/4096.
    The explicit local v2 switch selects broader layouts and tiled launches
    for evaluation; otherwise unmeasured layouts/compilation use the caller.
    Modulation is per-batch [B,1,D] (including chunk/unsqueeze batch gaps).
    Autograd retains the caller's original expression. Two pointwise Triton
    kernels surround native FP32 LayerNorm; torch.compile preserves that
    reduction through a functional custom op. There is no single-kernel
    reduction fast path or error-catching fallback after selection.
    """
    if not _supports_pointwise(residual, branch, (gate, scale, shift)):
        return None
    if not _v2_enabled() and not _has_measured_speedup(residual, branch):
        return None
    b, s, d = residual.shape
    r = torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)
    norm_input = torch.empty_like(r, dtype=torch.float32) if residual.dtype != torch.float32 else r
    y = torch.empty_like(r)
    block_size = triton.next_power_of_2(d)
    num_warps = 4 if d <= 2048 else 8
    tiles = 1
    if _v2_enabled():
        block_size, num_warps, tiles = _pointwise_config(b, s, d)
    _gated_residual_cast_kernel[(b * s * tiles,)](
        residual,
        branch,
        gate,
        r,
        norm_input,
        s,
        d,
        residual.stride(0),
        residual.stride(1),
        branch.stride(0),
        branch.stride(1),
        gate.stride(0),
        block_size,
        residual.dtype != torch.float32,
        tiles,
        program_offset=0,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    normalized = _native_norm_boundary(norm_input, eps)
    _cast_modulate_kernel[(b * s * tiles,)](
        normalized,
        scale,
        shift,
        y,
        s,
        d,
        scale.stride(0),
        shift.stride(0),
        block_size,
        tiles,
        program_offset=0,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return r, y
