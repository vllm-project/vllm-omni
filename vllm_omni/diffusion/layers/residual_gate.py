# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exact fused gated residual updates for diffusion models.

SANA-Video repeatedly evaluates ``residual + gate * update`` on very large
BF16 tensors.  Its first residual is a transposed-dense ``[B, S, D]`` view
with stride ``(S * D, 1, S)``, while the attention update is contiguous.  A
naive eager expression launches separate multiply and add kernels and makes
one side of the add use uncoalesced memory accesses.

The mixed-layout Triton kernel below transposes a 32x32 update tile in
registers, so both the contiguous update read and the transposed
residual/output traffic are coalesced.  A simpler linear kernel handles the
fully contiguous residual site.  Both kernels preserve eager's low-precision
materialization boundary between multiply and add, making FP16/BF16 outputs
bit exact.  Unsupported inputs and compiled regions retain the eager formula.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_TRANSPOSE_TILE = 32
_CONTIGUOUS_BLOCK_SIZE = 1024
_MAX_GRID_DIM = 65535
_MAX_INT32_INDEX = 2**31 - 1
_FAILED_RUNTIME_KEYS: set[tuple[int | None, torch.dtype]] = set()


@triton.jit
def _round_bf16_to_fp32(value):
    """RNE-round FP32 to BF16 precision while retaining an FP32 register."""
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    rounded = rounded_bits.to(tl.float32, bitcast=True)
    # Integer rounding is only defined for finite values.  Preserve Inf/NaN
    # through the product/add chain and let the final BF16 store perform the
    # same canonicalization as eager TensorIterator.
    is_non_finite = (bits & 0x7F800000) == 0x7F800000
    return tl.where(is_non_finite, value, rounded)


@triton.jit
def _round_fp16_to_fp32(value):
    """RNE-round FP32 to FP16 without allowing the cast to be folded away."""
    return tl.inline_asm_elementwise(
        asm="{ .reg .b16 h; cvt.rn.f16.f32 h, $1; cvt.f32.f16 $0, h; }",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _round_product_to_input_dtype(product, input_is_bf16: tl.constexpr):
    if input_is_bf16:
        return _round_bf16_to_fp32(product)
    return _round_fp16_to_fp32(product)


@triton.jit
def _residual_gate_add_contiguous_kernel(
    output_ptr,
    residual_ptr,
    update_ptr,
    gate_ptr,
    numel,
    batch_stride,
    tokens,
    gate_batch_stride,
    gate_token_stride,
    hidden_size: tl.constexpr,
    gate_is_batched: tl.constexpr,
    gate_is_tokenwise: tl.constexpr,
    input_is_bf16: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < numel
    hidden = offsets % hidden_size
    batch = offsets // batch_stride
    gate_batch = batch if gate_is_batched else 0
    if gate_is_tokenwise:
        token = (offsets // hidden_size) % tokens
    else:
        token = 0

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate_offsets = gate_batch * gate_batch_stride + token * gate_token_stride + hidden
    gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)

    # Eager materializes ``gate * update`` in the input dtype before the add.
    # Preserve that rounding point and prevent multiply-add contraction.
    product = _round_product_to_input_dtype(gate * update, input_is_bf16)
    tl.store(output_ptr + offsets, residual + product, mask=mask)


@triton.jit
def _residual_gate_add_transposed_kernel(
    output_ptr,
    residual_ptr,
    update_ptr,
    gate_ptr,
    tokens,
    hidden_size,
    gate_batch_stride,
    gate_token_stride,
    gate_is_batched: tl.constexpr,
    gate_is_tokenwise: tl.constexpr,
    input_is_bf16: tl.constexpr,
    block_tokens: tl.constexpr,
    block_hidden: tl.constexpr,
):
    token_base = tl.program_id(0) * block_tokens
    hidden_base = tl.program_id(1) * block_hidden
    batch = tl.program_id(2)
    token_offsets = token_base + tl.arange(0, block_tokens)
    hidden_offsets = hidden_base + tl.arange(0, block_hidden)
    batch_offset = batch * tokens * hidden_size

    # Read the update along its contiguous hidden dimension, then transpose the
    # register tile to line up with coalesced residual/output token traffic.
    update_offsets = batch_offset + token_offsets[:, None] * hidden_size + hidden_offsets[None, :]
    update_mask = (token_offsets[:, None] < tokens) & (hidden_offsets[None, :] < hidden_size)
    update = tl.load(update_ptr + update_offsets, mask=update_mask, other=0.0).to(tl.float32)
    update = tl.trans(update)

    transposed_offsets = batch_offset + hidden_offsets[:, None] * tokens + token_offsets[None, :]
    transposed_mask = (hidden_offsets[:, None] < hidden_size) & (token_offsets[None, :] < tokens)
    residual = tl.load(residual_ptr + transposed_offsets, mask=transposed_mask, other=0.0).to(tl.float32)
    gate_batch = batch if gate_is_batched else 0
    gate_batch_offset = gate_batch * gate_batch_stride
    if gate_is_tokenwise:
        gate_offsets = gate_batch_offset + token_offsets[:, None] * gate_token_stride + hidden_offsets[None, :]
        gate = tl.load(gate_ptr + gate_offsets, mask=update_mask, other=0.0).to(tl.float32)
        gate = tl.trans(gate)
        product = _round_product_to_input_dtype(gate * update, input_is_bf16)
    else:
        gate = tl.load(
            gate_ptr + gate_batch_offset + hidden_offsets,
            mask=hidden_offsets < hidden_size,
            other=0.0,
        ).to(tl.float32)
        product = _round_product_to_input_dtype(gate[:, None] * update, input_is_bf16)
    tl.store(output_ptr + transposed_offsets, residual + product, mask=transposed_mask)


def _has_supported_gate_shape(residual: torch.Tensor, gate: torch.Tensor) -> bool:
    batch, tokens, hidden_size = residual.shape
    return gate.shape[0] in (1, batch) and gate.shape[1] in (1, tokens) and gate.shape[2] == hidden_size


def _is_transposed_dense(residual: torch.Tensor) -> bool:
    batch, tokens, hidden_size = residual.shape
    del batch
    return (
        residual.stride() == (tokens * hidden_size, 1, tokens)
        and residual.shape[0] <= _MAX_GRID_DIM
        and triton.cdiv(tokens, _TRANSPOSE_TILE) <= _MAX_GRID_DIM
        and triton.cdiv(hidden_size, _TRANSPOSE_TILE) <= _MAX_GRID_DIM
    )


def _fits_int32_indexing(residual: torch.Tensor, gate: torch.Tensor) -> bool:
    gate_strides = gate.stride()
    if any(stride < 0 for stride in gate_strides):
        return False
    max_gate_offset = sum((size - 1) * stride for size, stride in zip(gate.shape, gate_strides, strict=True))
    return residual.numel() - 1 <= _MAX_INT32_INDEX and max_gate_offset <= _MAX_INT32_INDEX


def _can_use_fused_residual_gate_add(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> bool:
    """Return whether the lossless CUDA fast path supports these tensors."""
    if residual.ndim != 3 or update.ndim != 3 or gate.ndim != 3:
        return False
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and residual.is_cuda
        and update.is_cuda
        and gate.is_cuda
        and residual.dtype in _SUPPORTED_DTYPES
        and update.dtype == residual.dtype
        and gate.dtype == residual.dtype
        and not residual.requires_grad
        and not update.requires_grad
        and not gate.requires_grad
        and update.device == residual.device
        and gate.device == residual.device
        and residual.shape == update.shape
        and residual.numel() > 0
        and update.is_contiguous()
        # adaLN-single returns views from ``unbind``.  Their hidden dimension
        # is dense, but rows are separated by the six modulation vectors.
        and gate.stride(2) == 1
        and _has_supported_gate_shape(residual, gate)
        and _fits_int32_indexing(residual, gate)
        and (residual.is_contiguous() or _is_transposed_dense(residual))
    )


def _launch_fused_residual_gate_add(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    batch, tokens, hidden_size = residual.shape
    output = torch.empty_strided(
        residual.shape,
        residual.stride(),
        dtype=residual.dtype,
        device=residual.device,
    )
    gate_is_batched = gate.shape[0] == batch
    gate_is_tokenwise = gate.shape[1] == tokens
    input_is_bf16 = residual.dtype == torch.bfloat16

    with torch.accelerator.device_index(residual.device.index):
        if residual.is_contiguous():
            contiguous_grid = (triton.cdiv(residual.numel(), _CONTIGUOUS_BLOCK_SIZE),)
            _residual_gate_add_contiguous_kernel[contiguous_grid](
                output,
                residual,
                update,
                gate,
                residual.numel(),
                tokens * hidden_size,
                tokens,
                gate.stride(0),
                gate.stride(1),
                hidden_size=hidden_size,
                gate_is_batched=gate_is_batched,
                gate_is_tokenwise=gate_is_tokenwise,
                input_is_bf16=input_is_bf16,
                block_size=_CONTIGUOUS_BLOCK_SIZE,
                num_warps=8,
            )
            return output

        transposed_grid = (
            triton.cdiv(tokens, _TRANSPOSE_TILE),
            triton.cdiv(hidden_size, _TRANSPOSE_TILE),
            batch,
        )
        _residual_gate_add_transposed_kernel[transposed_grid](
            output,
            residual,
            update,
            gate,
            tokens,
            hidden_size,
            gate.stride(0),
            gate.stride(1),
            gate_is_batched=gate_is_batched,
            gate_is_tokenwise=gate_is_tokenwise,
            input_is_bf16=input_is_bf16,
            block_tokens=_TRANSPOSE_TILE,
            block_hidden=_TRANSPOSE_TILE,
            num_warps=8,
        )
    return output


def residual_gate_add(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Return ``residual + gate * update``, using a bit-exact CUDA fusion.

    The direct Triton call intentionally stays outside compiled regions:
    Inductor can already fuse the eager formula there, while an opaque kernel
    boundary would inhibit surrounding graph optimization.
    """
    if torch.compiler.is_compiling():
        return residual + gate * update

    runtime_key = (residual.device.index, residual.dtype)
    if runtime_key not in _FAILED_RUNTIME_KEYS and _can_use_fused_residual_gate_add(residual, update, gate):
        try:
            return _launch_fused_residual_gate_add(residual, update, gate)
        except Exception as exc:
            _FAILED_RUNTIME_KEYS.add(runtime_key)
            logger.warning_once(
                "Disabling fused residual-gate add on %s/%s after a runtime failure: %s",
                residual.device,
                residual.dtype,
                exc,
            )
    return residual + gate * update


__all__ = ["residual_gate_add"]
