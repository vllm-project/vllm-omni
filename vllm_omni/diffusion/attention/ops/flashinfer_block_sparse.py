# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-independent FlashInfer tile64 attention and hardware contracts.

Callers own sparse selection and any model-specific compression correction.
This module owns the BF16/Sage ABI and validates the actual input device.
"""

import math

import torch

from .block_sparse import block_map_to_indices


def validate_sparse_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    block_sizes: torch.Tensor | None,
) -> None:
    """Check structural metadata without reading device values on the host."""
    for tensor in (query, key, value):
        if tensor.ndim != 4 or tensor.shape[-1] != 128 or min(tensor.shape) < 1:
            raise ValueError("FlashInfer sparse attention expects nonempty BSHD inputs with head dimension 128")
        if tensor.stride(-1) != 1:
            raise ValueError("FlashInfer sparse attention requires a contiguous head dimension")
    if key.shape != value.shape or query.shape[::2] != key.shape[::2]:
        raise ValueError("FlashInfer sparse attention requires matching batch/head dimensions and matching K/V shapes")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise TypeError("FlashInfer sparse inputs must use BF16")
    batch, rows, heads, _ = query.shape
    blocks = (rows + 63) // 64
    if indices.ndim != 4 or indices.shape[:3] != (batch, heads, blocks):
        raise ValueError("Sparse indices must have shape [B,H,ceil(Sq/64),capacity]")
    if indices.shape[-1] < 1 or counts.shape != indices.shape[:3]:
        raise ValueError("Sparse counts must match the query blocks and capacity must be positive")
    metadata: tuple[torch.Tensor, ...] = (indices, counts)
    if block_sizes is not None:
        key_blocks = (key.shape[1] + 63) // 64
        if block_sizes.shape not in ((key_blocks,), (batch, key_blocks), (batch, heads, key_blocks)):
            raise ValueError("Block sizes must have shape [Kblocks], [B,Kblocks], or [B,H,Kblocks]")
        metadata += (block_sizes,)
    for tensor in metadata:
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise TypeError("Sparse metadata must be contiguous INT32")
    if any(t.device != query.device for t in (key, value, *metadata)):
        raise ValueError("FlashInfer sparse inputs and sparse metadata must share one device")


def validate_softmax_scale(softmax_scale: float) -> None:
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a real scalar")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")


def validate_flashinfer_sparse_capability(precision: str, capability: tuple[int, int] | None) -> None:
    """Match the public provider's hardware contract without loading its kernels."""
    if precision not in ("bf16", "sage"):
        raise ValueError("FlashInfer sparse precision must be bf16 or sage")
    supported = ((12, 0), (12, 1)) if precision == "bf16" else ((12, 0),)
    if capability not in supported:
        hardware = "SM120/SM121" if precision == "bf16" else "SM120"
        raise ValueError(f"FlashInfer {precision} tile64 attention requires {hardware}; got {capability}")


def require_flashinfer_sparse(precision: str, device: torch.device | None = None):
    """Fail before launch on unsupported hardware or missing optional APIs."""
    from vllm_omni.platforms import current_omni_platform

    if device is not None and device.type != "cuda":
        raise ValueError("FlashInfer sparse execution requires CUDA")
    device_id = device.index if device is not None and device.index is not None else 0
    validate_flashinfer_sparse_capability(precision, current_omni_platform.get_device_capability(device_id))
    name = "bsa_attn_sm120_blk64_sage_fwd" if precision == "sage" else "bsa_attn_sm120_blk64_fwd"
    try:
        from flashinfer.cute_dsl.sparse import bsa_attn_sm120

        return getattr(bsa_attn_sm120, name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"FlashInfer {precision} requires a build with {name}") from exc


def flashinfer_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    block_sizes: torch.Tensor | None,
    softmax_scale: float,
    *,
    precision: str,
) -> torch.Tensor:
    """Execute an explicit [B,H,ceil(Sq/64),ceil(Sk/64)] sparse map.

    BF16 BSHD inputs may have different query/key lengths, ragged tails and
    arbitrary per-batch/head selected blocks. No model metadata is required.
    """
    if precision not in ("bf16", "sage"):
        raise ValueError("FlashInfer sparse precision must be bf16 or sage")
    indices, counts = block_map_to_indices(block_map)
    validate_sparse_inputs(query, key, value, indices, counts, block_sizes)
    if block_map.shape[-1] != (key.shape[1] + 63) // 64:
        raise ValueError("block_map key blocks must match ceil(Sk/64)")
    validate_softmax_scale(softmax_scale)
    if precision == "sage":
        from .sage_block_sparse_attention import sage_block_sparse_attention

        return sage_block_sparse_attention(query, key, value, indices, counts, block_sizes, softmax_scale)
    kernel = require_flashinfer_sparse(precision, query.device)
    # BF16 consumes/returns BSHD and a tuple; Sage uses a separate BHSD ABI.
    output, _ = kernel(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        indices,
        indices.shape[-1],
        block_sizes=block_sizes,
        q2k_block_nums=counts,
        softmax_scale=softmax_scale,
    )
    return output.contiguous()
