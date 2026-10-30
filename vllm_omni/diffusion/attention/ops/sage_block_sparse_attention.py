# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sage fine attention with an explicit sparse-layout and prepared-Q contract.

The scheduler owns stream dependencies and sparse selection. This adapter owns
operand quantization, the CAKE kernel invocation, and output layout. Tensor-map
descriptors are owned by FlashInfer's by-value launch ABI, not by a global cache.
"""

from functools import cache

import torch

from .flashinfer_block_sparse import require_flashinfer_sparse, validate_softmax_scale, validate_sparse_inputs


@cache
def _empty_descriptor_workspace(device: torch.device) -> torch.Tensor:
    # The public API still accepts this argument; the by-value ABI never uses
    # its storage. There is no descriptor state shared by concurrent launches.
    return torch.empty(0, dtype=torch.uint8, device=device)


def sage_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    block_sizes: torch.Tensor | None,
    softmax_scale: float,
    *,
    prepared_q: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute Sage fine attention and return contiguous BF16 BSHD output.

    A prepared Q may be supplied after the caller joins its producer stream.
    Otherwise all three operands are quantized here. Ordered top-k indices
    are not contiguous blocks, and prefix rows have different counts: neither
    FlashInfer's uniform-count nor contiguous-index specialization is enabled.
    """
    validate_sparse_inputs(query, key, value, indices, counts, block_sizes)
    validate_softmax_scale(softmax_scale)
    if prepared_q is not None:
        batch, rows, heads, dim = query.shape
        expected = (((batch, heads, rows, dim), torch.int8), ((batch, heads, ((rows + 127) // 128) * 4), torch.float32))
        if len(prepared_q) != 2:
            raise ValueError("prepared_q must contain an INT8 tensor and FP32 scales")
        for tensor, (shape, dtype) in zip(prepared_q, expected, strict=True):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != query.device:
                raise ValueError("prepared_q must match the query shape, scale layout, and device")
            if not tensor.is_contiguous():
                raise ValueError("prepared_q must be contiguous")
    if query.device.type != "cuda":
        raise ValueError("Sage execution requires CUDA")
    bsa_attn_sm120_blk64_sage_fwd = require_flashinfer_sparse("sage", query.device)

    from .sage_quantization import quantize_sage_kv_sm120, quantize_sage_qkv_sm120

    q_bhsd, k_bhsd, v_bhsd = (t.transpose(1, 2) for t in (query, key, value))
    if prepared_q is None:
        quantized = quantize_sage_qkv_sm120(q_bhsd, k_bhsd, v_bhsd)
    else:
        k_int8, v_fp8, k_scale, v_scale = quantize_sage_kv_sm120(k_bhsd, v_bhsd)
        quantized = (prepared_q[0], k_int8, v_fp8, prepared_q[1], k_scale, v_scale)
    output = torch.empty_like(q_bhsd, memory_format=torch.contiguous_format)
    result = bsa_attn_sm120_blk64_sage_fwd(
        *quantized,
        indices,
        int(indices.shape[-1]),
        block_sizes=block_sizes,
        q2k_block_nums=counts,
        softmax_scale=softmax_scale,
        out=output,
        tma_descriptor_workspace=_empty_descriptor_workspace(query.device),
        uniform_block_count=False,
        contiguous_block_indices=False,
        backend="cake",
    )
    return result.transpose(1, 2).contiguous()
