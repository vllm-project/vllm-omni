# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""BF16 VSA on Ada and newer GPUs using FastVideo's portable Triton kernel."""

import math
from collections.abc import Callable
from functools import lru_cache

import torch

from vllm_omni.platforms import current_omni_platform

from .block_sparse import block_map_to_indices, split_tile128_map, validate_sparse_inputs


@lru_cache(maxsize=32)
def require_triton_sparse(
    device: torch.device,
) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    capability = current_omni_platform.get_device_capability(device.index) if device.type == "cuda" else None
    if capability is None or capability < (8, 0):
        raise ValueError(f"Triton BF16 block-sparse attention requires SM80 or newer; got {capability}")
    try:
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn_triton
    except ImportError as exc:
        raise ImportError("Triton VSA requires fastvideo-kernel; install vllm-omni[vsa]") from exc
    return block_sparse_attn_triton


def triton_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    block_sizes: torch.Tensor | None,
    softmax_scale: float,
    block_size: int,
    *,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if block_size not in (64, 128):
        raise ValueError("Triton VSA requires logical tile64 or tile128")
    if not math.isclose(softmax_scale, 128**-0.5, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Triton VSA requires head_size**-0.5 scaling")
    if query.shape[1] % block_size or key.shape[1] % block_size:
        raise ValueError("Triton VSA inputs must be padded to complete logical tiles")
    key_blocks = key.shape[1] // block_size
    if block_sizes is None or block_sizes.shape != (key_blocks,):
        raise ValueError("Triton VSA requires one valid-token count per KV tile")
    if block_map.shape[-1] != key_blocks:
        raise ValueError("Triton VSA block_map must match the KV tile count")
    indices, counts = block_map_to_indices(block_map)
    validate_sparse_inputs(query, key, value, indices, counts, block_sizes, block_size)
    if block_size == 128:
        block_map, block_sizes = split_tile128_map(block_map, block_sizes)
    else:
        block_map = block_map & (block_sizes > 0)[None, None, None, :]
    indices, counts = block_map_to_indices(block_map)
    kernel = require_triton_sparse(query.device)
    output, lse_log2 = kernel(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        indices,
        counts,
        block_sizes,
    )
    # The upstream kernel divides by a zero denominator for empty rows.
    # Define those rows as zero without hiding non-finite nonempty results.
    empty_rows = (counts == 0).repeat_interleave(64, -1)
    output = output.masked_fill(empty_rows.unsqueeze(-1), 0).transpose(1, 2).contiguous()
    if not return_lse:
        return output
    if lse_log2.shape != (query.shape[0], query.shape[2], query.shape[1]) or lse_log2.dtype != torch.float32:
        raise ValueError("Triton BSA must return FP32 base-2 LSE with shape [B,H,Sq]")
    lse = (lse_log2 * math.log(2)).masked_fill(empty_rows, -torch.inf).contiguous()
    return output, lse
