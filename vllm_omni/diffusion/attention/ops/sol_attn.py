# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SOL kernels, threshold routing, and normalization-weighted residual merge."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from vllm_omni.diffusion.attention.ops.block_sparse import (
    block_sparse_attn_bshd,
    mean_pool_tiles,
    resolve_block_sparse_provider,
    validate_softmax_scale,
)

if not hasattr(torch.ops.vllm_omni, "sol_attention"):

    @torch.library.custom_op("vllm_omni::sol_attention", mutates_args=())
    def _sol_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        kv_splits: int,
        sink_start: int,
        sink_tokens: int,
    ) -> torch.Tensor:
        from sol_attn import sol_attn

        return sol_attn(
            query,
            key,
            value,
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
            kv_splits=kv_splits,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )

    @_sol_attention.register_fake
    def _sol_attention_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        kv_splits: int,
        sink_start: int,
        sink_tokens: int,
    ) -> torch.Tensor:
        return torch.empty_like(query)


sol_attention = torch.ops.vllm_omni.sol_attention


def require_sol_bsa(
    provider: str, scale: float, device: torch.device
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    validate_softmax_scale(scale)
    if provider not in ("cudnn", "triton"):
        raise ValueError("SOL BSA requires an explicit cudnn or triton provider.")
    if provider == "triton" and not math.isclose(scale, 128**-0.5, rel_tol=0, abs_tol=1e-6):
        raise ValueError("SOL BSA Triton requires head_size**-0.5 scaling.")
    resolve_block_sparse_provider(provider, "bf16", device, torch.bfloat16, 128, 64)
    try:
        from sol_attn.triton_ref.preprocess import prepare
    except ImportError as error:
        raise ImportError("SOL BSA requires the pinned sol_attn package with triton_ref.preprocess.prepare.") from error
    if not callable(prepare):
        raise ImportError("SOL BSA requires callable triton_ref.preprocess.prepare.")
    return prepare


def _sol_bsa_block_map(
    query: torch.Tensor,
    key_centroids: torch.Tensor,
    threshold: torch.Tensor,
    sizes: torch.Tensor,
    scale: float,
    sink_start: int,
    sink_tokens: int,
    sink_block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SOL thresholds, neighboring tiles and dense query/key sink tiles."""
    pooled_query = mean_pool_tiles(query, sizes, 64)
    scores = pooled_query @ key_centroids.float().transpose(1, 2).transpose(-1, -2)
    selected = scores * (scale * math.log2(math.e)) > threshold.transpose(1, 2).unsqueeze(-1)
    positions = torch.arange(sizes.numel(), device=query.device)
    selected |= (positions[:, None] - positions[None, :]).abs() <= 1
    if sink_tokens:
        first, last = sink_start // 64, (sink_start + sink_tokens + 63) // 64
        selected[..., first:last, :] = True
        selected[..., first:last] = True
    if sink_block_mask is not None:
        selected |= sink_block_mask[:, None] | sink_block_mask[None, :]
    return selected


def _merge_sol_bsa_residual(
    query: torch.Tensor,
    key_centroids: torch.Tensor,
    value_sums: torch.Tensor,
    selected: torch.Tensor,
    sizes: torch.Tensor,
    exact_output: torch.Tensor,
    exact_lse: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference SOL merge with bounded query chunks and natural-log normalization.

    Upstream summarizes K by its mean and V by its sum. Block multiplicity
    belongs only in the residual denominator. This correctness-first path
    deliberately does not claim the speed of a fused SOL residual kernel.
    """
    output = torch.empty_like(query)
    keys = key_centroids.float().transpose(1, 2).transpose(-1, -2)
    values = value_sums.float().transpose(1, 2)
    log_sizes = sizes.float().log()
    for start in range(0, query.shape[1], 256):
        stop = min(start + 256, query.shape[1])
        scores = query[:, start:stop].float().transpose(1, 2) @ keys * scale
        exact_mask = selected[:, :, start // 64 : (stop + 63) // 64].repeat_interleave(64, -2)
        scores.masked_fill_(exact_mask[:, :, : stop - start], -torch.inf)
        residual_lse = (scores + log_sizes).logsumexp(-1)
        exact_lse_chunk = exact_lse[:, :, start:stop]
        joint_lse = torch.logaddexp(exact_lse_chunk, residual_lse)
        residual = (scores - joint_lse.unsqueeze(-1)).exp() @ values
        exact_weight = (exact_lse_chunk - joint_lse).exp().unsqueeze(-1)
        merged = exact_output[:, start:stop].float().transpose(1, 2) * exact_weight + residual
        output[:, start:stop] = merged.transpose(1, 2).to(query.dtype)
    return output


if not hasattr(torch.ops.vllm_omni, "sol_bsa_attention"):

    @torch.library.custom_op("vllm_omni::sol_bsa_attention", mutates_args=())
    def _sol_bsa_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        sink_start: int,
        sink_tokens: int,
        provider: str,
        sink_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prepare = require_sol_bsa(provider, scale, query.device)
        tokens = query.shape[1]
        blocks = (tokens + 63) // 64
        key_centroids, value_sums, threshold = prepare(
            query, key, value, tau=tau, scale=scale, thresh_type=thresh_type, tokens=tokens
        )
        key_centroids, value_sums = key_centroids[:, :blocks], value_sums[:, :blocks]
        sizes = (tokens - torch.arange(blocks, device=query.device, dtype=torch.int32) * 64).clamp(max=64)
        padding = blocks * 64 - tokens
        query_padded, key_padded, value_padded = (
            torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, padding)) if padding else tensor
            for tensor in (query, key, value)
        )
        selected = _sol_bsa_block_map(
            query_padded, key_centroids, threshold, sizes, scale, sink_start, sink_tokens, sink_block_mask
        )
        exact_output, exact_lse = block_sparse_attn_bshd(
            query_padded, key_padded, value_padded, selected, sizes, scale, provider=provider, return_lse=True
        )
        return _merge_sol_bsa_residual(
            query, key_centroids, value_sums, selected, sizes, exact_output, exact_lse, scale
        )

    @_sol_bsa_attention.register_fake
    def _sol_bsa_attention_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        sink_start: int,
        sink_tokens: int,
        provider: str,
        sink_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty_like(query)


sol_bsa_attention = torch.ops.vllm_omni.sol_bsa_attention
