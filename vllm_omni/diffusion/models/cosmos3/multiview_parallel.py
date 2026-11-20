# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Parallel topology validation and Ulysses attention for Multiview-AV."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from .multiview_attention import multiview_attention
from .multiview_flex_attention import MultiviewAttentionContext

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionParallelConfig


def validate_multiview_parallel_config(
    parallel_config: DiffusionParallelConfig,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int,
) -> None:
    """Validate multiview-only limits before constructing/loading the transformer or running collectives.

    ``DiffusionParallelConfig`` already enforces positive degrees,
    ``sequence_parallel_size == ulysses * ring * allgather`` and HSDP/TP exclusivity.
    """
    tp = parallel_config.tensor_parallel_size
    cp = parallel_config.ulysses_degree
    if parallel_config.cfg_parallel_size > 2:
        raise ValueError("Cosmos3 multiview requires cfg_parallel_size in (1, 2).")
    for name in ("ring_degree", "allgather_degree", "pipeline_parallel_size", "vae_patch_parallel_size"):
        if getattr(parallel_config, name) != 1:
            raise ValueError(f"Cosmos3 multiview requires {name}=1.")
    if cp > 1 and parallel_config.ulysses_mode != "strict":
        raise ValueError("Cosmos3 multiview requires ulysses_mode='strict'.")
    if num_key_value_heads <= 0 or num_attention_heads <= 0 or num_attention_heads % num_key_value_heads:
        raise ValueError("Cosmos3 multiview requires positive query/KV head counts with an integral GQA ratio.")
    for name, heads in (("query", num_attention_heads), ("KV", num_key_value_heads)):
        if heads % (tp * cp):
            raise ValueError(f"Cosmos3 multiview {name} heads ({heads}) must be divisible by TP × CP ({tp} × {cp}).")
    if intermediate_size <= 0 or intermediate_size % tp:
        raise ValueError(f"Cosmos3 multiview intermediate_size ({intermediate_size}) must be divisible by TP ({tp}).")


def _all_to_all(tensor: torch.Tensor, group: dist.ProcessGroup, scatter: int, gather: int) -> torch.Tensor:
    # Use the shared collective implementation; keep runtime imports lazy so
    # geometry/validation can also be checked without initializing vLLM.
    from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D

    return SeqAllToAll4D.apply(group, tensor.contiguous(), scatter, gather, False)


def multiview_ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    context: MultiviewAttentionContext,
    *,
    group: dist.ProcessGroup,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Attend over global tokens/local heads, then return local tokens/all heads.

    Inputs have shape [B, local_GEN, TP_local_heads, D]. UND K/V are replicated
    across CP ranks, already TP-local. Plans and sparse masks stay in global camera-major coordinates. CP padding
    is removed before dispatch and restored before the inverse exchange. Head
    divisibility is checked at load by ``validate_multiview_parallel_config``.
    """
    real_len = context.layout.gen_tokens
    local_len = (real_len + world_size - 1) // world_size
    if q.shape[1] != local_len:
        raise ValueError(
            f"Multiview Ulysses expected {local_len} local GEN tokens for {real_len} tokens / CP{world_size}, "
            f"got {q.shape[1]}. Check that the transformer sequence-sharding hooks are applied."
        )
    q_full = _all_to_all(q, group, 2, 1)
    k_full = _all_to_all(k, group, 2, 1)
    v_full = _all_to_all(v, group, 2, 1)
    kv_heads = k.shape[2] // world_size
    start = rank * kv_heads
    output = multiview_attention(
        q_full[:, :real_len],
        k_full[:, :real_len],
        v_full[:, :real_len],
        k_und[:, :, start : start + kv_heads],
        v_und[:, :, start : start + kv_heads],
        context,
    )
    padding = local_len * world_size - real_len
    if padding:
        output = torch.cat((output, output.new_zeros(output.shape[0], padding, *output.shape[2:])), dim=1)
    return _all_to_all(output, group, 1, 2)
