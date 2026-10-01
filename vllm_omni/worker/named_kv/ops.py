# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Custom op for named KV branch attention.

The op fuses K/V scatter-write and paged attention into a single graph node
so ``torch.compile`` treats it as opaque (no graph break, no eager island).
The K/V cache uses FA's native four-dimensional layout:

    key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: [num_blocks, block_size, num_kv_heads, head_size]

``slot_mapping`` carries flat slot indices (``block_id * block_size +
offset``); the op internally maps them to four-dimensional coordinates.

Per-layer FA parameters (``kv_cache_dtype`` string, ``k_scale``, ``v_scale``,
``softmax_scale``) are passed as individual primitive arguments because
``torch.library.custom_op`` only accepts tensor / int / float / bool / str
types (no dataclasses).

When ``scheduler_metadata`` is ``None`` (eager path), the op computes it
per-call via ``get_scheduler_metadata``.  When it is a pre-allocated tensor
(graph path), the executor updates it before execution and the op only reads
it. Metadata workspace mutation is therefore outside this op's schema.
"""

from __future__ import annotations

import torch


def _named_kv_branch_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    num_kv_heads: int,
    head_size: int,
    num_heads_q: int,
    block_size: int,
    scheduler_metadata: torch.Tensor | None,
    fa_version: int,
) -> None:
    # 1. Scatter-write K/V into the 4-D pool.
    from vllm._custom_ops import reshape_and_cache_flash

    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    # 2. Run paged attention (causal, varlen, single query per request).
    from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_scheduler_metadata,
    )

    # Fix degenerate strides on size-1 dims (matches FA backend behavior).
    fixed_k = canonicalize_singleton_dim_strides(key_cache)
    fixed_v = canonicalize_singleton_dim_strides(value_cache)

    # Compute scheduler_metadata.
    # - Eager path (scheduler_metadata is None): compute per-call.
    # - Graph path (scheduler_metadata is pre-allocated): use directly,
    #   do NOT recompute.  The executor fills the workspace before replay.
    if scheduler_metadata is None:
        computed_metadata = get_scheduler_metadata(
            batch_size=query_start_loc.shape[0] - 1,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            num_heads_q=num_heads_q,
            num_heads_kv=num_kv_heads,
            headdim=head_size,
            cache_seqlens=seq_lens,
            qkv_dtype=key_cache.dtype,
            cu_seqlens_q=query_start_loc,
            page_size=block_size,
            causal=True,
            window_size=(-1, -1),
            num_splits=1,
        )
        effective_metadata = computed_metadata
    else:
        effective_metadata = scheduler_metadata

    flash_attn_varlen_func(
        q=query,
        k=fixed_k,
        v=fixed_v,
        out=output,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_query_len,
        seqused_k=seq_lens,
        max_seqlen_k=max_seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        block_table=block_table,
        scheduler_metadata=effective_metadata,
        num_splits=1,
        fa_version=fa_version,
    )


def _named_kv_branch_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    num_kv_heads: int,
    head_size: int,
    num_heads_q: int,
    block_size: int,
    scheduler_metadata: torch.Tensor | None,
    fa_version: int,
) -> None:
    return None


if not hasattr(torch.ops.vllm_omni, "named_kv_branch_attention"):
    _named_kv_branch_attention_op = torch.library.custom_op(
        "vllm_omni::named_kv_branch_attention",
        mutates_args=("output", "key_cache", "value_cache"),
    )(_named_kv_branch_attention_impl)

    _named_kv_branch_attention_op.register_fake(_named_kv_branch_attention_fake)
else:
    _named_kv_branch_attention_op = torch.ops.vllm_omni.named_kv_branch_attention  # type: ignore[attr-defined]


__all__: list[str] = []
