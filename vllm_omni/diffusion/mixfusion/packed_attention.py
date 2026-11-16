# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed attention entry for MixFusion chunked layouts.

Image chunks are stacked on a local batch dimension while text stays
request-level, so the model cannot concatenate them before attention. This
attention-layer boundary gathers each request into its logical ``text +
image`` order, runs one packed forward, and scatters back to every stream
layout.

The per-step plan (carrying the model's ``AttentionLayout``) travels in
``extra[MIXFUSION_PLAN_KEY]`` and ``"request"``-kind streams pass Q/K/V
through ``extra[MIXFUSION_STREAMS_KEY]``. The positional
``query``/``key``/``value`` belong to the single ``"chunked"`` stream. The
output is a ``dict`` keyed by stream name.
"""

from __future__ import annotations

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout
from vllm_omni.diffusion.mixfusion.execution_plan import PackedExecutionPlan

MIXFUSION_PLAN_KEY = "mixfusion_plan"
MIXFUSION_STREAMS_KEY = "mixfusion_streams"


def _backend_supports_attention_mask(attn) -> bool:
    """Whether the attention layer's backend can consume a dense mask.

    Custom attention modules carry no backend and report no mask support
    (mirroring ``Attention._assert_metadata_compatible``).
    """
    _, backend, spec = attn.effective_attention()
    if backend is None:
        return False
    return bool(backend.supports_attention_mask(spec))


def _stream_rows(
    streams: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    layout: AttentionLayout,
    plan: PackedExecutionPlan,
    req_idx: int,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather request ``req_idx``'s rows in layout order."""
    queries: list[torch.Tensor] = []
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for spec in layout.streams:
        stream_query, stream_key, stream_value = streams[spec.name]
        if spec.kind == "request":
            length = plan.request_stream_lens[spec.name][req_idx]
            queries.append(stream_query[req_idx, :length])
            keys.append(stream_key[req_idx, :length])
            values.append(stream_value[req_idx, :length])
        else:
            chunk_start, chunk_end = plan.request_chunk_ranges[req_idx]
            req_query = stream_query[chunk_start:chunk_end]
            req_key = stream_key[chunk_start:chunk_end]
            req_value = stream_value[chunk_start:chunk_end]
            queries.append(req_query.reshape(-1, num_query_heads, head_dim))
            keys.append(req_key.reshape(-1, num_kv_heads, head_dim))
            values.append(req_value.reshape(-1, num_kv_heads, head_dim))
    return torch.cat(queries, dim=0), torch.cat(keys, dim=0), torch.cat(values, dim=0)


def _packed_dense_with_text_mask(
    attn,
    flat_queries: list[torch.Tensor],
    flat_keys: list[torch.Tensor],
    flat_values: list[torch.Tensor],
    layout: AttentionLayout,
    plan: PackedExecutionPlan,
    chunked_name: str,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Serve packed documents with one masked dense forward per image length.

    Documents sharing a chunked-stream length share a batch row; their
    request-level streams are padded to the group's longest and a
    ``[group, L]`` key mask hides the padding. Padded query rows are dropped
    by the per-request gather, so the flat row count and order still match
    ``plan.cu_seqlens_tensor`` and ``plan.output_indices`` is unchanged.
    """
    num_requests = len(flat_queries)
    groups: dict[int, list[int]] = {}
    for req_idx in range(num_requests):
        image_len = plan.request_stream_lens[chunked_name][req_idx]
        groups.setdefault(image_len, []).append(req_idx)

    per_request: list[torch.Tensor | None] = [None] * num_requests
    for indices in groups.values():
        widths = {
            spec.name: max(plan.request_stream_lens[spec.name][req_idx] for req_idx in indices)
            for spec in layout.streams
            if spec.kind == "request"
        }
        group_len = sum(
            widths.get(spec.name, plan.request_stream_lens[spec.name][indices[0]]) for spec in layout.streams
        )
        needs_mask = any(
            plan.request_stream_lens[spec.name][req_idx] < widths[spec.name]
            for req_idx in indices
            for spec in layout.streams
            if spec.kind == "request"
        )
        group_query = flat_queries[0].new_zeros(len(indices), group_len, num_query_heads, head_dim)
        group_key = flat_keys[0].new_zeros(len(indices), group_len, num_kv_heads, head_dim)
        group_value = flat_values[0].new_zeros(len(indices), group_len, num_kv_heads, head_dim)
        mask = None
        if needs_mask:
            mask = torch.zeros(len(indices), group_len, dtype=torch.bool, device=group_query.device)
        positions: list[torch.Tensor] = []
        for group_row, req_idx in enumerate(indices):
            source_offset = 0
            target_offset = 0
            real_positions: list[int] = []
            for spec in layout.streams:
                real_len = plan.request_stream_lens[spec.name][req_idx]
                width = widths.get(spec.name, real_len)
                target = slice(target_offset, target_offset + real_len)
                source = slice(source_offset, source_offset + real_len)
                group_query[group_row, target] = flat_queries[req_idx][source]
                group_key[group_row, target] = flat_keys[req_idx][source]
                group_value[group_row, target] = flat_values[req_idx][source]
                if mask is not None:
                    mask[group_row, target] = True
                real_positions.extend(range(target_offset, target_offset + real_len))
                source_offset += real_len
                target_offset += width
            positions.append(torch.tensor(real_positions, dtype=torch.long, device=group_query.device))

        # Fresh metadata without the plan key: Attention.forward runs the
        # ordinary kernel and its re-entry guard stays quiet.
        if mask is None:
            group_out = attn.forward(group_query, group_key, group_value, AttentionMetadata())
        else:
            metadata = AttentionMetadata(attn_mask=mask, extra={"attention_mask_mode": "padding"})
            group_out = attn.forward(group_query, group_key, group_value, metadata)
        for group_row, req_idx in enumerate(indices):
            per_request[req_idx] = group_out[group_row].index_select(0, positions[group_row])

    # Every request belongs to exactly one group; concatenating in request
    # order rebuilds the flat packed layout the varlen path would produce.
    return torch.cat([rows for rows in per_request if rows is not None], dim=0)


def run_mixfusion_packed_attention(
    attn,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> dict[str, torch.Tensor]:
    """Run joint attention over the streams declared by the model's layout.

    ``query``/``key``/``value`` hold the single ``"chunked"`` stream as
    ``[num_chunks, chunk_size, heads, head_dim]``; other streams ride
    ``attn_metadata.extra[MIXFUSION_STREAMS_KEY]`` as ``{name: (q, k, v)}``.
    Each request is recovered to its logical stream order before the kernel,
    so attention matches independent dense execution. Returns outputs keyed by
    stream name in each stream's input layout.
    """
    plan: PackedExecutionPlan = attn_metadata.extra[MIXFUSION_PLAN_KEY]
    layout = plan.layout
    chunked_name = next(spec.name for spec in layout.streams if spec.kind == "chunked")
    all_streams = {**attn_metadata.extra[MIXFUSION_STREAMS_KEY], chunked_name: (query, key, value)}

    num_query_heads = query.shape[2]
    num_kv_heads = key.shape[2]
    head_dim = query.shape[3]
    num_requests = len(plan.request_chunk_ranges)

    # Recover each request to its logical stream order, then stack every
    # request into the flat varlen layout the kernel expects.
    flat_queries: list[torch.Tensor] = []
    flat_keys: list[torch.Tensor] = []
    flat_values: list[torch.Tensor] = []
    for req_idx in range(num_requests):
        req_query, req_key, req_value = _stream_rows(
            all_streams,
            layout,
            plan,
            req_idx,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        flat_queries.append(req_query)
        flat_keys.append(req_key)
        flat_values.append(req_value)

    # A varlen-capable backend takes the whole flat packing in one kernel; a
    # dense backend that takes masks fuses documents of the same image length
    # by padding and masking their text stream. The capability is probed
    # without a kv dtype on purpose: a quantized policy (which rejects both
    # cu_seqlens and masks) then reaches ``forward_packed``, whose mask-free
    # exact-length buckets are the only path it can run.
    if not attn.supports_multi_doc_packed_varlen() and _backend_supports_attention_mask(attn):
        packed_hidden_states = _packed_dense_with_text_mask(
            attn,
            flat_queries,
            flat_keys,
            flat_values,
            layout,
            plan,
            chunked_name,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
    else:
        packed_query = torch.cat(flat_queries, dim=0)
        packed_key = torch.cat(flat_keys, dim=0)
        packed_value = torch.cat(flat_values, dim=0)
        del flat_queries, flat_keys, flat_values
        packed_hidden_states = attn.forward_packed(
            packed_query,
            packed_key,
            packed_value,
            cu_seqlens=plan.cu_seqlens_tensor,
            max_seqlen=plan.max_seq_len,
        )

    # Recover each stream's rows with one gather; request-level streams copy
    # into a zero buffer when their padded width exceeds the real length.
    outputs: dict[str, torch.Tensor] = {}
    for spec in layout.streams:
        gather, scatter = plan.output_indices[spec.name]
        rows = packed_hidden_states.index_select(0, gather)
        if spec.kind == "chunked":
            outputs[spec.name] = rows.view(plan.num_chunks, plan.chunk_size, num_query_heads, head_dim)
            continue
        padded = plan.stream_padded_lens[spec.name]
        if scatter is None:
            outputs[spec.name] = rows.view(num_requests, padded, num_query_heads, head_dim)
        else:
            buffer = rows.new_zeros(num_requests * padded, num_query_heads, head_dim)
            buffer.index_copy_(0, scatter, rows)
            outputs[spec.name] = buffer.view(num_requests, padded, num_query_heads, head_dim)
    return outputs
