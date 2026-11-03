# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed attention entry for MixFusion chunked layouts.

The MixFusion layout stacks image chunks on a local batch dimension while
text stays request-level, so the two streams cannot be concatenated by the
model before attention (their batch dimensions differ). This module is the
attention-layer boundary that turns the layout into the standard flat
packed row: it gathers each request back into its logical ``text + image``
token order, runs one packed forward, and scatters the result back to every
declared stream layout -- the same pre/post boundary shape the
sequence-parallel strategies use around their all-to-all.

The model declares its streams through
:meth:`~vllm_omni.diffusion.mixfusion.adapter.MixFusionAdapter.build_attention_layout`;
the armed ``MixFusionStep`` carries that layout, the per-step packing plan
travels in ``extra[MIXFUSION_PLAN_KEY]``, and every ``"request"``-kind stream
passes its Q/K/V through ``extra[MIXFUSION_STREAMS_KEY]``. The positional
``query``/``key``/``value`` arguments belong to the single ``"chunked"``
stream, because that is the tensor shape the attention caller already has.
The output is a ``dict`` keyed by stream name.
"""

from __future__ import annotations

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout
from vllm_omni.diffusion.mixfusion.execution_plan import PackedExecutionPlan

MIXFUSION_PLAN_KEY = "mixfusion_plan"
MIXFUSION_STREAMS_KEY = "mixfusion_streams"


def _backend_supports_attention_mask(attn) -> bool:
    """Whether the attention layer's backend can consume a dense mask.

    ``Attention.attn_backend`` is the backend class and ``attn_spec`` its
    resolved spec; custom attention modules carry no backend and therefore
    report no mask support (mirroring ``Attention._assert_metadata_compatible``).
    """
    backend = getattr(attn, "attn_backend", None)
    if backend is None:
        return False
    return bool(backend.supports_attention_mask(getattr(attn, "attn_spec", None)))


def _active_layout() -> AttentionLayout:
    if is_forward_context_available():
        step = get_forward_context().mixfusion_step
        if step is not None and step.layout is not None:
            return step.layout
    raise ValueError(
        "MixFusion packed attention requires an armed MixFusionStep with an AttentionLayout (step.layout)."
    )


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
            if stream_query.shape[0] <= req_idx or stream_query.shape[1] < length:
                raise ValueError(
                    f"Stream {spec.name!r} of request {req_idx} has fewer rows than its {length}-token length."
                )
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

    Backends that cannot take cu_seqlens but can take a dense mask do not need
    one forward per document: documents with the same chunked-stream length can
    share a batch row, pad only their request-level streams (text) to the
    group's longest, and keep every row the padding introduced out of attention
    with a ``[group, L]`` Boolean key mask. Padded query rows are computed but
    dropped by the per-request gather, so the result keeps the flat row count
    and order of ``plan.cu_seqlens_tensor`` and the caller's output recovery
    (``plan.output_indices``) is unchanged.
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
    ``[num_chunks, chunk_size, heads, head_dim]``; the other streams ride
    ``attn_metadata.extra[MIXFUSION_STREAMS_KEY]`` as
    ``{name: (q, k, v)}``; the packing layout is
    ``attn_metadata.extra[MIXFUSION_PLAN_KEY]``.

    Each request is recovered to its logical stream order right before the
    kernel so the attention semantics match independent dense execution.
    Returns the attention outputs keyed by stream name in each stream's input
    layout (chunk-stacked or padded request-level).
    """
    plan: PackedExecutionPlan = attn_metadata.extra[MIXFUSION_PLAN_KEY]
    layout = _active_layout()
    streams = attn_metadata.extra.get(MIXFUSION_STREAMS_KEY)
    if streams is None:
        raise ValueError(
            f"MixFusion packed attention requires the extra streams in attn_metadata.extra[{MIXFUSION_STREAMS_KEY!r}]"
        )

    chunked_names = [spec.name for spec in layout.streams if spec.kind == "chunked"]
    if len(chunked_names) != 1:
        raise ValueError(
            f"MixFusion packed attention needs exactly one chunked stream to bind query/key/value, got {chunked_names}."
        )
    chunked_name = chunked_names[0]
    if chunked_name in streams:
        raise ValueError(f"Chunked stream {chunked_name!r} takes its Q/K/V positionally; do not also pass it in extra.")
    missing = [name for name in layout.names if name != chunked_name and name not in streams]
    if missing:
        raise ValueError(f"MixFusion packed attention is missing Q/K/V for streams {missing}.")
    all_streams = {**streams, chunked_name: (query, key, value)}

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

    if not attn.supports_multi_doc_packed_varlen() and _backend_supports_attention_mask(attn):
        # A varlen-capable backend takes the whole flat packing in one kernel.
        # A backend that only understands dense batches can still fuse
        # documents of the same image length by padding their text stream and
        # masking the padding, which keeps one forward per image shape instead
        # of one per distinct text length.
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
        # Varlen-capable backends get one packed kernel; every other backend
        # is served by equal-length bucketed dense attention that keeps the
        # same document isolation. Either way the result is the flat layout.
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
