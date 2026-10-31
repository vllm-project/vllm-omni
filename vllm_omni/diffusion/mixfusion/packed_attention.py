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

    joint_query = torch.cat(flat_queries, dim=0)
    joint_key = torch.cat(flat_keys, dim=0)
    joint_value = torch.cat(flat_values, dim=0)
    del flat_queries, flat_keys, flat_values

    # One packed forward: the attention layer routes varlen-capable backends
    # to a single varlen kernel and serves every other backend with
    # equal-length bucketed dense attention that keeps the same document
    # isolation. Either way the result is the flat joint layout.
    joint_hidden_states = attn.forward_packed(
        joint_query,
        joint_key,
        joint_value,
        cu_seqlens=plan.cu_seqlens_tensor,
        max_seqlen=plan.max_seq_len,
    )

    # Recover each stream's rows with one gather; request-level streams copy
    # into a zero buffer when their padded width exceeds the real length.
    outputs: dict[str, torch.Tensor] = {}
    for spec in layout.streams:
        gather, scatter = plan.output_indices[spec.name]
        rows = joint_hidden_states.index_select(0, gather)
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
