# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed attention entry for MixFusion chunked layouts.

The MixFusion layout stacks image chunks on a local batch dimension while
text stays request-level, so the two streams cannot be concatenated by the
model before attention (their batch dimensions differ). This module is the
attention-layer boundary that turns the layout into the standard flat
packed row: it gathers each request back into ``text + full image``,
runs one packed forward, and scatters the result back to the two stream
layouts -- the same pre/post boundary shape the sequence-parallel
strategies use around their all-to-all.

Models opt in purely through ``AttentionMetadata``: image Q/K/V are the
``query``/``key``/``value`` arguments, text Q/K/V ride the ``joint_query``/
``joint_key``/``joint_value`` channel (the SP-joint convention), and the
per-step packing plan travels in ``extra[MIXFUSION_PLAN_KEY]``. The output
is a ``(image, text)`` tuple in the two stream layouts.
"""

from __future__ import annotations

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.mixfusion.layout import MixFusionPlan

MIXFUSION_PLAN_KEY = "mixfusion_plan"


def run_mixfusion_packed_attention(
    attn,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run joint attention over chunked image and request-level text streams.

    ``query``/``key``/``value`` hold the image chunks as
    ``[num_chunks, chunk_size, heads, head_dim]``; the text streams ride
    ``attn_metadata.joint_query``/``joint_key``/``joint_value`` as
    ``[num_requests, seq_len_txt, heads, head_dim]``; the packing layout is
    ``attn_metadata.extra[MIXFUSION_PLAN_KEY]``.

    Each request is recovered to ``text + full image`` right before the
    kernel so the attention semantics match independent dense execution.
    Returns ``(image, text)`` attention outputs in the input stream layouts
    (chunk-stacked and request-level respectively).
    """
    plan: MixFusionPlan = attn_metadata.extra[MIXFUSION_PLAN_KEY]
    txt_query = attn_metadata.joint_query
    txt_key = attn_metadata.joint_key
    txt_value = attn_metadata.joint_value
    if txt_query is None or txt_key is None or txt_value is None:
        raise ValueError(
            "MixFusion packed attention requires the text streams in attn_metadata.joint_query/joint_key/joint_value"
        )

    num_query_heads = query.shape[2]
    num_kv_heads = key.shape[2]
    head_dim = query.shape[3]
    num_requests = txt_query.shape[0]

    # Recover each request to ``text + full image``: text rows in request
    # order, image rows from the chunk range the plan assigned to it.
    flat_queries: list[torch.Tensor] = []
    flat_keys: list[torch.Tensor] = []
    flat_values: list[torch.Tensor] = []
    for req_idx, (chunk_start, chunk_end) in enumerate(plan.request_chunk_ranges):
        txt_len = int(plan.real_txt_lens[req_idx])
        req_img_query = query[chunk_start:chunk_end].reshape(-1, num_query_heads, head_dim)
        req_img_key = key[chunk_start:chunk_end].reshape(-1, num_kv_heads, head_dim)
        req_img_value = value[chunk_start:chunk_end].reshape(-1, num_kv_heads, head_dim)
        flat_queries.append(torch.cat([txt_query[req_idx, :txt_len], req_img_query], dim=0))
        flat_keys.append(torch.cat([txt_key[req_idx, :txt_len], req_img_key], dim=0))
        flat_values.append(torch.cat([txt_value[req_idx, :txt_len], req_img_value], dim=0))

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

    # Recover rows with one gather each: text rows in request order, image
    # rows in chunk order. No mask -> the text gather is already the padded
    # layout, just reshape; otherwise copy into a zero buffer at the
    # precomputed padded positions.
    txt_rows = joint_hidden_states.index_select(0, plan.txt_out_idx)
    img_rows = joint_hidden_states.index_select(0, plan.img_out_idx)
    if plan.txt_pad_idx is None:
        txt_attn_output = txt_rows.view(num_requests, plan.seq_len_txt, num_query_heads, head_dim)
    else:
        txt_buffer = txt_rows.new_zeros(num_requests * plan.seq_len_txt, num_query_heads, head_dim)
        txt_buffer.index_copy_(0, plan.txt_pad_idx, txt_rows)
        txt_attn_output = txt_buffer.view(num_requests, plan.seq_len_txt, num_query_heads, head_dim)
    img_attn_output = img_rows.view(query.shape[0], plan.chunk_size, num_query_heads, head_dim)

    return img_attn_output, txt_attn_output
