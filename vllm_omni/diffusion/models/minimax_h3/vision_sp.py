# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in SP8 vision, preserving full replicated downstream output contracts.

Batch is deliberately disabled. Weights and module boundaries remain owned by
encoder.py, retaining its offloader hooks. This path has been validated on A5 fixed-six
inputs; A2 and other shapes still require their own bitwise runtime validation.
"""

import torch
import torch.nn.functional as F

from .vision_parallel import VisionAllToAll, build_vision_plan, gather_vision_outputs


def segment_attention(q, k, v):
    """Preserve the A5 short-segment SDPA head shape validated by the probe."""
    heads = q.shape[1]
    if q.shape[2] <= 1024 and heads < 8:
        inputs = [t.repeat(1, 8 // heads, 1, 1).transpose(1, 2).contiguous().transpose(1, 2) for t in (q, k, v)]
        return F.scaled_dot_product_attention(*inputs, dropout_p=0.0, is_causal=False)[:, :heads]
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def forward_sp8(model, pixels, grid, *, group):
    """Shard whole spatial merger groups, exchange heads, then gather all outputs."""
    lengths = tuple(h * w for t, h, w in grid.cpu().tolist() for _ in range(t))
    plan = build_vision_plan(lengths, 8, num_heads=model.config.num_heads, merge_size=model.spatial_merge_size)
    transport = VisionAllToAll(plan, group=group)
    ids = plan.rank_token_indices[transport.rank]
    select = torch.tensor(ids, dtype=torch.long, device=pixels.device)
    hidden = model.patch_embed(pixels.index_select(0, select))
    hidden = hidden + model.fast_pos_embed_interpolate(grid).index_select(0, select)
    rope = model.rot_pos_emb(grid).reshape(sum(lengths), -1)
    emb = torch.cat((rope, rope), dim=-1)
    cos, sin = emb.cos().index_select(0, select), emb.sin().index_select(0, select)
    deepstack = []
    # Invoke each original block so DLO pre/post hooks still stage its weights.
    for number, block in enumerate(model.blocks):
        hidden = block(hidden, cu_seqlens=None, position_embeddings=(cos, sin), sequence_parallel=transport)
        if number in model.deepstack_visual_indexes:
            index = model.deepstack_visual_indexes.index(number)
            deepstack.append(model.deepstack_merger_list[index](hidden))
    final = model.merger(hidden)
    return gather_vision_outputs((final, deepstack), plan, group=group)


def attention_sp8(attention, hidden, position_embeddings, transport, rotate):
    """Attention body called inside the original offloader-managed block."""
    qkv = attention.qkv(hidden)
    q, k, v = qkv.reshape(hidden.shape[0], 3, attention.num_heads, attention.head_dim).permute(1, 0, 2, 3).unbind(0)
    q, k = rotate(q, k, *position_embeddings)
    q, k, v = [transport.to_heads(tensor).transpose(0, 1).unsqueeze(0) for tensor in (q, k, v)]
    splits = [torch.split(tensor, transport.plan.segment_lengths, dim=2) for tensor in (q, k, v)]
    outputs = [segment_attention(qi, ki, vi) for qi, ki, vi in zip(*splits)]
    full = torch.cat(outputs, dim=2).squeeze(0).transpose(0, 1).contiguous()
    local = transport.to_tokens(full).reshape(hidden.shape[0], attention.num_heads * attention.head_dim)
    return attention.proj(local)
