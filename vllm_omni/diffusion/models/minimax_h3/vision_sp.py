# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in SP8 vision, preserving full replicated downstream output contracts.

Batch is deliberately disabled. Weights and module boundaries remain owned by
encoder.py, retaining its offloader hooks. Expanding the supported input layouts
requires model-level bitwise validation on the target NPU.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .vision_parallel import VisionAllToAll, build_vision_plan, gather_vision_outputs

if TYPE_CHECKING:
    from .encoder import MiniMaxH3Qwen3VLVisionAttention, MiniMaxH3Qwen3VLVisionModel

# The short-segment path was validated with at least this many heads; below it
# the head shard is tiled up and sliced back. Tiled heads are independent copies,
# so the retained rows are unchanged.
_MIN_SHORT_SEGMENT_HEADS = 8


def segment_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Run SDPA per segment, padding short segments to the validated head count."""
    heads = q.shape[1]
    if q.shape[2] <= 1024 and heads < _MIN_SHORT_SEGMENT_HEADS:
        repeat = -(-_MIN_SHORT_SEGMENT_HEADS // heads)
        inputs = [t.repeat(1, repeat, 1, 1).transpose(1, 2).contiguous().transpose(1, 2) for t in (q, k, v)]
        return F.scaled_dot_product_attention(*inputs, dropout_p=0.0, is_causal=False)[:, :heads]
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def forward_sp8(
    model: MiniMaxH3Qwen3VLVisionModel,
    pixels: torch.Tensor,
    grid: torch.Tensor,
    *,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Shard whole spatial merger groups, exchange heads, then gather all outputs."""
    lengths = tuple(h * w for t, h, w in grid.cpu().tolist() for _ in range(t))
    plan = build_vision_plan(lengths, 8, num_heads=model.config.num_heads, merge_size=model.spatial_merge_size)
    transport = VisionAllToAll(plan, group=group)
    ids = plan.rank_token_indices[transport.rank]
    select = torch.tensor(ids, dtype=torch.long, device=pixels.device)
    hidden = model.patch_embed(pixels.index_select(0, select))
    position = model.fast_pos_embed_interpolate(grid, token_indices=ids)
    hidden = hidden + position
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


def attention_sp8(
    attention: MiniMaxH3Qwen3VLVisionAttention,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    transport: VisionAllToAll,
    rotate: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
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
