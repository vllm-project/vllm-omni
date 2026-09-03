# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched multi-request packing for MiniMax H3 step-wise execution.

Request mode forwards exactly one packed sequence per denoise step. Step mode
(continuous batching) may hold several requests at once, so this module
concatenates their packed layouts into a single sequence and rebuilds
``cu_seqlens`` with one document per request plus any nonempty alignment-padding
tail. Attention therefore never crosses a request boundary, while every request
keeps its own RoPE coordinates, token tags, and timesteps.
A batch of one is not rebuilt at all -- it delegates to the request-mode layout.

Row order is the request order, so the concatenated video velocity returned by
the DiT can be split back per request by row count alone -- which is exactly
what the runner does with ``StepRequestState.latents``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .denoise_loop import (
    MINIMAX_H3_AUDIO_ROW_WIDTH,
    MINIMAX_H3_VIDEO_ROW_WIDTH,
    MiniMaxH3DenoiseBranch,
)


def minimax_h3_batched_forward_kwargs(
    *,
    branches: Sequence[MiniMaxH3DenoiseBranch],
    video_rows: Sequence[torch.Tensor],
    audio_rows: Sequence[torch.Tensor],
    t_video: Sequence[float],
    t_audio: Sequence[float],
    imgvid_cond_timesteps: Sequence[float],
    audio_ref_cond_timesteps: Sequence[float],
) -> dict[str, Any]:
    """Build one DiT forward kwargs dict covering every request in the batch.

    A single request delegates to :meth:`MiniMaxH3DenoiseBranch.forward_kwargs`,
    so step mode and request mode are the same call rather than two layouts that
    happen to agree -- which also keeps ``video_token_layout`` (the sparse-
    attention hint) on the single-request step path.
    """
    if len(branches) == 1:
        return branches[0].forward_kwargs(
            video_rows=video_rows[0],
            audio_rows=audio_rows[0],
            t_video=t_video[0],
            t_audio=t_audio[0],
            imgvid_cond_timestep=imgvid_cond_timesteps[0],
            audio_ref_cond_timestep=audio_ref_cond_timesteps[0],
        )

    device = branches[0].device
    total_seq = sum(branch.seq_len for branch in branches)

    img_pos_parts: list[torch.Tensor] = []
    audio_pos_parts: list[torch.Tensor] = []
    text_pos_parts: list[torch.Tensor] = []
    update_parts: list[torch.Tensor] = []
    tag_parts: list[torch.Tensor] = []
    position_id_parts: list[torch.Tensor] = []
    text_embed_parts: list[torch.Tensor] = []
    cu_bounds: list[int] = [0]
    refiner_bounds: list[int] = [0]
    max_seqlen = 0
    refiner_max_seqlen = 0

    # Non-media rows (text and alignment padding) inherit their request's video
    # timestep, matching the single-request packed-sequence semantics.
    timesteps = torch.empty(total_seq, dtype=torch.float32, device=device)

    seq_offset = 0
    text_offset = 0
    for index, branch in enumerate(branches):
        img_pos = branch.img_pos_dev + seq_offset
        audio_pos = branch.audio_pos_dev + seq_offset
        img_pos_parts.append(img_pos)
        audio_pos_parts.append(audio_pos)
        text_pos_parts.append(branch.text_pos_dev + seq_offset)
        update_parts.append(branch.update_mask_dev)
        tag_parts.append(branch.token_tags_dev)
        position_id_parts.append(branch.img_position_ids_dev)
        text_embed_parts.append(branch.text_embeddings_dev)

        timesteps[seq_offset : seq_offset + branch.seq_len] = float(t_video[index])
        timesteps[img_pos[branch.update_mask_dev]] = float(t_video[index])
        timesteps[img_pos[~branch.update_mask_dev]] = float(imgvid_cond_timesteps[index])
        timesteps[audio_pos[branch.audio_update_mask_dev]] = float(t_audio[index])
        timesteps[audio_pos[~branch.audio_update_mask_dev]] = float(audio_ref_cond_timesteps[index])

        # Empty documents are omitted because not every varlen kernel accepts
        # repeated interior boundaries.
        real_end = seq_offset + branch.used_len
        padded_end = seq_offset + branch.seq_len
        cu_bounds.append(real_end)
        if padded_end != real_end:
            cu_bounds.append(padded_end)
        max_seqlen = max(max_seqlen, branch.used_len, branch.seq_len - branch.used_len)

        text_offset += branch.text_len
        refiner_bounds.append(text_offset)
        refiner_max_seqlen = max(refiner_max_seqlen, branch.text_len)
        seq_offset += branch.seq_len

    img_pos_all = torch.cat(img_pos_parts)
    audio_pos_all = torch.cat(audio_pos_parts)

    x = torch.zeros(1, total_seq, MINIMAX_H3_VIDEO_ROW_WIDTH, dtype=torch.float32, device=device)
    x[0].index_copy_(0, img_pos_all, torch.cat([rows.to(device=device, dtype=torch.float32) for rows in video_rows]))
    audio_x = torch.zeros(1, total_seq, MINIMAX_H3_AUDIO_ROW_WIDTH, dtype=torch.float32, device=device)
    audio_x[0].index_copy_(
        0,
        audio_pos_all,
        torch.cat([rows.to(device=device, dtype=torch.float32) for rows in audio_rows]),
    )

    unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)
    return {
        "x": x,
        "audio_x": audio_x,
        "img_position_ids": torch.cat(position_id_parts)[None],
        "update_mask": torch.cat(update_parts),
        "token_tags": torch.cat(tag_parts),
        "skip_mask_out_condition": False,
        "prompt_embeds": torch.cat(text_embed_parts),
        "img_pos_info": {"position_ids": img_pos_all},
        "audio_pos_info": {"position_ids": audio_pos_all},
        "text_pos_info": {"position_ids": torch.cat(text_pos_parts)},
        "img_pos_for_infer_output_info": {"position_ids": img_pos_all},
        "packed_seq_params": {
            "cu_seqlens_q": torch.tensor(cu_bounds, dtype=torch.int32, device=device),
            "max_seqlen_q": max_seqlen,
            # Attention needs the request count on the host to know its valid
            # rows are block-diagonal rather than a prefix. It describes the
            # whole forward, so the DiT reads it once here and applies it to the
            # token refiner too -- the refiner's text-only bounds cannot be told
            # apart from a single padded request on their own.
            "num_requests": len(branches),
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.tensor(refiner_bounds, dtype=torch.int32, device=device),
            "max_seqlen_q": refiner_max_seqlen,
        },
        "unique_timesteps": unique_timesteps,
        "inverse_indices": inverse_indices,
    }


__all__ = [
    "minimax_h3_batched_forward_kwargs",
]
