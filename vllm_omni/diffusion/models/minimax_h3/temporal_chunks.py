# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small compatibility loop for the released MiniMax-H3 temporal decoder."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def decode_temporal_chunks(
    model: nn.Module,
    latent: torch.Tensor,
    callback: Callable[[torch.Tensor], None] | None,
) -> torch.Tensor:
    """Decode H3 temporal clips and synchronously publish committed frames.

    The released VAE exposes the temporal clip primitives but materializes the
    final tensor.  This is the same loop with the write point replaced by a
    frames-only callback.  ``callback=None`` retains the complete-output path.
    """
    if latent.ndim != 5 or not bool(getattr(model, "use_3d_conv", False)):
        raise ValueError("MiniMax-H3 temporal chunk decode requires a rank-5 3D latent")

    token_drop = int(model.token_drop)
    chunk_size = int(model.tokens_chunk_size)
    overlap_tokens = int(model.token_overlap)
    ratio_t = int(model.vae_ratio_t)
    pre_padding = int(model.frame_pre_padding)

    isolated_first = bool(model.isolated_first_frame and pre_padding == 0)
    isolated_last = bool(model.isolated_last_frame)
    z_head = latent[:, :, :1] if isolated_first else None
    z_tail = latent[:, :, -1:] if isolated_last else None
    start = int(isolated_first)
    stop = int(latent.shape[2]) - int(isolated_last)
    latent = latent[:, :, start:stop]

    pseudo_tokens = int(latent.shape[2]) + token_drop
    pad_tokens = (-pseudo_tokens) % chunk_size
    if pad_tokens:
        latent = torch.cat((latent, latent[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)), dim=2)
    num_chunks = (pseudo_tokens + pad_tokens) // chunk_size - int(token_drop > 0)
    if num_chunks <= 0:
        raise ValueError("MiniMax-H3 temporal chunk plan is empty")

    total_frames, pad_frames, output_frames = model._decode_temporal_output_frame_plan(
        latent, z_head, z_tail, num_chunks, pad_tokens
    )
    output_frames = int(output_frames)
    if output_frames <= 0:
        raise ValueError("MiniMax-H3 temporal chunk plan has no output frames")

    collected: list[torch.Tensor] = []
    overlap: torch.Tensor | None = None
    written = 0

    def emit(part: torch.Tensor) -> None:
        nonlocal written
        frames = int(part.shape[2])
        if frames <= 0 or written >= output_frames:
            return
        frames = min(frames, output_frames - written)
        part = part[:, :, :frames]
        if callback is None:
            collected.append(part)
        else:
            callback(part)
        written += frames

    chunk_frames = chunk_size * ratio_t
    split_count = int(token_drop > 0) + 1
    for index in range(num_chunks):
        begin = index * chunk_size
        end = begin + chunk_size + overlap_tokens
        clip = latent[:, :, begin:end]
        if index == 0 and z_head is not None:
            clip = torch.cat((z_head, clip), dim=2)
        if index == num_chunks - 1 and z_tail is not None:
            clip = torch.cat((clip, z_tail), dim=2)

        decoded = model._adaptive_decode(clip)
        for split in range(split_count):
            begin = split * chunk_frames
            end = min(begin + chunk_frames, int(decoded.shape[2]))
            part = decoded[:, :, begin:end]
            part = part[:, :, pre_padding:]
            if split == 0:
                if overlap is not None:
                    part = model.blend(overlap, part, int(model.frame_overlap), dim=-3)
                    overlap = None
                emit(part)
            else:
                overlap = part.contiguous()

        if index == num_chunks - 1:
            if overlap is not None:
                emit(overlap)
                overlap = None

    if written != output_frames:
        raise RuntimeError(f"MiniMax-H3 temporal decode emitted {written}/{output_frames} frames")
    if callback is not None:
        return latent.new_empty((0,))
    return torch.cat(collected, dim=2) if collected else latent.new_empty((0,))
