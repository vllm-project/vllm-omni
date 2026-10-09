# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed-sequence layouts for TaoMate-H3 streaming phases and the audio teacher.

A streaming phase is one future-free T2VA document ``[text | audio | video | pad]``
built with the shared MiniMax-H3 builder and then re-timed onto the session's
global RoPE timeline:

* the current prompt is right-aligned to the request's media time origin;
* video latents sit at their absolute positions (the release's 5/3-scaled
  (1, 4, 4, 4, 4) spacing continued across requests);
* audio latents sit at their absolute 40 Hz index.

The audio teacher runs audio-only documents ``[text | (reference audio) | audio | pad]``
whose builders are ported here because the shared builder always packs a video
target. Both return the layout dictionary that ``MiniMaxH3DenoiseBranch``
consumes, so the upstream DiT forward path is reused unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
    MINIMAX_H3_SEQ_ALIGN,
    _axis_from_sqrt_area,
    minimax_h3_packed_sequence,
)

from .geometry import StreamPhase, video_temporal_position, video_temporal_positions

_PATCH_H = 2
_PATCH_W = 2


def _pad_len(used: int, seq_len: int | None) -> int:
    if seq_len is None:
        return ((used + MINIMAX_H3_SEQ_ALIGN - 1) // MINIMAX_H3_SEQ_ALIGN) * MINIMAX_H3_SEQ_ALIGN
    if seq_len < used or seq_len % MINIMAX_H3_SEQ_ALIGN:
        raise ValueError(f"seq_len {seq_len} must cover {used} used rows and be a multiple of {MINIMAX_H3_SEQ_ALIGN}")
    return seq_len


def prompt_rope_start(*, text_len: int, media_time_origin: int, video_latent_offset: int) -> float:
    """Right-align the current prompt to the request's global media origin."""
    return float(media_time_origin + video_temporal_position(video_latent_offset) - text_len)


def taomate_phase_packed_layout(
    *,
    text_len: int,
    phase: StreamPhase,
    latent_h: int,
    latent_w: int,
    media_time_origin: int,
    video_latent_offset: int,
    audio_latent_offset: int,
    seq_len: int | None = None,
) -> dict[str, Any]:
    """One streaming phase document on the global timeline.

    ``video_latent_offset`` / ``audio_latent_offset`` are the session-global
    latent indices at which the current *request* starts; ``phase`` carries the
    request-local offsets of this chunk.
    """
    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=phase.video_latent_count,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=phase.audio_latent_count,
        audio_channel=2,
        include_keyframe_cond=False,
        seq_len=seq_len,
    )
    grid: torch.Tensor = packed["img_position_ids"]
    text_pos: torch.Tensor = packed["text_pos"].view(-1)
    img_pos: torch.Tensor = packed["img_pos"].view(-1)
    audio_pos: torch.Tensor = packed["audio_pos"].view(-1)
    frame_rows = (latent_h // _PATCH_H) * (latent_w // _PATCH_W)
    if int(img_pos.numel()) != phase.video_latent_count * frame_rows:
        raise RuntimeError("chunk video rows do not match the latent geometry")

    start = prompt_rope_start(
        text_len=text_len,
        media_time_origin=media_time_origin,
        video_latent_offset=video_latent_offset,
    )
    grid[text_pos, 0] = start + torch.arange(text_len, dtype=grid.dtype)
    positions = video_temporal_positions(video_latent_offset + phase.video_latent_start, phase.video_latent_count)
    video_times = torch.tensor([float(media_time_origin + position) for position in positions], dtype=grid.dtype)
    grid[img_pos, 0] = video_times.repeat_interleave(frame_rows)
    audio_by_channel = audio_pos.view(2, phase.audio_latent_count)
    audio_times = media_time_origin + torch.arange(
        audio_latent_offset + phase.audio_latent_start,
        audio_latent_offset + phase.audio_latent_stop,
        dtype=grid.dtype,
    )
    grid[audio_by_channel[0], 0] = audio_times
    grid[audio_by_channel[1], 0] = audio_times
    packed["taomate_times"] = {
        "text_time_start": start,
        "phase_video_time_start": float(video_times[0]),
        "phase_audio_time_start": float(audio_times[0]),
    }
    return packed


def _audio_only_common(
    *,
    text_len: int,
    latent_h: int,
    latent_w: int,
    ref_audio_t: int,
    audio_t: int,
    reference_time_start: float,
    target_time_start: float,
    audio_channel: int,
    seq_len: int | None,
) -> dict[str, Any]:
    if min(text_len, audio_t, latent_h, latent_w, audio_channel) <= 0 or ref_audio_t < 0:
        raise ValueError("audio-only layout dimensions must be positive")
    if latent_h % _PATCH_H or latent_w % _PATCH_W:
        raise ValueError("latent_h and latent_w must be divisible by the 2x2 patch")
    ref_rows = ref_audio_t * audio_channel
    target_rows = audio_t * audio_channel
    used = text_len + ref_rows + target_rows
    seq_len = _pad_len(used, seq_len)
    text_sl = slice(0, text_len)
    ref_sl = slice(text_len, text_len + ref_rows)
    target_sl = slice(ref_sl.stop, ref_sl.stop + target_rows)

    grid = torch.zeros(seq_len, 3, dtype=torch.float64)
    grid[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
    if ref_audio_t:
        grid[ref_sl, 0] = (float(reference_time_start) + torch.arange(ref_audio_t, dtype=torch.float64)).repeat(
            audio_channel
        )
    grid[target_sl, 0] = (float(target_time_start) + torch.arange(audio_t, dtype=torch.float64)).repeat(audio_channel)
    sqrt_area = np.sqrt(latent_h * latent_w)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    for audio_sl, temporal_rows in ((ref_sl, ref_audio_t), (target_sl, audio_t)):
        if temporal_rows:
            grid[audio_sl.start : audio_sl.start + temporal_rows, 2] = float(w_grid[0])
            grid[audio_sl.start + temporal_rows : audio_sl.stop, 2] = float(w_grid[-1])

    audio_pos = torch.arange(ref_sl.start, target_sl.stop)
    audio_update_mask = torch.zeros(ref_rows + target_rows, dtype=torch.bool)
    audio_update_mask[ref_rows:] = True
    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1
    token_tags[audio_pos] = 2
    ph, pw = latent_h // _PATCH_H, latent_w // _PATCH_W
    return {
        "seq_len": torch.tensor(seq_len),
        "img_pos": torch.empty(0, dtype=torch.long),
        "audio_pos": audio_pos,
        "audio_update_mask": audio_update_mask,
        "text_pos": torch.arange(text_len),
        "update_mask": torch.zeros(0, dtype=torch.bool),
        "img_position_ids": grid,
        "token_tags": token_tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
        # No video rows: the layout metadata is informational only.
        "latent_grid": torch.tensor([0, ph, pw], dtype=torch.int64),
        "video_row_start": torch.tensor(target_sl.stop, dtype=torch.int64),
    }


def taomate_audio_only_packed_layout(
    *,
    text_len: int,
    audio_t: int,
    latent_h: int,
    latent_w: int,
    audio_channel: int = 2,
    seq_len: int | None = None,
) -> dict[str, Any]:
    """The teacher's first-request text/audio document without video tokens."""
    return _audio_only_common(
        text_len=text_len,
        latent_h=latent_h,
        latent_w=latent_w,
        ref_audio_t=0,
        audio_t=audio_t,
        reference_time_start=float(text_len),
        target_time_start=float(text_len),
        audio_channel=audio_channel,
        seq_len=seq_len,
    )


def taomate_audio_only_frozen_prefix_packed_layout(
    *,
    text_len: int,
    ref_audio_t: int,
    audio_t: int,
    latent_h: int,
    latent_w: int,
    reference_time_start: int,
    target_time_start: int,
    audio_channel: int = 2,
    seq_len: int | None = None,
) -> dict[str, Any]:
    """The teacher's continuation document with a clean, read-only reference tail."""
    if ref_audio_t <= 0:
        raise ValueError("frozen-prefix layout requires ref_audio_t > 0")
    return _audio_only_common(
        text_len=text_len,
        latent_h=latent_h,
        latent_w=latent_w,
        ref_audio_t=ref_audio_t,
        audio_t=audio_t,
        reference_time_start=float(reference_time_start),
        target_time_start=float(target_time_start),
        audio_channel=audio_channel,
        seq_len=seq_len,
    )


__all__ = [
    "prompt_rope_start",
    "taomate_audio_only_frozen_prefix_packed_layout",
    "taomate_audio_only_packed_layout",
    "taomate_phase_packed_layout",
]
