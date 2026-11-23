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

The audio teacher uses the shared Ref2VA builder with zero video latents for
``[text | (reference audio) | audio | pad]`` documents. Only the streaming
timeline is applied here; structural fields share the MiniMax-H3 builder.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
    MINIMAX_H3_SEQ_ALIGN,
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)

from .geometry import StreamPhase, video_temporal_position, video_temporal_positions

_PATCH_H = 2
_PATCH_W = 2


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
    used = text_len + (ref_audio_t + audio_t) * audio_channel
    if seq_len is not None and (seq_len < used or seq_len % MINIMAX_H3_SEQ_ALIGN):
        raise ValueError(f"seq_len {seq_len} must cover {used} used rows and be a multiple of {MINIMAX_H3_SEQ_ALIGN}")
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=text_len,
        latent_t=0,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        audio_channel=audio_channel,
        ref_blocks=({"kind": "audio", "ref_audio_t": ref_audio_t},) if ref_audio_t else (),
        seq_len=seq_len,
    )
    cursor = text_len
    for length, start in ((ref_audio_t, reference_time_start), (audio_t, target_time_start)):
        rows = length * audio_channel
        packed["img_position_ids"][cursor : cursor + rows, 0] = (
            float(start) + torch.arange(length, dtype=torch.float64)
        ).repeat(audio_channel)
        cursor += rows
    return packed


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
