# SPDX-License-Identifier: Apache-2.0
"""Pure packing, mRoPE, and hashing helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence

import torch

from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
    compute_mrope_position_ids_action,
    compute_mrope_position_ids_vision,
)


def iter_ar_chunk_ranges(start_frame: int, num_frames: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield the training-aligned partition ``[1, C, C, ...]``."""
    if start_frame < 0 or num_frames < 0 or start_frame > num_frames:
        raise ValueError(f"Invalid Cosmos-Dreams frame range [{start_frame}, {num_frames})")
    if chunk_size <= 0:
        raise ValueError(f"Cosmos-Dreams chunk_size must be positive, got {chunk_size}")
    frame = start_frame
    while frame < num_frames:
        if frame == 0:
            chunk_end = 1
        else:
            chunk_end = 1 + ((frame - 1) // chunk_size + 1) * chunk_size
        chunk_end = min(chunk_end, num_frames)
        yield frame, chunk_end
        frame = chunk_end


def iter_clean_commit_frames(
    chunk_start: int,
    chunk_end: int,
    *,
    target_frame: int,
    terminal_request: bool,
) -> Iterator[tuple[int, int]]:
    """Yield ``(local, absolute)`` clean-refresh frames in commit order.

    The globally final frame has no downstream reader and is omitted only for
    a terminal request. Every other frame is refreshed individually so later
    frames in the same denoised chunk see clean, committed history.
    """
    if chunk_start < 0 or chunk_end <= chunk_start or target_frame < chunk_end:
        raise ValueError(
            f"Invalid Cosmos-Dreams clean-commit range: chunk=[{chunk_start}, {chunk_end}), target={target_frame}"
        )
    for local_idx, frame_idx in enumerate(range(chunk_start, chunk_end)):
        if terminal_request and frame_idx == target_frame - 1:
            continue
        yield local_idx, frame_idx


def interleave_action_vision_tokens(
    action_tokens: torch.Tensor,
    vision_tokens: torch.Tensor,
) -> torch.Tensor:
    """Pack per-frame hidden states as ``[action, vision]`` supertokens.

    Args:
        action_tokens: ``[B, T, A, D]``.
        vision_tokens: ``[B, T, P, D]``.
    """
    if action_tokens.ndim != 4 or vision_tokens.ndim != 4:
        raise ValueError(
            "Cosmos-Dreams interleaving expects action [B,T,A,D] and vision [B,T,P,D], "
            f"got {tuple(action_tokens.shape)} and {tuple(vision_tokens.shape)}"
        )
    if action_tokens.shape[:2] != vision_tokens.shape[:2] or action_tokens.shape[-1] != vision_tokens.shape[-1]:
        raise ValueError(
            "Cosmos-Dreams action/vision batch, frame, and hidden dimensions must match; "
            f"got {tuple(action_tokens.shape)} and {tuple(vision_tokens.shape)}"
        )
    return torch.cat([action_tokens, vision_tokens], dim=2).flatten(1, 2)


def split_interleaved_action_vision_tokens(
    tokens: torch.Tensor,
    *,
    num_frames: int,
    action_tokens_per_frame: int,
    vision_tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`interleave_action_vision_tokens`."""
    if tokens.ndim != 3:
        raise ValueError(f"Cosmos-Dreams packed tokens must have shape [B,S,D], got {tuple(tokens.shape)}")
    tokens_per_frame = action_tokens_per_frame + vision_tokens_per_frame
    expected = num_frames * tokens_per_frame
    if tokens.shape[1] != expected:
        raise ValueError(f"Cosmos-Dreams packed length must be {expected}, got {tokens.shape[1]}")
    framed = tokens.view(tokens.shape[0], num_frames, tokens_per_frame, tokens.shape[-1])
    return framed[:, :, :action_tokens_per_frame], framed[:, :, action_tokens_per_frame:]


def build_interleaved_mrope_position_ids(
    *,
    frame_start: int,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    text_temporal_offset: int,
    temporal_modality_margin: int,
    fps: float,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    enable_fps_modulation: bool = True,
    action_tokens_per_frame: int = 4,
    null_action_frames: Iterable[int] = (),
) -> torch.Tensor:
    """Build reference-compatible mRoPE IDs in interleaved supertoken order.

    Real action sub-tokens span the interval ending at their associated latent
    frame. The first null-action supertoken in an AR unit is co-located with
    its vision frame. Later all-null supertokens retain the architectural real-
    action IDs used by the reference packer, even though their values are zero.
    """
    if frame_start < 0 or num_frames <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError(
            "Cosmos-Dreams mRoPE dimensions must be positive and frame_start non-negative; "
            f"got start={frame_start}, frames={num_frames}, grid={grid_h}x{grid_w}"
        )
    if fps <= 0 or base_fps <= 0:
        raise ValueError(f"Cosmos-Dreams FPS values must be positive, got fps={fps}, base_fps={base_fps}")
    if action_tokens_per_frame <= 0:
        raise ValueError(f"Cosmos-Dreams action_tokens_per_frame must be positive, got {action_tokens_per_frame}")

    null_frames = {int(frame) for frame in null_action_frames}
    patch_count = grid_h * grid_w
    base_offset = float(text_temporal_offset + temporal_modality_margin)
    vision_ids, _ = compute_mrope_position_ids_vision(
        grid_t=num_frames,
        grid_h=grid_h,
        grid_w=grid_w,
        temporal_offset=base_offset,
        fps=fps,
        base_fps=base_fps,
        temporal_compression_factor=temporal_compression_factor,
        base_temporal_compression_factor=temporal_compression_factor,
        enable_fps_modulation=enable_fps_modulation,
        start_frame_offset=frame_start,
    )
    # Each action token represents one pixel-rate step ending at its latent
    # frame. Flattening the frame/action axes gives the base helper the exact
    # sequence it expects; Dreams only owns the final per-frame interleaving.
    action_start_offset = frame_start * action_tokens_per_frame - action_tokens_per_frame + 1
    action_ids, _ = compute_mrope_position_ids_action(
        grid_t=num_frames * action_tokens_per_frame,
        temporal_offset=base_offset,
        action_fps=fps,
        base_fps=base_fps,
        base_temporal_compression_factor=temporal_compression_factor,
        enable_fps_modulation=enable_fps_modulation,
        start_frame_offset=action_start_offset,
    )

    action_ids = action_ids.view(3, num_frames, action_tokens_per_frame)
    vision_ids = vision_ids.view(3, num_frames, patch_count)
    if frame_start in null_frames:
        # The first null-action supertoken in an AR unit is colocated with its
        # vision frame; later null frames retain the architectural action IDs.
        action_ids[0, 0].fill_(vision_ids[0, 0, 0])
    return torch.cat([action_ids, vision_ids], dim=2).flatten(1, 2)


def zero_null_action_values(
    value: torch.Tensor,
    *,
    num_frames: int,
    tokens_per_frame: int,
    action_tokens_per_frame: int,
    null_frame_indexes: Sequence[int],
) -> torch.Tensor:
    """Zero V (not K) for null action slots before persistent storage."""
    if value.ndim != 4:
        raise ValueError(f"Cosmos-Dreams K/V must have shape [B,S,H,D], got {tuple(value.shape)}")
    if value.shape[1] != num_frames * tokens_per_frame:
        raise ValueError(
            "Cosmos-Dreams K/V length does not match frame geometry: "
            f"length={value.shape[1]}, frames={num_frames}, tokens_per_frame={tokens_per_frame}"
        )
    if not null_frame_indexes:
        return value
    result = value.clone()
    positions: list[int] = []
    for frame in null_frame_indexes:
        if frame < 0 or frame >= num_frames:
            raise ValueError(f"Cosmos-Dreams null action frame {frame} is outside [0, {num_frames})")
        start = frame * tokens_per_frame
        positions.extend(range(start, start + action_tokens_per_frame))
    result[:, positions] = 0
    return result


def prompt_token_hash(token_ids: Sequence[int] | torch.Tensor) -> str:
    """Stable SHA-256 over prompt token IDs, independent of tensor dtype."""
    if isinstance(token_ids, torch.Tensor):
        values = [int(value) for value in token_ids.detach().cpu().reshape(-1).tolist()]
    else:
        values = [int(value) for value in token_ids]
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
