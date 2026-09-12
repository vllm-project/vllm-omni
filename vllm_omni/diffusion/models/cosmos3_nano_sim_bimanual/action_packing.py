# SPDX-License-Identifier: Apache-2.0
"""Action-specific token packing, mRoPE, and null-value helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


def interleave_action_vision_tokens(
    action_tokens: torch.Tensor,
    vision_tokens: torch.Tensor,
) -> torch.Tensor:
    """Pack per-frame hidden states as ``[action, vision]`` supertokens."""

    if action_tokens.ndim != 4 or vision_tokens.ndim != 4:
        raise ValueError(
            "Cosmos3-Nano-Sim-Bimanual interleaving expects action [B,T,A,D] and vision [B,T,P,D], "
            f"got {tuple(action_tokens.shape)} and {tuple(vision_tokens.shape)}"
        )
    if action_tokens.shape[:2] != vision_tokens.shape[:2] or action_tokens.shape[-1] != vision_tokens.shape[-1]:
        raise ValueError(
            "Cosmos3-Nano-Sim-Bimanual action/vision batch, frame, and hidden dimensions must match; "
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
        raise ValueError(f"Cosmos3-Nano-Sim-Bimanual packed tokens must have shape [B,S,D], got {tuple(tokens.shape)}")
    tokens_per_frame = action_tokens_per_frame + vision_tokens_per_frame
    expected = num_frames * tokens_per_frame
    if tokens.shape[1] != expected:
        raise ValueError(f"Cosmos3-Nano-Sim-Bimanual packed length must be {expected}, got {tokens.shape[1]}")
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
    action_tokens_per_frame: int = 4,
    null_action_frames: Iterable[int] = (),
) -> torch.Tensor:
    """Build reference-compatible mRoPE IDs in interleaved supertoken order."""

    if frame_start < 0 or num_frames <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError(
            "Cosmos3-Nano-Sim-Bimanual mRoPE dimensions must be positive and frame_start non-negative; "
            f"got start={frame_start}, frames={num_frames}, grid={grid_h}x{grid_w}"
        )
    if fps <= 0 or base_fps <= 0:
        raise ValueError(f"Cosmos3-Nano-Sim-Bimanual FPS values must be positive, got fps={fps}, base_fps={base_fps}")
    if action_tokens_per_frame <= 0:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Bimanual action_tokens_per_frame must be positive, got {action_tokens_per_frame}"
        )

    null_frames = {int(frame) for frame in null_action_frames}
    patch_count = grid_h * grid_w
    base_offset = float(text_temporal_offset + temporal_modality_margin)
    frame_stride = float(base_fps) / float(fps)
    position_parts: list[torch.Tensor] = []
    h_ids = torch.arange(grid_h, dtype=torch.float32).view(-1, 1).expand(-1, grid_w).flatten()
    w_ids = torch.arange(grid_w, dtype=torch.float32).view(1, -1).expand(grid_h, -1).flatten()

    for local_frame in range(num_frames):
        absolute_frame = frame_start + local_frame
        vision_t = base_offset + absolute_frame * frame_stride
        if absolute_frame in null_frames and local_frame == 0:
            action_t = torch.full((action_tokens_per_frame,), vision_t, dtype=torch.float32)
        else:
            substep = frame_stride / action_tokens_per_frame
            action_t = (
                vision_t - frame_stride + substep * torch.arange(1, action_tokens_per_frame + 1, dtype=torch.float32)
            )
        zeros = torch.zeros(action_tokens_per_frame, dtype=torch.float32)
        action_ids = torch.stack([action_t, zeros, zeros], dim=0)
        vision_ids = torch.stack(
            [torch.full((patch_count,), vision_t, dtype=torch.float32), h_ids, w_ids],
            dim=0,
        )
        position_parts.extend((action_ids, vision_ids))

    return torch.cat(position_parts, dim=1)


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
        raise ValueError(f"Cosmos3-Nano-Sim-Bimanual K/V must have shape [B,S,H,D], got {tuple(value.shape)}")
    if value.shape[1] != num_frames * tokens_per_frame:
        raise ValueError(
            "Cosmos3-Nano-Sim-Bimanual K/V length does not match frame geometry: "
            f"length={value.shape[1]}, frames={num_frames}, tokens_per_frame={tokens_per_frame}"
        )
    if not null_frame_indexes:
        return value
    result = value.clone()
    positions: list[int] = []
    for frame in null_frame_indexes:
        if frame < 0 or frame >= num_frames:
            raise ValueError(f"Cosmos3-Nano-Sim-Bimanual null action frame {frame} is outside [0, {num_frames})")
        start = frame * tokens_per_frame
        positions.extend(range(start, start + action_tokens_per_frame))
    result[:, positions] = 0
    return result


__all__ = [
    "build_interleaved_mrope_position_ids",
    "interleave_action_vision_tokens",
    "split_interleaved_action_vision_tokens",
    "zero_null_action_values",
]
