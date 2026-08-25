# SPDX-License-Identifier: Apache-2.0
"""Pure-vision token packing and shared mRoPE for Transfer conditioning."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import compute_mrope_position_ids_vision


def build_shared_vision_mrope_position_ids(
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
) -> torch.Tensor:
    """Build the one vision-ID sequence shared by control and target RGB."""

    if frame_start < 0 or num_frames <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError(
            "Cosmos-Dreams-Transfer mRoPE dimensions must be positive and frame_start non-negative; "
            f"got start={frame_start}, frames={num_frames}, grid={grid_h}x{grid_w}."
        )
    if fps <= 0 or base_fps <= 0:
        raise ValueError(f"Cosmos-Dreams-Transfer FPS values must be positive, got fps={fps}, base_fps={base_fps}.")
    position_ids, _ = compute_mrope_position_ids_vision(
        num_frames,
        grid_h,
        grid_w,
        temporal_offset=text_temporal_offset + temporal_modality_margin,
        fps=fps,
        base_fps=base_fps,
        temporal_compression_factor=temporal_compression_factor,
        enable_fps_modulation=enable_fps_modulation,
        start_frame_offset=frame_start,
    )
    return position_ids


def pack_pure_vision_tokens(vision_tokens: torch.Tensor) -> torch.Tensor:
    """Flatten ``[B,T,P,D]`` vision frames without conditioning tokens."""

    if vision_tokens.ndim != 4:
        raise ValueError(
            f"Cosmos-Dreams-Transfer vision tokens must have shape [B,T,P,D], got {tuple(vision_tokens.shape)}."
        )
    return vision_tokens.flatten(1, 2)


def unpack_pure_vision_tokens(
    tokens: torch.Tensor,
    *,
    num_frames: int,
    vision_tokens_per_frame: int,
) -> torch.Tensor:
    """Restore ``[B,T,P,D]`` from a pure-vision packed sequence."""

    if tokens.ndim != 3:
        raise ValueError(f"Cosmos-Dreams-Transfer packed tokens must have shape [B,S,D], got {tuple(tokens.shape)}.")
    expected = num_frames * vision_tokens_per_frame
    if tokens.shape[1] != expected:
        raise ValueError(f"Cosmos-Dreams-Transfer packed length must be {expected}, got {tokens.shape[1]}.")
    return tokens.view(tokens.shape[0], num_frames, vision_tokens_per_frame, tokens.shape[-1])


__all__ = [
    "build_shared_vision_mrope_position_ids",
    "pack_pure_vision_tokens",
    "unpack_pure_vision_tokens",
]
