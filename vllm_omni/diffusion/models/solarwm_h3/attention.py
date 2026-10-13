# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarWM-H3 segmented attention with raw, window-relative video KV.

Adapted from SolarWM's ``minimax_h3/sgf_attention.py`` (Apache-2.0).
Condition and audio rows are recomputed on each call. Only clean video K/V,
after QK normalization and before either positional transform, enter the cache.
Consequently a sliding window can give the same history new local MM-RoPE
positions without repeatedly rotating, or otherwise changing, stored tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import torch
import torch.nn.functional as F

from .camera import prope_qkv_separate
from .layout import H3PackedLayout, native_video_position_grid

Rotary = tuple[torch.Tensor, torch.Tensor]
RawKVEntry = tuple[int, torch.Tensor, torch.Tensor]


class SolarRawKVCache(Protocol):
    """Transaction boundary shared by direct and runner-owned raw KV caches."""

    def history(self, layer: int, chunk_index: int) -> Sequence[RawKVEntry]: ...

    def stage(self, layer: int, chunk_index: int, key: torch.Tensor, value: torch.Tensor) -> None: ...


@dataclass(frozen=True)
class SolarAttentionContext:
    """One five-latent-frame call, before Ulysses shards its packed rows.

    Camera slots are the anchor followed by the rollout's latent video frames.
    A successful clean forward stages each layer; the caller commits the whole
    chunk only after every layer has completed. No denoising probe stages KV.
    """

    layout: H3PackedLayout
    viewmats: torch.Tensor
    intrinsics: torch.Tensor
    chunk_index: int
    cache: SolarRawKVCache
    commit_cache: bool = False
    prefix_rotary: Rotary | None = None
    window_rotaries: dict[int, Rotary] | None = None


def prepare_solar_rotaries(control: SolarAttentionContext, rope: Callable[[torch.Tensor], torch.Tensor]):
    """Prepare the six bounded local windows with the loaded native H3 RoPE."""
    layout = control.layout
    prefix = int(layout.condition_indices.numel()) + int(layout.audio_indices.numel())

    def rotary(positions: torch.Tensor) -> Rotary:
        frequencies = rope(positions.unsqueeze(0))
        return frequencies.cos(), frequencies.sin()

    rotations = {}
    for frames in (5, 10, 15, 20, 25, 30):
        positions = native_video_position_grid(
            frames,
            layout.latent_height,
            layout.latent_width,
            patch_size=layout.patch_size,
            origin=float(layout.text_indices.numel()),
            device=layout.position_ids.device,
        )
        rotations[frames] = rotary(positions)
    return replace(
        control,
        prefix_rotary=rotary(layout.position_ids[:prefix]),
        window_rotaries=rotations,
    )


def apply_mm_rope(hidden: torch.Tensor, rotary: Rotary) -> torch.Tensor:
    """H3's half-split rotary arithmetic, retaining its unrotated suffix."""
    cos, sin = rotary
    rotary_dim = cos.shape[-1]
    if cos.shape != sin.shape or cos.shape[0] != hidden.shape[1] or rotary_dim > hidden.shape[-1]:
        raise ValueError("SolarWM MM-RoPE shape does not match its attention segment")
    if rotary_dim % 2:
        raise ValueError("SolarWM MM-RoPE requires an even rotary dimension")
    selected, passthrough = hidden[..., :rotary_dim], hidden[..., rotary_dim:]
    first, second = selected.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos = cos.to(device=hidden.device, dtype=hidden.dtype)[None, :, None, :]
    sin = sin.to(device=hidden.device, dtype=hidden.dtype)[None, :, None, :]
    return torch.cat((selected * cos + rotated * sin, passthrough), dim=-1).contiguous()


def _slice_rotary(rotary: Rotary, start: int, stop: int) -> Rotary:
    return rotary[0][start:stop], rotary[1][start:stop]


def _cat_rotary(first: Rotary, second: Rotary) -> Rotary:
    return torch.cat((first[0], second[0])), torch.cat((first[1], second[1]))


def _sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    # PyTorch dispatches SDPA to the available exact dense attention kernel.
    return F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0, is_causal=False
    ).transpose(1, 2)


def solar_rollout_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    control: SolarAttentionContext,
    layer: int,
    attend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = _sdpa,
) -> torch.Tensor:
    """Attend full packed rows with TP/Ulysses-local heads, in ``[B,S,H,D]``.

    The three query groups have different visibility: conditions see only
    conditions, audio sees conditions and audio, video sees that prefix plus
    the preceding five clean chunks and its own five-frame block. This avoids
    a quadratic dense attention mask while preserving the reference operation.
    """
    layout = control.layout
    condition_rows = int(layout.condition_indices.numel())
    prefix = condition_rows + int(layout.audio_indices.numel())
    rows_per_frame = int(layout.rows_per_video_frame)
    chunk_rows = 5 * rows_per_frame
    expected = prefix + chunk_rows
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("SolarWM attention requires equal [B,S,H,D] Q/K/V tensors")
    if query.shape[0] != 1 or query.shape[1] != expected or query.shape[-1] != 128:
        raise ValueError(f"SolarWM attention requires [1,{expected},H,128], got {tuple(query.shape)}")
    if condition_rows <= 0 or rows_per_frame <= 0:
        raise ValueError("SolarWM attention requires a nonempty condition prefix and spatial grid")
    if control.chunk_index < 0:
        raise ValueError("SolarWM chunk_index must be non-negative")
    if control.prefix_rotary is None or control.window_rotaries is None:
        raise ValueError("SolarWM rotaries must be prepared before attention")
    if control.commit_cache and torch.is_grad_enabled():
        raise RuntimeError("SolarWM raw KV commits require a no-grad clean forward")
    frames_needed = 1 + (control.chunk_index + 1) * 5
    if (
        control.viewmats.ndim != 4
        or control.intrinsics.ndim != 4
        or control.viewmats.shape[:2] != control.intrinsics.shape[:2]
        or control.viewmats.shape[0] != query.shape[0]
        or control.viewmats.shape[1] < frames_needed
    ):
        raise ValueError("SolarWM camera metadata must cover the anchor and current latent chunk")

    prefix_view = torch.eye(4, device=query.device, dtype=torch.float32).repeat(1, prefix, 1, 1)
    prefix_intrinsics = torch.eye(3, device=query.device, dtype=torch.float32).repeat(1, prefix, 1, 1)
    anchor_rows = layout.condition_video_indices.to(device=query.device, dtype=torch.long)
    prefix_view[:, anchor_rows] = control.viewmats[:, :1].to(device=query.device)
    prefix_intrinsics[:, anchor_rows] = control.intrinsics[:, :1].to(device=query.device)

    def camera_for_frames(start: int, stop: int):
        return (
            control.viewmats[:, 1 + start : 1 + stop].to(query.device).repeat_interleave(rows_per_frame, dim=1),
            control.intrinsics[:, 1 + start : 1 + stop].to(query.device).repeat_interleave(rows_per_frame, dim=1),
        )

    def calculate(q, k, v, qrotary, krotary, qcamera, kcamera):
        q = apply_mm_rope(q, qrotary)
        k = apply_mm_rope(k, krotary)
        q, k, v, output_basis = prope_qkv_separate(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            query_viewmats=qcamera[0],
            query_K=qcamera[1],
            key_viewmats=kcamera[0],
            key_K=kcamera[1],
        )
        output = attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return output_basis(output.transpose(1, 2)).transpose(1, 2)

    prefix_rotary = control.prefix_rotary
    condition_rotary = _slice_rotary(prefix_rotary, 0, condition_rows)
    condition_camera = prefix_view[:, :condition_rows], prefix_intrinsics[:, :condition_rows]
    prefix_camera = prefix_view, prefix_intrinsics
    outputs = [
        calculate(
            query[:, :condition_rows],
            key[:, :condition_rows],
            value[:, :condition_rows],
            condition_rotary,
            condition_rotary,
            condition_camera,
            condition_camera,
        )
    ]
    if prefix > condition_rows:
        outputs.append(
            calculate(
                query[:, condition_rows:prefix],
                key[:, :prefix],
                value[:, :prefix],
                _slice_rotary(prefix_rotary, condition_rows, prefix),
                prefix_rotary,
                (prefix_view[:, condition_rows:prefix], prefix_intrinsics[:, condition_rows:prefix]),
                prefix_camera,
            )
        )

    index = control.chunk_index
    first = max(0, index - 5)
    history = control.cache.history(layer, index)
    if [entry[0] for entry in history] != list(range(first, index)):
        raise RuntimeError(f"SolarWM raw KV history differs at layer={layer}, chunk={index}")
    expected_history_shape = (query.shape[0], chunk_rows, query.shape[2], query.shape[3])
    for _, history_key, history_value in history:
        if (
            history_key.shape != expected_history_shape
            or history_value.shape != expected_history_shape
            or history_key.device != key.device
            or history_value.device != value.device
            or history_key.dtype != key.dtype
            or history_value.dtype != value.dtype
        ):
            raise ValueError("SolarWM raw KV history geometry, dtype or device differs from the current chunk")
    history_keys = torch.cat([entry[1] for entry in history], dim=1) if history else key[:, :0]
    history_values = torch.cat([entry[2] for entry in history], dim=1) if history else value[:, :0]
    current_key, current_value = key[:, prefix:], value[:, prefix:]
    visible_key = torch.cat((key[:, :prefix], history_keys, current_key), dim=1)
    visible_value = torch.cat((value[:, :prefix], history_values, current_value), dim=1)
    local_frames = (index - first + 1) * 5
    rotary = control.window_rotaries[local_frames]
    current_rotary = _slice_rotary(rotary, (local_frames - 5) * rows_per_frame, local_frames * rows_per_frame)
    window_camera = camera_for_frames(first * 5, (index + 1) * 5)
    key_camera = tuple(torch.cat((p, w), dim=1) for p, w in zip(prefix_camera, window_camera, strict=True))
    outputs.append(
        calculate(
            query[:, prefix:],
            visible_key,
            visible_value,
            current_rotary,
            _cat_rotary(prefix_rotary, rotary),
            camera_for_frames(index * 5, (index + 1) * 5),
            key_camera,
        )
    )
    if control.commit_cache:
        control.cache.stage(layer, index, current_key, current_value)
    return torch.cat(outputs, dim=1)
