# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Camera arithmetic adapted from SolarWM (Apache-2.0) and PRoPE (MIT).
# Copyright 2026 SolarWM Contributors.
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
# The full third-party license notices are in the adjacent NOTICE file.
# Source: Junchao-cs/SolarWM@ce1da4e7705391eda8eeda6016c0fd3f614b975e
"""SolarWM-H3 camera preparation and the head-sliced projective transform."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .layout import FullLengthGeometry, full_length_geometry

WAN_FIXED_FX = 969.6969696969696 / (960.0 * 2.0)
WAN_FIXED_FY = 969.6969696969696 / (540.0 * 2.0)


def invert_se3(value: torch.Tensor) -> torch.Tensor:
    """Invert rigid transforms without a general matrix inverse."""
    output = torch.zeros_like(value)
    rotation = value[..., :3, :3].transpose(-1, -2)
    output[..., :3, :3] = rotation
    output[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation, value[..., :3, 3])
    output[..., 3, 3] = 1
    return output


@dataclass(frozen=True)
class CameraCondition:
    # These poses have NOT yet had logd4 applied. PRoPE applies it exactly once.
    viewmats: torch.Tensor
    intrinsics: torch.Tensor
    geometry: FullLengthGeometry


def prepare_camera(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    num_frames: int,
    height: int = 768,
    width: int = 1344,
    intrinsics_normalized: bool = False,
    device: torch.device | str | None = None,
) -> CameraCondition:
    """Convert absolute source cameras to anchor-plus-rollout camera rows.

    ``intrinsics`` may be a single 3x3 matrix or one per source frame, in source
    pixel coordinates unless explicitly marked normalized. The checkpoint uses
    fixed focal values inside PRoPE, but source intrinsics are still validated.
    Tail cameras repeat the last *latent-aligned* camera rather than the last
    physical frame. The separate image anchor is always first-frame relative.
    """
    geometry = full_length_geometry(num_frames)
    views = torch.as_tensor(c2w, dtype=torch.float32, device=device)
    if views.ndim == 3:
        views = views.unsqueeze(0)
    if views.ndim != 4 or views.shape[1:] != (num_frames, 4, 4) or not bool(torch.isfinite(views).all()):
        raise ValueError("camera c2w must be finite [T,4,4] or [B,T,4,4] with one pose per frame")
    bottom = views.new_tensor((0, 0, 0, 1)).expand_as(views[..., 3, :])
    rotation = views[..., :3, :3]
    identity = torch.eye(3, dtype=views.dtype, device=views.device).expand_as(rotation)
    if (
        not torch.allclose(views[..., 3, :], bottom, atol=1e-4, rtol=0)
        or not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-4, rtol=0)
        or not torch.allclose(torch.linalg.det(rotation), views.new_ones(views.shape[:2]), atol=1e-4, rtol=0)
    ):
        raise ValueError("camera c2w must contain proper rigid transforms with no scale or axis flip")

    K = torch.as_tensor(intrinsics, dtype=torch.float32, device=views.device)
    if K.ndim == 2:
        K = K[None, None]
    elif K.ndim == 3:
        K = K[None]
    if (
        K.ndim != 4
        or K.shape[-2:] != (3, 3)
        or K.shape[0] not in (1, views.shape[0])
        or K.shape[1] not in (1, num_frames)
        or not bool(torch.isfinite(K).all())
    ):
        raise ValueError("camera intrinsics must be finite [3,3], [T,3,3] or [B,T,3,3]")
    K = K.expand(views.shape[0], num_frames, -1, -1).clone()
    if height <= 0 or width <= 0:
        raise ValueError("source image dimensions must be positive")
    if not intrinsics_normalized:
        K[..., 0, :] /= width
        K[..., 1, :] /= height
    if (
        not bool((K[..., (0, 1), (0, 1)] > 0).all())
        or not torch.allclose(K[..., 2, :], K.new_tensor((0, 0, 1)).expand_as(K[..., 2, :]), atol=1e-5, rtol=0)
        or not bool((K[..., 0, 1].abs() <= 1e-5).all())
        or not bool((K[..., 1, 0].abs() <= 1e-5).all())
    ):
        raise ValueError("camera intrinsics must be no-skew pinhole matrices with positive focal lengths")
    relative = invert_se3(views) @ views[:, :1]
    relative[:, 0] = torch.eye(4, dtype=views.dtype, device=views.device)
    indices = torch.tensor((0, *geometry.camera_indices), dtype=torch.long, device=views.device)
    return CameraCondition(
        relative.index_select(1, indices).contiguous(), K.index_select(1, indices).contiguous(), geometry
    )


def logd4_relative_viewmats(viewmats: torch.Tensor) -> torch.Tensor:
    """Compress relative translation with zero-safe ``log(1+d)/(4*d)``."""
    if viewmats.shape[-2:] != (4, 4) or not viewmats.is_floating_point():
        raise ValueError("view matrices must be floating tensors ending in [4,4]")
    compute_dtype = torch.float64 if viewmats.dtype == torch.float64 else torch.float32
    translation = viewmats[..., :3, 3].to(compute_dtype)
    radius = torch.linalg.vector_norm(translation, dim=-1, keepdim=True)
    scale = torch.where(radius > 0, torch.log1p(radius) / (4 * radius.clamp_min(torch.finfo(compute_dtype).tiny)), 0)
    output = viewmats.clone()
    output[..., :3, 3] = (translation * scale).to(viewmats.dtype)
    return output


def _suffix_transforms(viewmats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fixed-focal checkpoint ignores principal point and source focal."""
    projection = torch.zeros_like(viewmats)
    projection[..., 0, 0] = WAN_FIXED_FX
    projection[..., 1, 1] = WAN_FIXED_FY
    projection[..., 2, 2] = projection[..., 3, 3] = 1
    inverse_K = torch.zeros_like(projection)
    # Invert the dtype-rounded focal values, as the released implementation
    # does. Inverting the Python constants first changes BF16 K/V by 1 ULP.
    inverse_K[..., 0, 0] = 1 / projection[..., 0, 0]
    inverse_K[..., 1, 1] = 1 / projection[..., 1, 1]
    inverse_K[..., 2, 2] = inverse_K[..., 3, 3] = 1
    # Retain the released contraction order, including BF16 intermediate
    # rounding; generic matmul can fuse that rounding differently.
    projection = torch.einsum("...ij,...jk->...ik", projection, viewmats)
    inverse = torch.einsum("...ij,...jk->...ik", invert_se3(viewmats), inverse_K)
    return projection.transpose(-1, -2), inverse, projection


def _project_suffix(features: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    suffix = features[..., 96:128]
    projected = torch.einsum("bsij,bhspj->bhspi", matrix.to(features), suffix.reshape(*suffix.shape[:-1], 8, 4))
    return torch.cat((features[..., :96], projected.flatten(-2)), -1)


def prope_qkv_separate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_viewmats: torch.Tensor,
    query_K: torch.Tensor,  # noqa: N803 - upstream camera argument name
    key_viewmats: torch.Tensor,
    key_K: torch.Tensor,  # noqa: N803 - upstream camera argument name
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Apply native MM-RoPE first, then call this on BHSD Q/K/V tensors.

    Cached K has already been RMS-normalized but has not had either positional
    transform applied. Cached V has not had PRoPE applied. The returned callable
    restores the query camera basis after attention and before output projection.
    """
    if (
        q.ndim != 4
        or k.ndim != 4
        or k.shape != v.shape
        or q.shape[:2] != k.shape[:2]
        or q.shape[-1] != 128
        or k.shape[-1] != 128
    ):
        raise ValueError("H3 PRoPE requires BHSD queries and matching BHSD keys/values, D=128")
    for features, views, K in ((q, query_viewmats, query_K), (k, key_viewmats, key_K)):
        if views.shape != (features.shape[0], features.shape[2], 4, 4) or K.shape != (
            features.shape[0],
            features.shape[2],
            3,
            3,
        ):
            raise ValueError("H3 PRoPE cameras must have one matrix per attention row")
        if views.device != features.device or K.device != features.device:
            raise ValueError("H3 PRoPE cameras and features must be on the same device")
    # Cast BEFORE logd4 to reproduce the released implementation's rounding.
    query_matrix, _, output_matrix = _suffix_transforms(logd4_relative_viewmats(query_viewmats.to(q.dtype)))
    _, kv_matrix, _ = _suffix_transforms(logd4_relative_viewmats(key_viewmats.to(k.dtype)))

    def output_basis(output: torch.Tensor) -> torch.Tensor:
        return _project_suffix(output, output_matrix)

    return (
        _project_suffix(q, query_matrix),
        _project_suffix(k, kv_matrix),
        _project_suffix(v, kv_matrix),
        output_basis,
    )
