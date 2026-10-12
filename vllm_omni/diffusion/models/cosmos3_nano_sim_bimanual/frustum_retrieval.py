# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Camera-volume overlap selection using an 8x8x8 volume grid.

Geometry is CPU float64; returned frame IDs refer to the full chronological
archive. Selection never changes model actions or positional encodings.
"""

from __future__ import annotations

import math
from typing import Any

import torch


class FrustumHistorySelector:
    def __init__(self, geometry: dict[str, Any], frames: int, compression: int, fps: float) -> None:
        if not isinstance(geometry, dict):
            raise ValueError("frustum_geometry must be an object")
        required = {"c2w", "fps", "image_size", "near", "far"}
        if required - geometry.keys():
            raise ValueError(f"Missing frustum geometry fields: {sorted(required - geometry.keys())}")
        if float(geometry["fps"]) != fps:
            raise ValueError("Frustum geometry FPS must match inference")
        self.sizes = tuple(
            geometry.get(k, v) for k, v in [("sink_size", 8), ("retrieved_size", 32), ("recent_size", 32)]
        )
        if any(type(v) is not int or v < 0 for v in self.sizes):
            raise ValueError("Frustum history sizes must be nonnegative integers")
        size = geometry["image_size"]
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError("Frustum image_size must be positive [width, height]")
        self.width, self.height = size
        self.near, self.far = float(geometry["near"]), float(geometry["far"])
        if not math.isfinite(self.far) or not 0 < self.near < self.far:
            raise ValueError("Frustum depths must satisfy finite 0 < near < far")
        poses = torch.as_tensor(geometry["c2w"], dtype=torch.float64, device="cpu").detach().clone()
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) <= (frames - 1) * compression:
            raise ValueError("Frustum poses must be [RGB_frames,4,4] covering the complete rollout")
        indices = torch.arange(frames) * compression
        self.poses = poses[indices]
        if ("intrinsics" in geometry) == ("horizontal_fov_degrees" in geometry):
            raise ValueError("Supply exactly one of intrinsics and horizontal_fov_degrees")
        if "intrinsics" in geometry:
            k = torch.as_tensor(geometry["intrinsics"], dtype=torch.float64, device="cpu").detach().clone()
            if k.shape == (3, 3):
                k = k.expand(frames, -1, -1).clone()
            elif k.shape == (len(poses), 3, 3):
                k = k[indices]
            else:
                raise ValueError("Frustum intrinsics must be [3,3] or [RGB_frames,3,3]")
        else:
            fov = geometry["horizontal_fov_degrees"]
            if type(fov) not in (int, float) or not math.isfinite(fov) or not 0 < fov < 180:
                raise ValueError("Frustum horizontal FOV must be in (0,180)")
            focal = self.width / (2 * math.tan(math.radians(fov) / 2))
            k = (
                torch.tensor([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]], dtype=torch.float64)
                .expand(frames, -1, -1)
                .clone()
            )
        self.k = k
        rotation = self.poses[:, :3, :3]
        if (
            not torch.isfinite(self.poses).all()
            or not torch.isfinite(k).all()
            or not torch.allclose(
                self.poses[:, 3],
                torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(frames, -1),
                atol=1e-6,
                rtol=0,
            )
            or not torch.allclose(
                rotation.transpose(1, 2) @ rotation,
                torch.eye(3, dtype=torch.float64).expand(frames, -1, -1),
                atol=1e-4,
                rtol=0,
            )
            or not torch.allclose(
                torch.linalg.det(rotation), torch.ones(frames, dtype=torch.float64), atol=1e-4, rtol=0
            )
        ):
            raise ValueError("Frustum poses must be finite rigid OpenCV c2w transforms")
        if (
            not torch.allclose(
                k[:, 2], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(frames, -1), atol=1e-8, rtol=0
            )
            or not torch.allclose(k[:, 1, 0], torch.zeros(frames, dtype=torch.float64), atol=1e-8, rtol=0)
            or (k[:, 0, 0] <= 0).any()
            or (k[:, 1, 1] <= 0).any()
        ):
            raise ValueError("Frustum intrinsics must be upper triangular with positive focal lengths")
        q = (torch.arange(8, dtype=torch.float64) + 0.5) / 8
        near3 = torch.tensor(self.near, dtype=torch.float64).pow(3)
        depth_volume = torch.tensor(self.far, dtype=torch.float64).pow(3) - near3
        self.volumes = (self.width * self.height / (k[:, 0, 0] * k[:, 1, 1])) * depth_volume / 3
        u, v, z = torch.meshgrid(q * self.width, q * self.height, (near3 + q * depth_volume).pow(1 / 3), indexing="ij")
        pixels = torch.stack((u.flatten(), v.flatten(), torch.ones(u.numel(), dtype=torch.float64)), dim=-1)
        camera = (pixels @ torch.linalg.inv(k).transpose(1, 2)) * z.flatten()[None, :, None]
        self.points = camera @ rotation.transpose(1, 2) + self.poses[:, None, :3, 3]
        if not torch.isfinite(self.points).all() or not torch.isfinite(self.volumes).all() or (self.volumes <= 0).any():
            raise ValueError("Frustum volumes and samples must be finite and positive")

    def _inside(self, points: torch.Tensor, targets: list[int]) -> torch.Tensor:
        poses = self.poses[targets]
        camera = (points - poses[:, None, :3, 3]) @ poses[:, :3, :3]
        pixels = camera @ self.k[targets].transpose(1, 2)
        z = camera[..., 2]
        return (
            (
                (z >= self.near)
                & (z <= self.far)
                & (pixels[..., 0] >= 0)
                & (pixels[..., 0] <= self.width * z)
                & (pixels[..., 1] >= 0)
                & (pixels[..., 1] <= self.height * z)
            )
            .to(torch.float64)
            .mean(-1)
        )

    def scores(self, queries: list[int], history: list[int]) -> torch.Tensor:
        result = torch.zeros(len(history), dtype=torch.float64)
        for begin in range(0, len(history), 64):
            ids = history[begin : begin + 64]
            volume = self.volumes[ids]
            total = torch.zeros(len(ids), dtype=torch.float64)
            for query in queries:
                qvolume = self.volumes[query]
                intersection = (
                    self._inside(self.points[query : query + 1], ids) * qvolume
                    + self._inside(self.points[ids], [query]) * volume
                ) / 2
                intersection = torch.minimum(intersection, torch.minimum(qvolume, volume))
                total += (intersection / (qvolume + volume - intersection)).clamp(0, 1)
            result[begin : begin + len(ids)] = total / len(queries)
        return result

    def select(self, start: int, end: int) -> list[int]:
        if not 0 <= start < end <= len(self.poses):
            raise ValueError("Frustum chunk must be within the supplied trajectory")
        sinks, retrieved, recent = self.sizes
        sink_end = min(sinks, start)
        recent_start = max(sink_end, start - recent)
        candidates = list(range(sink_end, recent_start))
        if len(candidates) > retrieved:
            ranked = (
                sorted(
                    zip(candidates, self.scores(list(range(start, end)), candidates).tolist(), strict=True),
                    key=lambda item: (item[1], item[0]),
                    reverse=True,
                )
                if retrieved
                else []
            )
            candidates = [i for i, _ in ranked[:retrieved]]
        return sorted([*range(sink_end), *candidates, *range(recent_start, start)])


def resolve_frustum_geometry(
    geometry: dict[str, Any],
    *,
    fps: float,
    num_frames: int,
    camera_poses: Any = None,
    action: Any = None,
    action_space: str | None = None,
    domain_id: Any = None,
    normalization_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve pose-source configuration without modifying model conditioning."""
    import numpy as np

    if not isinstance(geometry, dict):
        raise ValueError("frustum_geometry must be an object")
    result = dict(geometry)
    source = result.pop("pose_source", None)
    if source is None:
        return result
    if "c2w" in result:
        raise ValueError("Specify c2w or pose_source, not both")
    if source == "camera_batch" and "fps" in result:
        raise ValueError("Camera-batch FPS must not be overridden in geometry")
    if "fps" in result and float(result["fps"]) != fps:
        raise ValueError("Frustum geometry FPS must match inference")
    result["fps"] = fps
    if source == "camera_batch":
        if camera_poses is None:
            raise ValueError("camera_batch requires RGB-aligned absolute camera poses")
        poses = torch.as_tensor(camera_poses, device="cpu", dtype=torch.float64)
        if poses.shape != (num_frames, 4, 4):
            raise ValueError("camera_batch poses must match the inference RGB-frame count")
        result["c2w"] = poses.tolist()
        return result
    if source != "agibot_unified_v1_chunk16":
        raise ValueError(f"Unsupported frustum pose_source: {source!r}")
    if action_space != "normalized_unified_v1" or action is None or normalization_stats is None:
        raise ValueError("Action-derived poses require normalized_unified_v1 actions and checkpoint statistics")
    rows = torch.as_tensor(action, dtype=torch.float32, device="cpu")
    if rows.shape != (num_frames - 1, 59) or not torch.isfinite(rows).all():
        raise ValueError("Camera recovery requires finite [num_frames-1, 59] actions")
    if domain_id is None:
        raise ValueError("Action-derived poses require explicit AgiBot/camera domains")
    domains = torch.as_tensor(domain_id)
    if domains.ndim > 1 or (domains.ndim == 1 and len(domains) != len(rows)):
        raise ValueError("Expected a scalar domain or one domain per action row")
    if domains.dtype == torch.bool or not ((domains == 15) | (domains == 2)).all():
        raise ValueError("Camera recovery supports only AgiBot/camera domains")
    stats = normalization_stats["global"]
    lo = torch.tensor(stats["q01"], dtype=torch.float32)[:9]
    hi = torch.tensor(stats["q99"], dtype=torch.float32)[:9]
    if lo.shape != (9,) or hi.shape != (9,) or not torch.isfinite(lo).all() or not torch.isfinite(hi).all():
        raise ValueError("Invalid camera normalization statistics")
    raw = (
        torch.sinh(rows[:, :9] * rows.new_tensor(math.asinh(1.0))) * ((hi - lo).clamp(min=1e-8) / 2) + (hi + lo) / 2
    ).numpy()
    if not np.isfinite(raw).all():
        raise ValueError("Camera inverse normalization produced nonfinite values")
    # Column-based rot6d, projected onto proper rotations before composition.
    first, second = raw[:, 3:6], raw[:, 6:9]
    matrices = np.stack((first, second, np.cross(first, second)), axis=-1)
    u, _, vt = np.linalg.svd(matrices)
    reflected = np.linalg.det(u @ vt) < 0
    u[reflected, :, -1] *= -1
    rotations = u @ vt
    poses = np.repeat(np.eye(4)[None], num_frames, axis=0)
    for index in range(len(rows)):
        delta = np.eye(4)
        delta[:3, :3] = rotations[index]
        delta[:3, 3] = raw[index, :3]
        poses[index + 1] = poses[16 * (index // 16)] @ delta
    result["c2w"] = poses.tolist()
    return result
