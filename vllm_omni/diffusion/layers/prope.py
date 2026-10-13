# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Projective positional encoding (PRoPE) for camera-conditioned attention.

With ``P = lift(K) @ viewmat`` the ``image <- world`` projection of a token's
camera, PRoPE multiplies every 4-channel group of a query by ``P^T``, of a key
or value by ``P^-1``, and of the attention output by ``P``. Attention logits
then depend only on the relative transform between the query and key cameras.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PRoPECameras:
    """Per-frame PRoPE transforms, each ``[frames, 4, 4]``.

    ``query`` is ``P^T``; ``key`` is ``P^-1`` and applies to keys and values;
    ``output`` is ``P``.
    """

    query: torch.Tensor
    key: torch.Tensor
    output: torch.Tensor

    @classmethod
    def from_views(cls, viewmats: torch.Tensor, intrinsics: torch.Tensor, *, dtype: torch.dtype) -> PRoPECameras:
        """Build the transforms from ``camera <- world`` view matrices and normalized intrinsics.

        Args:
            viewmats (torch.Tensor): rigid ``[frames, 4, 4]`` world-to-camera matrices.
            intrinsics (torch.Tensor): ``[frames, 3, 3]`` normalized intrinsics. Only
                the focal lengths enter ``P``; the principal point is dropped.
            dtype (torch.dtype): dtype the transforms are computed and stored in.

        Returns:
            PRoPECameras: the per-frame query, key/value and output transforms.
        """
        if viewmats.ndim != 3 or viewmats.shape[1:] != (4, 4):
            raise ValueError(f"viewmats must be [frames, 4, 4], got {tuple(viewmats.shape)}.")
        if intrinsics.shape != (viewmats.shape[0], 3, 3):
            raise ValueError(f"intrinsics must be [{viewmats.shape[0]}, 3, 3], got {tuple(intrinsics.shape)}.")
        viewmats = viewmats.to(dtype)
        intrinsics = intrinsics.to(dtype)
        focal = torch.zeros_like(intrinsics)
        focal[:, 0, 0] = intrinsics[:, 0, 0]
        focal[:, 1, 1] = intrinsics[:, 1, 1]
        focal[:, 2, 2] = 1.0
        inverse_focal = torch.zeros_like(focal)
        inverse_focal[:, 0, 0] = 1.0 / focal[:, 0, 0]
        inverse_focal[:, 1, 1] = 1.0 / focal[:, 1, 1]
        inverse_focal[:, 2, 2] = 1.0
        projection = torch.einsum("...ij,...jk->...ik", _lift(focal), viewmats)
        inverse = torch.einsum("...ij,...jk->...ik", _invert_rigid(viewmats), _lift(inverse_focal))
        return cls(query=projection.transpose(-1, -2), key=inverse, output=projection)

    def __len__(self) -> int:
        return self.query.shape[0]

    def frames(self, start: int, stop: int) -> PRoPECameras:
        """Return the transforms of latent frames ``[start, stop)``."""
        if not 0 <= start < stop <= len(self):
            raise ValueError(f"camera frames [{start}, {stop}) are outside the {len(self)} available frames.")
        return PRoPECameras(query=self.query[start:stop], key=self.key[start:stop], output=self.output[start:stop])

    def to(self, device: torch.device) -> PRoPECameras:
        return PRoPECameras(query=self.query.to(device), key=self.key.to(device), output=self.output.to(device))


def apply_prope(features: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
    """Multiply every 4-channel group of each token by its frame's 4x4 transform.

    Args:
        features (torch.Tensor): ``[batch, frames * tokens_per_frame, heads, head_dim]``.
        transforms (torch.Tensor): ``[frames, 4, 4]``.

    Returns:
        torch.Tensor: the transformed features, same shape as ``features``.
    """
    batch, tokens, heads, head_dim = features.shape
    frames = transforms.shape[0]
    if tokens % frames or head_dim % 4:
        raise ValueError(f"cannot split {tokens} tokens over {frames} frames into 4-channel groups of {head_dim}.")
    grouped = features.reshape(batch, frames, tokens // frames, heads, head_dim // 4, 4)
    return torch.einsum("fij,bfthgj->bfthgi", transforms.to(features.dtype), grouped).reshape(features.shape)


def _lift(matrices: torch.Tensor) -> torch.Tensor:
    lifted = torch.zeros(matrices.shape[:-2] + (4, 4), device=matrices.device)
    lifted[..., :3, :3] = matrices
    lifted[..., 3, 3] = 1.0
    return lifted.to(matrices.dtype)


def _invert_rigid(transforms: torch.Tensor) -> torch.Tensor:
    rotation_inverse = transforms[..., :3, :3].transpose(-1, -2)
    inverse = torch.zeros_like(transforms)
    inverse[..., :3, :3] = rotation_inverse
    inverse[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_inverse, transforms[..., :3, 3])
    inverse[..., 3, 3] = 1.0
    return inverse
