# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PRoPE: attention logits must depend only on the cameras' relative poses."""

import pytest
import torch

from vllm_omni.diffusion.layers.prope import PRoPECameras, apply_prope

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_FRAMES, _TOKENS, _HEADS, _HEAD_DIM = 2, 3, 2, 8


def _rigid(frames: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rotation, _ = torch.linalg.qr(torch.randn(frames, 3, 3, generator=generator, dtype=torch.float64))
    rotation = rotation * torch.linalg.det(rotation).sign().view(-1, 1, 1)
    transform = torch.eye(4, dtype=torch.float64).repeat(frames, 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = torch.randn(frames, 3, generator=generator, dtype=torch.float64)
    return transform


def _intrinsics(frames: int) -> torch.Tensor:
    # A focal length float32 cannot represent, so a float32 detour in the float64 math shows up.
    focal = 969.6969696969696 / 1920.0
    intrinsics = torch.tensor([[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float64)
    return intrinsics.repeat(frames, 1, 1)


def _features(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, _FRAMES * _TOKENS, _HEADS, _HEAD_DIM, generator=generator, dtype=torch.float64)


def test_logits_are_invariant_to_a_shared_change_of_world_frame() -> None:
    query, key = _features(1), _features(2)

    def logits(viewmats: torch.Tensor) -> torch.Tensor:
        cameras = PRoPECameras.from_views(viewmats, _intrinsics(_FRAMES), dtype=torch.float64)
        return torch.einsum("bshd,bthd->bhst", apply_prope(query, cameras.query), apply_prope(key, cameras.key))

    viewmats = _rigid(_FRAMES, seed=3)
    world = _rigid(1, seed=4)[0]
    moved = logits(viewmats @ world)
    torch.testing.assert_close(moved, logits(viewmats))
    assert not torch.allclose(moved, torch.einsum("bshd,bthd->bhst", query, key))


def test_output_transform_undoes_the_value_transform_of_the_same_camera() -> None:
    cameras = PRoPECameras.from_views(_rigid(_FRAMES, seed=5), _intrinsics(_FRAMES), dtype=torch.float64)
    value = _features(6)
    torch.testing.assert_close(apply_prope(apply_prope(value, cameras.key), cameras.output), value, rtol=0, atol=1e-12)
