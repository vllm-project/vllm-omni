# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for latent-mask serialization helpers and the temporal-mask node."""

import json

import pytest
import torch
from comfyui_vllm_omni.nodes import VLLMOmniMiniMaxH3TemporalMask
from comfyui_vllm_omni.utils.latent_mask import scalar_mask_to_json, video_mask_to_json

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_scalar_mask_to_json():
    assert scalar_mask_to_json(0.5) == "0.5"
    assert scalar_mask_to_json(1.0) == "1.0"
    with pytest.raises(ValueError):
        scalar_mask_to_json(1.5)
    with pytest.raises(ValueError):
        scalar_mask_to_json(-0.1)


def test_video_mask_to_json_downsamples_spatial_mask():
    # A 2D mask keeps its rank; only the spatial axes shrink by the VAE stride.
    grid = json.loads(video_mask_to_json(torch.full((120, 160), 0.5)))
    assert len(grid) == 8  # ceil(120 / 16)
    assert len(grid[0]) == 10
    assert grid[0][0] == 0.5


def test_video_mask_to_json_keeps_frame_axis():
    # A 3D mask keeps one slice per frame; the server maps frames to latents.
    grid = json.loads(video_mask_to_json(torch.full((22, 120, 160), 0.5)))
    assert len(grid) == 22
    assert len(grid[0]) == 8
    assert len(grid[0][0]) == 10


def test_video_mask_to_json_area_averages_and_rounds():
    mask = torch.zeros(16, 32)
    mask[:, 16:] = 1.0
    mask[0, :16] = 1.0 / 3.0
    assert json.loads(video_mask_to_json(mask)) == [[0.0208, 1.0]]


def test_video_mask_to_json_fits_upload_limit():
    # A soft full-resolution 1344x768 mask serialized raw would exceed 8 MiB.
    payload = video_mask_to_json(torch.rand(768, 1344))
    assert len(payload.encode()) < 8 * 1024 * 1024 // 100


def test_video_mask_to_json_rejects_wrong_rank():
    with pytest.raises(ValueError):
        video_mask_to_json(torch.zeros(4))
    with pytest.raises(ValueError):
        video_mask_to_json(torch.zeros(1, 1, 1, 1))


@pytest.mark.parametrize(
    ("mode", "duration", "preserve_fraction", "frames", "prefix"),
    [
        # 5 s source extended to 10 s: 240 frames align to 243; the 120 source
        # frames snap down to a 107-frame prefix (32 latents on the server).
        ("extension", 10.0, 0.5, 243, 107),
        # 5 s output keeps 20% of the source: 24 frames snap down to 22.
        ("continuation", 5.0, 0.2, 124, 22),
    ],
)
def test_temporal_mask_is_frame_space(mode, duration, preserve_fraction, frames, prefix):
    images = torch.zeros(120, 8, 8, 3)
    mask, preview_fps, preview_images, preview_mask = VLLMOmniMiniMaxH3TemporalMask().build(
        images, 24.0, duration, mode, preserve_fraction
    )
    assert mask.shape == (frames, 1, 1)
    assert bool((mask[:prefix] == 0).all())
    assert bool((mask[prefix:] == 1).all())
    assert preview_fps == 24.0
    assert preview_images.shape[0] == frames
    assert torch.equal(preview_mask, mask)
