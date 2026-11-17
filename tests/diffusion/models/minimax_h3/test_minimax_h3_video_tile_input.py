# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Synthetic CPU coverage for rank-local input; collectives are simulated."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class DiagonalGaussianDistribution:
    calls = 0
    moments: torch.Tensor | None = None

    def __init__(self, moments):
        type(self).moments = moments

    def sample(self):
        type(self).calls += 1
        moments = type(self).moments
        assert moments is not None
        return moments[:, :3]


class EncoderFCN3D(nn.Module):
    def forward(self, value):
        return torch.cat((value, torch.zeros_like(value)), dim=1)


@dataclass
class _Normalize:
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


class _Processor:
    def __init__(self, crop_mode):
        self.transform = _Normalize()
        self.crop_mode = crop_mode

    def get_suitable_video_length(self, length):
        return length

    def _align_to_total_patch_size(self, height, width):
        return height // 2 * 2, width // 2 * 2


class _Checkpoint(nn.Module):
    def __init__(self, crop_mode="top_left"):
        super().__init__()
        self.encoder = EncoderFCN3D()
        self.parameter = nn.Parameter(torch.zeros(1))
        self.processor = _Processor(crop_mode)
        self.clip_length = 4
        self.token_drop = 0
        self.vae_ratio = 1
        self.encoder_tiling = True
        self.parallel_tiling = False
        self.stack_tiling = False
        self.encoder_parallel = False
        self.inputs = []
        self.blends = []
        self.eval()

    def split_tiles(self, length, is_decoder=False):
        starts = list(range(0, length - 3, 2))
        return starts, [4] * len(starts), [2] * (len(starts) - 1)

    def encode(self, value):
        self.inputs.append(value.clone())
        return self.encoder(value)

    def blend(self, a, b, extent, dim):
        # Corresponding synthetic overlaps are identical. Returning b leaves
        # a closed-form full-video reference independent of the merge loop.
        self.blends.append((extent, dim))
        return b

    def encode_base(self, value):
        return DiagonalGaussianDistribution(self.encoder(value)).sample()


def _vae(crop_mode="top_left"):
    vae = object.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(vae)
    vae.model = _Checkpoint(crop_mode)
    vae.parallel_size = 1
    vae.config_dict = {"latent_channels": 3, "latents_mean": [0] * 3, "latents_std": [1] * 3}
    DiagonalGaussianDistribution.calls = 0
    DiagonalGaussianDistribution.moments = None
    return vae


def _frames(length=7):
    return np.random.default_rng(123).integers(0, 256, (length, 9, 11, 3), dtype=np.uint8)


def _reference(vae, frames):
    result = vae._stream_prepare_video_tensor(frames, torch.device("cpu"))
    height, width = vae.model.processor._align_to_total_patch_size(*frames.shape[1:3])
    center = vae.model.processor.crop_mode == "center"
    top, left = ((frames.shape[1] - height) // 2, (frames.shape[2] - width) // 2) if center else (0, 0)
    return result[:, :, top : top + height, left : left + width].unsqueeze(0)


@pytest.mark.parametrize("crop_mode", ["top_left", "center"])
@pytest.mark.parametrize("length", [1, 4, 7, 13])
def test_tile_input_preserves_pixels_padding_and_one_posterior(crop_mode, length):
    vae, frames = _vae(crop_mode), _frames(length)
    expected = _reference(vae, frames)
    latent = vae._stream_encode_video_tiles(frames, torch.device("cpu"))
    assert latent is not None
    moments = DiagonalGaussianDistribution.moments
    assert moments is not None
    torch.testing.assert_close(moments[:, :3], expected, rtol=0, atol=0)
    torch.testing.assert_close(latent, expected.squeeze(0).half(), rtol=0, atol=0)
    assert DiagonalGaussianDistribution.calls == 1
    assert all(value.shape == (1, 3, 4, 4, 4) for value in vae.model.inputs)
    assert all(extent == 2 and dim in (-2, -1) for extent, dim in vae.model.blends)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_only_assigned_tiles_are_uploaded_and_each_clip_gathers(monkeypatch, rank):
    vae, frames = _vae(), _frames(7)
    expected = _reference(vae, frames)
    vae.model.parallel_tiling = True
    group = object()
    vae._native_parallel_state = lambda: {"sp_rank": rank, "sp_size": 3, "sp_process_group": group}
    barriers: list[dict] = []
    gathered: list[int] = []
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.vae.dist.barrier", lambda **kwargs: barriers.append(kwargs)
    )
    y_idx, _, _ = vae.model.split_tiles(expected.shape[-2])
    x_idx, _, _ = vae.model.split_tiles(expected.shape[-1])
    all_inputs = []

    def gather(tasks, tile_count):
        start = len(gathered) * 4
        inputs = [expected[:, :, start : start + 4, y : y + 4, x : x + 4] for y in y_idx for x in x_idx]
        assert tile_count == len(inputs)
        for actual, reference in zip(tasks, inputs[rank::3], strict=True):
            torch.testing.assert_close(actual[:, :3], reference, rtol=0, atol=0)
        all_inputs.extend(inputs[rank::3])
        gathered.append(tile_count)
        return [vae.model.encoder(value) for value in inputs]

    vae.model._all_gather_tiled_results = gather
    latent = vae._stream_encode_video_tiles(frames, torch.device("cpu"))
    assert latent is not None
    assert len(barriers) == len(gathered) == 2
    assert len(vae.model.inputs) == len(all_inputs) == 8
    for actual, reference in zip(vae.model.inputs, all_inputs, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(latent, expected.squeeze(0).half(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "attribute",
    ["isolated_first_frame", "isolated_last_frame", "isolated_key_frame", "stack_tiling", "encoder_parallel"],
)
def test_unsupported_checkpoint_uses_ordinary_preparation(attribute):
    vae = _vae()
    setattr(vae.model, attribute, True)
    assert vae._stream_encode_video_tiles(_frames(), torch.device("cpu")) is None
    assert vae.model.inputs == []


def test_token_drop_precedes_the_single_posterior():
    vae, frames = _vae(), _frames()
    vae.model.token_drop = 1
    expected = _reference(vae, frames)[:, :, :-1]
    result = vae._stream_encode_video_tiles(frames, torch.device("cpu"))
    torch.testing.assert_close(result, expected.squeeze(0).half(), rtol=0, atol=0)
    assert DiagonalGaussianDistribution.calls == 1


def test_distinct_tile_outputs_preserve_native_weighted_blend_neighbors():
    vae = _vae()
    frames = np.zeros((4, 6, 6, 3), dtype=np.uint8)
    encoded: list[torch.Tensor] = []

    def encode(value):
        tile = torch.full((1, 6, 4, 4, 4), float(len(encoded) * 10))
        encoded.append(tile)
        return tile

    def blend(a, b, extent, dim):
        weight_shape = [1] * b.ndim
        weight_shape[dim] = extent
        weight = (torch.arange(extent) / extent).view(weight_shape)
        overlap = a.narrow(dim, a.shape[dim] - extent, extent) * (1 - weight)
        overlap = overlap + b.narrow(dim, 0, extent) * weight
        return torch.cat((overlap, b.narrow(dim, extent, b.shape[dim] - extent)), dim=dim)

    vae.model.encode = encode
    vae.model.blend = blend
    result = vae._stream_encode_video_tiles(frames, torch.device("cpu"))
    # Closed-form native 2x2 result for tiles 0/10/20/30 and overlap 2.
    # In particular, horizontal blending reads the original left tile after
    # vertical blending, rather than the already blended/cropped output.
    expected = torch.tensor(
        [
            [0, 0, 0, 5, 10, 10],
            [0, 0, 0, 5, 10, 10],
            [0, 0, 20, 15, 10, 10],
            [10, 10, 20, 20, 20, 20],
            [20, 20, 20, 25, 30, 30],
            [20, 20, 20, 25, 30, 30],
        ],
        dtype=torch.float16,
    ).expand(3, 4, 6, 6)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_enabled_encode_does_not_construct_a_complete_device_video(monkeypatch):
    vae = _vae()
    monkeypatch.setenv("VLLM_OMNI_H3_VAE_STREAM_TILE", "1")
    monkeypatch.setenv("VLLM_OMNI_VAE_ENCODE_LEGACY_PREP", "0")

    def unexpected(*args):
        pytest.fail("eligible tile input must not prepare a complete video")

    vae._stream_prepare_video_tensor = unexpected
    rows, shape = vae.encode_video(_frames())
    assert shape == (8, 8, 10)
    assert rows.shape == (8 * 4 * 5, 12)
