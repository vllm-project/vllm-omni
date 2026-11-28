# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Synthetic CPU coverage for rank-local input; collectives are simulated."""

from dataclasses import dataclass
from functools import partial

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
    patch_size = 2

    def __init__(self, crop_mode):
        self.transform = _Normalize()
        self.crop_mode = crop_mode

    def get_suitable_video_length(self, length):
        return length

    def _align_to_total_patch_size(self, height, width):
        return height // self.patch_size * self.patch_size, width // self.patch_size * self.patch_size


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


def _frames(length=7, *, size=(9, 11)):
    # Deterministic synthetic RGB frames generated here without media files.
    return np.random.default_rng(123).integers(0, 256, (length, *size, 3), dtype=np.uint8)


def _reference(vae, frames):
    result = vae._stream_prepare_video_tensor(frames, torch.device("cpu"))
    height, width = vae.model.processor._align_to_total_patch_size(*frames.shape[1:3])
    center = vae.model.processor.crop_mode == "center"
    top, left = ((frames.shape[1] - height) // 2, (frames.shape[2] - width) // 2) if center else (0, 0)
    return result[:, :, top : top + height, left : left + width].unsqueeze(0)


@pytest.mark.parametrize(("crop_mode", "offset"), [("top_left", 0), ("center", 1)])
@pytest.mark.parametrize("length", [1, 4, 7, 13])
def test_tile_input_preserves_pixels_padding_and_one_posterior(crop_mode, offset, length):
    vae, frames = _vae(crop_mode), _frames(length, size=(11, 15))
    vae.model.processor.patch_size = 4
    prepared = vae._stream_prepare_video_tensor(frames, torch.device("cpu"))
    # Alignment trims three pixels on each axis, shifting both center offsets by one.
    expected = prepared[:, :, offset : offset + 8, offset : offset + 12].unsqueeze(0)
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
    collectives = []
    gathered: list[int] = []

    def agree(failed, *, group):
        assert group is vae._native_parallel_state()["sp_process_group"]
        assert failed.item() == 0
        collectives.append("agree")

    monkeypatch.setattr("vllm_omni.diffusion.models.minimax_h3.vae.dist.all_reduce", agree)
    y_idx, _, _ = vae.model.split_tiles(expected.shape[-2])
    x_idx, _, _ = vae.model.split_tiles(expected.shape[-1])
    all_inputs = []

    def gather(tasks, tile_count):
        collectives.append("gather")
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
    assert collectives == ["agree", "gather", "agree", "gather", "agree"]
    assert len(vae.model.inputs) == len(all_inputs) == 8
    for actual, reference in zip(vae.model.inputs, all_inputs, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(latent, expected.squeeze(0).half(), rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["local", "peer", "reduce", "gather"])
def test_parallel_tile_failure_before_blend(monkeypatch, failure):
    vae = _vae()
    group = object()
    vae.model.parallel_tiling = True
    vae._native_parallel_state = lambda: {"sp_rank": 0, "sp_size": 2, "sp_process_group": group}
    collectives = []

    def fail_encode(value):
        raise RuntimeError("local tile encode failed")

    def agree(failed, *, group):
        collectives.append("agree")
        assert failed.item() == int(failure == "local")
        if failure == "peer":
            failed.fill_(1)
        if failure == "reduce":
            raise RuntimeError("status reduction failed")

    def gather(tasks, tile_count):
        collectives.append("gather")
        raise RuntimeError("tile gather failed")

    if failure == "local":
        vae.model.encode = fail_encode
    vae.model._all_gather_tiled_results = gather
    monkeypatch.setattr("vllm_omni.diffusion.models.minimax_h3.vae.dist.all_reduce", agree)
    if failure in ("reduce", "gather"):
        with pytest.raises(RuntimeError, match="failed"):
            vae._stream_encode_video_tiles(_frames(), torch.device("cpu"))
    else:
        assert vae._stream_encode_video_tiles(_frames(), torch.device("cpu")) is None
    assert collectives == (["agree", "gather"] if failure == "gather" else ["agree"])
    assert DiagonalGaussianDistribution.calls == 0


@pytest.mark.parametrize("failure", ["blend", "sample"])
@pytest.mark.parametrize("rank", [0, 1])
def test_parallel_tile_failure_after_gather(monkeypatch, failure, rank):
    vae = _vae()
    process_group = object()
    vae.model.parallel_tiling = True
    vae._native_parallel_state = lambda: {"sp_rank": rank, "sp_size": 2, "sp_process_group": process_group}
    collectives = []
    failure_agreement = 2 if failure == "blend" else 3

    def fail(*args, **kwargs):
        raise RuntimeError(f"rank 0 {failure} failed")

    def agree(failed, *, group):
        assert group is process_group
        is_failure_agreement = collectives.count("agree") + 1 == failure_agreement
        assert failed.item() == int(rank == 0 and is_failure_agreement)
        if rank == 1 and is_failure_agreement:
            failed.fill_(1)
        collectives.append("agree")

    def gather(tasks, tile_count):
        assert len(tasks) * 2 == tile_count
        collectives.append("gather")
        return tasks * 2

    if rank == 0:
        target = vae.model if failure == "blend" else DiagonalGaussianDistribution
        monkeypatch.setattr(target, failure, fail)
    vae.model._all_gather_tiled_results = gather
    monkeypatch.setattr("vllm_omni.diffusion.models.minimax_h3.vae.dist.all_reduce", agree)

    assert vae._stream_encode_video_tiles(_frames(), torch.device("cpu")) is None
    expected = ["agree", "gather", "agree"]
    if failure == "sample":
        expected.extend(["gather", "agree"])
    assert collectives == expected
    assert DiagonalGaussianDistribution.calls == int(rank == 1 and failure == "sample")


def test_encoder_subclass_can_stream_tiles():
    class EncoderSubclass(EncoderFCN3D):
        pass

    vae, frames = _vae(), _frames()
    vae.model.encoder = EncoderSubclass()
    latent = vae._stream_encode_video_tiles(frames, torch.device("cpu"))
    torch.testing.assert_close(latent, _reference(vae, frames).squeeze(0).half(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "attribute",
    ["isolated_first_frame", "isolated_last_frame", "isolated_key_frame", "stack_tiling", "encoder_parallel"],
)
def test_unsupported_checkpoint_uses_ordinary_preparation(attribute):
    vae = _vae()
    setattr(vae.model, attribute, True)
    assert vae._stream_encode_video_tiles(_frames(), torch.device("cpu")) is None
    assert vae.model.inputs == []


def test_missing_posterior_lookup_falls_back():
    vae = _vae()
    vae.model.encode_base = partial(vae.model.encode_base)
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


@pytest.mark.parametrize("flag", ["1", " True ", "on"])
def test_enabled_encode_does_not_construct_a_complete_device_video(monkeypatch, flag):
    vae = _vae()
    monkeypatch.setenv("VLLM_OMNI_H3_VAE_STREAM_TILE", flag)
    monkeypatch.setenv("VLLM_OMNI_VAE_ENCODE_LEGACY_PREP", "0")

    def unexpected(*args):
        pytest.fail("eligible tile input must not prepare a complete video")

    vae._stream_prepare_video_tensor = unexpected
    rows, shape = vae.encode_video(_frames())
    assert shape == (8, 8, 10)
    assert rows.shape == (8 * 4 * 5, 12)


def test_enabled_encode_falls_back_to_whole_video_preparation(monkeypatch):
    vae, frames = _vae(), _frames()
    vae.model.stack_tiling = True
    monkeypatch.setenv("VLLM_OMNI_H3_VAE_STREAM_TILE", "1")
    monkeypatch.setenv("VLLM_OMNI_VAE_ENCODE_LEGACY_PREP", "0")
    messages = []
    monkeypatch.setattr("vllm_omni.diffusion.models.minimax_h3.vae.logger.info_once", messages.append)
    expected = vae._stream_prepare_video_tensor(frames, torch.device("cpu"))

    def encode_videos(videos, *, use_fp16_latent):
        assert use_fp16_latent
        assert isinstance(videos, list)
        (prepared,) = videos
        torch.testing.assert_close(prepared, expected, rtol=0, atol=0)
        return [prepared[:, :, :8, :10].half()]

    vae.model.encode_videos = encode_videos
    rows, shape = vae.encode_video(frames)
    assert shape == (8, 8, 10)
    assert rows.shape == (8 * 4 * 5, 12)
    assert vae.model.inputs == []
    assert messages == ["MiniMax H3 video tile streaming skipped: unsupported checkpoint encoder layout."]


def test_fallback_after_sampling_reuses_seeded_rng_state(monkeypatch):
    vae, frames = _vae(), _frames()
    monkeypatch.setenv("VLLM_OMNI_VAE_ENCODE_LEGACY_PREP", "0")

    def stream_then_fall_back(frames, device):
        torch.randn(1, device=device)
        return None

    def encode_videos(videos, *, use_fp16_latent):
        return [torch.randn_like(videos[0][:, :, :8, :10]).half()]

    vae._stream_encode_video_tiles = stream_then_fall_back
    vae.model.encode_videos = encode_videos
    monkeypatch.setenv("VLLM_OMNI_H3_VAE_STREAM_TILE", "0")
    expected, expected_shape = vae.encode_video(frames)
    monkeypatch.setenv("VLLM_OMNI_H3_VAE_STREAM_TILE", "1")
    actual, actual_shape = vae.encode_video(frames)
    assert actual_shape == expected_shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
