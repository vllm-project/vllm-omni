# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage21 import (
    DistributedAutoencoderKLQwenImage21,
)
from vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _make_vae(cls=AutoencoderKLQwenImage21, **overrides):
    """Toy VAE with the 2.1 residual architecture; 4x spatial compression."""
    kwargs = dict(
        base_dim=16,
        decoder_base_dim=16,
        z_dim=4,
        dim_mult=[1, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False, False],
        is_residual=True,
        in_channels=4,
        out_channels=4,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
        scale_factor_spatial=None,
        scale_factor_temporal=None,
    )
    kwargs.update(overrides)
    return cls(**kwargs)


def test_spatial_compression_ratio_derived_from_architecture():
    # One 2x downsampling stage per `temperal_downsample` entry.
    vae = _make_vae()
    assert vae.spatial_compression_ratio == 4

    # The real Qwen-Image-2.1 checkpoint ships scale_factor_spatial=8, but its
    # four downsampling stages compress 16x; the architecture wins.
    with torch.device("meta"):
        vae = AutoencoderKLQwenImage21(
            base_dim=96,
            decoder_base_dim=144,
            z_dim=64,
            dim_mult=[1, 2, 4, 8, 8],
            num_res_blocks=2,
            temperal_downsample=[False, True, True, True],
            is_residual=True,
            in_channels=4,
            out_channels=4,
            scale_factor_spatial=8,
        )
    assert vae.spatial_compression_ratio == 16


def test_tiled_decode_output_shape_and_closeness():
    torch.manual_seed(0)
    vae = _make_vae()
    z = torch.randn(1, 4, 1, 16, 16)
    ratio = vae.spatial_compression_ratio

    with torch.no_grad():
        ref = vae._decode(z, return_dict=False)[0]
        vae.enable_tiling(
            tile_sample_min_height=32,
            tile_sample_min_width=32,
            tile_sample_stride_height=24,
            tile_sample_stride_width=24,
        )
        tiled = vae.decode(z, return_dict=False)[0]

    # Regression: with the wrong (config-derived) ratio the tiled output was
    # cropped to a fraction of the sample size.
    assert tiled.shape == (1, 4, 1, 16 * ratio, 16 * ratio)
    assert tiled.shape == ref.shape

    # Tiled decode blends overlapping tiles, so it is not bitwise-identical to
    # untiled decode; on this toy model the seam error stays small.
    mse = ((ref - tiled) ** 2).mean().item()
    psnr = 10 * math.log10(4.0 / mse)
    assert psnr > 20.0

    # Tiled output is clamped to [-1, 1] like the untiled path.
    assert tiled.min() >= -1.0 and tiled.max() <= 1.0


def test_distributed_tile_hooks_match_tiled_decode():
    """tile_split -> tile_exec -> tile_merge must reproduce tiled_decode exactly."""
    torch.manual_seed(0)
    vae = _make_vae()
    vae.enable_tiling(
        tile_sample_min_height=32,
        tile_sample_min_width=32,
        tile_sample_stride_height=24,
        tile_sample_stride_width=24,
    )
    distributed = _make_vae(DistributedAutoencoderKLQwenImage21)
    distributed.enable_tiling(
        tile_sample_min_height=32,
        tile_sample_min_width=32,
        tile_sample_stride_height=24,
        tile_sample_stride_width=24,
    )
    distributed.load_state_dict(vae.state_dict())

    z = torch.randn(1, 4, 1, 16, 16)
    with torch.no_grad():
        tiled = vae.tiled_decode(z, return_dict=False)[0]

        tasks, grid_spec = DistributedAutoencoderKLQwenImage21.tile_split(distributed, z)
        # 16x16 latent, 8x8 latent tiles, stride 6 -> range(0, 16, 6) = 3 stops/axis.
        assert len(tasks) == 9
        assert grid_spec.grid_shape == (3, 3)
        assert grid_spec.split_dims == (3, 4)
        # Every tile carries all frames (T=1 for images).
        assert all(len(task.tensor) == z.shape[2] for task in tasks)

        coord_tensor_map = {task.grid_coord: distributed.tile_exec(task) for task in tasks}
        merged = distributed.tile_merge(coord_tensor_map, grid_spec)

    assert merged.shape == tiled.shape
    torch.testing.assert_close(merged, tiled)


def test_tiled_encode_output_shape_and_closeness():
    """tiled_encode is engaged for image-conditioned generation when tiling is on."""
    torch.manual_seed(0)
    vae = _make_vae()
    x = torch.randn(1, 4, 1, 64, 64)

    with torch.no_grad():
        ref = vae._encode(x)
        vae.enable_tiling(
            tile_sample_min_height=32,
            tile_sample_min_width=32,
            tile_sample_stride_height=24,
            tile_sample_stride_width=24,
        )
        tiled = vae._encode(x)

    ratio = vae.spatial_compression_ratio
    assert tiled.shape == (1, 8, 1, 64 // ratio, 64 // ratio)
    assert tiled.shape == ref.shape
    # Blended tile borders deviate from untiled encode but stay small.
    assert (ref - tiled).abs().max().item() < 1.0


def test_encode_decode_round_trip_shape():
    torch.manual_seed(0)
    vae = _make_vae()
    x = torch.randn(1, 4, 1, 32, 32)
    with torch.no_grad():
        latents = vae.encode(x, return_dict=False)[0].mode()
        decoded = vae.decode(latents, return_dict=False)[0]
    ratio = vae.spatial_compression_ratio
    assert latents.shape == (1, 4, 1, 32 // ratio, 32 // ratio)
    assert decoded.shape == x.shape


@pytest.mark.parametrize("persistent", [False, True])
def test_adaptive_oom_restores_tile_configuration(monkeypatch, persistent):
    vae = _make_vae()
    vae.enable_tiling(
        tile_sample_min_height=512,
        tile_sample_min_width=512,
        tile_sample_stride_height=384,
        tile_sample_stride_width=384,
    )
    attempts = []
    z = torch.zeros(1, 4, 1, 4, 4)

    def decode_tile(z, return_dict):
        attempts.append(vae.tile_sample_min_height)
        if persistent or vae.tile_sample_min_height > 128:
            raise torch.OutOfMemoryError("injected tile OOM")
        return (z,)

    monkeypatch.setattr(vae, "tiled_decode", decode_tile)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    if persistent:
        with pytest.raises(torch.OutOfMemoryError):
            vae._tiled_decode_adaptive(z, return_dict=False)
    else:
        assert vae._tiled_decode_adaptive(z, return_dict=False)[0] is z
    assert attempts == [512, 256, 128]
    assert (
        vae.tile_sample_min_height,
        vae.tile_sample_min_width,
        vae.tile_sample_stride_height,
        vae.tile_sample_stride_width,
    ) == (512, 512, 384, 384)


def test_adaptive_oom_clears_feat_map_when_retries_exhausted(monkeypatch):
    """Final min-tile OOM must not leave decoder activations on the device."""
    vae = _make_vae()
    vae.enable_tiling(
        tile_sample_min_height=512,
        tile_sample_min_width=512,
        tile_sample_stride_height=384,
        tile_sample_stride_width=384,
    )
    vae.clear_cache()
    z = torch.zeros(1, 4, 1, 4, 4)
    leaked = torch.ones(2, 2, device="cpu")  # stand-in for a cached activation

    def decode_tile(z, return_dict):
        # Mimic a partial tiled decode that already wrote into _feat_map before OOM.
        vae._feat_map[0] = leaked
        raise torch.OutOfMemoryError("injected tile OOM")

    monkeypatch.setattr(vae, "tiled_decode", decode_tile)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    with pytest.raises(torch.OutOfMemoryError):
        vae._tiled_decode_adaptive(z, return_dict=False)

    assert all(slot is None for slot in vae._feat_map)
    assert all(slot is None for slot in vae._enc_feat_map)
    assert (
        vae.tile_sample_min_height,
        vae.tile_sample_min_width,
        vae.tile_sample_stride_height,
        vae.tile_sample_stride_width,
    ) == (512, 512, 384, 384)


def test_tiled_encode_oom_clears_enc_feat_map(monkeypatch):
    """Tiled _encode must clear encoder activations if tiled_encode OOMs."""
    vae = _make_vae()
    vae.enable_tiling(
        tile_sample_min_height=32,
        tile_sample_min_width=32,
        tile_sample_stride_height=24,
        tile_sample_stride_width=24,
    )
    vae.clear_cache()
    leaked = torch.ones(2, 2, device="cpu")
    x = torch.zeros(1, 4, 1, 64, 64)

    def encode_tile(x_in):
        vae._enc_feat_map[0] = leaked
        raise torch.OutOfMemoryError("injected tiled encode OOM")

    monkeypatch.setattr(vae, "tiled_encode", encode_tile)

    with pytest.raises(torch.OutOfMemoryError):
        vae._encode(x)

    assert all(slot is None for slot in vae._enc_feat_map)
    assert all(slot is None for slot in vae._feat_map)


# Single-frame paths -> expected output shape. The toy VAE below has one temporal
# stage so the downsample3d/upsample3d and first_chunk branches are exercised.
_SINGLE_FRAME_PATHS = {
    "decode": (1, 4, 1, 64, 64),
    "tiled_decode": (1, 4, 1, 64, 64),
    "tile_exec": (1, 4, 1, 64, 64),
    "encode": (1, 8, 1, 16, 16),
    "tiled_encode": (1, 8, 1, 16, 16),
}


def _make_single_frame_vae(path):
    cls = DistributedAutoencoderKLQwenImage21 if path == "tile_exec" else AutoencoderKLQwenImage21
    torch.manual_seed(0)
    vae = _make_vae(cls, temperal_downsample=[False, True])
    if path in ("tiled_decode", "tile_exec", "tiled_encode"):
        vae.enable_tiling(
            tile_sample_min_height=32,
            tile_sample_min_width=32,
            tile_sample_stride_height=24,
            tile_sample_stride_width=24,
        )
    return vae


def _run_single_frame_path(vae, path):
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        if path in ("encode", "tiled_encode"):
            return vae._encode(torch.randn(1, 4, 1, 64, 64, generator=generator))
        z = torch.randn(1, 4, 1, 16, 16, generator=generator)
        if path == "tile_exec":
            tasks, grid_spec = vae.tile_split(z)
            return vae.tile_merge({task.grid_coord: vae.tile_exec(task) for task in tasks}, grid_spec)
        return vae.decode(z, return_dict=False)[0]


def _wrap_coder_forwards(monkeypatch, vae, inject_cache):
    """Wrap encoder/decoder forward; return the feature caches they ran with."""
    caches = []
    for name in ("encoder", "decoder"):
        module = getattr(vae, name)
        num_convs = vae._cached_conv_counts[name]

        def forward(x, feat_cache=None, feat_idx=None, _forward=module.forward, _num_convs=num_convs, **kwargs):
            if inject_cache:
                # Fresh per-call cache, as the cached single-frame path used.
                feat_cache, feat_idx = [None] * _num_convs, [0]
            caches.append(feat_cache)
            return _forward(x, feat_cache=feat_cache, feat_idx=feat_idx, **kwargs)

        monkeypatch.setattr(module, "forward", forward)
    return caches


def _num_cached_tensors(feat_cache):
    return sum(isinstance(entry, torch.Tensor) for entry in feat_cache or [])


@pytest.mark.parametrize("path", list(_SINGLE_FRAME_PATHS))
def test_single_frame_paths_hold_no_feature_cache(monkeypatch, path):
    """The 2D convs never read the temporal cache, so no conv input is cloned into it."""
    vae = _make_single_frame_vae(path)
    caches = _wrap_coder_forwards(monkeypatch, vae, inject_cache=False)

    out = _run_single_frame_path(vae, path)

    assert out.shape == _SINGLE_FRAME_PATHS[path]
    assert caches, "encoder/decoder was not called"
    assert all(cache is None for cache in caches)
    assert _num_cached_tensors(getattr(vae, "_feat_map", None)) == 0
    assert _num_cached_tensors(getattr(vae, "_enc_feat_map", None)) == 0


@pytest.mark.parametrize("path", list(_SINGLE_FRAME_PATHS))
def test_single_frame_paths_match_cached_reference(monkeypatch, path):
    vae = _make_single_frame_vae(path)
    out = _run_single_frame_path(vae, path)
    assert torch.equal(out, _run_single_frame_path(vae, path))

    caches = _wrap_coder_forwards(monkeypatch, vae, inject_cache=True)
    ref = _run_single_frame_path(vae, path)

    # The reference really took the cached branches, and dropping them is bit-identical.
    assert caches and all(_num_cached_tensors(cache) > 0 for cache in caches)
    assert ref.shape == out.shape
    assert torch.equal(out, ref)
