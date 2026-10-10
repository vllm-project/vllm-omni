# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standalone LiDAR VAE decoder contracts, causal streaming, and checkpoint loading."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.cosmos3.lidar import (
    Cosmos3LidarDecoder,
    lidar_decoder_args,
    postprocess_lidar_decoder_output,
    validate_lidar_decoder_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def config():
    return {
        "dtype": "float32",
        "sample_posterior": False,
        "apply_validity_mask": True,
        "fps": 10.0,
        "latent_channels": 2,
        "spatial_compression": [16, 16],
        "temporal_compression_factor": 1,
        "streaming_chunk_frames": 2,
        "streaming_context_frames": 3,
        "range_projection": {
            "native_height": 128,
            "semantic_width": 1800,
            "model_width": 1808,
            "model_width_transform": "circular_pad",
            "intensity_encoding": "unit",
            "min_range_m": 5.0,
            "max_range_m": 105.0,
        },
        "network_config": {
            "resolution": [128, 1808],
            "patch_size": [2, 2],
            "in_channels": 3,
            "out_channels": 3,
            "z_dim": 2,
            "base_channels": 4,
            "depths": [0, 0, 0, 1],
            "num_heads": [1, 1, 1, 1],
            "dilation": [1, 1, 1, 1],
            "temporal_downsample": [False, False, False],
        },
    }


@pytest.mark.parametrize("masked", [False, True])
def test_physical_channels_validity_cut_and_exact_width_crop(config, masked):
    config["apply_validity_mask"] = masked
    raw = torch.zeros(1, 3, 1, 128, 1808)
    raw[:, 0] = torch.linspace(-2, 2, 1808)
    raw[:, 1] = torch.linspace(2, -2, 1808)
    raw[:, 2, :, :, 4:7] = torch.tensor([-1.0, 0.0, 1.0])
    result = postprocess_lidar_decoder_output(raw, config)
    assert result.shape == (1, 3, 1, 128, 1800)
    assert result.dtype == torch.float32 and result.is_contiguous()
    validity = raw[:, 2:3, ..., 4:-4].sigmoid()
    ranges = (raw[:, :1, ..., 4:-4].clamp(-1, 1) + 1) * 50 + 5
    intensity = (raw[:, 1:2, ..., 4:-4].clamp(-1, 1) + 1) / 2
    if masked:
        keep = validity >= 0.5
        assert result[0, 2, 0, 0, :3].tolist() == [0, 1, 1]
        ranges = ranges.masked_fill(~keep, 0)
        intensity = intensity.masked_fill(~keep, 0)
        validity = keep.float()
    torch.testing.assert_close(result, torch.cat((ranges, intensity, validity), 1), rtol=0, atol=0)
    config["range_projection"]["validity_threshold"] = 0.75
    if masked:
        assert postprocess_lidar_decoder_output(raw, config)[0, 2, 0, 0, :3].eq(0).all()


@pytest.mark.parametrize("threshold", [0, 1, True, -1, float("nan"), float("inf"), "0.5"])
def test_invalid_validity_threshold_fails(config, threshold):
    config["range_projection"]["validity_threshold"] = threshold
    with pytest.raises(ValueError, match="validity_threshold"):
        validate_lidar_decoder_config(config)


def test_asymmetric_decoder_resolves_reference_overrides(config):
    network = config["network_config"]
    network.update(
        decoder_depths=[1, 1, 1],
        decoder_num_heads=[1, 2, 4],
        decoder_dilation=[1, 2, 3],
        decoder_temporal_upsample=[False, False],
        out_patch_size=[4, 4],
    )
    validate_lidar_decoder_config(config)
    args = lidar_decoder_args(network)
    assert args["depths"] == [1, 1, 1] and args["num_heads"] == [1, 2, 4]
    assert args["dilation"] == [1, 2, 3]
    assert args["temporal_downsample"] == args["temporal_upsample"] == [False, False]
    network.pop("decoder_temporal_upsample")
    with pytest.raises(ValueError, match="asymmetric"):
        validate_lidar_decoder_config(config)


@pytest.mark.parametrize(
    "override",
    [
        {"temporal_upsample": [True, False, False]},
        {"temporal_mixer": "conv"},
        {"bottleneck_3d": True},
        {"out_channels": 2},
        {"predict_validity": True},
        {"out_patch_size": [4, 4]},
        {"decoder_num_heads": [1, 1]},
    ],
)
def test_unsupported_decoder_topology_fails_at_construction(config, override):
    config["network_config"].update(override)
    with pytest.raises(ValueError):
        Cosmos3LidarDecoder(config)


@pytest.mark.parametrize("context", [None, 2, 3])
@pytest.mark.parametrize("frames", [1, 2, 5])
def test_streaming_affine_cache_eviction_partial_chunk_and_request_isolation(config, context, frames):
    config["streaming_context_frames"] = context
    model = Cosmos3LidarDecoder(config).float()
    model.latent_mean.fill_(2)
    model.latent_std.fill_(3)
    with torch.no_grad():
        model.post_quant_conv.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
        model.post_quant_conv.bias.zero_()
    seen = []

    class Decode(torch.nn.Module):
        def forward(self, z, coords, temporal_kv_cache, return_temporal_kv_cache):
            assert z.dtype == coords.dtype == torch.float32 and return_temporal_kv_cache
            old = temporal_kv_cache["time"][0] if temporal_kv_cache else z.new_empty(1, 1, 0, 1)
            marker = z[:1, :1, :, :1, :1].flatten(3)
            seen.append((marker.flatten().tolist(), old.flatten().tolist()))
            cache = torch.cat((old, marker), 2)
            return z.new_zeros(1, 3, z.shape[2], 128, 1808), {"time": (cache, cache)}

    model.decoder = Decode()
    latents = torch.arange(frames).reshape(1, 1, frames, 1, 1).expand(1, 2, frames, 8, 113).float()
    rng = torch.random.get_rng_state()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.decode(latents)
        repeat = model.decode(latents)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert output.shape == (1, 3, frames, 128, 1800)
    torch.testing.assert_close(repeat, output, rtol=0, atol=0)
    calls = (frames + 1) // 2
    assert seen[:calls] == seen[calls:] and seen[0][1] == []
    for index, (current, old) in enumerate(seen[:calls]):
        start = index * 2
        expected = (torch.arange(start, min(start + 2, frames)) * 3 + 2).tolist()
        assert current == expected
        keep = start if context is None else min(start, context - len(current))
        assert old == (torch.arange(start - keep, start) * 3 + 2).tolist()


@pytest.fixture
def artifact(config, tmp_path):
    folder = tmp_path / "lidar_vae"
    folder.mkdir()
    model = Cosmos3LidarDecoder(config).float()
    state = model.state_dict()
    for value in state.values():
        value.fill_(1.0001)
    state.update({"encoder.unused": torch.ones(1), "quant_conv.unused": torch.ones(1)})
    (folder / "config.json").write_text(json.dumps(config))
    save_file(state, folder / "diffusion_pytorch_model.safetensors")
    return tmp_path, state


def test_decoder_selective_exact_fp32_loading(config, artifact, monkeypatch):
    import safetensors

    path, state = artifact
    safe_open = safetensors.safe_open
    reads = []

    @contextmanager
    def tracked(*args, **kwargs):
        with safe_open(*args, **kwargs) as handle:

            def tensor(name):
                assert not name.startswith(("encoder.", "quant_conv."))
                reads.append(name)
                return handle.get_tensor(name)

            yield SimpleNamespace(keys=handle.keys, get_slice=handle.get_slice, get_tensor=tensor)

    monkeypatch.setattr(safetensors, "safe_open", tracked)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        loaded = Cosmos3LidarDecoder.from_pretrained(str(path), config, torch.device("cpu"))
    finally:
        torch.set_default_dtype(previous)
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert set(reads) == set(loaded.state_dict())
    for name, value in loaded.state_dict().items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert not hasattr(loaded, "encoder")


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype", "stats", "unexpected", "config"])
def test_invalid_decoder_artifact_fails(config, artifact, corruption):
    path, state = artifact
    if corruption == "missing":
        state.pop("post_quant_conv.weight")
    elif corruption == "shape":
        state["post_quant_conv.weight"] = torch.ones(1)
    elif corruption == "dtype":
        state["post_quant_conv.weight"] = state["post_quant_conv.weight"].bfloat16()
    elif corruption == "stats":
        state["latent_std"].zero_()
    elif corruption == "unexpected":
        state["decoder.unexpected"] = torch.ones(1)
    else:
        config = copy.deepcopy(config)
        config["fps"] = 11
    save_file(state, path / "lidar_vae/diffusion_pytorch_model.safetensors")
    with pytest.raises((ValueError, RuntimeError)):
        Cosmos3LidarDecoder.from_pretrained(str(path), config, torch.device("cpu"))


def test_decoder_real_streaming_matches_full_causal_network(config):
    config["streaming_context_frames"] = None
    model = Cosmos3LidarDecoder(config).float().eval()
    model.latent_mean.fill_(0.125)
    model.latent_std.fill_(1.25)
    latents = torch.randn(1, 2, 3, 8, 113)
    with torch.inference_mode():
        z = latents * model.latent_std + model.latent_mean
        z = model.post_quant_conv(z.permute(0, 2, 1, 3, 4).flatten(0, 1))
        z = z.reshape(1, 3, 2, 8, 113).permute(0, 2, 1, 3, 4)
        full = postprocess_lidar_decoder_output(model.decoder(z, model.coords), config)
        chunked = model.decode(latents)
    torch.testing.assert_close(chunked, full, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("context", [None, 3, 4])
@pytest.mark.parametrize("frames", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("batch", [1, 2])
def test_real_local_attention_decoder_streaming(
    config, asymmetric, context, frames, batch, monkeypatch, bottleneck_3d=False
):
    from vllm_omni.diffusion.models.cosmos3.lidar_encoder import neighborhood_attention as attention
    from vllm_omni.diffusion.models.cosmos3.lidar_encoder import transformer_vae
    from vllm_omni.diffusion.models.cosmos3.lidar_encoder.encoding import generate_polar_coords

    # Execute all real upsampling/temporal/spatial blocks on a small grid that
    # permits eager CPU FlexAttention. Topology still uses the artifact resolver.
    network = copy.deepcopy(config["network_config"])
    network.update(
        resolution=[32, 48],
        base_channels=8,
        depths=[2, 2, 1, 1],
        num_heads=[1, 1, 2, 2],
        dilation=[2, 2, 1, 1],
        window_size=[3, 3],
        bottleneck_3d=bottleneck_3d,
        bottleneck_3d_causal_time=bottleneck_3d,
        bottleneck_3d_rope=bottleneck_3d,
    )
    if asymmetric:
        network.update(
            decoder_depths=[2, 1, 1],
            decoder_num_heads=[1, 1, 2],
            decoder_dilation=[2, 1, 1],
            decoder_temporal_upsample=[False, False],
            out_patch_size=[4, 4],
        )
    with torch.random.fork_rng():
        torch.manual_seed(721)
        model = transformer_vae.Decoder(**lidar_decoder_args(network)).float().eval()
        # Zero-initialized residual projections otherwise hide attention errors.
        for name, parameter in model.named_parameters():
            if name.endswith("out_proj.weight"):
                torch.nn.init.normal_(parameter, std=0.1)
        z = torch.randn(batch, 2, frames, 2, 3)
    coords = generate_polar_coords(32, 48)
    keys_before = set(model.state_dict())
    seen = set()
    real_attention = transformer_vae.neighborhood_attention_2d

    def track(query, key, value, **kwargs):
        seen.add((query.shape[0], tuple(query.shape[1:3]), tuple(kwargs["dilation"])))
        return real_attention(query, key, value, **kwargs)

    monkeypatch.setattr(transformer_vae, "neighborhood_attention_2d", track)
    attention._get_block_mask.cache_clear()

    def stream():
        cache, chunks = None, []
        for start in range(0, frames, 3):
            chunk = z[:, :, start : start + 3]
            if cache is not None and context is not None:
                keep = context - chunk.shape[2]
                cache = {key: (k[:, :, -keep:], v[:, :, -keep:]) for key, (k, v) in cache.items()} if keep else None
            output, cache = model(chunk, coords, temporal_kv_cache=cache, return_temporal_kv_cache=True)
            assert cache and all(key.startswith(("mid_temporal.", "mid_3d.", "up_levels.temporal_")) for key in cache)
            if context is not None:
                assert all(k.shape[2] <= context for k, _ in cache.values())
            chunks.append(output)
        return torch.cat(chunks, 2)

    rng = torch.random.get_rng_state()
    with torch.inference_mode():
        actual = stream()
        misses = attention._get_block_mask.cache_info().misses
        repeated = stream()
        assert attention._get_block_mask.cache_info().misses == misses
        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
        if context is None:
            torch.testing.assert_close(actual, model(z, coords), rtol=1e-4, atol=1e-4)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert actual.shape == (batch, 3, frames, 32, 48) and actual.dtype == torch.float32
    assert set(model.state_dict()) == keys_before
    assert any(dilation == (2, 2) for _, _, dilation in seen)
    assert len({hw for _, hw, _ in seen}) == (2 if asymmetric else 3)
    if frames > 3 and frames % 3:
        assert {folded for folded, _, _ in seen} >= {batch * 3, batch * (frames % 3)}


@pytest.mark.parametrize("asymmetric", [False, True])
def test_local_attention_preserves_joint_3d_bottleneck_streaming(config, asymmetric, monkeypatch):
    test_real_local_attention_decoder_streaming(config, asymmetric, None, 4, 1, monkeypatch, bottleneck_3d=True)


def test_real_decoder_wrapper_with_local_attention(config):
    from vllm_omni.diffusion.models.cosmos3.lidar_encoder.transformer_vae import CircularNeighborhoodSelfAttentionBlock

    # Lowest upsampling level is small enough for eager CPU attention while
    # the wrapper still exercises the physical 128x1808 -> 128x1800 contract.
    config["network_config"].update(depths=[0, 0, 1, 1], window_size=[3, 3])
    config["streaming_context_frames"] = None
    with torch.random.fork_rng():
        torch.manual_seed(721)
        model = Cosmos3LidarDecoder(config).float().eval()
        for block in model.modules():
            if isinstance(block, CircularNeighborhoodSelfAttentionBlock):
                torch.nn.init.normal_(block.out_proj.weight, std=0.1)
        latents = torch.randn(1, 2, 1, 8, 113)
    model.latent_mean.fill_(0.125)
    model.latent_std.fill_(1.25)
    rng = torch.random.get_rng_state()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model.decode(latents)
    with torch.inference_mode():
        z = model.post_quant_conv((latents * 1.25 + 0.125)[:, :, 0]).unsqueeze(2)
        expected = postprocess_lidar_decoder_output(model.decoder(z, model.coords), config)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert actual.shape == (1, 3, 1, 128, 1800) and actual.dtype == torch.float32
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("shape", [(0, 2, 1, 8, 113), (1, 2, 0, 8, 113), (1, 3, 1, 8, 113), (1, 2, 1, 8, 112)])
def test_decoder_rejects_invalid_latent_geometry(config, shape):
    with pytest.raises(ValueError, match="latents"):
        Cosmos3LidarDecoder(config).decode(torch.zeros(shape))
