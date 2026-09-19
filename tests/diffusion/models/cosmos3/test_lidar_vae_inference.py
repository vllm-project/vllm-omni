# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference compatibility and causal execution of the LiDAR VAE."""

import hashlib
import json

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.lidar_encoder.encoding import generate_polar_coords
from vllm_omni.diffusion.models.cosmos3.lidar_encoder.transformer_vae import (
    CausalTemporalAttention,
    Decoder,
    Encoder,
    Joint3DSelfAttention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def network_config(joint=False):
    return dict(
        resolution=[32, 48],
        patch_size=[2, 2],
        z_dim=2,
        base_channels=8,
        depths=[2, 2, 1, 1],
        num_heads=[1, 1, 2, 2],
        dilation=[2, 2, 1, 1],
        window_size=[3, 3],
        temporal_downsample=[False, False, False],
        bottleneck_3d=joint,
        bottleneck_3d_rope=joint,
        bottleneck_3d_causal_time=joint,
        bottleneck_3d_max_t=4,
    )


def make_model(kind, **overrides):
    config = network_config()
    config.update(overrides)
    if kind == "encoder":
        return Encoder(in_channels=3, **config).float().eval()
    return Decoder(out_channels=3, **config).float().eval()


def activate_residuals(model):
    # Freshly initialized zero residuals hide attention and FFN regressions.
    with torch.no_grad():
        for parameter in model.parameters():
            if not torch.count_nonzero(parameter):
                torch.nn.init.normal_(parameter, std=0.05)


@pytest.mark.parametrize(
    "kind,joint,expected",
    [
        ("encoder", False, "b13774847a0213db8a137f2ae7d0d1d19b64984ce0a70889853035bff33c6661"),
        ("encoder", True, "bb2469b3de04ad6ed2060da81f778858d8e9accbee0006c591da2cd623d19b81"),
        ("decoder", False, "c4fe1474091402ef0e1f4f77528aeb3a91739db86419d59af19d856e6f424869"),
        ("decoder", True, "a171cabe84ef5053003f677020a130e263cc5a2d854f34c27ec7d98fe19c6031"),
        ("asymmetric", False, "9df4566a22bf8ed3777b5879328faedecebf9cbdbd64a82a101cc2d736382754"),
        ("asymmetric", True, "193fac1120597365e0bde5e934587c58383225ff5aebabca0cff448ceba1739b"),
    ],
)
def test_checkpoint_key_and_shape_inventory(kind, joint, expected):
    # Frozen from the implementation before inference-only cleanup. Unlike a
    # save/reload roundtrip, this also detects changes to checkpoint key names.
    config = network_config(joint)
    if kind == "asymmetric":
        config.update(
            depths=[2, 1, 1],
            num_heads=[1, 1, 2],
            dilation=[2, 1, 1],
            out_patch_size=[4, 4],
            temporal_downsample=[False, False],
        )
    model = make_model(kind, **config)
    inventory = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    assert hashlib.sha256(json.dumps(inventory).encode()).hexdigest() == expected, inventory


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
@pytest.mark.parametrize(
    "override,match",
    [
        ({"temporal_downsample": [True, False, False]}, "temporal resampling"),
        ({"temporal_downsample": [False]}, "temporal resampling"),
        ({"temporal_mixer": "conv"}, "temporal_mixer"),
        ({"bottleneck_3d": True}, "3D bottlenecks"),
        ({"bottleneck_3d": True, "bottleneck_3d_rope": True}, "3D bottlenecks"),
        ({"bottleneck_3d": True, "bottleneck_3d_causal_time": True}, "3D bottlenecks"),
        ({"positional_embedding": "fourier"}, "positional_embedding"),
        ({"dropout": 1.1}, "dropout"),
    ],
)
def test_unsupported_modes_fail_at_construction(kind, override, match):
    with pytest.raises(ValueError, match=match):
        make_model(kind, **override)


def test_decoder_validates_effective_upsampling_and_external_stem():
    # Explicit upsampling overrides the legacy downsampling argument.
    make_model("decoder", temporal_downsample=[True] * 3, temporal_upsample=[False] * 3)
    with pytest.raises(ValueError, match="temporal resampling"):
        make_model("decoder", temporal_upsample=[True, False, False])
    with pytest.raises(ValueError, match="patchifying stem"):
        make_model("decoder", stem_patchify=True)


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
def test_legacy_inert_options_preserve_initialization_and_eval_outputs(kind):
    options = dict(dropout=0.75, mapping_depth=9, temporal_conv_kernel=7)
    options["temporal_first_frame_special" if kind == "encoder" else "temporal_expand_wan_style"] = True
    with torch.random.fork_rng():
        torch.manual_seed(721)
        original = make_model(kind)
        expected_rng = torch.random.get_rng_state()
        torch.manual_seed(721)
        compatible = make_model(kind, **options)
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
        torch.testing.assert_close(compatible.state_dict(), original.state_dict(), rtol=0, atol=0)
        activate_residuals(original)
        compatible.load_state_dict(original.state_dict(), strict=True)
        shape = (1, 3, 2, 32, 48) if kind == "encoder" else (1, 2, 2, 2, 3)
        inputs = torch.randn(shape)
    coords = generate_polar_coords(32, 48)
    with torch.inference_mode():
        torch.testing.assert_close(compatible(inputs, coords), original(inputs, coords), rtol=0, atol=0)


@pytest.mark.parametrize("joint", [False, True])
def test_encoder_full_and_streaming_are_causal(joint):
    with torch.random.fork_rng():
        torch.manual_seed(721)
        model = make_model("encoder", **network_config(joint))
        activate_residuals(model)
        pixels = torch.randn(2, 3, 4, 32, 48)
    coords = generate_polar_coords(32, 48)
    with torch.inference_mode():
        full = model(pixels, coords)
        first, cache = model.forward_stream(pixels[:, :, :3], coords)
        last, cache = model.forward_stream(pixels[:, :, 3:], coords, cache)
        torch.testing.assert_close(torch.cat([first, last], dim=2), full, rtol=1e-4, atol=1e-4)
        pixels[:, :, -1] += 10
        changed = model(pixels, coords)
        torch.testing.assert_close(changed[:, :, :-1], full[:, :, :-1], rtol=0, atol=0)
        assert not torch.equal(changed[:, :, -1], full[:, :, -1])
        with pytest.raises(ValueError, match="unused temporal KV cache keys"):
            model.forward_stream(pixels[:, :, :1], coords, {"unknown": next(iter(cache.values()))})


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_joint_3d_cache_rollover_matches_window_replay(chunk_size):
    with torch.random.fork_rng():
        torch.manual_seed(721)
        model = Joint3DSelfAttention(dim=16, num_heads=1, max_t=4, len_h=2, len_w=3).eval()
        activate_residuals(model)
        inputs = torch.randn(2, 8, 2, 3, 16)
    coords = generate_polar_coords(2, 3)
    cache = None
    with torch.inference_mode():
        for start in range(0, inputs.shape[1], chunk_size):
            end = min(start + chunk_size, inputs.shape[1])
            output, cache = model.forward_stream(inputs[:, start:end], coords, cache)
            window = inputs[:, max(0, end - model.max_t) : end]
            expected = model(window, coords)[:, -(end - start) :]
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
            assert all(tensor.shape == (2, 1, min(end, 4), 6, 16) for tensor in cache)
        with pytest.raises(ValueError, match="exceeds bottleneck_3d_max_t"):
            model.forward_stream(inputs[:, :5], coords)


def test_temporal_sdpa_batch_splitting_preserves_full_and_cached_attention():
    with torch.random.fork_rng():
        torch.manual_seed(721)
        model = CausalTemporalAttention(dim=8, num_heads=2).eval()
        activate_residuals(model)
        inputs = torch.randn(5, 4, 8)
    with torch.inference_mode():
        expected_full = model(inputs)
        _, prefix = model.forward_stream(inputs[:, :2])
        expected_chunk, expected_cache = model.forward_stream(inputs[:, 2:], prefix)
        model._SDPA_MAX_BATCH = 2
        torch.testing.assert_close(model(inputs), expected_full, rtol=1e-6, atol=1e-6)
        chunk, cache = model.forward_stream(inputs[:, 2:], prefix)
        torch.testing.assert_close(chunk, expected_chunk, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
