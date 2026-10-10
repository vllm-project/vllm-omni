# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standalone LiDAR VAE encoder normalization, streaming, and checkpoint loading."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.cosmos3.lidar import (
    Cosmos3LidarEncoder,
    postprocess_lidar_decoder_output,
    prepare_lidar_encoder_input,
    validate_lidar_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def lidar_config() -> dict:
    return {
        "dtype": "float32",
        "sample_posterior": False,
        "apply_validity_mask": True,
        "fps": 10.0,
        "latent_channels": 128,
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
            "max_range_m": 100.0,
        },
        "network_config": {
            "resolution": [128, 1808],
            "patch_size": [2, 2],
            "depths": [3, 3, 3, 3],
            "temporal_downsample": [False, False, False],
            "z_dim": 128,
            "in_channels": 3,
        },
    }


def numeric_frames(sweeps=3):
    frames = torch.zeros(3, sweeps, 128, 1800)
    frames[0] = 52.5
    frames[1:] = 1
    return frames


def test_physical_normalization_circular_padding_and_validity():
    frames = numeric_frames(1)
    frames[0, 0, 0, :4] = torch.tensor([0.0, 4.9, 5.0, 100.1])
    frames[2, 0, 1, 0] = 0
    frames[1, 0, 2, 0] = 0.25
    normalized = prepare_lidar_encoder_input(frames, lidar_config()["range_projection"])
    assert normalized.shape == (1, 3, 1, 128, 1808)
    torch.testing.assert_close(normalized[..., :4], normalized[..., 1800:1804])
    torch.testing.assert_close(normalized[..., -4:], normalized[..., 4:8])
    assert normalized[0, :, 0, 0, 4].tolist() == [-1, -1, 0]
    assert normalized[0, :, 0, 0, 6].tolist() == [-1, 1, 1]
    assert normalized[0, :, 0, 0, 7].tolist() == [-1, -1, 0]
    assert normalized[0, :, 0, 1, 4].tolist() == [-1, -1, 0]
    assert normalized[0, 1, 0, 2, 4].item() == -0.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "bfloat16"),
        ("sample_posterior", True),
        ("sample_posterior", 0),
        ("apply_validity_mask", 1),
        ("fps", 0),
        ("spatial_compression", [8, 8]),
        ("temporal_compression_factor", 4),
        ("streaming_context_frames", 1),
    ],
)
def test_rejects_incompatible_encoder_metadata(field, value):
    config = lidar_config()
    validate_lidar_config(config)
    config[field] = value
    with pytest.raises(ValueError):
        validate_lidar_config(config)


def fake_encoder_model(apply_validity_mask=True):
    model = object.__new__(Cosmos3LidarEncoder)
    torch.nn.Module.__init__(model)
    model.config = lidar_config()
    model.config["apply_validity_mask"] = apply_validity_mask
    model.coords = torch.zeros(1, 2, 128, 1808)
    model.latent_mean = torch.tensor(0.25)
    model.latent_std = torch.tensor(0.5)
    calls = []

    class Encoder(torch.nn.Module):
        def forward_stream(self, pixels, coords, cache):
            assert pixels.dtype == coords.dtype == torch.float32
            assert not torch.is_autocast_enabled("cpu")
            calls.append((pixels.shape[2], 0 if cache is None else cache["temporal"][0].shape[2]))
            channels = torch.cat((pixels[:, :1, :, :1, :1], torch.full_like(pixels[:, :1, :, :1, :1], 999)), dim=1)
            # Fake log variance is deliberately huge: only the posterior mean is encoded.
            return channels, {"temporal": (torch.zeros(1, 1, 9, 1), torch.zeros(1, 1, 9, 1))}

    model.encoder = Encoder()
    model.quant_conv = torch.nn.Identity()
    return model, calls


@pytest.mark.parametrize("apply_validity_mask", [False, True])
def test_encoder_uses_fp32_posterior_mean_chunk_context_and_latent_affine(apply_validity_mask):
    model, calls = fake_encoder_model(apply_validity_mask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(numeric_frames(5))
    assert calls == [(2, 0), (2, 1), (1, 2)]
    assert result.shape == (1, 1, 5, 1, 1)
    torch.testing.assert_close(result, torch.full_like(result, -0.5))
    assert not hasattr(model, "decode")


def test_encoder_accepts_batched_decoder_output():
    model, _ = fake_encoder_model()
    decoded = postprocess_lidar_decoder_output(torch.zeros(1, 3, 5, 128, 1808), model.config)
    assert decoded.shape == (1, 3, 5, 128, 1800)
    torch.testing.assert_close(model(decoded), model(decoded[0]))

    frames = torch.stack((numeric_frames(5), numeric_frames(5)))
    frames[1, 0] = 24.0
    result = model(frames)
    assert result.shape == (2, 1, 5, 1, 1)
    for index in range(2):
        torch.testing.assert_close(result[index : index + 1], model(frames[index]))
    assert not torch.equal(result[0], result[1])


@pytest.mark.parametrize(
    "shape",
    [
        (3, 128, 1800),
        (1, 1, 3, 1, 128, 1800),
        (0, 3, 1, 128, 1800),
        (4, 1, 128, 1800),
        (1, 2, 1, 128, 1800),
        (3, 0, 128, 1800),
        (3, 1, 127, 1800),
        (3, 1, 128, 1799),
        (1, 3, 1, 128, 1808),
    ],
)
def test_encoder_rejects_invalid_frame_geometry(shape):
    model, calls = fake_encoder_model()
    with pytest.raises(ValueError, match="Expected nonempty LiDAR frames"):
        prepare_lidar_encoder_input(torch.zeros(shape), model.config["range_projection"])
    with pytest.raises(ValueError, match="Expected nonempty LiDAR frames"):
        model(torch.zeros(shape))
    assert calls == []


@pytest.mark.parametrize("model_width", [6, 10])
def test_encoder_padding_uses_projection_widths(model_width):
    frames = torch.ones(3, 1, 1, 6)
    frames[0] = 52.5
    frames[1] = torch.linspace(0, 1, 6)
    projection = {
        "model_width": model_width,
        "semantic_width": 6,
        "native_height": 1,
        "min_range_m": 5,
        "max_range_m": 100,
    }
    actual = prepare_lidar_encoder_input(frames, projection)
    half_padding = (model_width - 6) // 2
    columns = [(index - half_padding) % 6 for index in range(model_width)]
    assert actual.shape == (1, 3, 1, 1, model_width)
    torch.testing.assert_close(actual[0, 1, 0, 0], (frames[1, 0, 0] * 2 - 1)[columns])
    assert actual[0, 0].eq(0).all() and actual[0, 2].eq(1).all()


class TinyLidarEncoder(Cosmos3LidarEncoder):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.config = config
        self.encoder = torch.nn.Linear(1, config["network_config"].get("base_channels", 1))
        self.quant_conv = torch.nn.Linear(1, 1)
        self.register_buffer("coords", torch.zeros(1, 2, 1, 1))
        self.register_buffer("latent_mean", torch.zeros(1))
        self.register_buffer("latent_std", torch.ones(1))


@pytest.fixture
def lidar_vae_artifact(tmp_path):
    config = lidar_config()
    component = {**config, "network_config": {**config["network_config"], "base_channels": 4, "decoder_depths": None}}
    folder = tmp_path / "lidar_vae"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(component))
    model = TinyLidarEncoder(component).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0001)  # Detect rounding through the pipeline's BF16 default.
    state = {
        **model.state_dict(),
        "decoder.weight": torch.ones(1),
        "post_quant_conv.weight": torch.ones(1),
    }
    save_file(state, folder / "diffusion_pytorch_model.safetensors")
    return tmp_path, config, component, state


def test_encoder_artifact_inventory_and_fp32_loading(lidar_vae_artifact, monkeypatch):
    import safetensors

    path, config, component, state = lidar_vae_artifact
    safe_open = safetensors.safe_open
    reads = []

    @contextmanager
    def encoder_only_open(filename, **kwargs):
        assert kwargs == {"framework": "pt", "device": "cpu"}
        with safe_open(filename, **kwargs) as handle:

            def get_slice(name):
                assert not name.startswith(("decoder.", "post_quant_conv."))
                return handle.get_slice(name)

            def get_tensor(name):
                assert not name.startswith(("decoder.", "post_quant_conv."))
                reads.append(name)
                return handle.get_tensor(name)

            yield SimpleNamespace(keys=handle.keys, get_slice=get_slice, get_tensor=get_tensor)

    monkeypatch.setattr(safetensors, "safe_open", encoder_only_open)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        loaded = TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))
    finally:
        torch.set_default_dtype(previous)
    assert loaded.config == component
    assert loaded.encoder.out_features == 4
    assert set(reads) == loaded.state_dict().keys()
    for name, tensor in loaded.state_dict().items():
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert not hasattr(loaded, "decoder") and not hasattr(loaded, "decode")


def test_encoder_resolves_vae_from_hub(lidar_vae_artifact, monkeypatch):
    from vllm_omni.model_executor.model_loader import weight_utils

    path, config, _, _ = lidar_vae_artifact

    def download(*, model_name_or_path, cache_dir, allow_patterns, require_all):
        assert model_name_or_path == "test-org/joint-lidar-model"
        assert cache_dir is None
        assert allow_patterns == ["lidar_vae/config.json", "lidar_vae/diffusion_pytorch_model.safetensors"]
        assert require_all is True
        return str(path)

    monkeypatch.setattr(weight_utils, "download_weights_from_hf_specific", download)
    loaded = TinyLidarEncoder.from_pretrained("test-org/joint-lidar-model", config, torch.device("cpu"))
    assert loaded.encoder.out_features == 4


@pytest.mark.parametrize("missing", ["config.json", "diffusion_pytorch_model.safetensors"])
def test_encoder_requires_vae_files(lidar_vae_artifact, missing):
    path, config, _, _ = lidar_vae_artifact
    (path / "lidar_vae" / missing).unlink()
    with pytest.raises(ValueError, match="Incomplete joint artifact: lidar_vae/config.json"):
        TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))


@pytest.mark.parametrize("missing", ["encoder.weight", "quant_conv.weight", "coords", "latent_mean", "latent_std"])
def test_encoder_requires_complete_encoder_state(lidar_vae_artifact, missing):
    path, config, _, state = lidar_vae_artifact
    del state[missing]
    save_file(state, path / "lidar_vae/diffusion_pytorch_model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))


@pytest.mark.parametrize(
    "name,value,error,message",
    [
        ("encoder.weight", torch.ones(2, 1), RuntimeError, "size mismatch"),
        ("latent_mean", torch.zeros(2), RuntimeError, "size mismatch"),
        ("encoder.weight", torch.ones(4, 1, dtype=torch.bfloat16), ValueError, "must be FP32"),
        ("quant_conv.weight", torch.ones(1, 1, dtype=torch.float16), ValueError, "must be FP32"),
        ("coords", torch.zeros(1, 2, 1, 1, dtype=torch.bfloat16), ValueError, "must be FP32"),
        ("latent_std", torch.ones(1, dtype=torch.int64), ValueError, "must be FP32"),
        ("latent_mean", torch.tensor([float("nan")]), ValueError, "positive standard deviations"),
        ("latent_mean", torch.tensor([float("inf")]), ValueError, "positive standard deviations"),
        ("latent_std", torch.tensor([float("nan")]), ValueError, "positive standard deviations"),
        ("latent_std", torch.tensor([float("inf")]), ValueError, "positive standard deviations"),
        ("latent_std", torch.tensor([0.0]), ValueError, "positive standard deviations"),
        ("latent_std", torch.tensor([-1.0]), ValueError, "positive standard deviations"),
        ("encoder.unexpected", torch.ones(1), ValueError, "Unexpected LiDAR encoder tensors"),
        ("optimizer.step", torch.ones(1), ValueError, "Unexpected LiDAR encoder tensors"),
    ],
)
def test_encoder_rejects_invalid_vae_encoder_state(lidar_vae_artifact, name, value, error, message):
    path, config, _, state = lidar_vae_artifact
    state[name] = value
    save_file(state, path / "lidar_vae/diffusion_pytorch_model.safetensors")
    with pytest.raises(error, match=message):
        TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))


@pytest.mark.parametrize(
    "field", ["fps", "network_config", "range_projection", "apply_validity_mask", "decoder_depths", "missing_default"]
)
def test_encoder_rejects_conflicting_vae_metadata(lidar_vae_artifact, field):
    path, config, component, _ = lidar_vae_artifact
    if field == "network_config":
        component[field]["depths"] = [1, 1, 1, 1]
    elif field == "range_projection":
        component[field] = {**component[field], "max_range_m": 105.0}
    elif field == "decoder_depths":
        config["network_config"][field] = [1, 1, 1, 1]
    elif field == "missing_default":
        config["network_config"]["decoder_depths"] = None
        del component["network_config"]["decoder_depths"]
    else:
        component[field] = False if field == "apply_validity_mask" else 11.0
    (path / "lidar_vae/config.json").write_text(json.dumps(component))
    with pytest.raises(ValueError, match="metadata disagrees"):
        TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))


@pytest.mark.parametrize("field", ["dtype", "sample_posterior", "apply_validity_mask"])
def test_encoder_requires_vae_policy_metadata(lidar_vae_artifact, field):
    path, config, component, _ = lidar_vae_artifact
    del component[field]
    (path / "lidar_vae/config.json").write_text(json.dumps(component))
    with pytest.raises(ValueError, match="missing LiDAR metadata"):
        TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))


def test_encoder_loads_real_architecture_with_saved_constructor_defaults(tmp_path):
    import inspect

    from vllm_omni.diffusion.models.cosmos3.lidar_encoder.transformer_vae import Encoder

    config = lidar_config()
    config["latent_channels"] = 4
    config["network_config"].update(z_dim=4, base_channels=4, depths=[1] * 4, num_heads=[1] * 4)
    arguments = inspect.signature(Encoder).bind(**config["network_config"])
    arguments.apply_defaults()
    network = json.loads(json.dumps(arguments.arguments))
    # Constructor defaults and decoder-only overrides are absent from deployment metadata.
    component = {**config, "network_config": {**network, "decoder_depths": None, "out_channels": 3}}
    model = Cosmos3LidarEncoder(component).float()
    state = model.state_dict()
    state["latent_mean"].fill_(0.1234567)
    state["latent_std"].fill_(0.9876543)
    state["encoder.tokenizer.0.weight"].fill_(0.1234567)
    folder = tmp_path / "lidar_vae"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(component))
    save_file({**state, "decoder.weight": torch.ones(1)}, folder / "diffusion_pytorch_model.safetensors")
    loaded = Cosmos3LidarEncoder.from_pretrained(str(tmp_path), config, torch.device("cpu"))
    assert loaded.config == component
    assert loaded.state_dict().keys() == state.keys()
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
