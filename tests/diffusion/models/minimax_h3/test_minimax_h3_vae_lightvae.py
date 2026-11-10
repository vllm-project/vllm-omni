# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the portable checkpoint module without importing the serving stack."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _api(component="checkpoint"):
    package = "h3_lightvae_checkpoint_test_target"
    root = Path(__file__).resolve().parents[4]
    directory = root / "vllm_omni/diffusion/models/minimax_h3/lightvae"
    if package not in sys.modules:
        namespace = types.ModuleType(package)
        namespace.__path__ = [str(directory)]
        sys.modules[package] = namespace
    name = f"{package}.{component}"
    if name not in sys.modules:
        path = directory / f"{component}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def _native_checkpoint():
    nn = torch.nn
    model = nn.Module()
    model.encoder = nn.Linear(2, 2)
    model.quant_conv = nn.Conv3d(2, 2, 1)
    model.post_quant_conv = nn.Conv3d(2, 2, 1)
    decoder = nn.Module()
    decoder.x_embedder = nn.Linear(2, 16)
    decoder.register_tokens = nn.Parameter(torch.zeros(1, 4, 16))
    decoder.mask_token = nn.Parameter(torch.zeros(1, 1, 16))
    decoder.norm_out = nn.LayerNorm(16, eps=1e-5)
    decoder.proj_out = nn.Linear(16, 3)
    decoder.pos_embed = nn.Module()
    decoder.pos_embed.n_dim = 3
    decoder.pos_embed.angle_scale = 6.283185307179586
    decoder.pos_embed.register_buffer("inv_freq", torch.tensor([1.0]), persistent=False)
    decoder.transformer_blocks = nn.ModuleList()
    for _ in range(4):
        block = nn.Module()
        block.norm1, block.norm2 = nn.RMSNorm(16, eps=1e-5), nn.RMSNorm(16, eps=1e-5)
        block.scale1 = nn.Parameter(torch.ones(16))
        block.scale2 = nn.Parameter(torch.ones(16))
        block.attn = nn.Module()
        block.attn.to_qkv, block.attn.to_out = nn.Linear(16, 48), nn.Linear(16, 16)
        block.attn.dim_head = 8
        block.attn.norm_q = nn.RMSNorm(8, eps=1e-5, elementwise_affine=False)
        block.attn.norm_k = nn.RMSNorm(8, eps=1e-5, elementwise_affine=False)
        block.ff = nn.Module()
        block.ff.w1, block.ff.w2 = nn.Linear(16, 32), nn.Linear(16, 16)
        decoder.transformer_blocks.append(block)
    model.decoder = decoder
    config = {
        "latent_channels": 2,
        "latents_mean": [0.0, 0.0],
        "latents_std": [1.0, 1.0],
        "decoder_num_layers": 4,
        "decoder_num_attention_heads": 2,
        "decoder_attention_head_dim": 8,
        "decoder_norm_eps": 1e-5,
        "decoder_rope_theta": 100.0,
        "decoder_rope_dim_ratio": 0.75,
        "clip_length": 17,
        "token_drop": 3,
    }
    expected, state = {}, {}
    for key, tensor in model.state_dict().items():
        if not key.startswith(("decoder.", "post_quant_conv.")) or key == "decoder.mask_token":
            continue
        if key.startswith("decoder.transformer_blocks."):
            parts = key.split(".")
            if int(parts[2]) not in (1, 3):
                continue
            parts[2] = str((1, 3).index(int(parts[2])))
            key = ".".join(parts)
        tensor = tensor.clone() + 10
        expected[key] = tensor
        source = key.replace("decoder.x_embedder.", "decoder.proj_in.")
        if ".attn.to_qkv." in source:
            packed = tensor.reshape(2, 3, 8, *tensor.shape[1:])
            for index, part in enumerate(("q", "k", "v")):
                state[source.replace(".to_qkv.", f".to_{part}.")] = (
                    packed[:, index].reshape(16, *tensor.shape[1:]).contiguous()
                )
        elif ".ff.w1." in source:
            gate, up = tensor.chunk(2, dim=0)
            state[source.replace(".ff.w1.", ".ff.net.0.proj.")] = torch.cat((up, gate))
        else:
            source = source.replace(".attn.to_out.", ".attn.to_out.0.").replace(".ff.w2.", ".ff.net.2.")
            state[source] = tensor
    checkpoint = _api().H3LiteDecoderCheckpoint(config, (1, 3), state)
    return model, checkpoint, expected


@pytest.mark.parametrize("default_device", ["cpu", "meta"])
def test_native_install_loads_all_lite_weights_without_changing_encoder(default_device):
    model, checkpoint, expected = _native_checkpoint()
    encoder = {key: tensor.clone() for key, tensor in model.encoder.state_dict().items()}
    # Pipeline construction may set an accelerator default device while the
    # native VAE and checkpoint are deliberately still loaded on the CPU.
    with torch.device(default_device):
        _api("native").install_h3_lite_decoder(model, checkpoint, native_config=checkpoint.config)
    assert len(model.decoder.transformer_blocks) == 2
    actual = model.state_dict()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key
    for key, value in encoder.items():
        assert torch.equal(model.encoder.state_dict()[key], value)
    assert torch.count_nonzero(model.decoder.mask_token) == 0


def test_native_install_rejects_incompatible_normalization():
    model, checkpoint, _ = _native_checkpoint()
    config = dict(checkpoint.config, latents_mean=[1.0, 1.0])
    with pytest.raises(ValueError, match="latents_mean"):
        _api("native").install_h3_lite_decoder(model, checkpoint, native_config=config)
    assert len(model.decoder.transformer_blocks) == 4


def test_native_install_validates_runtime_when_native_config_omits_decoder_fields():
    """Released H3 component configs leave decoder semantics in the source class."""
    model, checkpoint, expected = _native_checkpoint()
    config = {
        key: value
        for key, value in checkpoint.config.items()
        if key not in ("decoder_norm_eps", "decoder_rope_theta", "decoder_rope_dim_ratio")
    }
    _api("native").install_h3_lite_decoder(model, checkpoint, native_config=config)
    assert len(model.decoder.transformer_blocks) == 2
    for key, value in expected.items():
        assert torch.equal(model.state_dict()[key], value), key


def test_native_install_rejects_missing_weights():
    model, checkpoint, _ = _native_checkpoint()
    checkpoint.state_dict.pop("decoder.proj_in.weight")
    with pytest.raises(ValueError, match="x_embedder.weight"):
        _api("native").install_h3_lite_decoder(model, checkpoint, native_config=checkpoint.config)


@pytest.mark.parametrize("target", ["norm_out", "norm1", "norm2", "norm_q", "norm_k"])
@pytest.mark.parametrize("omit_config", [False, True])
def test_native_install_rejects_runtime_norm_epsilon_before_mutation(target, omit_config):
    model, checkpoint, _ = _native_checkpoint()
    block = model.decoder.transformer_blocks[1]
    norm = (
        model.decoder.norm_out
        if target == "norm_out"
        else getattr(block.attn if target.startswith("norm_") else block, target)
    )
    norm.eps = 0.1
    config = dict(checkpoint.config)
    if omit_config:
        config.pop("decoder_norm_eps")
    before = model.decoder.x_embedder.weight.detach().clone()
    with pytest.raises(ValueError, match="norm_eps"):
        _api("native").install_h3_lite_decoder(model, checkpoint, native_config=config)
    assert len(model.decoder.transformer_blocks) == 4
    assert torch.equal(before, model.decoder.x_embedder.weight)


@pytest.mark.parametrize("failure", ["theta", "ratio", "buffer", "axes", "angle", "missing"])
def test_native_install_rejects_incompatible_rope(failure):
    model, checkpoint, _ = _native_checkpoint()
    config = dict(checkpoint.config)
    if failure == "theta":
        config["decoder_rope_theta"] = 10000.0
    elif failure == "ratio":
        config["decoder_rope_dim_ratio"] = 0.5
    elif failure == "buffer":
        model.decoder.pos_embed.inv_freq.fill_(0.5)
    elif failure == "axes":
        model.decoder.pos_embed.n_dim = 2
    elif failure == "angle":
        model.decoder.pos_embed.angle_scale = 1.0
    else:
        del model.decoder.pos_embed
    with pytest.raises(ValueError, match="rope"):
        _api("native").install_h3_lite_decoder(model, checkpoint, native_config=config)
    assert len(model.decoder.transformer_blocks) == 4


def test_native_install_requires_cpu_and_supported_structure():
    model, checkpoint, _ = _native_checkpoint()
    model.decoder = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="structure"):
        _api("native").install_h3_lite_decoder(model, checkpoint, native_config=checkpoint.config)
    model, checkpoint, _ = _native_checkpoint()
    model.to("meta")
    with pytest.raises(ValueError, match="CPU"):
        _api("native").install_h3_lite_decoder(model, checkpoint, native_config=checkpoint.config)


@pytest.fixture
def lite_checkpoint_factory(tmp_path):
    def create(*, changes=None, state=None, nested=False, metadata=None):
        config = {
            "family": "h3",
            "latent_channels": 2,
            "latents_mean": [0.0, 0.0],
            "latents_std": [1.0, 1.0],
            "decoder_num_layers": 4,
            "decoder_num_attention_heads": 2,
            "decoder_attention_head_dim": 2,
            "clip_length": 17,
            "token_drop": 3,
            "kept_layers": [1, 3],
        }
        config.update(changes or {})
        architecture = config
        if nested:
            architecture = {
                "teacher_config": {k: v for k, v in config.items() if k != "kept_layers"},
                "kept_layers": config["kept_layers"],
                "tile_sample_min_height": 256,
                "tile_sample_min_overlap_height": 64,
            }
        path = tmp_path / "lite.safetensors"
        save_file(
            state
            if state is not None
            else {
                "decoder.transformer_blocks.0.scale1": torch.ones(4),
                "decoder.transformer_blocks.1.scale1": torch.full((4,), 2.0),
                "post_quant_conv.weight": torch.ones(2, 2, 1, 1, 1),
            },
            path,
            metadata=metadata if metadata is not None else {"architecture": json.dumps(architecture)},
        )
        return path

    return create


@pytest.mark.parametrize("nested", [False, True])
def test_compact_indices_do_not_reapply_teacher_pruning(lite_checkpoint_factory, nested):
    checkpoint = _api().read_h3_lite_decoder_checkpoint(lite_checkpoint_factory(nested=nested))
    assert checkpoint.kept_layers == (1, 3)
    assert checkpoint.state_dict["decoder.transformer_blocks.0.scale1"].tolist() == [1.0] * 4
    assert checkpoint.state_dict["decoder.transformer_blocks.1.scale1"].tolist() == [2.0] * 4
    assert all(value.device.type == "cpu" for value in checkpoint.state_dict.values())


@pytest.mark.parametrize("suffix", ["weight", "bias"])
def test_qkv_is_packed_per_head_including_bias(suffix):
    state = {}
    for part, offset in (("q", 0), ("k", 10), ("v", 20)):
        value = (torch.arange(4) + offset).float()
        if suffix == "weight":
            value = value[:, None].expand(4, 2).clone()
        state[f"decoder.transformer_blocks.0.attn.to_{part}.{suffix}"] = value
    result = _api().convert_h3_lite_decoder_state_dict(state, num_heads=2)
    actual = result[f"decoder.transformer_blocks.0.attn.to_qkv.{suffix}"]
    expected = torch.tensor([0, 1, 10, 11, 20, 21, 2, 3, 12, 13, 22, 23]).float()
    assert torch.equal(actual if suffix == "bias" else actual[:, 0], expected)
    if suffix == "weight":
        x = torch.tensor([[0.25, -0.5]])
        projected = torch.nn.functional.linear(x, actual).reshape(1, 2, 3, 2)
        separate = [
            torch.nn.functional.linear(x, state[f"decoder.transformer_blocks.0.attn.to_{p}.weight"]).reshape(1, 2, 2)
            for p in ("q", "k", "v")
        ]
        torch.testing.assert_close(projected, torch.stack(separate, dim=2), atol=1e-4, rtol=1e-4)


def test_mlp_swaps_up_gate_including_bias():
    state = {
        "decoder.transformer_blocks.0.ff.net.0.proj.weight": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "decoder.transformer_blocks.0.ff.net.0.proj.bias": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "decoder.transformer_blocks.0.ff.net.2.weight": torch.ones(1, 2),
        "decoder.transformer_blocks.0.attn.to_out.0.weight": torch.ones(2, 2),
        "decoder.proj_in.weight": torch.ones(2, 1),
        "decoder.proj_out.bias": torch.ones(3),
        "decoder.norm_out.weight": torch.ones(2),
        "decoder.register_tokens": torch.ones(1, 4, 2),
        "post_quant_conv.bias": torch.ones(1),
    }
    converted = _api().convert_h3_lite_decoder_state_dict(state, num_heads=2)
    assert converted["decoder.transformer_blocks.0.ff.w1.bias"].tolist() == [3, 4, 1, 2]
    assert converted["decoder.transformer_blocks.0.ff.w1.weight"][:, 0].tolist() == [
        3,
        4,
        1,
        2,
    ]
    assert set(converted) == {
        "decoder.transformer_blocks.0.ff.w1.weight",
        "decoder.transformer_blocks.0.ff.w1.bias",
        "decoder.transformer_blocks.0.ff.w2.weight",
        "decoder.transformer_blocks.0.attn.to_out.weight",
        "decoder.x_embedder.weight",
        "decoder.proj_out.bias",
        "decoder.norm_out.weight",
        "decoder.register_tokens",
        "post_quant_conv.bias",
    }


@pytest.mark.parametrize("kept", [[], [1, 1], [3, 1], [-1], [4], [True]])
def test_invalid_teacher_indices_are_rejected(lite_checkpoint_factory, kept):
    with pytest.raises(ValueError, match="kept_layers"):
        _api().read_h3_lite_decoder_checkpoint(lite_checkpoint_factory(changes={"kept_layers": kept}))


@pytest.mark.parametrize("kept", [[0], [3], [0, 3]])
def test_kept_layers_accept_boundary_indices(lite_checkpoint_factory, kept):
    state = {
        f"decoder.transformer_blocks.{index}.scale1": torch.ones(4)
        for index in range(len(kept))
    }
    state["post_quant_conv.weight"] = torch.ones(2, 2, 1, 1, 1)
    checkpoint = _api().read_h3_lite_decoder_checkpoint(
        lite_checkpoint_factory(changes={"kept_layers": kept}, state=state)
    )
    assert checkpoint.kept_layers == tuple(kept)


@pytest.mark.parametrize("depth", [0, -1, True, 1])
def test_compact_decoder_depth_boundaries(lite_checkpoint_factory, depth):
    state = {
        "decoder.transformer_blocks.0.scale1": torch.ones(4),
        "post_quant_conv.weight": torch.ones(2, 2, 1, 1, 1),
    }
    changes = {"decoder_num_layers": depth}
    if type(depth) is int and depth == 1:
        changes["kept_layers"] = [0]
        checkpoint = _api().read_h3_lite_decoder_checkpoint(
            lite_checkpoint_factory(changes=changes, state=state)
        )
        assert checkpoint.kept_layers == (0,)
    else:
        with pytest.raises(ValueError, match="decoder_num_layers"):
            _api().read_h3_lite_decoder_checkpoint(lite_checkpoint_factory(changes=changes))


@pytest.mark.parametrize("metadata", [{}, {"architecture": "invalid"}, {"architecture": "[]"}])
def test_missing_or_invalid_architecture_is_rejected(lite_checkpoint_factory, metadata):
    with pytest.raises(ValueError, match="architecture"):
        _api().read_h3_lite_decoder_checkpoint(lite_checkpoint_factory(metadata=metadata))


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"family": "wan21"}, "family"),
        ({"latents_std": [1.0, 0.0]}, "latents_std"),
        ({"latents_mean": [0.0]}, "latents_mean"),
        ({"clip_length": 9}, "clip_length"),
        ({"clip_length": 16}, "clip_length"),
        ({"clip_length": 18}, "clip_length"),
        ({"token_drop": 0}, "token_drop"),
        ({"token_drop": 4}, "token_drop"),
    ],
)
def test_incompatible_architecture_is_rejected(lite_checkpoint_factory, changes, field):
    with pytest.raises(ValueError, match=field):
        _api().read_h3_lite_decoder_checkpoint(lite_checkpoint_factory(changes=changes))


def test_quantized_weights_are_rejected(lite_checkpoint_factory):
    path = lite_checkpoint_factory(state={"decoder.proj_in.weight": torch.ones(2, 2, dtype=torch.int8)})
    with pytest.raises(ValueError, match="decoder.proj_in.weight"):
        _api().read_h3_lite_decoder_checkpoint(path)


def test_unknown_compact_checkpoint_tensor_is_rejected(lite_checkpoint_factory):
    path = lite_checkpoint_factory(
        state={
            "decoder.transformer_blocks.0.scale1": torch.ones(4),
            "encoder.conv_in.weight": torch.ones(1, 1, 1, 1, 1),
        }
    )
    with pytest.raises(ValueError, match="unsupported checkpoint tensor"):
        _api().read_h3_lite_decoder_checkpoint(path)


def test_qkv_shape_and_missing_projection_are_rejected():
    key = "decoder.transformer_blocks.0.attn.to_q.weight"
    with pytest.raises(ValueError, match="to_[qkv]"):
        _api().convert_h3_lite_decoder_state_dict({key: torch.ones(4, 2)}, num_heads=2)
    state = {key.replace("to_q", f"to_{part}"): torch.ones(rows, 2) for part, rows in (("q", 4), ("k", 3), ("v", 4))}
    with pytest.raises(ValueError, match="to_[qkv]"):
        _api().convert_h3_lite_decoder_state_dict(state, num_heads=2)


def test_unknown_decoder_parameter_is_not_silently_dropped():
    with pytest.raises(ValueError, match="unknown"):
        _api().convert_h3_lite_decoder_state_dict({"decoder.unknown.weight": torch.ones(2)}, num_heads=2)


def test_optional_path_and_explicit_invalid_values(tmp_path):
    api = _api()
    assert api.resolve_h3_lightvae_decoder_path({}) is None
    assert api.resolve_h3_lightvae_decoder_path(None) is None
    path = str(tmp_path / "lite.safetensors")
    assert api.resolve_h3_lightvae_decoder_path({"lightvae_decoder_path": path}) == path
    for value in (None, "", "  ", 1, True):
        with pytest.raises(ValueError, match="lightvae_decoder_path"):
            api.resolve_h3_lightvae_decoder_path({"lightvae_decoder_path": value})


@pytest.mark.parametrize("decoder_source", ["decoder-nvfp4.safetensors", {"model": "released-decoder"}])
def test_lightvae_rejects_conflicting_decoder_selection(decoder_source):
    with pytest.raises(ValueError, match="cannot be combined with video_vae_decoder"):
        _api().resolve_h3_lightvae_decoder_path(
            {"lightvae_decoder_path": "lite.safetensors", "video_vae_decoder": decoder_source}
        )


@pytest.fixture
def lite_directory_factory(tmp_path):
    def create(
        *,
        student_layers=26,
        latents=24,
        decode_config=None,
        config_changes=None,
        state=None,
        shards=1,
    ):
        component = tmp_path / "lite26"
        component.mkdir()
        config = {
            "_class_name": "AutoencoderKLMiniMaxH3",
            "latent_channels": latents,
            "decoder_num_layers": student_layers,
            "decoder_num_attention_heads": 2,
            "decoder_attention_head_dim": 2,
            "decoder_norm_eps": 1e-5,
            "decoder_rope_theta": 100.0,
            "decoder_rope_dim_ratio": 0.75,
            "latents_mean": [0.0] * latents,
            "latents_std": [1.0] * latents,
            "clip_length": 17,
            "token_drop": 3,
        }
        config.update(config_changes or {})
        (component / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if state is None:
            state = {
                f"decoder.transformer_blocks.{index}.scale1": torch.ones(2)
                for index in range(student_layers)
            }
            state["post_quant_conv.weight"] = torch.ones(1)
        if shards == 1:
            save_file(state, component / "diffusion_pytorch_model.safetensors")
        else:
            weight_map = {}
            keys = list(state)
            for shard_index in range(shards):
                shard_keys = keys[shard_index::shards]
                shard_name = f"diffusion_pytorch_model-{shard_index + 1:05d}.safetensors"
                save_file({key: state[key] for key in shard_keys}, component / shard_name)
                weight_map.update({key: shard_name for key in shard_keys})
            (component / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
            )
        if decode_config is not None:
            (component / "decode_config.json").write_text(json.dumps(decode_config), encoding="utf-8")
        return component

    return create


def test_diffusers_directory_maps_student_to_teacher_depth(lite_directory_factory):
    path = lite_directory_factory(
        shards=2,
        decode_config={
            "student_tile_size": 272,
            "student_long_axis_tile": 208,
            "student_tile_overlap": 64,
            "student_short_axis_overlap": 0,
            "student_long_axis_overlap": 16,
            "image_context_tokens": 5,
            "temporal_decode_protocol": "released-segment-prepadding-v3",
            "tile_layout": "native",
        },
    )
    checkpoint = _api().read_h3_lite_decoder_checkpoint(path, teacher_num_layers=36)
    assert checkpoint.config["decoder_num_layers"] == 36
    assert checkpoint.kept_layers == tuple(range(26))
    assert len(checkpoint.state_dict) == 27
    assert checkpoint.decode_profile.tile_size == 272
    assert checkpoint.decode_profile.long_axis_tile == 208
    assert checkpoint.decode_profile.short_axis_overlap == 0
    assert checkpoint.decode_profile.long_axis_overlap == 16
    assert checkpoint.decode_profile.image_context_tokens == 5
    assert checkpoint.decode_profile.temporal_decode_protocol == "released-segment-prepadding-v3"


def test_diffusers_directory_uses_official_vae_config_as_fallback(lite_directory_factory):
    path = lite_directory_factory(
        decode_config={"student_tile_size": 272},
        config_changes={"vae_tile_size": 256, "vae_tile_overlap_min": 64},
    )
    checkpoint = _api().read_h3_lite_decoder_checkpoint(path, teacher_num_layers=36)
    profile = checkpoint.decode_profile
    assert profile.tile_size == 272
    assert profile.tile_overlap == 64
    assert profile.long_axis_tile == 256
    assert profile.short_axis_overlap == 64
    assert profile.long_axis_overlap == 64
    assert profile.image_tile_size == 256
    assert profile.image_long_axis_tile == 256


def test_lightvae_geometry_is_anisotropic_and_frame_aware():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(2, 2)
            self.vae_ratio = 16
            self.tokens_chunk_size = 5
            self.decoder_tile_size = 256
            self.decoder_tile_overlap_min = 64

    model = _Model()
    profile = _api().H3LiteDecodeProfile(
        tile_size=272,
        tile_overlap=64,
        long_axis_tile=208,
        short_axis_overlap=0,
        long_axis_overlap=16,
        image_tile_size=256,
        image_long_axis_tile=256,
    )
    native = _api("native")
    assert native._lightvae_geometry(model, profile, 5, 768, 1344) == (272, 208, 0, 16)
    assert native._lightvae_geometry(model, profile, 5, 1344, 768) == (208, 272, 16, 0)
    assert native._lightvae_geometry(model, profile, 5, 768, 768) == (272, 272, 16, 16)
    assert native._lightvae_geometry(model, profile, 1, 768, 1344) == (256, 256, 64, 64)


def test_lightvae_decoder_tile_count_uses_release_geometry():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(2, 2)
            self.vae_ratio = 16
            self.tokens_chunk_size = 5
            self.decoder_tile_size = 256
            self.decoder_tile_overlap_min = 64
            self._h3_lite_decode_profile = _api().H3LiteDecodeProfile(
                tile_size=272,
                tile_overlap=64,
                long_axis_tile=208,
                short_axis_overlap=0,
                long_axis_overlap=16,
                tile_layout="native",
            )

    latent = torch.zeros(1, 24, 9, 48, 84)
    native = _api("native")
    assert native.lightvae_decoder_tile_count(_Model(), latent) == 21


def test_diffusers_directory_requires_teacher_depth(lite_directory_factory):
    path = lite_directory_factory()
    with pytest.raises(ValueError, match="teacher_num_layers"):
        _api().read_h3_lite_decoder_checkpoint(path)


def test_diffusers_directory_rejects_student_deeper_than_teacher(lite_directory_factory):
    path = lite_directory_factory(student_layers=26)
    with pytest.raises(ValueError, match="exceeds teacher depth"):
        _api().read_h3_lite_decoder_checkpoint(path, teacher_num_layers=12)


@pytest.mark.parametrize("teacher", [0, -1, True])
def test_diffusers_directory_rejects_invalid_teacher_depth(lite_directory_factory, teacher):
    path = lite_directory_factory(student_layers=2)
    with pytest.raises(ValueError, match="teacher_num_layers"):
        _api().read_h3_lite_decoder_checkpoint(path, teacher_num_layers=teacher)


def test_diffusers_directory_accepts_student_equal_to_teacher(lite_directory_factory):
    path = lite_directory_factory(student_layers=2)
    checkpoint = _api().read_h3_lite_decoder_checkpoint(path, teacher_num_layers=2)
    assert checkpoint.kept_layers == (0, 1)
    assert checkpoint.config["decoder_num_layers"] == 2


def test_diffusers_directory_uses_explicit_kept_layer_mapping(lite_directory_factory):
    path = lite_directory_factory(student_layers=2, config_changes={"kept_layers": [0, 2]})
    checkpoint = _api().read_h3_lite_decoder_checkpoint(path, teacher_num_layers=36)
    assert checkpoint.kept_layers == (0, 2)


def test_diffusers_directory_rejects_missing_or_bad_config(lite_directory_factory):
    path = lite_directory_factory()
    (path / "config.json").unlink()
    with pytest.raises(ValueError, match="config.json"):
        _api().read_h3_lite_decoder_checkpoint(path, teacher_num_layers=36)


@pytest.mark.parametrize(
    "decode_config,field",
    [
        ({"student_tile_size": 0}, "student_tile_size"),
        ({"student_tile_overlap": -1}, "student_tile_overlap"),
        ({"student_long_axis_tile": 0}, "student_long_axis_tile"),
        ({"student_short_axis_overlap": -1}, "student_short_axis_overlap"),
        ({"student_long_axis_overlap": -1}, "student_long_axis_overlap"),
        ({"image_context_tokens": 0}, "image_context_tokens"),
        ({"temporal_decode_protocol": "unknown"}, "temporal_decode_protocol"),
        ({"tile_layout": "unknown"}, "tile_layout"),
        ({"tile_layout": ""}, "tile_layout"),
    ],
)
def test_diffusers_directory_rejects_bad_decode_profile(lite_directory_factory, decode_config, field):
    path = lite_directory_factory(decode_config=decode_config)
    with pytest.raises(ValueError, match=field):
        _api().read_h3_lite_decoder_checkpoint(path, teacher_num_layers=36)


def test_decode_profile_accepts_minimum_positive_values(lite_directory_factory):
    path = lite_directory_factory(
        decode_config={
            "student_tile_size": 1,
            "student_long_axis_tile": 1,
            "student_tile_overlap": 0,
            "student_short_axis_overlap": 0,
            "student_long_axis_overlap": 0,
            "image_context_tokens": 1,
            "temporal_decode_protocol": "released-segment-prepadding-v3",
            "tile_layout": "native",
        }
    )
    checkpoint = _api().read_h3_lite_decoder_checkpoint(path, teacher_num_layers=36)
    assert checkpoint.decode_profile.tile_size == 1
    assert checkpoint.decode_profile.long_axis_tile == 1
    assert checkpoint.decode_profile.tile_overlap == 0


def test_apply_decode_profile_is_scoped_to_the_native_vae():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(2, 2)
            self.vae_ratio = 8
            self.tokens_chunk_size = 5
            self.decoder_tile_size = 256
            self.decoder_tile_overlap_min = 64

    model = _Model()
    profile = _api().H3LiteDecodeProfile(
        tile_size=272,
        tile_overlap=64,
        long_axis_tile=208,
        short_axis_overlap=0,
        long_axis_overlap=16,
        image_context_tokens=5,
        temporal_decode_protocol="released-segment-prepadding-v3",
    )
    _api("native").apply_h3_lite_decode_profile(model, profile)
    assert model._h3_lite_decode_profile is profile
    assert model.decoder_tile_size == 272
    assert model.decoder_tile_overlap_min == 0
    assert model.decoder_long_axis_tile == 208
    assert model.decoder_short_axis_overlap == 0
    assert model.decoder_long_axis_overlap == 16
    assert model.decoder_image_overlap == 64
    assert not hasattr(model, "official_vae_tile_size")


def test_apply_decode_profile_is_atomic_on_failure():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(2, 2)
            self.vae_ratio = 8
            self.tokens_chunk_size = 5
            self.decoder_tile_size = 256
            self.decoder_tile_overlap_min = 64

    model = _Model()
    profile = _api().H3LiteDecodeProfile(tile_size=272, tile_overlap=65)
    with pytest.raises(ValueError, match="decoder_tile_overlap_min"):
        _api("native").apply_h3_lite_decode_profile(model, profile)
    assert model.decoder_tile_size == 256
    assert model.decoder_tile_overlap_min == 64
    assert not hasattr(model, "_h3_lite_decode_profile")


def test_apply_decode_profile_accepts_minimum_ratio_tile():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(2, 2)
            self.vae_ratio = 8
            self.tokens_chunk_size = 5
            self.decoder_tile_size = 256
            self.decoder_tile_overlap_min = 64

    model = _Model()
    profile = _api().H3LiteDecodeProfile(tile_size=8, tile_overlap=0)
    _api("native").apply_h3_lite_decode_profile(model, profile)
    assert model.decoder_tile_size == 8
    assert model.decoder_tile_overlap_min == 0


def test_apply_decode_profile_rejects_non_multiple_tile():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(2, 2)
            self.vae_ratio = 8
            self.tokens_chunk_size = 5
            self.decoder_tile_size = 256
            self.decoder_tile_overlap_min = 64

    with pytest.raises(ValueError, match="multiple of the spatial ratio"):
        _api("native").apply_h3_lite_decode_profile(
            _Model(), _api().H3LiteDecodeProfile(tile_size=7, tile_overlap=0)
        )


def test_apply_decode_profile_rejects_token_window_mismatch():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(2, 2)
            self.vae_ratio = 8
            self.tokens_chunk_size = 4
            self.decoder_tile_size = 256
            self.decoder_tile_overlap_min = 64

    profile = _api().H3LiteDecodeProfile(
        tile_size=272,
        tile_overlap=64,
        image_context_tokens=5,
        temporal_decode_protocol="released-segment-prepadding-v3",
    )
    with pytest.raises(ValueError, match="token window"):
        _api("native").apply_h3_lite_decode_profile(_Model(), profile)


def test_path_aliases_are_mutually_exclusive(tmp_path):
    api = _api()
    path = str(tmp_path / "lite")
    assert api.resolve_h3_lightvae_decoder_path({"vae_path": path}) == path
    assert api.resolve_h3_lightvae_decoder_path({"vae_decoder_path": path}) == path
    with pytest.raises(ValueError, match="use only one"):
        api.resolve_h3_lightvae_decoder_path(
            {"vae_path": path, "lightvae_decoder_path": path}
        )
