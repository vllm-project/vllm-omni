# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from tests.diffusion.models.wan2_2 import test_lingbot_world_attention as attention_tests

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "lingbot_world_weight_index_names.json.fixture"
_SHAPE_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "lingbot_world_official_shapes.json.fixture"


@pytest.fixture(autouse=True)
def _inference_context():
    with torch.inference_mode():
        yield


def _tiny_model(
    module,
    *,
    num_layers: int = 2,
    num_frames_per_block: int = 1,
    sliding_window_num_frames: int = 3,
):
    return module.CausalLingBotWorldTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=2,
        in_channels=36,
        out_channels=2,
        text_dim=6,
        freq_dim=4,
        ffn_dim=8,
        num_layers=num_layers,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=16,
        sink_size=1,
        num_frames_per_block=num_frames_per_block,
        sliding_window_num_frames=sliding_window_num_frames,
        local_attn_size=-1,
    )


def _cache(module, model, *, max_tokens: int | None = None):
    if max_tokens is None:
        window_frames = (
            model.config.local_attn_size
            if model.config.local_attn_size != -1
            else model.config.sliding_window_num_frames
        )
        max_tokens = window_frames * 2 * 2
    return module.allocate_lingbot_cache(
        batch_size=1,
        num_layers=len(model.blocks),
        max_tokens=max_tokens,
        num_local_heads=2,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _checkpoint_parameter_specs(model) -> dict[str, tuple[torch.Size, torch.dtype]]:
    specs: dict[str, tuple[torch.Size, torch.dtype]] = {}
    for name, parameter in model.named_parameters():
        marker = ".self_attn.qkv."
        if marker not in name:
            specs[name] = (parameter.shape, parameter.dtype)
            continue
        shard_shape = torch.Size((parameter.shape[0] // 3, *parameter.shape[1:]))
        for projection in ("q", "k", "v"):
            specs[name.replace(marker, f".self_attn.{projection}.")] = (shard_shape, parameter.dtype)
    return specs


def _checkpoint_weights_for_model(model) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    checkpoint_weights: dict[str, torch.Tensor] = {}
    expected_parameters: dict[str, torch.Tensor] = {}
    next_value = 1
    for name, parameter in model.named_parameters():
        marker = ".self_attn.qkv."
        if marker not in name:
            value = torch.full_like(parameter, next_value)
            checkpoint_weights[name] = value
            expected_parameters[name] = value
            next_value += 1
            continue
        shards = []
        shard_shape = (parameter.shape[0] // 3, *parameter.shape[1:])
        for projection in ("q", "k", "v"):
            value = torch.full(shard_shape, next_value, dtype=parameter.dtype, device=parameter.device)
            checkpoint_weights[name.replace(marker, f".self_attn.{projection}.")] = value
            shards.append(value)
            next_value += 1
        expected_parameters[name] = torch.cat(shards)
    return checkpoint_weights, expected_parameters


def _self_cache_snapshot(cache) -> list[tuple[torch.Tensor, torch.Tensor, tuple[int, int, int | None, int]]]:
    return [attention_tests._cache_snapshot(layer_cache) for layer_cache in cache.self_attention]


def _assert_self_cache_unchanged(cache, snapshot) -> None:
    for layer_cache, layer_snapshot in zip(cache.self_attention, snapshot, strict=True):
        attention_tests._assert_cache_unchanged(layer_cache, layer_snapshot)


def test_tiny_transformer_runs_four_chunks_with_explicit_cache_commit_and_camera_path() -> None:
    torch.manual_seed(7)
    module = attention_tests._load_module()
    model = _tiny_model(module).eval()
    cache = _cache(module, model)
    encoder_hidden_states = torch.randn(1, 3, 6)
    camera_hidden_states = torch.randn(1, 6 * 8 * 8, 1, 4, 4)
    cross_key_outputs = [attention_tests._record_outputs(block.cross_attn.k) for block in model.blocks]
    outputs = []

    assert model.patch_embedding.in_channels == 36
    assert model.patch_embedding_wancamctrl.in_features == 6 * 8 * 8 * 1 * 2 * 2

    for start_frame in range(4):
        hidden_states = torch.randn(1, 36, 1, 4, 4)
        timestep = torch.tensor([float(start_frame + 1)])
        snapshot = _self_cache_snapshot(cache)

        transient_output = model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            camera_hidden_states,
            cache=cache,
            start_frame=start_frame,
            update_cache=False,
        )

        assert transient_output.shape == (1, 2, 1, 4, 4)
        _assert_self_cache_unchanged(cache, snapshot)
        assert all(layer_cache is not None for layer_cache in cache.cross_attention)

        committed_output = model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            camera_hidden_states,
            cache=cache,
            start_frame=start_frame,
            update_cache=True,
        )
        torch.testing.assert_close(committed_output, transient_output)
        outputs.append(committed_output)

        expected_token_end = (start_frame + 1) * 2 * 2
        assert all(layer_cache.absolute_end == expected_token_end for layer_cache in cache.self_attention)

    assert len(outputs) == 4
    assert all(len(outputs) == 1 for outputs in cross_key_outputs)

    alternate_cache = _cache(module, model)
    alternate_output = model(
        torch.zeros(1, 36, 1, 4, 4),
        torch.tensor([1.0]),
        encoder_hidden_states,
        torch.zeros_like(camera_hidden_states),
        cache=alternate_cache,
        start_frame=0,
        update_cache=False,
    )
    camera_output = model(
        torch.zeros(1, 36, 1, 4, 4),
        torch.tensor([1.0]),
        encoder_hidden_states,
        torch.ones_like(camera_hidden_states),
        cache=_cache(module, model),
        start_frame=0,
        update_cache=False,
    )
    assert not torch.equal(camera_output, alternate_output)


@pytest.mark.parametrize("frames", [1, 6], ids=("partial", "multiple_blocks"))
def test_forward_rejects_chunks_that_do_not_equal_configured_block_size(frames: int) -> None:
    module = attention_tests._load_module()
    model = _tiny_model(
        module,
        num_layers=1,
        num_frames_per_block=3,
        sliding_window_num_frames=6,
    )
    cache = _cache(module, model)
    snapshot = _self_cache_snapshot(cache)
    patch_inputs = attention_tests._record_inputs(model.patch_embedding)

    with pytest.raises(ValueError, match="exactly 3 post-patch frames"):
        model(
            torch.randn(1, 36, frames, 4, 4),
            torch.tensor([1.0]),
            torch.randn(1, 3, 6),
            torch.randn(1, 6 * 8 * 8, frames, 4, 4),
            cache=cache,
            start_frame=0,
            update_cache=True,
        )

    assert patch_inputs == []
    _assert_self_cache_unchanged(cache, snapshot)
    assert cache.cross_attention == [None]


def test_forward_accepts_exactly_one_configured_frame_block() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(
        module,
        num_layers=1,
        num_frames_per_block=3,
        sliding_window_num_frames=6,
    )

    output = model(
        torch.randn(1, 36, 3, 4, 4),
        torch.tensor([1.0]),
        torch.randn(1, 3, 6),
        torch.randn(1, 6 * 8 * 8, 3, 4, 4),
        cache=_cache(module, model),
        start_frame=0,
        update_cache=False,
    )

    assert output.shape == (1, 2, 3, 4, 4)


def test_transformer_allocates_request_cache_from_its_configured_geometry() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(
        module,
        num_layers=2,
        num_frames_per_block=3,
        sliding_window_num_frames=6,
    )

    cache = model.allocate_cache(
        batch_size=2,
        latent_height=8,
        latent_width=12,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(cache.self_attention) == 2
    assert len(cache.cross_attention) == 2
    assert cache.cross_attention == [None, None]
    assert all(layer.key.shape == (2, 6 * 4 * 6, 2, 2) for layer in cache.self_attention)
    assert all(layer.value.shape == (2, 6 * 4 * 6, 2, 2) for layer in cache.self_attention)


def test_video_patch_embedding_uses_temporal_height_width_token_order() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1, num_frames_per_block=2)
    hidden_states = torch.zeros(1, 36, 2, 2, 4)
    hidden_states[0, 0] = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        ]
    )
    with torch.no_grad():
        model.patch_embedding.weight.zero_()
        model.patch_embedding.bias.zero_()
        model.patch_embedding.weight[0, 0, 0, 0, 0] = 1

    tokens = model.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

    torch.testing.assert_close(tokens[0, :, 0], torch.tensor([1.0, 3.0, 9.0, 11.0]))


def test_unpatchify_restores_two_frame_channel_and_spatial_order() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1, num_frames_per_block=2)
    hidden_states = (
        torch.stack(
            (
                torch.arange(0, 8),
                torch.arange(10, 18),
                torch.arange(20, 28),
                torch.arange(30, 38),
            )
        )
        .unsqueeze(0)
        .float()
    )
    expected = torch.tensor(
        [
            [
                [[0, 2, 10, 12], [4, 6, 14, 16]],
                [[20, 22, 30, 32], [24, 26, 34, 36]],
            ],
            [
                [[1, 3, 11, 13], [5, 7, 15, 17]],
                [[21, 23, 31, 33], [25, 27, 35, 37]],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    output = model._unpatchify(
        hidden_states,
        batch_size=1,
        frames=2,
        height=1,
        width=2,
    )

    torch.testing.assert_close(output, expected)


def test_head_modulation_broadcasts_distinct_condition_per_frame() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1, num_frames_per_block=2)
    model.head.norm = torch.nn.Identity()
    with torch.no_grad():
        model.head.modulation.zero_()
        model.head.head.weight.zero_()
        model.head.head.bias.zero_()
        model.head.head.weight[:, 0] = 1
    hidden_states = torch.ones(1, 4, model.dim)
    timestep_embedding = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]]])
    expected = torch.tensor([21.0, 21.0, 41.0, 41.0]).view(1, 4, 1).expand(1, 4, 8)

    output = model.head(hidden_states, timestep_embedding)

    torch.testing.assert_close(output, expected)


def test_constructor_defaults_match_official_checkpoint_config() -> None:
    module = attention_tests._load_module()
    parameters = inspect.signature(module.CausalLingBotWorldTransformer3DModel.__init__).parameters
    expected = {
        "in_channels": 36,
        "out_channels": 16,
        "num_layers": 40,
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "patch_size": (1, 2, 2),
        "text_dim": 4096,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "rope_max_seq_len": 1024,
        "eps": 1e-6,
        "cross_attn_norm": True,
        "sink_size": 9,
        "num_frames_per_block": 3,
        "sliding_window_num_frames": 18,
        "local_attn_size": -1,
    }

    assert {name: parameters[name].default for name in expected} == expected


def test_transformer_exposes_parameter_dtype_for_pipeline_runtime() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(module)

    assert model.dtype == next(model.parameters()).dtype


def test_transformer_declares_regional_compile_block() -> None:
    module = attention_tests._load_module()

    assert module.CausalLingBotWorldTransformer3DModel._repeated_blocks == ["LingBotAttentionBlock"]


def test_lingbot_rms_norm_uses_global_tp_square_mean(monkeypatch) -> None:
    module = attention_tests._load_module()
    reduced_values: list[torch.Tensor] = []

    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)

    def all_reduce(value: torch.Tensor) -> torch.Tensor:
        reduced_values.append(value.detach().clone())
        return value * 2

    monkeypatch.setattr(module, "tensor_model_parallel_all_reduce", all_reduce)
    norm = module._LingBotRMSNorm(hidden_size=2, eps=0.0)
    value = torch.tensor([[3.0, 4.0]])

    output = norm(value)

    torch.testing.assert_close(output, value / math.sqrt(12.5))
    assert len(reduced_values) == 1
    torch.testing.assert_close(reduced_values[0], torch.tensor([[25.0]]))


def test_attention_rejects_heads_not_divisible_by_tp_size(monkeypatch) -> None:
    module = attention_tests._load_module()
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 3)

    with pytest.raises(ValueError, match="num_heads=2.*tp_size=3"):
        module.LingBotSelfAttention(dim=4, num_heads=2)


def test_constructor_rejects_unsupported_qk_norm_with_config_error() -> None:
    module = attention_tests._load_module()

    with pytest.raises(ValueError, match="qk_norm.*rms_norm_across_heads"):
        module.CausalLingBotWorldTransformer3DModel(qk_norm="layer_norm")


@pytest.mark.parametrize("field", ["image_dim", "added_kv_proj_dim", "pos_embed_seq_len"])
def test_constructor_rejects_non_null_image_embedding_fields(field: str) -> None:
    module = attention_tests._load_module()

    with pytest.raises(ValueError, match=field):
        module.CausalLingBotWorldTransformer3DModel(**{field: 4})


def test_constructor_rejects_unsupported_quantization_with_runtime_error() -> None:
    module = attention_tests._load_module()

    with pytest.raises(RuntimeError, match="quant_config.*not supported"):
        module.CausalLingBotWorldTransformer3DModel(quant_config=object())


def test_from_config_accepts_diffusers_metadata_and_normalizes_patch_size() -> None:
    module = attention_tests._load_module()

    class ConfigProbe(module.CausalLingBotWorldTransformer3DModel):
        def __init__(self, **kwargs) -> None:
            self.received_kwargs = kwargs

    config = {
        "_class_name": "CausalLingBotWorldTransformer3DModel",
        "_diffusers_version": "0.35.0.dev0",
        "patch_size": [1, 2, 2],
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "in_channels": 36,
        "out_channels": 16,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 13824,
        "num_layers": 40,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "image_dim": None,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 1024,
        "pos_embed_seq_len": None,
        "qk_norm": "rms_norm_across_heads",
        "sink_size": 9,
        "num_frames_per_block": 3,
        "sliding_window_num_frames": 18,
        "local_attn_size": -1,
    }

    model = ConfigProbe.from_config(config, prefix="transformer")

    assert model.received_kwargs["patch_size"] == (1, 2, 2)
    assert model.received_kwargs["num_layers"] == 40
    assert model.received_kwargs["prefix"] == "transformer"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("in_channels", 35),
        ("num_layers", 39),
        ("num_attention_heads", 32),
        ("attention_head_dim", 64),
        ("ffn_dim", 10240),
        ("num_frames_per_block", 4),
        ("sliding_window_num_frames", 24),
        ("sink_size", 0),
    ],
)
def test_from_config_rejects_checkpoint_topology_drift(field: str, value: object) -> None:
    module = attention_tests._load_module()
    config = {
        "_class_name": "CausalLingBotWorldTransformer3DModel",
        "_diffusers_version": "0.35.0.dev0",
        "patch_size": [1, 2, 2],
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "in_channels": 36,
        "out_channels": 16,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 13824,
        "num_layers": 40,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "image_dim": None,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 1024,
        "pos_embed_seq_len": None,
        "qk_norm": "rms_norm_across_heads",
        "sink_size": 9,
        "num_frames_per_block": 3,
        "sliding_window_num_frames": 18,
        "local_attn_size": -1,
    }
    config[field] = value

    with pytest.raises(ValueError, match=field):
        module.CausalLingBotWorldTransformer3DModel.from_config(config)


def test_from_config_ignores_non_semantic_checkpoint_metadata() -> None:
    module = attention_tests._load_module()
    config = {
        "_class_name": "CausalLingBotWorldTransformer3DModel",
        "architectures": ["CausalLingBotWorldTransformer3DModel"],
        "future_diffusers_metadata": {"version": 1},
    }

    with torch.device("meta"):
        model = module.CausalLingBotWorldTransformer3DModel.from_config(config)

    assert model.config.in_channels == 36
    assert model.config.num_layers == 40


def test_load_weights_uses_parameter_loaders_and_rejects_unknown_model_keys() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1)
    checkpoint_weights, expected_parameters = _checkpoint_weights_for_model(model)
    loader_calls: list[str] = []
    qkv_weight = model.blocks[0].self_attn.qkv.weight

    def record_loader(param: torch.Tensor, loaded_weight: torch.Tensor, shard_id: str) -> None:
        loader_calls.append(shard_id)
        shard_index = {"q": 0, "k": 1, "v": 2}[shard_id]
        shard_size = param.shape[0] // 3
        param.data.narrow(0, shard_index * shard_size, shard_size).copy_(loaded_weight)

    qkv_weight.weight_loader = record_loader
    loaded = model.load_weights(iter(checkpoint_weights.items()))

    assert loaded == set(checkpoint_weights)
    assert loader_calls == ["q", "k", "v"]
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected_parameters[name])

    first_name = next(iter(checkpoint_weights))
    partial = _tiny_model(module, num_layers=1)
    partial_loaded = partial.load_weights([(first_name, checkpoint_weights[first_name])])
    assert partial_loaded == {first_name}

    with pytest.raises(KeyError, match="unexpected_model.weight"):
        model.load_weights([("unexpected_model.weight", torch.ones(1))])


def test_load_weights_consumes_checkpoint_iterator_incrementally() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1)
    first_name, first_param = next(iter(model.named_parameters()))
    loaded_weight = torch.full_like(first_param, 17)

    def weights():
        yield first_name, loaded_weight
        torch.testing.assert_close(first_param, loaded_weight)

    assert model.load_weights(weights()) == {first_name}


def test_checkpoint_weight_index_fixture_matches_model_namespaces() -> None:
    module = attention_tests._load_module()
    fixture = json.loads(_FIXTURE_PATH.read_text())
    model = _tiny_model(module, num_layers=1)
    parameter_names = set(_checkpoint_parameter_specs(model))

    assert {name.split(".", 1)[0] for name in parameter_names} == set(fixture["top_level_modules"])
    assert {name for name in parameter_names if not name.startswith("blocks.")} == set(fixture["non_block_parameters"])
    assert {name.removeprefix("blocks.0.") for name in parameter_names if name.startswith("blocks.0.")} == set(
        fixture["block_parameter_suffixes"]
    )
    expanded_names = set(fixture["non_block_parameters"])
    expanded_names.update(
        f"blocks.{layer}.{suffix}" for layer in range(40) for suffix in fixture["block_parameter_suffixes"]
    )
    canonical_names = "\n".join(sorted(expanded_names))

    assert len(expanded_names) == fixture["index_parameter_count"] == 1421
    assert hashlib.sha256(canonical_names.encode()).hexdigest() == fixture["canonical_name_set_sha256"]
    assert fixture["canonical_name_to_shard_sha256"] == (
        "e1addc27d2b1fad6f10226a23a265ee3e3c61e74dce4d5859727bdf180a67992"
    )
    assert list(fixture["shards"]) == [
        f"diffusion_pytorch_model-{index:05}-of-00008.safetensors" for index in range(1, 9)
    ]
    assert sum(shard["parameter_count"] for shard in fixture["shards"].values()) == 1421
    assert all(len(shard["parameter_names_sha256"]) == 64 for shard in fixture["shards"].values())


def test_official_default_shapes_match_public_safetensors_header_fixture() -> None:
    module = attention_tests._load_module()
    fixture = json.loads(_SHAPE_FIXTURE_PATH.read_text())

    with torch.device("meta"):
        model = module.CausalLingBotWorldTransformer3DModel()
    parameter_specs = _checkpoint_parameter_specs(model)
    expected_dtype = {"F32": torch.float32}[fixture["dtype"]]

    assert len(fixture["shard_representatives"]) == 8
    assert set(fixture["shard_representatives"].values()) <= set(fixture["parameter_shapes"])

    for name, expected_shape in fixture["parameter_shapes"].items():
        shape, dtype = parameter_specs[name]
        assert tuple(shape) == tuple(expected_shape), name
        assert dtype == expected_dtype
