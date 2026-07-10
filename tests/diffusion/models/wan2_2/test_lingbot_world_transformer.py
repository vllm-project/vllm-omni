# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from tests.diffusion.models.wan2_2 import test_lingbot_world_attention as attention_tests

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "lingbot_world_weight_index_names.json.fixture"


def _tiny_model(module, *, num_layers: int = 2):
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
        num_frames_per_block=1,
        sliding_window_num_frames=3,
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


def test_from_config_accepts_diffusers_metadata_and_normalizes_patch_size() -> None:
    module = attention_tests._load_module()
    config = {
        "_class_name": "CausalLingBotWorldTransformer3DModel",
        "_diffusers_version": "0.35.0.dev0",
        "patch_size": [1, 2, 2],
        "num_attention_heads": 2,
        "attention_head_dim": 2,
        "in_channels": 36,
        "out_channels": 2,
        "text_dim": 6,
        "freq_dim": 4,
        "ffn_dim": 8,
        "num_layers": 1,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "image_dim": None,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 16,
        "pos_embed_seq_len": None,
        "qk_norm": "rms_norm_across_heads",
        "sink_size": 1,
        "num_frames_per_block": 1,
        "sliding_window_num_frames": 3,
        "local_attn_size": -1,
    }

    model = module.CausalLingBotWorldTransformer3DModel.from_config(config, prefix="transformer")

    assert model.config.patch_size == (1, 2, 2)
    assert model.config.num_layers == 1


def test_from_config_rejects_unknown_checkpoint_fields() -> None:
    module = attention_tests._load_module()

    with pytest.raises(ValueError, match="unknown_field"):
        module.CausalLingBotWorldTransformer3DModel.from_config({"unknown_field": 1})


def test_load_weights_uses_parameter_loaders_and_rejects_unknown_model_keys() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1)
    expected = {name: torch.full_like(param, index + 1) for index, (name, param) in enumerate(model.named_parameters())}
    loader_calls = []
    q_weight = model.blocks[0].self_attn.q.weight

    def record_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        loader_calls.append(param)
        param.data.copy_(loaded_weight)

    q_weight.weight_loader = record_loader
    loaded = model.load_weights(iter(expected.items()))

    assert loaded == set(expected)
    assert loader_calls == [q_weight]
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected[name])

    first_name = next(iter(expected))
    partial = _tiny_model(module, num_layers=1)
    partial_loaded = partial.load_weights([(first_name, expected[first_name])])
    assert partial_loaded == {first_name}
    assert set(dict(partial.named_parameters())) - partial_loaded

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


def test_forward_rejects_cache_capacity_that_ignores_configured_window() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1)
    hidden_states = torch.randn(1, 36, 1, 4, 4)
    camera_hidden_states = torch.randn(1, 6 * 8 * 8, 1, 4, 4)

    with pytest.raises(ValueError, match="capacity.*12"):
        model(
            hidden_states,
            torch.tensor([1.0]),
            torch.randn(1, 3, 6),
            camera_hidden_states,
            cache=_cache(module, model, max_tokens=16),
            start_frame=0,
            update_cache=True,
        )


def test_checkpoint_weight_index_fixture_matches_model_namespaces() -> None:
    module = attention_tests._load_module()
    fixture = json.loads(_FIXTURE_PATH.read_text())
    model = _tiny_model(module, num_layers=1)
    parameter_names = set(dict(model.named_parameters()))

    assert {name.split(".", 1)[0] for name in parameter_names} == set(fixture["top_level_modules"])
    assert {name for name in parameter_names if not name.startswith("blocks.")} == set(fixture["non_block_parameters"])
    assert {name.removeprefix("blocks.0.") for name in parameter_names if name.startswith("blocks.0.")} == set(
        fixture["block_parameter_suffixes"]
    )
    assert fixture["index_parameter_count"] == 40 * len(fixture["block_parameter_suffixes"]) + len(
        fixture["non_block_parameters"]
    )
