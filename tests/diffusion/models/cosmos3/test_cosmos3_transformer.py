# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _single_rank_tensor_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the TP metadata required by vLLM parallel linear layers."""
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    from vllm_omni.diffusion.models.cosmos3 import (
        transformer_cosmos3,
        transformer_cosmos3_edge,
    )

    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(transformer_cosmos3, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(transformer_cosmos3_edge, "get_tensor_model_parallel_world_size", lambda: 1)


@pytest.fixture
def accelerator_device() -> torch.device:
    """Provide an accelerator device, skipping when none is available."""
    if current_omni_platform.get_device_count() == 0:
        pytest.skip("Accelerator required for this test")
    return current_omni_platform.get_torch_device(0)


def _tiny_cosmos3_config(**overrides):
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 16,
        "vocab_size": 32,
        "latent_patch_size": 1,
        "latent_channel": 2,
        "rope_scaling": {"mrope_section": [1, 1, 0]},
    }
    config.update(overrides)
    return config


def _tiny_cosmos3_edge_config(**overrides):
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import (
        COSMOS3_EDGE_BACKBONE_TYPE,
    )

    config = _tiny_cosmos3_config(
        latent_patch_size=2,
        latent_channel=48,
        temporal_compression_factor=4,
        backbone_type=COSMOS3_EDGE_BACKBONE_TYPE,
        qk_norm_for_text=False,
    )
    config.update(overrides)
    return config


def test_mrope_position_ids_cover_text_video_sound_and_action() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        compute_mrope_position_ids_action,
        compute_mrope_position_ids_sound,
        compute_mrope_position_ids_text,
        compute_mrope_position_ids_vision,
    )

    text_ids, text_offset = compute_mrope_position_ids_text(num_tokens=3, temporal_offset=5)
    assert text_ids.tolist() == [[5, 6, 7], [5, 6, 7], [5, 6, 7]]
    assert text_offset == 8

    vision_ids, vision_offset = compute_mrope_position_ids_vision(2, 2, 3, temporal_offset=10, fps=None)
    assert vision_ids.shape == (3, 12)
    assert vision_ids[0].tolist() == [10] * 6 + [11] * 6
    assert vision_offset == 12

    modulated_ids, modulated_offset = compute_mrope_position_ids_vision(
        2,
        1,
        1,
        temporal_offset=10,
        fps=12.0,
        base_fps=24.0,
        temporal_compression_factor=4,
    )
    torch.testing.assert_close(modulated_ids[0], torch.tensor([10.0, 12.0]))
    assert modulated_offset == 13

    sound_ids, sound_offset = compute_mrope_position_ids_sound(3, temporal_offset=10, sound_latent_fps=25.0)
    torch.testing.assert_close(sound_ids[0], torch.tensor([10.0, 10.96, 11.92]))
    assert sound_offset == 12

    action_ids, action_offset = compute_mrope_position_ids_action(3, temporal_offset=10, action_fps=None)
    assert action_ids.tolist() == [[11, 12, 13], [0, 0, 0], [0, 0, 0]]
    assert action_offset == 14


def test_timestep_embedder_stores_frequencies_in_fp32() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        TimestepEmbedder,
    )

    embedder = TimestepEmbedder(hidden_size=8, frequency_embedding_size=16)

    assert embedder.freqs.dtype == torch.float32


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("qk_norm_for_diffusion", False),
        ("qk_norm_for_text", False),
        ("position_embedding_type", "rotary"),
        ("unified_3d_mrope_reset_spatial_ids", False),
        ("joint_attn_implementation", "one_way"),
    ],
)
def test_validate_supported_config_rejects_unsupported_flags(key: str, value) -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    with pytest.raises(ValueError, match=f"{key}="):
        Cosmos3VFMTransformer._validate_supported_config({key: value})
    Cosmos3VFMTransformer._validate_supported_config({})
    Cosmos3VFMTransformer._validate_supported_config(None)


def test_edge_config_resolves_nemotron_defaults() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import (
        Cosmos3EdgeVFMTransformer,
    )

    model = Cosmos3EdgeVFMTransformer(
        SimpleNamespace(
            tf_model_config=_tiny_cosmos3_edge_config(
                layer_norm_epsilon=1e-5,
                rope_scaling={},
                rope_parameters={"mrope_section": [24, 20, 20]},
            ),
            dtype=torch.float32,
        )
    )

    assert model.rms_norm_eps == 1e-5
    assert model.rope_theta == 100_000_000
    assert model.mrope_section == [24, 20, 20]
    assert model.latent_channel_size == 48
    assert model.latent_patch_size == 2
    assert model.temporal_compression_factor == 4


def test_edge_config_requires_backbone_type() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import (
        Cosmos3EdgeVFMTransformer,
    )

    config = _tiny_cosmos3_edge_config()
    config.pop("backbone_type")

    with pytest.raises(ValueError, match="must declare backbone_type"):
        Cosmos3EdgeVFMTransformer(
            SimpleNamespace(
                tf_model_config=config,
                dtype=torch.float32,
            )
        )


def test_edge_und_and_gen_use_relu2_and_no_und_qk_norm() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import (
        Cosmos3EdgeVFMTransformer,
        Cosmos3Relu2MLP,
    )

    model = Cosmos3EdgeVFMTransformer(
        SimpleNamespace(
            tf_model_config=_tiny_cosmos3_edge_config(
                num_hidden_layers=1,
                use_k_norm_und_for_gen=False,
            ),
            dtype=torch.float32,
        )
    )

    layer = model.language_model.layers[0]
    gen_layer = model.gen_layers[0]
    assert isinstance(layer.mlp, Cosmos3Relu2MLP)
    assert not hasattr(layer.mlp, "gate_proj")
    assert isinstance(gen_layer.mlp, Cosmos3Relu2MLP)
    assert not hasattr(gen_layer.mlp, "gate_proj")
    assert not hasattr(layer.self_attn, "norm_q")
    assert not hasattr(layer.self_attn, "norm_k")
    assert layer.self_attn.k_norm_und_for_gen is None


def test_edge_creates_gen_k_norm_when_enabled() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import RMSNorm
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import (
        Cosmos3EdgeVFMTransformer,
    )

    model = Cosmos3EdgeVFMTransformer(
        SimpleNamespace(
            tf_model_config=_tiny_cosmos3_edge_config(
                num_hidden_layers=1,
                qk_norm_for_diffusion=True,
                use_k_norm_und_for_gen=True,
            ),
            dtype=torch.float32,
        )
    )

    k_norm = model.language_model.layers[0].self_attn.k_norm_und_for_gen
    assert model.use_k_norm_und_for_gen is True
    assert isinstance(k_norm, RMSNorm)
    assert isinstance(model.gen_layers[0].cross_attention.norm_q, RMSNorm)
    assert isinstance(model.gen_layers[0].cross_attention.norm_k, RMSNorm)


def test_edge_supports_no_gen_qk_norm_variant() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import (
        Cosmos3EdgeVFMTransformer,
    )

    model = Cosmos3EdgeVFMTransformer(
        SimpleNamespace(
            tf_model_config=_tiny_cosmos3_edge_config(
                num_hidden_layers=1,
                qk_norm_for_diffusion=False,
                use_k_norm_und_for_gen=False,
            ),
            dtype=torch.float32,
        )
    )

    assert model.qk_norm_for_diffusion is False
    assert model.language_model.layers[0].self_attn.k_norm_und_for_gen is None
    assert model.gen_layers[0].cross_attention.qk_norm is False
    assert not hasattr(model.gen_layers[0].cross_attention, "norm_q")
    assert not hasattr(model.gen_layers[0].cross_attention, "norm_k")


def test_edge_validates_required_relu2_weights() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import (
        Cosmos3EdgeVFMTransformer,
    )

    model = object.__new__(Cosmos3EdgeVFMTransformer)
    nn.Module.__init__(model)
    model.num_hidden_layers = 1
    model.qk_norm_for_diffusion = True
    model.use_k_norm_und_for_gen = True

    complete = {
        "transformer.language_model.layers.0.mlp.up_proj.weight",
        "transformer.language_model.layers.0.mlp.down_proj.weight",
        "transformer.language_model.layers.0.self_attn.k_norm_und_for_gen.weight",
        "transformer.gen_layers.0.mlp.up_proj.weight",
        "transformer.gen_layers.0.mlp.down_proj.weight",
    }
    model.validate_loaded_weights(complete)

    mlp_weights = complete - {
        "transformer.language_model.layers.0.self_attn.k_norm_und_for_gen.weight",
    }
    for missing_weight in mlp_weights:
        with pytest.raises(ValueError, match="missing required weights"):
            model.validate_loaded_weights(complete - {missing_weight})

    missing_k_norm = complete - {"transformer.language_model.layers.0.self_attn.k_norm_und_for_gen.weight"}
    with pytest.raises(ValueError, match=r"self_attn\.k_norm_und_for_gen"):
        model.validate_loaded_weights(missing_k_norm)

    model.use_k_norm_und_for_gen = False
    model.validate_loaded_weights(missing_k_norm)


def test_edge_gen_cached_k_is_normalized_but_reasoner_uses_raw_k() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import RMSNorm
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import (
        Cosmos3EdgeCausalAttention,
    )

    class ScaleProjection(nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.scale = scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.scale

    class IdentityOutput(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class RawKAttention(nn.Module):
        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            del v
            return q + k

    attn = object.__new__(Cosmos3EdgeCausalAttention)
    nn.Module.__init__(attn)
    attn.hidden_size = 4
    attn.num_heads = 2
    attn.num_kv_heads = 2
    attn.head_dim = 2
    attn.num_heads_local = 2
    attn.num_kv_heads_local = 2
    attn.to_q = ScaleProjection(1.0)
    attn.to_k = ScaleProjection(1.0)
    attn.to_v = ScaleProjection(1.0)
    attn.to_out = IdentityOutput()
    attn.k_norm_und_for_gen = RMSNorm(2, eps=1e-5)
    attn.attn = RawKAttention()

    hidden_states = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]]],
        dtype=torch.float32,
    )
    freqs_cos = torch.ones(1, 2, 1, 2)
    freqs_sin = torch.zeros(1, 2, 1, 2)

    reasoner_out, gen_k, gen_v = attn(hidden_states, freqs_cos, freqs_sin)
    attn.to_k.scale = 3.0
    scaled_reasoner_out, scaled_gen_k, scaled_gen_v = attn(hidden_states, freqs_cos, freqs_sin)

    torch.testing.assert_close(gen_k, scaled_gen_k)
    torch.testing.assert_close(gen_v, scaled_gen_v)
    assert not torch.allclose(reasoner_out, scaled_reasoner_out)


def test_transformer_sharding_offload_and_patch_round_trip_contracts() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3LanguageModel, Cosmos3VFMTransformer
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import (
        Cosmos3EdgeLanguageModel,
        Cosmos3EdgeVFMTransformer,
    )

    for transformer_cls, language_model_cls in (
        (Cosmos3VFMTransformer, Cosmos3LanguageModel),
        (Cosmos3EdgeVFMTransformer, Cosmos3EdgeLanguageModel),
    ):
        model = object.__new__(transformer_cls)
        nn.Module.__init__(model)
        model.language_model = nn.Module()
        model.language_model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
        model.gen_layers = nn.ModuleList([nn.Linear(2, 2)])
        model.norm_moe_gen = nn.LayerNorm(2)

        matched = [
            name
            for name, module in model.named_modules()
            if any(condition(name, module) for condition in model._hsdp_shard_conditions)
        ]
        assert matched == ["language_model.layers.0", "language_model.layers.1", "gen_layers.0"]
        assert transformer_cls._layerwise_offload_blocks_attrs == ["gen_layers"]
        assert language_model_cls._layerwise_offload_blocks_attrs == ["layers"]
        assert transformer_cls._repeated_blocks == ["Cosmos3GenDecoderLayer"]

        model.latent_patch_size = 2
        model.latent_channel_size = 3
        latents = torch.arange(1 * 3 * 1 * 3 * 5, dtype=torch.float32).reshape(1, 3, 1, 3, 5)
        torch.testing.assert_close(
            model.unpatchify(model.patchify(latents, t=1, h=3, w=5), t=1, h=3, w=5),
            latents,
        )


def test_forward_returns_video_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

    output = transformer_cosmos3.Cosmos3VFMTransformer(
        SimpleNamespace(tf_model_config=_tiny_cosmos3_config(), dtype=torch.float32)
    )(
        hidden_states=torch.zeros(1, 2, 1, 2, 2),
        timestep=torch.tensor([1.0]),
        text_ids=torch.tensor([[1, 2]], dtype=torch.long),
        text_mask=torch.ones(1, 2, dtype=torch.long),
        video_shape=(1, 2, 2),
        fps=24.0,
    )

    assert tuple(output.shape) == (1, 2, 1, 2, 2)


def test_qwen_gen_mlp_remains_gated() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3GatedMLP, Cosmos3VFMTransformer

    model = Cosmos3VFMTransformer(
        SimpleNamespace(tf_model_config=_tiny_cosmos3_config(num_hidden_layers=1), dtype=torch.float32)
    )

    assert isinstance(model.gen_layers[0].mlp, Cosmos3GatedMLP)
    assert hasattr(model.gen_layers[0].mlp, "gate_proj")


def test_model_cpu_offload_swaps_back_to_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

    class ToyReasonerLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, hidden: torch.Tensor, freqs):
            del freqs
            kv = hidden.new_zeros(hidden.shape[0], hidden.shape[1], 1, 1)
            return hidden + self.weight.to(hidden.dtype), kv, kv

    class ToyGeneratorLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, hidden: torch.Tensor, **kwargs):
            del kwargs
            return hidden + self.weight.to(hidden.dtype)

    model = transformer_cosmos3.Cosmos3VFMTransformer(
        SimpleNamespace(tf_model_config=_tiny_cosmos3_config(), dtype=torch.float32)
    )
    model.language_model.layers = nn.ModuleList([ToyReasonerLayer()])
    model.gen_layers = nn.ModuleList([ToyGeneratorLayer()])
    model.enable_model_cpu_offload(device=torch.device("cpu"), pin_memory=False)

    assert model._model_cpu_offload_enabled
    assert set(model._model_cpu_offload_components()) == {"reasoner", "generator"}
    assert model.device == torch.device("cpu")

    output = model(
        hidden_states=torch.zeros(1, 2, 1, 2, 2),
        timestep=torch.tensor([1.0]),
        text_ids=torch.tensor([[1, 2]], dtype=torch.long),
        text_mask=torch.ones(1, 2, dtype=torch.long),
        video_shape=(1, 2, 2),
        fps=24.0,
    )

    assert tuple(output.shape) == (1, 2, 1, 2, 2)
    # The reasoner runs once then the generator component stays resident for GEN.
    assert model._active_model_cpu_offload_component == "generator"

    model.disable_model_cpu_offload()
    assert not model._model_cpu_offload_enabled


def test_model_cpu_offload_moves_reasoner_and_generator_between_cpu_and_device(
    monkeypatch: pytest.MonkeyPatch, accelerator_device: torch.device
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

    model = transformer_cosmos3.Cosmos3VFMTransformer(
        SimpleNamespace(tf_model_config=_tiny_cosmos3_config(), dtype=torch.float32)
    )
    model.language_model.layers = nn.ModuleList([nn.Linear(2, 2)])
    model.gen_layers = nn.ModuleList([nn.Linear(2, 2)])
    model.to(accelerator_device)

    reasoner_param = model.language_model.layers[0].weight
    generator_param = model.gen_layers[0].weight

    model.enable_model_cpu_offload(device=accelerator_device, pin_memory=False)

    assert model._model_cpu_offload_enabled
    # On enable, every group is parked on CPU until a phase activates it.
    assert reasoner_param.device.type == "cpu"
    assert generator_param.device.type == "cpu"

    model._activate_model_cpu_offload_component("reasoner")
    assert reasoner_param.device == accelerator_device
    assert generator_param.device.type == "cpu"

    model._activate_model_cpu_offload_component("generator")
    assert reasoner_param.device.type == "cpu"
    assert generator_param.device == accelerator_device

    model.disable_model_cpu_offload()
    assert not model._model_cpu_offload_enabled
    # Disable restores both components onto the device.
    assert reasoner_param.device == accelerator_device
    assert generator_param.device == accelerator_device


def test_forward_accepts_transfer_control_latents(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

    model = transformer_cosmos3.Cosmos3VFMTransformer(
        SimpleNamespace(tf_model_config=_tiny_cosmos3_config(), dtype=torch.float32)
    )
    hidden_states = torch.zeros(1, 2, 1, 2, 2)
    output = model(
        hidden_states=hidden_states,
        timestep=torch.tensor([1.0]),
        text_ids=torch.tensor([[1, 2]], dtype=torch.long),
        text_mask=torch.ones(1, 2, dtype=torch.long),
        video_shape=(1, 2, 2),
        fps=24.0,
        control_latents=[torch.ones_like(hidden_states), torch.full_like(hidden_states, 2.0)],
    )

    assert tuple(output.shape) == tuple(hidden_states.shape)
    with pytest.raises(ValueError, match="control latent shape"):
        model(
            hidden_states=hidden_states,
            timestep=torch.tensor([1.0]),
            text_ids=torch.tensor([[1, 2]], dtype=torch.long),
            text_mask=torch.ones(1, 2, dtype=torch.long),
            video_shape=(1, 2, 2),
            fps=24.0,
            control_latents=[torch.zeros(1, 2, 2, 2, 2)],
        )


def test_forward_gathers_gen_tokens_before_unpatchify(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    class RecordingGather(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, hidden_gen: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return hidden_gen

    monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))
    model = transformer_cosmos3.Cosmos3VFMTransformer(
        SimpleNamespace(tf_model_config=_tiny_cosmos3_config(), dtype=torch.float32)
    )
    gather = RecordingGather()
    model.gen_sp_gather = gather

    output = model(
        hidden_states=torch.zeros(1, 2, 1, 2, 2),
        timestep=torch.tensor([1.0]),
        text_ids=torch.tensor([[1, 2]], dtype=torch.long),
        text_mask=torch.ones(1, 2, dtype=torch.long),
        video_shape=(1, 2, 2),
        fps=24.0,
    )

    assert gather.calls == 1
    assert tuple(output.shape) == (1, 2, 1, 2, 2)


def test_sound_and_action_modules_follow_config() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    tiny = _tiny_cosmos3_config()
    no_modal = Cosmos3VFMTransformer(SimpleNamespace(tf_model_config=tiny, dtype=torch.float32))
    with_sound = Cosmos3VFMTransformer(
        SimpleNamespace(tf_model_config=tiny, dtype=torch.float32),
        sound_gen=True,
        sound_dim=5,
        sound_latent_fps=40.0,
    )
    with_action = Cosmos3VFMTransformer(
        SimpleNamespace(
            tf_model_config={**tiny, "action_gen": True, "max_action_dim": 6, "num_embodiment_domains": 9},
            dtype=torch.float32,
        )
    )

    assert no_modal.sound_gen is False
    assert no_modal.action_gen is False
    assert not hasattr(no_modal, "audio_proj_in")
    assert not hasattr(no_modal, "action_proj_in")
    assert with_sound.sound_dim == 5
    assert with_sound.sound_latent_fps == 40.0
    assert with_sound.audio_proj_in.in_features == 5
    assert with_action.action_dim == 6
    assert with_action.action_proj_in.num_domains == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sound_gen": True},
        {"sound_gen": True, "sound_dim": 5},
        {"sound_gen": True, "sound_latent_fps": 40.0},
    ],
)
def test_transformer_requires_sound_dim_and_fps_when_sound_gen_true(kwargs: dict[str, Any]) -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    with pytest.raises(ValueError, match=r"requires an explicit sound_dim and sound_latent_fps"):
        Cosmos3VFMTransformer(
            SimpleNamespace(tf_model_config=_tiny_cosmos3_config(), dtype=torch.float32),
            **kwargs,
        )


def test_sound_and_action_pack_unpack_validate_shapes() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.sound_dim = 3
    model.action_dim = 3

    sound = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    action = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    torch.testing.assert_close(model.unpack_sound(model.pack_sound(sound)), sound)
    torch.testing.assert_close(model.unpack_action(model.pack_action(action)), action)

    with pytest.raises(ValueError, match="channel mismatch"):
        model.pack_sound(torch.zeros(1, 4, 2))
    with pytest.raises(ValueError, match="dimension mismatch"):
        model.pack_action(torch.zeros(1, 2, 4))


@pytest.mark.parametrize(
    ("config", "transformer_kwargs", "extra_kwargs", "expected_shapes"),
    [
        (
            _tiny_cosmos3_config(),
            {"sound_gen": True, "sound_dim": 3, "sound_latent_fps": 24.0},
            {"sound_latents": torch.zeros(1, 3, 4)},
            [(1, 2, 1, 2, 2), (1, 3, 4)],
        ),
        (
            _tiny_cosmos3_config(action_gen=True, max_action_dim=3, num_embodiment_domains=4),
            {},
            {"action_latents": torch.zeros(1, 5, 3), "action_domain_ids": torch.tensor([2])},
            [(1, 2, 1, 2, 2), (1, 5, 3)],
        ),
    ],
)
def test_forward_returns_video_plus_optional_modality_predictions(
    monkeypatch: pytest.MonkeyPatch,
    config,
    transformer_kwargs,
    extra_kwargs,
    expected_shapes,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

    output = transformer_cosmos3.Cosmos3VFMTransformer(
        SimpleNamespace(tf_model_config=config, dtype=torch.float32),
        **transformer_kwargs,
    )(
        hidden_states=torch.zeros(1, 2, 1, 2, 2),
        timestep=torch.tensor([1.0]),
        text_ids=torch.tensor([[1, 2]], dtype=torch.long),
        text_mask=torch.ones(1, 2, dtype=torch.long),
        video_shape=(1, 2, 2),
        fps=24.0,
        action_noisy_mask=torch.ones(1, 5, 1),
        **extra_kwargs,
    )

    assert isinstance(output, tuple)
    assert [tuple(tensor.shape) for tensor in output] == expected_shapes


def test_forward_with_sound_ulysses_error_mentions_combined_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 as cosmos3_module

    model = cosmos3_module.Cosmos3VFMTransformer(
        SimpleNamespace(tf_model_config=_tiny_cosmos3_config(), dtype=torch.float32),
        sound_gen=True,
        sound_dim=3,
        sound_latent_fps=40.0,
    )
    monkeypatch.setattr(cosmos3_module, "_get_ulysses_state", lambda: (2, 0, None))

    with pytest.raises(ValueError, match=r"GEN sequence length \(3 = video tokens 2 \+ sound tokens 1\)"):
        model(
            hidden_states=torch.zeros(1, 2, 1, 1, 2),
            timestep=torch.tensor([1.0]),
            text_ids=torch.tensor([[1, 2]], dtype=torch.long),
            text_mask=torch.ones(1, 2, dtype=torch.long),
            video_shape=(1, 1, 2),
            fps=24.0,
            sound_latents=torch.zeros(1, 3, 1),
        )


def test_sound_latent_frames_padded_for_sequence_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    model = object.__new__(transformer_cosmos3.Cosmos3VFMTransformer)
    model.latent_patch_size = 2
    vs = (3, 16, 16)

    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_world_size", lambda: 1)
    assert model.sound_latent_frames_for_sequence_parallel(video_shape=vs, sound_frames=97) == 97

    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_world_size", lambda: 2)
    assert model.sound_latent_frames_for_sequence_parallel(video_shape=vs, sound_frames=97) == 98
    assert model.sound_latent_frames_for_sequence_parallel(video_shape=vs, sound_frames=98) == 98

    # ulysses=4, with a transfer control folded into the vision base.
    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_world_size", lambda: 4)
    padded = model.sound_latent_frames_for_sequence_parallel(video_shape=vs, sound_frames=97, num_vision_items=2)
    assert (2 * 192 + padded) % 4 == 0


def test_compute_rope_freqs_places_text_video_action_and_sound_positions() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    class FakeRotary:
        def __init__(self) -> None:
            self.position_ids: list[torch.Tensor] = []

        def __call__(self, x, position_ids):
            del x
            self.position_ids.append(position_ids.detach().cpu())
            batch, seq = position_ids.shape[1], position_ids.shape[2]
            return torch.zeros(batch, seq, 4), torch.ones(batch, seq, 4)

    rotary = FakeRotary()
    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.language_model = SimpleNamespace(rotary_emb=rotary)
    model.temporal_modality_margin = 100
    model.base_fps = 24.0
    model.temporal_compression_factor = 4
    model.temporal_compression_factor_sound = 1
    model.sound_latent_fps = 25.0
    model.enable_fps_modulation = False

    freqs_und, freqs_gen = model._compute_rope_freqs(
        text_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long),
        t=2,
        hp=1,
        wp=1,
        fps=24.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    text_pos, vision_pos = rotary.position_ids
    assert text_pos[:, 0, :].tolist() == [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert vision_pos[0, 0].tolist() == [102, 103]
    assert freqs_und[0].shape == (2, 3, 1, 4)
    assert freqs_gen[0].shape == (2, 2, 1, 4)

    rotary.position_ids.clear()
    model._compute_rope_freqs(
        text_mask=torch.tensor([[1, 1]], dtype=torch.long),
        t=2,
        hp=1,
        wp=1,
        fps=24.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        t_action=2,
        action_start_frame_offset=1,
        t_sound=1,
    )

    _, gen_pos = rotary.position_ids
    assert gen_pos.shape == (3, 1, 5)
    assert gen_pos[0, 0].tolist() == [102, 103, 103, 104, 102]

    rotary.position_ids.clear()
    model._compute_rope_freqs(
        text_mask=torch.tensor([[1, 1]], dtype=torch.long),
        t=2,
        hp=1,
        wp=1,
        fps=24.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_vision_items=3,
        share_vision_temporal_positions=True,
    )
    _, shared_gen_pos = rotary.position_ids
    assert shared_gen_pos[0, 0].tolist() == [102, 103, 102, 103, 102, 103]

    rotary.position_ids.clear()
    model._compute_rope_freqs(
        text_mask=torch.tensor([[1, 1]], dtype=torch.long),
        t=2,
        hp=1,
        wp=1,
        fps=24.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_vision_items=3,
        share_vision_temporal_positions=False,
    )
    _, offset_gen_pos = rotary.position_ids
    assert offset_gen_pos[0, 0].tolist() == [102, 103, 104, 105, 106, 107]
