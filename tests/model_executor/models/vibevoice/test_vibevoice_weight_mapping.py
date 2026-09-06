# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for online Microsoft-to-HF VibeVoice weight-name mapping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.vibevoice.vibevoice import _build_vibevoice_weights_mapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _config(*, acoustic_stages: int = 7, semantic_stages: int = 7):
    return SimpleNamespace(
        audio_config=SimpleNamespace(depths=[3] * acoustic_stages),
        semantic_model_config=SimpleNamespace(depths=[3] * semantic_stages),
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Semantic Encoder.
        (
            "model.semantic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight",
            "model.semantic_tokenizer_encoder.stem.conv.conv.weight",
        ),
        (
            "model.semantic_tokenizer.encoder.downsample_layers.3.0.conv.conv.weight",
            "model.semantic_tokenizer_encoder.conv_layers.2.conv.conv.weight",
        ),
        (
            "model.semantic_tokenizer.encoder.stages.6.7.mixer.conv.conv.conv.weight",
            "model.semantic_tokenizer_encoder.conv_layers.5.stage.7.mixer.conv.weight",
        ),
        (
            "model.semantic_tokenizer.encoder.head.conv.conv.weight",
            "model.semantic_tokenizer_encoder.head.conv.weight",
        ),
        # Acoustic Encoder and Decoder.
        (
            "model.acoustic_tokenizer.encoder.stages.0.0.ffn.linear1.weight",
            "model.audio_tower.encoder.stem.stage.0.ffn.linear1.weight",
        ),
        (
            "model.acoustic_tokenizer.encoder.downsample_layers.6.0.conv.conv.weight",
            "model.audio_tower.encoder.conv_layers.5.conv.conv.weight",
        ),
        (
            "model.acoustic_tokenizer.decoder.upsample_layers.0.0.conv.conv.weight",
            "model.audio_tower.decoder.stem.conv.conv.weight",
        ),
        (
            "model.acoustic_tokenizer.decoder.upsample_layers.4.0.convtr.convtr.weight",
            "model.audio_tower.decoder.conv_layers.3.convtr.convtr.weight",
        ),
        (
            "model.acoustic_tokenizer.decoder.stages.2.0.ffn.linear2.weight",
            "model.audio_tower.decoder.conv_layers.1.stage.0.ffn.linear2.weight",
        ),
        (
            "model.acoustic_tokenizer.decoder.head.conv.conv.bias",
            "model.audio_tower.decoder.head.conv.bias",
        ),
        # Diffusion Head.
        (
            "model.prediction_head.t_embedder.mlp.0.weight",
            "model.diffusion_head.timestep_proj.layer_1.weight",
        ),
        (
            "model.prediction_head.t_embedder.mlp.2.bias",
            "model.diffusion_head.timestep_proj.layer_2.bias",
        ),
        (
            "model.prediction_head.layers.3.adaLN_modulation.1.weight",
            "model.diffusion_head.layers.3.linear.weight",
        ),
        (
            "model.prediction_head.final_layer.adaLN_modulation.1.weight",
            "model.diffusion_head.final_layer.linear_1.weight",
        ),
        (
            "model.prediction_head.final_layer.linear.weight",
            "model.diffusion_head.final_layer.linear_2.weight",
        ),
        (
            "model.prediction_head.cond_proj.weight",
            "model.diffusion_head.cond_proj.weight",
        ),
        # Connectors and latent normalization factors.
        (
            "model.acoustic_connector.fc1.weight",
            "model.multi_modal_projector.linear_1.weight",
        ),
        (
            "model.acoustic_connector.norm.weight",
            "model.multi_modal_projector.act.weight",
        ),
        (
            "model.semantic_connector.fc2.bias",
            "model.semantic_connector.linear_2.bias",
        ),
        ("model.speech_scaling_factor", "model.latent_scaling_factor"),
        ("model.speech_bias_factor", "model.latent_bias_factor"),
        # Language-model names already match the runtime hierarchy.
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
        ),
        # A converted HF key must pass through unchanged.
        (
            "model.audio_tower.encoder.conv_layers.0.stage.0.ffn.linear1.weight",
            "model.audio_tower.encoder.conv_layers.0.stage.0.ffn.linear1.weight",
        ),
    ],
)
def test_official_weight_names_map_to_runtime_schema(source: str, expected: str):
    mapper = _build_vibevoice_weights_mapper(_config())

    assert mapper.apply_list([source]) == [expected]


def test_mapper_generates_stage_index_rules_from_config():
    mapper = _build_vibevoice_weights_mapper(_config(acoustic_stages=3, semantic_stages=2))

    assert mapper.apply_list(["model.acoustic_tokenizer.encoder.stages.2.0.gamma"]) == [
        "model.audio_tower.encoder.conv_layers.1.stage.0.gamma"
    ]
    assert mapper.apply_list(["model.semantic_tokenizer.encoder.stages.1.0.gamma"]) == [
        "model.semantic_tokenizer_encoder.conv_layers.0.stage.0.gamma"
    ]

    # There is no generated stage-3 rule in a three-stage Acoustic config.
    # The generic acoustic rename still applies, making a malformed/incompatible
    # checkpoint visible to AutoWeightsLoader instead of silently dropping it.
    assert mapper.apply_list(["model.acoustic_tokenizer.encoder.stages.3.0.gamma"]) == [
        "model.audio_tower.encoder.stages.3.0.gamma"
    ]


@pytest.mark.parametrize("child_name", ["audio_config", "semantic_model_config"])
def test_mapper_rejects_missing_tokenizer_depths(child_name: str):
    config = _config()
    setattr(config, child_name, SimpleNamespace(depths=[]))

    with pytest.raises(ValueError, match=rf"{child_name}\.depths"):
        _build_vibevoice_weights_mapper(config)


def test_mapper_preserves_tensor_identity():
    mapper = _build_vibevoice_weights_mapper(_config())
    tensor = torch.randn(2, 3)

    [(name, mapped_tensor)] = list(mapper.apply([("model.acoustic_connector.fc1.weight", tensor)]))

    assert name == "model.multi_modal_projector.linear_1.weight"
    assert mapped_tensor is tensor
