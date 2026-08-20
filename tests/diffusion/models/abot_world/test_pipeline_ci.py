# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small CPU tests for ABot-World tensor geometry and checkpoint mapping."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.abot_world.transformer import (
    ABotSimpleAdapter,
    ABotWorldCausalTransformer3DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _tiny_transformer() -> ABotWorldCausalTransformer3DModel:
    return ABotWorldCausalTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=1,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=8,
        ffn_dim=16,
        num_layers=1,
        downscale_factor_control_adapter=2,
    )


def test_action_adapter_preserves_patch_token_geometry() -> None:
    adapter = ABotSimpleAdapter(dim=8, downscale_factor=2, control_in_dim=2)
    tokens = torch.zeros(1, 3 * 4 * 4, 8)
    actions = torch.zeros(1, 2, 3, 16, 16)

    output = adapter(tokens, actions, num_frames=3, spatial_tokens=16)

    assert output.shape == tokens.shape


def test_timestep_projection_expands_framewise_values_to_tokens() -> None:
    model = _tiny_transformer()
    temb, projection = model._timestep_embeddings(
        torch.tensor([[0.0, 500.0, 500.0]]),
        batch_size=1,
        frames=3,
        dtype=torch.float32,
    )

    assert temb.shape == (1, 3, 8)
    assert projection.shape == (1, 3, 6, 8)
    assert not torch.equal(projection[:, 0], projection[:, 1])


def test_unknown_checkpoint_weight_is_rejected() -> None:
    model = _tiny_transformer()

    with pytest.raises(KeyError, match="Unexpected ABot model weight"):
        model.load_weights([("model.not_a_real_parameter", torch.zeros(1))])


def test_official_generator_prefix_is_normalized() -> None:
    model = _tiny_transformer()
    weight = torch.ones_like(model.blocks[0].norm3.weight)

    loaded = model.load_weights([("generator.model.blocks.0.norm3.weight", weight)])

    assert "blocks.0.norm3.weight" in loaded
    assert torch.equal(model.blocks[0].norm3.weight, weight)
