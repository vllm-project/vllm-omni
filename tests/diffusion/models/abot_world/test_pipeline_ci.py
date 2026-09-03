# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small CPU tests for ABot-World tensor geometry and checkpoint mapping."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.abot_world.pipeline_abot_world import ABotWorldCausalPipeline
from vllm_omni.diffusion.models.abot_world.abot_world_transformer import (
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


def test_realtime_vae_decode_keeps_state_across_chunks() -> None:
    class Decoder:
        def __call__(self, hidden_states, *, feat_cache, feat_idx, first_chunk):
            history = feat_cache[0] or 0
            feat_cache[0] = history + 1
            feat_idx[0] += 1
            frames = 1 if first_chunk else 4
            return hidden_states.new_full((*hidden_states.shape[:2], frames, *hidden_states.shape[3:]), history / 10)

    vae = SimpleNamespace(
        _cached_conv_counts={"decoder": 1},
        _execution_context=nullcontext,
        config=SimpleNamespace(patch_size=None),
        decoder=Decoder(),
        post_quant_conv=lambda value: value,
    )
    pipeline = SimpleNamespace(vae=vae)
    chunk = torch.zeros(1, 1, 3, 2, 2)

    first, cache = ABotWorldCausalPipeline._decode_realtime_chunk(pipeline, chunk, None)
    continued, _ = ABotWorldCausalPipeline._decode_realtime_chunk(pipeline, chunk, cache)
    restarted, _ = ABotWorldCausalPipeline._decode_realtime_chunk(pipeline, chunk, None)

    assert first.shape[2] == restarted.shape[2] == 9
    assert continued.shape[2] == 12
    assert continued[:, :, 0].eq(0.3).all()
    assert restarted[:, :, 0].eq(0).all()
