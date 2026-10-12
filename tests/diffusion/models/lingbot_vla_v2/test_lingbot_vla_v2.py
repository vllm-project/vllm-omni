# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline tests on a real tiny LingBot inference graph and local processors."""

import copy

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.lingbot_vla_v2.config import LingBotVLAV2Config
from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingBotVLAV2ForActionPrediction
from vllm_omni.diffusion.models.lingbot_vla_v2.processor_lingbot_vla_v2 import LingBotVLAV2Processor

from .conftest import observation, tiny_backbone, tiny_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_real_inference_repeatability_and_noise_ownership(model, assets):
    proc = LingBotVLAV2Processor(tiny_config(), assets)
    inputs = proc.build_model_inputs(observation())
    noise = torch.randn(1, 4, 55)
    before = noise.clone()
    a = model.sample_actions(**inputs, noise=noise)
    altered = copy.deepcopy(inputs)
    altered["state"] += 0.3
    b = model.sample_actions(**altered, noise=noise)
    again = model.sample_actions(**inputs, noise=noise)
    torch.testing.assert_close(noise, before, rtol=0, atol=0)
    torch.testing.assert_close(a, again, rtol=0, atol=0)
    assert not torch.equal(a, b)
    assert np.isfinite(proc.build_model_outputs(a)).all()
    assert proc.build_model_outputs(a).shape == (4, 7)
    assert model._align_query_tokens().shape == (8, 64)


def test_processor_gripper_round_trip_and_nonidentity_stats(assets):
    cfg = tiny_config()
    proc = LingBotVLAV2Processor(cfg, assets)
    raw = torch.randn(1, 4, 7)
    normalized = proc.normalize(raw, "action")
    source, target = cfg.slot_indices("action")
    canonical = torch.zeros(1, 4, 55)
    canonical[..., target] = normalized[..., source]
    assert target[-1] == 28
    torch.testing.assert_close(torch.from_numpy(proc.build_model_outputs(canonical)), raw[0])


def test_strict_weights_and_persistent_buffers(model):
    state = {"model." + k: v.clone() for k, v in model.state_dict().items()}
    model.load_weights(state.items())
    state.pop(next(iter(state)))
    with pytest.raises(ValueError, match="Missing"):
        model.load_weights(state.items())
    with pytest.raises(ValueError, match="Unexpected"):
        model.load_weights([("model.invalid.weight", torch.ones(2))])


def test_grid_changes_match_fresh_vision():
    from transformers.models.qwen3_vl import Qwen3VLVisionConfig

    from vllm_omni.diffusion.models.lingbot_vla_v2.qwen3vl_in_vla import Qwen3VLVisionAttention

    cfg = Qwen3VLVisionConfig(hidden_size=32, num_heads=2)
    cfg._attn_implementation = "sdpa"
    layer = Qwen3VLVisionAttention(cfg).eval()
    fresh = copy.deepcopy(layer)
    hidden, pos = torch.randn(12, 32), (torch.ones(12, 16), torch.zeros(12, 16))
    layer(hidden, torch.tensor([0, 4, 12]), position_embeddings=pos)
    actual = layer(hidden, torch.tensor([0, 8, 12]), position_embeddings=pos)
    expected = fresh(hidden, torch.tensor([0, 8, 12]), position_embeddings=pos)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "override",
    [{"rtc_config": {}}, {"use_relative_actions": True}, {"use_cudagraph_prefix": True}, {"dtype": "bfloat16"}],
)
def test_unsupported_checkpoint_capabilities(override):
    from dataclasses import asdict

    raw = asdict(tiny_config()) | override
    with pytest.raises(ValueError):
        LingBotVLAV2Config.from_dict(raw)


@pytest.mark.parametrize(
    "field,value", [("state", [0.0]), ("prompt", ""), ("observation.images.top", np.ones((32, 32, 3)) * 255)]
)
def test_invalid_observation(assets, field, value):
    proc = LingBotVLAV2Processor(tiny_config(), assets)
    obs = observation()
    obs[field] = value
    with pytest.raises(ValueError):
        proc.build_model_inputs(obs)


@torch.no_grad()
def test_final_adaptive_norm_returns_finite_actions(assets):
    config = tiny_config(final_norm_adanorm=True)
    model = LingBotVLAV2ForActionPrediction(config, vlm_config=tiny_backbone()).float().eval()
    processor = LingBotVLAV2Processor(config, assets)

    result = model.sample_actions(**processor.build_model_inputs(observation()), noise=torch.zeros(1, 4, 55))

    assert result.shape == (1, 4, 55)
    assert torch.isfinite(result).all()


def test_final_adaptive_norm_requires_time_conditioning():
    with pytest.raises(ValueError, match="requires adanorm_time"):
        tiny_config(final_norm_adanorm=True, adanorm_time=False)


def test_unknown_router_activation_is_rejected():
    with pytest.raises(ValueError, match="router_activation"):
        tiny_config(router_activation="sigmod")
