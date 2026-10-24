# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline tests on a real tiny LingBot inference graph and local processors."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.lingbot_vla_v2.config import LingBotVLAV2Config
from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingBotVLAV2ForActionPrediction
from vllm_omni.diffusion.models.lingbot_vla_v2.processor_lingbot_vla_v2 import LingBotVLAV2Processor
from vllm_omni.diffusion.models.lingbot_vla_v2.qwen3vl_in_vla import Qwen3VLModel

from .conftest import observation, tiny_backbone, tiny_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CAMERA_KEYS = ("observation.images.top", "observation.images.left", "observation.images.right")
CAMERA_FRAMES = {
    key: np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    for index, key in enumerate(CAMERA_KEYS)
}


def make_camera_config(camera_keys: tuple[str, ...]) -> LingBotVLAV2Config:
    features = {"observation.state": {"type": "STATE", "shape": [7]}}
    features.update({key: {"type": "VISUAL", "shape": [3, 32, 32]} for key in camera_keys})
    return tiny_config(input_features=features)


class RecordingRopeIndex:
    """Forwards to Qwen3-VL get_rope_index and records the image tokens, grids and mask it pairs."""

    def __init__(self, backbone: Qwen3VLModel) -> None:
        self.backbone = backbone
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw: None,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((mm_token_type_ids == 1, image_grid_thw, attention_mask.bool()))
        return Qwen3VLModel.get_rope_index(
            self.backbone,
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )


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


@pytest.mark.parametrize("present_keys", [CAMERA_KEYS[:1], CAMERA_KEYS[::2], CAMERA_KEYS[1:]])
@torch.no_grad()
def test_missing_camera_takes_no_rope_grid_and_matches_a_policy_without_it(
    model: LingBotVLAV2ForActionPrediction,
    assets: Path,
    monkeypatch: pytest.MonkeyPatch,
    present_keys: tuple[str, ...],
) -> None:
    partial = (
        LingBotVLAV2ForActionPrediction(make_camera_config(CAMERA_KEYS), vlm_config=tiny_backbone()).float().eval()
    )
    reference = (
        LingBotVLAV2ForActionPrediction(make_camera_config(present_keys), vlm_config=tiny_backbone()).float().eval()
    )
    partial.load_state_dict(model.state_dict())
    reference.load_state_dict(model.state_dict())
    backbone = partial.qwenvl_with_expert.qwenvl.model
    merge_size = backbone.visual.spatial_merge_size
    rope_index = RecordingRopeIndex(backbone)
    monkeypatch.setattr(backbone, "get_rope_index", rope_index)
    obs = observation()
    obs.pop("observation.images.top")
    obs.update({key: CAMERA_FRAMES[key] for key in present_keys})
    partial_inputs = LingBotVLAV2Processor(partial.config, assets).build_model_inputs(obs)
    reference_inputs = LingBotVLAV2Processor(reference.config, assets).build_model_inputs(obs)
    noise = torch.randn(1, 4, 55, generator=torch.Generator().manual_seed(0))

    actions = partial.sample_actions(**partial_inputs, noise=noise)
    expected = reference.sample_actions(**reference_inputs, noise=noise)

    ((image_tokens, image_grid_thw, attention_mask),) = rope_index.calls
    assert len(image_grid_thw) == len(present_keys), "one MRoPE grid per present camera"
    assert image_tokens.sum() == (image_grid_thw.prod(-1) // merge_size**2).sum(), "image tokens match the grids"
    assert attention_mask[image_tokens].all(), "every MRoPE image token is attended"
    torch.testing.assert_close(actions, expected, msg="numerically equivalent to a policy without the camera")


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
    [
        {"rtc_config": {}},
        {"use_relative_actions": True},
        {"use_cudagraph_prefix": True},
        {"dtype": "bfloat16"},
        {"qwen3vl_use_vision_boundaries": False},
    ],
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
