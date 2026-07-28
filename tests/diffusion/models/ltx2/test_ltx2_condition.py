# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LTX-2 / LTX-2.3 multi-anchor frame conditioning (FLF2V / FMLF)."""

import pytest
import torch

from vllm_omni.diffusion.models.ltx2.ltx2_components import LTX2_COMPONENT_PROFILE
from vllm_omni.diffusion.models.ltx2.ltx2_condition import (
    LTX2VideoCondition,
    LTXMultiAnchorConditioningMixin,
    apply_first_frame_conditioning,
    preprocess_conditions,
)
from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2ConditionPipeline, LTX2Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_condition_pipeline_composes_multi_anchor_mixin_over_ltx2():
    assert issubclass(LTX2ConditionPipeline, LTXMultiAnchorConditioningMixin)
    assert issubclass(LTX2ConditionPipeline, LTX2Pipeline)
    assert issubclass(LTX2ConditionPipeline, LTXRuntime)


def test_condition_pipeline_inherits_runtime_contract():
    assert LTX2ConditionPipeline.pipeline_kind == "one_stage"
    assert LTX2ConditionPipeline.component_profile is LTX2_COMPONENT_PROFILE
    # Anchors resolve from the first request only -> batching disabled.
    assert LTX2ConditionPipeline.supports_request_batch is False
    assert LTX2ConditionPipeline.support_image_input is True
    assert LTX2ConditionPipeline.unified_text_image_entry is True


def test_condition_pipeline_overrides_conditioning_hooks():
    for hook in (
        "_prepare_video_latents_stage",
        "_step_video_latents_i2v",
        "_denoise_timestep_kwargs",
        "_prepare_denoise_context_for_cfg",
        "_video_guidance_model_sigma",
        "_unpack_and_denormalize_stage",
        "forward",
    ):
        assert getattr(LTX2ConditionPipeline, hook) is getattr(LTXMultiAnchorConditioningMixin, hook)


def test_condition_pipeline_registered():
    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, _DIFFUSION_POST_PROCESS_FUNCS

    assert _DIFFUSION_MODELS["LTX2ConditionPipeline"] == ("ltx2", "pipeline_ltx2", "LTX2ConditionPipeline")
    assert _DIFFUSION_POST_PROCESS_FUNCS["LTX2ConditionPipeline"] == "get_ltx2_post_process_func"


class _StubVideoProcessor:
    def preprocess_video(self, frames, height, width):
        return torch.zeros(1, 3, 1, height, width)


def _cond(index=0, strength=1.0):
    return LTX2VideoCondition(frames=object(), index=index, strength=strength)


def test_preprocess_conditions_rejects_empty():
    with pytest.raises(ValueError, match="at least one condition"):
        preprocess_conditions([], _StubVideoProcessor(), 64, 64, 8, device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_preprocess_conditions_rejects_out_of_range_strength(strength):
    with pytest.raises(ValueError, match="strength"):
        preprocess_conditions(
            [_cond(strength=strength)], _StubVideoProcessor(), 64, 64, 8, device="cpu", dtype=torch.float32
        )


def test_preprocess_conditions_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="outside the valid range"):
        preprocess_conditions([_cond(index=8)], _StubVideoProcessor(), 64, 64, 8, device="cpu", dtype=torch.float32)


def test_preprocess_conditions_resolves_negative_index():
    _, strengths, indices, pixel_frames = preprocess_conditions(
        [_cond(index=-1, strength=0.5)], _StubVideoProcessor(), 64, 64, 8, device="cpu", dtype=torch.float32
    )
    assert indices == [7]
    assert strengths == [0.5]
    assert pixel_frames == [1]


def test_apply_first_frame_conditioning_pins_frame_zero_and_skips_keyframes():
    latent_height, latent_width = 2, 2
    tokens_per_frame = latent_height * latent_width
    num_frames = 2
    latents = torch.zeros(1, num_frames * tokens_per_frame, 4)
    conditioning_mask = torch.zeros(1, num_frames * tokens_per_frame)
    frame0 = torch.full((1, tokens_per_frame, 4), 5.0)
    keyframe = torch.full((1, tokens_per_frame, 4), 9.0)

    out, mask, clean = apply_first_frame_conditioning(
        latents,
        conditioning_mask,
        [frame0, keyframe],
        [1.0, 1.0],
        [0, 1],  # index 0 applied in place; index 1 is a keyframe -> skipped here
        latent_height=latent_height,
        latent_width=latent_width,
    )

    assert torch.all(out[:, :tokens_per_frame] == 5.0)
    assert torch.all(mask[:, :tokens_per_frame] == 1.0)
    assert torch.all(clean[:, :tokens_per_frame] == 5.0)
    # The non-zero (keyframe) index is NOT overwritten in place; it is appended separately.
    assert torch.all(out[:, tokens_per_frame:] == 0.0)
    assert torch.all(mask[:, tokens_per_frame:] == 0.0)


def test_prepare_keyframe_coords_places_anchor_at_pixel_frame():
    pipe = object.__new__(LTX2ConditionPipeline)
    pipe.transformer_spatial_patch_size = 1
    pipe.transformer_temporal_patch_size = 1
    pipe.vae_spatial_compression_ratio = 32
    pipe.vae_temporal_compression_ratio = 8

    # latent_idx=2 -> pixel_frame_idx = (2 - 1) * 8 + 1 = 9
    coords = pipe._prepare_keyframe_coords(
        keyframe_latent_num_frames=1,
        keyframe_latent_height=1,
        keyframe_latent_width=1,
        pixel_frame_idx=9,
        num_pixel_frames=1,
        fps=24.0,
        device=torch.device("cpu"),
    )

    assert coords.shape[0] == 1
    assert coords.shape[1] == 3
    assert coords.shape[-1] == 2
    torch.testing.assert_close(coords[0, 0, 0, 0], torch.tensor(9.0 / 24.0))
    torch.testing.assert_close(coords[0, 0, 0, 1], torch.tensor(10.0 / 24.0))
