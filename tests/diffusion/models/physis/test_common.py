# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch
from diffusers.video_processor import VideoProcessor

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.models.physis.common import (
    build_physis_cell_plan,
    prepare_video_output,
    validate_physis_request,
)
from vllm_omni.diffusion.postprocess.device_reduction import reduce_video_to_uint8_frames
from vllm_omni.experimental.ar_diffusion.cell_schedule import CellSchedule, Ordering, build_cell_plan
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("frames", [81, 121, 162])
def test_fixed_recipe_accepts_verified_frame_counts_and_optional_defaults(frames):
    params = OmniDiffusionSamplingParams(num_frames=frames, seed=17)
    validate_physis_request(params, num_frames=frames)
    assert params.seed == 17


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("height", 512),
        ("width", 1024),
        ("num_frames", 1),
        ("num_inference_steps", 50),
        ("guidance_scale", 7.0),
        ("guidance_scale_2", 3.0),
        ("num_outputs_per_prompt", 2),
        ("do_classifier_free_guidance", True),
        ("sigmas", [1.0, 0.0]),
    ],
)
def test_rejects_explicit_unsupported_sampling_settings(name, value):
    params = OmniDiffusionSamplingParams(num_frames=81)
    setattr(params, name, value)
    with pytest.raises(ValueError):
        validate_physis_request(params, num_frames=81)


def test_ltx_model_frame_rate_is_separate_from_output_playback_rate():
    params = OmniDiffusionSamplingParams(num_frames=121, fps=16.0, frame_rate=24.0)
    validate_physis_request(params, num_frames=121, frame_rate=24.0)
    params.frame_rate = 16.0
    with pytest.raises(ValueError, match="frame_rate=24"):
        validate_physis_request(params, num_frames=121, frame_rate=24.0)


def test_optional_fixed_settings_can_be_unspecified():
    params = OmniDiffusionSamplingParams(num_frames=None)
    validate_physis_request(params, num_frames=81)


def test_explicit_recipe_is_accepted():
    params = OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=4,
        guidance_scale=1.0,
        num_outputs_per_prompt=1,
        seed=0,
    )
    validate_physis_request(params, num_frames=81)


@pytest.mark.parametrize("chunks", [19, 29, 14, 48])
@pytest.mark.parametrize("groups", [1, 2])
def test_shared_plan_preserves_model_schedule_and_task_order(chunks, groups):
    plan = build_physis_cell_plan(
        chunks=chunks, step_replicas=4, layer_groups=groups, blocks=8, blocks_per_rank=8 // groups
    )
    expected = build_cell_plan(
        CellSchedule(
            chunks=chunks - 1,
            num_denoise_steps=4,
            step_replicas=4,
            layer_groups=groups,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=18,
            blocks=8,
            blocks_per_rank=8 // groups,
        )
    )
    assert plan.schedule == expected.schedule
    assert plan.num_slots == expected.num_slots
    assert all(
        plan.task(slot, rank) == expected.task(slot, rank)
        for slot in range(plan.num_slots)
        for rank in range(plan.world)
    )


@pytest.mark.parametrize("output_type", ["np", "latent"])
def test_prepare_video_output_leaves_legacy_output_unchanged(output_type):
    output = DiffusionOutput(output=torch.zeros(1, 3, 1, 2, 2))
    assert prepare_video_output(output, output_type=output_type) is output
    assert output.media is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("output_type", ["np", "latent"])
def test_video_output_preserves_legacy_values(dtype, output_type):
    video = torch.linspace(-2, 2, 257, dtype=dtype).reshape(1, 1, 1, 1, 257).repeat(1, 3, 2, 1, 1)
    output = DiffusionOutput(
        media=DiffusionMediaOutput(
            video=VideoMediaOutput(
                tensor=video,
                spec=VideoTensorSpec(
                    layout=VideoTensorLayout.BCTHW,
                    encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                    value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
                ),
            )
        )
    )
    prepared = prepare_video_output(output, output_type=output_type)
    if output_type == "latent":
        assert prepared.media is None
        assert prepared.output is video
        return
    typed = prepared.media.video
    assert typed.tensor is video
    assert typed.spec.preserve_input_dtype
    actual = reduce_video_to_uint8_frames(
        typed.tensor, do_denormalize=True, preserve_input_dtype=typed.spec.preserve_input_dtype
    )
    floats = VideoProcessor(vae_scale_factor=8).postprocess_video(video, output_type="np")
    expected = np.rint(np.clip(floats, 0, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(actual.numpy(), expected)
