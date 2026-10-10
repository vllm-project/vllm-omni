# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Duration limits apply before encoding and to transported encoder payloads."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import (
    prepare_encoder_inputs,
    resolve_minimax_h3_shape,
)
from vllm_omni.model_executor.models.minimax_h3.long_video import validate_encoded_frame_limit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("mode,limit", [("full", 30), ("continuation", 300), (None, 300)])
@pytest.mark.parametrize("input_kind", ["duration", "duration_seconds", "target", "num_frames"])
def test_long_video_limits_cover_all_duration_inputs(mode, limit, input_kind):
    extra: dict[str, Any] = {"long_video": True}
    if mode is not None:
        extra["long_video_mode"] = mode
    sampling = OmniDiffusionSamplingParams(width=64, height=64, fps=24, extra_args=extra)

    def set_duration(seconds):
        if input_kind == "target":
            extra["target"] = {"duration_seconds": seconds}
        elif input_kind == "num_frames":
            sampling.num_frames = seconds * 24
        else:
            extra[input_kind] = seconds

    set_duration(limit)
    _, _, frames, _, _ = resolve_minimax_h3_shape("ref2va", sampling, None)
    assert frames >= limit * 24
    validate_encoded_frame_limit(extra, "ref2va", frames)
    with pytest.raises(OmniClientError, match="request limit"):
        validate_encoded_frame_limit(extra, "ref2va", frames + 17)
    set_duration(limit + 1)
    with pytest.raises(OmniClientError, match="duration"):
        resolve_minimax_h3_shape("ref2va", sampling, None)


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_full_mode_rejects_300_seconds_and_accepts_comparison(task):
    sampling = OmniDiffusionSamplingParams(
        width=64,
        height=64,
        fps=24,
        extra_args={"long_video": True, "long_video_mode": "full", "duration": 300, "aspect_ratio": "1:1"},
    )
    image = Image.new("RGB", (64, 64)) if task == "fl2va" else None
    with pytest.raises(OmniClientError, match="30"):
        resolve_minimax_h3_shape(task, sampling, image)
    sampling.extra_args["duration"] = 20
    assert resolve_minimax_h3_shape(task, sampling, image)[2] == 481


@pytest.mark.parametrize("mode", ["full", "continuation"])
def test_mode_alone_does_not_bypass_long_video_opt_in(mode):
    sampling = OmniDiffusionSamplingParams(
        width=64,
        height=64,
        extra_args={"long_video_mode": mode, "duration": 20},
    )
    with pytest.raises(OmniClientError, match="15"):
        resolve_minimax_h3_shape("ref2va", sampling, None)


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_continuation_edit_rejected_before_loading_source(kind):
    sampling = OmniDiffusionSamplingParams(
        width=64,
        height=64,
        extra_args={"long_video": True, "task": "ref2va", "duration": 20},
    )
    with pytest.raises(OmniClientError, match="continuation does not support latent-mask editing"):
        prepare_encoder_inputs(
            {
                "prompt": "test",
                "multi_modal_data": {
                    "image": Image.new("RGB", (64, 64)),
                    f"source_{kind}": "must-not-be-loaded",
                    f"{kind}_noise_mask": 0.5,
                },
            },
            sampling,
        )


def test_locked_audio_rejects_audio_edit_before_loading_source():
    sampling = OmniDiffusionSamplingParams(
        width=64,
        height=64,
        extra_args={"task": "t2va", "aspect_ratio": "1:1", "audio_mode": "lock_source"},
    )
    with pytest.raises(OmniClientError, match="cannot be combined"):
        prepare_encoder_inputs(
            {
                "prompt": "test",
                "multi_modal_data": {
                    "audio": (torch.zeros(8000), 8000),
                    "source_audio": "must-not-be-loaded",
                    "audio_noise_mask": 0.5,
                },
            },
            sampling,
        )


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_external_encoder_cannot_bypass_continuation_edit_validation(kind):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(step_execution=False)
    conditioning = SimpleNamespace(
        task="ref2va",
        num_frames=481,
        video_edit_clean_rows=torch.zeros(1) if kind == "video" else None,
        audio_edit_clean_rows=torch.zeros(1) if kind == "audio" else None,
    )
    sampling = OmniDiffusionSamplingParams(extra_args={"long_video": True})
    with pytest.raises(OmniClientError, match="continuation does not support latent-mask editing"):
        pipeline._prepare_encoder_conditioning_inputs(conditioning, sampling)


def test_external_encoder_cannot_bypass_full_duration_limit():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(step_execution=False)
    sampling = OmniDiffusionSamplingParams(extra_args={"long_video": True, "long_video_mode": "full"})
    with pytest.raises(OmniClientError, match="30-second request limit"):
        pipeline._prepare_encoder_conditioning_inputs(SimpleNamespace(task="ref2va", num_frames=7213), sampling)
