# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy
from pathlib import Path

import pytest
import yaml

from vllm_omni.diffusion.models.physis.common import validate_physis_request
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("deployment", "profile", "frames"),
    [
        ("wan", "wan21_1p3b", 81),
        ("wan", "wan21_14b", 81),
        ("wan", "wan22_5b", 121),
        ("ltx", None, 121),
        ("h3", None, 162),
    ],
)
@pytest.mark.parametrize("frame_input", ["omitted", "explicit", "nested", "invalid", "nested_invalid"])
def test_deployment_video_request_frame_defaults(deployment, profile, frames, frame_input):
    path = Path(__file__).resolve().parents[4] / "vllm_omni" / "deploy" / f"physis_{deployment}.yaml"
    stage = yaml.safe_load(path.read_text(encoding="utf-8"))["stages"][0]
    if profile is not None:
        stage["model_config"]["physis_profile"] = profile
    defaults = OmniDiffusionSamplingParams(**stage.get("default_sampling_params", {}))
    original_frames = defaults.num_frames
    params = copy.deepcopy(defaults)
    fields = {}
    if frame_input != "omitted":
        fields["num_frames"] = 1 if "invalid" in frame_input else frames
        if frame_input.startswith("nested"):
            fields = {"video_params": fields}
    request = VideoGenerationRequest(prompt="A corgi running on the beach.", **fields)
    video_params = request.resolve_video_params()
    # The serving layer copies stage defaults and only overwrites provided frames.
    if video_params.num_frames is not None:
        params.num_frames = video_params.num_frames
    if "invalid" in frame_input:
        with pytest.raises(ValueError, match=rf"num_frames={frames}; received 1"):
            validate_physis_request(params, num_frames=frames)
    else:
        validate_physis_request(params, num_frames=frames)
        assert params.num_frames == frames
    assert defaults.num_frames == original_frames
