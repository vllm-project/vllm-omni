# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.stable_audio_3.pipeline_stable_audio_3 import StableAudio3Pipeline
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "window",
    [
        {"audio_start_in_s": 5.0, "audio_end_in_s": 5.0},  # zero length
        {"audio_start_in_s": 5.0, "audio_end_in_s": 1.0},  # end before start
        {"audio_start_in_s": 0.0, "audio_end_in_s": 1e-6},  # shorter than one sample
        {"audio_start_in_s": 0.0, "audio_end_in_s": math.nan},
        {"audio_start_in_s": 0.0, "audio_end_in_s": math.inf},
        {"audio_start_in_s": 0.0, "audio_end_in_s": 381.0},
        {"audio_start_in_s": -1.0},
        {"audio_start_in_s": math.nan},
    ],
)
def test_forward_rejects_invalid_audio_window(window):
    # Validation runs before any model component is touched, so a bare stub is enough;
    # if it is skipped, forward reaches the stub's missing attributes and raises AttributeError.
    pipeline = SimpleNamespace(max_audio_seconds=StableAudio3Pipeline.max_audio_seconds, sample_rate=44100)
    req = SimpleNamespace(sampling_params=OmniDiffusionSamplingParams(extra_args=window), prompts=["rain"])
    with pytest.raises(ValueError):
        StableAudio3Pipeline.forward(pipeline, req)
