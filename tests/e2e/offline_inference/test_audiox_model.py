# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

# Prefer a tiny/random checkpoint for CI.
# Override in CI if needed: AUDIOX_TEST_MODEL=<model-or-local-path>
models = [os.environ.get("AUDIOX_TEST_MODEL", "zhangj1an/audiox_random")]


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_audiox_model(model_name: str):
    m = Omni(model=model_name, model_class_name="AudioXPipeline")

    # Keep runtime short for CI.
    seconds_total = 2.0
    # AudioXPipeline always emits 44.1 kHz stereo (advertised via class-level
    # ``audio_sample_rate``); the trimmed output should match this rate.
    sample_rate = 44100

    outputs = m.generate(
        prompts={"prompt": "A dog barking in a quiet park."},
        sampling_params_list=OmniDiffusionSamplingParams(
            num_inference_steps=4,
            guidance_scale=6.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            num_outputs_per_prompt=1,
            extra_args={
                "audiox_task": "t2a",
                "seconds_start": 0.0,
                "seconds_total": seconds_total,
            },
        ),
    )

    assert outputs is not None
    first_output = outputs[0]
    assert first_output.final_output_type == "audio"
    assert hasattr(first_output, "request_output") and first_output.request_output

    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput)
    assert req_out.final_output_type == "audio"
    assert hasattr(req_out, "multimodal_output") and req_out.multimodal_output

    audio = req_out.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    # audio shape: (batch, channels, samples)
    assert audio.ndim == 3
    assert audio.shape[0] == 1
    assert audio.shape[1] == 2
    assert audio.shape[2] > 0
    expected_samples = int(seconds_total * sample_rate)
    assert abs(audio.shape[2] - expected_samples) <= 2 * 1024
