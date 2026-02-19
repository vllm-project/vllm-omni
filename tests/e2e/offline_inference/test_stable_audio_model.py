import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.utils import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

# Use random weights model for CI testing (small, no authentication required)
models = ["linyueqian/stable_audio_random"]


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("model_name", models)
def test_stable_audio_model(model_name: str):
    m = Omni(model=model_name)

    # Use minimal settings for testing
    # Generate a short 2-second audio clip with minimal inference steps
    audio_start_in_s = 0.0
    audio_end_in_s = 2.0  # Short duration for fast testing
    sample_rate = 44100  # Stable Audio uses 44100 Hz

    outputs = m.generate(
        prompts={
            "prompt": "The sound of a dog barking",
            "negative_prompt": "Low quality.",
        },
        sampling_params_list=OmniDiffusionSamplingParams(
            num_inference_steps=4,  # Minimal steps for speed
            guidance_scale=7.0,
            generator=torch.Generator("cuda").manual_seed(42),
            num_outputs_per_prompt=1,
            extra_args={
                "audio_start_in_s": audio_start_in_s,
                "audio_end_in_s": audio_end_in_s,
            },
        ),
    )

    # Extract audio from OmniRequestOutput
    assert outputs is not None
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    assert hasattr(first_output, "request_output") and first_output.request_output

    req_out = first_output.request_output[0]
    assert isinstance(req_out, OmniRequestOutput)
    assert req_out.final_output_type == "audio"
    assert hasattr(req_out, "multimodal_output") and req_out.multimodal_output
    audio = req_out.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    # audio shape: (batch, channels, samples)
    # For stable-audio-open-1.0: sample_rate=44100, so 2 seconds = 88200 samples
    assert audio.ndim == 3
    assert audio.shape[0] == 1  # batch size
    assert audio.shape[1] == 2  # stereo channels
    expected_samples = int((audio_end_in_s - audio_start_in_s) * sample_rate)
    assert audio.shape[2] == expected_samples  # 88200 samples for 2 seconds
