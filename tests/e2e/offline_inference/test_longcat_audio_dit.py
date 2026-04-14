# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.utils import hardware_test
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

MODEL_PATH = os.environ.get(
    "LONGCAT_AUDIO_DIT_MODEL_PATH",
    "meituan-longcat/LongCat-AudioDiT-1B",
)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
def test_longcat_audio_dit_text_to_audio():
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model weights not found at {MODEL_PATH}")

    audio_end_in_s = 2.0
    sample_rate = 24000
    latent_hop = 2048
    expected_samples = int(audio_end_in_s * sample_rate // latent_hop) * latent_hop

    omni = Omni(model=MODEL_PATH)
    try:
        outputs = omni.generate(
            prompts={
                "prompt": "A calm ocean wave ambience with soft wind in the background.",
                "negative_prompt": "distorted, clipping, noisy",
            },
            sampling_params_list=OmniDiffusionSamplingParams(
                num_inference_steps=2,
                guidance_scale=4.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
                num_outputs_per_prompt=1,
                extra_args={
                    "audio_end_in_s": audio_end_in_s,
                },
            ),
        )

        assert outputs is not None
        first_output = outputs[0]
        assert hasattr(first_output, "request_output") and first_output.request_output

        req_out = first_output.request_output
        assert isinstance(req_out, OmniRequestOutput)
        assert req_out.final_output_type == "audio"
        assert req_out.multimodal_output is not None

        audio = req_out.multimodal_output.get("audio")
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 2
        assert audio.shape[0] == 1
        assert audio.shape[1] == expected_samples
    finally:
        omni.close()
