# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

MODEL = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
]


@hardware_test(res={"cuda": "L4"})
def test_moss_soundeffect_v2_model() -> None:
    # Keep runtime short for CI.
    seconds_total = 5.0
    # MOSS-SoundEffect-v2.0 emits 48 kHz mono audio.
    sample_rate = 48000

    with OmniRunner(MODEL) as runner:
        outputs = runner.omni.generate(
            prompts={"prompt": "A dog barking in a quiet park."},
            sampling_params_list=OmniDiffusionSamplingParams(
                num_inference_steps=10,
                guidance_scale=6.0,
                seed=42,
                extra_args={
                    "sigma_shift": 7.0,
                    "audio_end_in_s": seconds_total,
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

        audio = req_out.multimodal_output.get("audio")
        assert isinstance(audio, np.ndarray)
        # audio shape: (batch, channels, samples)
        assert audio.ndim == 3
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        assert audio.shape[2] > 0
        expected_samples = int(seconds_total * sample_rate)
        assert abs(audio.shape[2] - expected_samples) <= 2 * 1024
