# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline tests for OmniVoice TTS model with text input and audio output.

Uses GPUGenerationWorker for both stages (iterative unmasking + DAC decoder).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "k2-fsa/OmniVoice"
STAGE_CONFIG = get_deploy_config_path("omnivoice.yaml")

# (model, stage_config_path, extra_omni_kwargs) — see ``omni_runner`` in tests.helpers.fixtures.runtime
_OMNI_RUNNER_PARAM = (
    MODEL,
    STAGE_CONFIG,
    {
        "trust_remote_code": True,
        "log_stats": True,
    },
)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_omnivoice_text_to_audio(omni_runner: OmniRunner) -> None:
    """
    Test OmniVoice text-to-audio generation via offline Omni runner.
    Deploy Setting: omnivoice.yaml (enforce_eager=true)
    Input Modal: text
    Output Modal: audio
    """
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    prompts = {"prompt": "Hello, this is a test for text to audio."}
    sampling_params_list = [OmniDiffusionSamplingParams()]

    outputs = list(omni_runner.omni.generate(prompts, sampling_params_list=sampling_params_list))

    assert len(outputs) > 0, "No outputs generated"

    # Check final output has audio
    final_output = outputs[-1]
    ro = final_output.request_output
    assert ro is not None, "No request_output"

    mm = getattr(ro, "multimodal_output", None)
    if not mm and ro.outputs:
        mm = getattr(ro.outputs[0], "multimodal_output", None)

    assert mm is not None, "No multimodal_output"
    assert "audio" in mm, f"No 'audio' key in multimodal_output: {mm.keys()}"

    audio = mm["audio"]
    if isinstance(audio, np.ndarray):
        audio_np = audio
    else:
        audio_np = audio.cpu().numpy().squeeze()

    assert audio_np.size > 0, "Audio output is empty"
    rms = np.sqrt(np.mean(audio_np**2))
    assert rms > 0.01, f"Audio RMS too low ({rms:.4f}), likely silence"

    print(f"Generated audio: {len(audio_np) / 24000:.2f}s, rms={rms:.4f}")
