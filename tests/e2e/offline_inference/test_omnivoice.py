# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline tests for OmniVoice TTS model with text input and audio output.

Uses GPUGenerationWorker for both stages (iterative unmasking + DAC decoder).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import OmniRunner
from tests.utils import hardware_test

MODEL = "k2-fsa/OmniVoice"


def get_stage_config():
    return str(
        Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / "omnivoice.yaml"
    )


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_omnivoice_text_to_audio() -> None:
    """
    Test OmniVoice text-to-audio generation via offline Omni runner.
    Deploy Setting: omnivoice.yaml (enforce_eager=true)
    Input Modal: text
    Output Modal: audio
    """
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    with OmniRunner(
        MODEL,
        stage_configs_path=get_stage_config(),
        trust_remote_code=True,
        log_stats=True,
    ) as runner:
        prompts = {"prompt": "Hello, this is a test for text to audio."}

        sampling_params_list = [OmniDiffusionSamplingParams()]

        outputs = list(runner.omni.generate(prompts, sampling_params_list=sampling_params_list))

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
