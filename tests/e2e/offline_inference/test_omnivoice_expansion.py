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
import soundfile as sf
from vllm.multimodal.media.audio import load_audio

from tests.helpers.mark import hardware_test
from tests.helpers.media import convert_audio_file_to_text, get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "k2-fsa/OmniVoice"
DEPLOY_CONFIG = get_deploy_config_path("omnivoice.yaml")
_OMNIVOICE_REF_AUDIO_SEED = 102

# OmniRunner tuple: model, deploy config path, extra Omni kwargs.
_OMNI_RUNNER_PARAM = (
    MODEL,
    DEPLOY_CONFIG,
    {
        "trust_remote_code": True,
        "log_stats": True,
    },
)

pytestmark = [
    pytest.mark.slow,
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
    ro = final_output
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


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_omnivoice_ref_audio_without_ref_text(omni_runner: OmniRunner, tmp_path) -> None:
    """Reference audio without a transcript is transcribed before cloning."""
    ref_audio_path = get_asset_path("qwen3_tts/clone_2.wav")
    audio_signal, sample_rate = load_audio(str(ref_audio_path), sr=None)
    prompts = {
        "prompt": "hello",
        "multi_modal_data": {
            "audio": (audio_signal.astype(np.float32), sample_rate),
        },
        "mm_processor_kwargs": {"sample_rate": sample_rate},
    }

    def generate_audio():
        sampling_params_list = [OmniDiffusionSamplingParams(extra_args={"seed": _OMNIVOICE_REF_AUDIO_SEED})]
        outputs = list(omni_runner.omni.generate(prompts, sampling_params_list=sampling_params_list))
        assert outputs, "No outputs generated"
        final_output = outputs[-1]
        ro = final_output.request_output
        assert ro is not None, "No request_output"
        mm = getattr(ro, "multimodal_output", None)
        if not mm and ro.outputs:
            mm = getattr(ro.outputs[0], "multimodal_output", None)
        assert mm is not None, "No multimodal_output"

        audio = mm.get("audio")
        assert audio is not None, "No audio output"
        audio_np = audio if isinstance(audio, np.ndarray) else audio.cpu().numpy().squeeze()
        return np.asarray(audio_np).squeeze(), int(mm.get("sr", 24000))

    audio_np, output_sample_rate = generate_audio()
    assert audio_np.size > 1, "Audio output is empty"
    assert np.isfinite(audio_np).all(), "Audio contains non-finite samples"
    assert np.unique(audio_np).size > 1, "Audio has no sample variation"
    rms = np.sqrt(np.mean(audio_np**2))
    assert rms > 0.01, f"Audio RMS too low ({rms:.4f}), likely white noise or silence"
    assert output_sample_rate == 24000

    repeated_audio, repeated_sample_rate = generate_audio()
    assert repeated_sample_rate == output_sample_rate
    assert repeated_audio.shape == audio_np.shape
    np.testing.assert_allclose(repeated_audio, audio_np, rtol=1e-5, atol=1e-4)

    output_path = tmp_path / "omnivoice_ref_audio_only.wav"
    sf.write(str(output_path), audio_np, output_sample_rate)
    transcript = convert_audio_file_to_text(str(output_path), language="en").lower()
    assert "hello" in transcript, f"Generated audio transcript does not contain 'hello': {transcript!r}"
