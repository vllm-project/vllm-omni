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

from tests.helpers.mark import hardware_test
from tests.helpers.media import test_asset_path
from tests.helpers.runtime import OmniRunner

MODEL = os.environ.get("VLLM_OMNI_TEST_OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

try:
    from transformers import HiggsAudioV2TokenizerModel  # noqa: F401

    _HAS_VOICE_CLONE = True
except ImportError:
    _HAS_VOICE_CLONE = False


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


@pytest.mark.skipif(not _HAS_VOICE_CLONE, reason="Voice cloning requires transformers>=5.3.0")
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_omnivoice_offline_voice_clone() -> None:
    """Regression test for OmniTextPrompt parsing in the offline diffusion path.

    Voice-cloning fields arrive under ``multi_modal_data["audio"]`` and
    ``mm_processor_kwargs["ref_text"]`` when the prompt is built by
    ``Omni.generate`` (offline ``end2end.py`` shape) rather than as the
    top-level ``ref_audio`` / ``ref_text`` keys ``serving_speech.py``
    populates for ``/v1/audio/speech``. Earlier the diffusion pipeline only
    read top-level keys and otherwise fell through to ``str(prompt)``,
    yielding ~30 s of audio reciting the stringified dict for a sub-3 s
    utterance (#3392). This test asserts the generated audio duration is
    in the input ballpark — the regression is caught by length alone, no
    ASR needed.

    The reference clip is the vendored ``tests/assets/qwen3_tts/clone_2.wav``
    (real human speech) with the matching transcript so the in-context
    cloning has clean signal — synthetic ref_audio with mismatched
    ref_text triggers a separate model-level failure mode (the model
    echoes ref_text fragments) that's unrelated to this prompt-parsing
    regression.
    """
    import soundfile as sf

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    ref_path = test_asset_path("qwen3_tts/clone_2.wav")
    ref_audio_np, sr = sf.read(str(ref_path), dtype="float32")
    if ref_audio_np.ndim == 2:
        ref_audio_np = ref_audio_np.mean(axis=1)
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    text = "Hello, this is a short test."

    with OmniRunner(
        MODEL,
        stage_configs_path=get_stage_config(),
        trust_remote_code=True,
        log_stats=True,
    ) as runner:
        prompts = {
            "prompt": text,
            "multi_modal_data": {"audio": (ref_audio_np, sr)},
            "mm_processor_kwargs": {"ref_text": ref_text, "sample_rate": sr},
        }
        sampling_params_list = [OmniDiffusionSamplingParams()]
        outputs = list(runner.omni.generate(prompts, sampling_params_list=sampling_params_list))

    assert outputs, "No outputs generated"
    final_output = outputs[-1]
    ro = final_output.request_output
    assert ro is not None, "No request_output"

    mm = getattr(ro, "multimodal_output", None)
    if not mm and ro.outputs:
        mm = getattr(ro.outputs[0], "multimodal_output", None)
    assert mm is not None and "audio" in mm, "No audio in multimodal_output"

    audio = mm["audio"]
    audio_np = audio if isinstance(audio, np.ndarray) else audio.cpu().numpy().squeeze()
    duration_s = len(audio_np) / 24000
    print(f"Voice clone audio: {duration_s:.2f}s for input {text!r}")
    # Pre-#3392 the dict was stringified and recited end-to-end (~30 s for
    # a similar prompt). The 10 s upper bound rides through OmniVoice's
    # stochastic length while still catching the regression.
    assert duration_s < 10.0, (
        f"Audio is {duration_s:.2f}s for a short prompt — pipeline is likely "
        f"reciting the stringified prompt dict instead of just the input."
    )
