# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E test for VibeVoice-1.5B eager offline TTS (native HF checkpoint).

Calls Omni directly because the existing TTS client helpers do not expose the
raw completion token IDs needed to verify same-step audio at the length cap.
"""

import os

import pytest
import soundfile as sf
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.vibevoice.prompt import render_vibevoice_prompt

MODEL = os.environ.get("VIBEVOICE_MODEL", "vibevoice/VibeVoice-1.5B-hf")
REFERENCE = os.path.join(os.path.dirname(__file__), "../../assets/qwen3_tts/clone_2.wav")
AUDIO_TOKEN_ID = 151654
SAMPLES_PER_TOKEN = 3200

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [(MODEL, get_deploy_config_path("vibevoice.yaml"))], indirect=True),
]


def _audio(completion) -> torch.Tensor:
    mm = completion.multimodal_output
    # The output processor maps the model's internal model_outputs key to audio.
    audio = mm.get("audio")
    assert audio is not None, f"No public audio field, got {list(mm)}"
    sample_rate = mm.get("sr")
    assert sample_rate is not None, "Missing output sample rate"
    sample_rate = torch.as_tensor(sample_rate)
    assert sample_rate.numel() == 1 and sample_rate.item() == 24000
    if isinstance(audio, list):
        assert audio, "Empty audio chunk list"
        audio = torch.cat([torch.as_tensor(x).reshape(-1) for x in audio], dim=0)
    assert isinstance(audio, torch.Tensor)
    assert audio.dtype == torch.float32 and audio.ndim == 1 and audio.numel() > 0
    assert torch.isfinite(audio).all()
    return audio


def _requests(texts):
    waveform, sample_rate = sf.read(REFERENCE, dtype="float32")
    return [
        {"prompt": render_vibevoice_prompt(text), "multi_modal_data": {"audio": (waveform, sample_rate)}}
        for text in texts
    ]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_vibevoice_ragged_batch_natural_stop(omni_runner: OmniRunner) -> None:
    """Four ragged requests decode concurrently; each stops naturally with one chunk per audio token."""
    texts = [
        "Hello, this is a short test.",
        "The quick brown fox jumps over the lazy dog, and then it runs far away into the quiet forest.",
        "Speech synthesis converts written text into natural sounding audio.",
        "Good morning everyone.",
    ]
    outputs = omni_runner.omni.generate(_requests(texts), [SamplingParams(max_tokens=512)])
    assert len(outputs) == len(texts)
    for output in outputs:
        completion = output.outputs[0]
        audio = _audio(completion)
        assert completion.finish_reason == "stop"
        assert audio.numel() == completion.token_ids.count(AUDIO_TOKEN_ID) * SAMPLES_PER_TOKEN
        assert 0.5 < audio.numel() / 24000 < 30.0
        assert audio.abs().max() > 1e-3


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_vibevoice_length_cap_keeps_last_chunk(omni_runner: OmniRunner) -> None:
    """An audio token sampled at max_tokens still yields its waveform chunk."""
    (output,) = omni_runner.omni.generate(_requests(["Hello, this is a short test."]), [SamplingParams(max_tokens=5)])
    completion = output.outputs[0]
    assert completion.finish_reason == "length"
    assert len(completion.token_ids) == 5
    assert completion.token_ids[-1] == AUDIO_TOKEN_ID, (
        "The capped request must end on an AUDIO token to exercise terminal audio emission."
    )
    assert _audio(completion).numel() == completion.token_ids.count(AUDIO_TOKEN_ID) * SAMPLES_PER_TOKEN > 0


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_vibevoice_no_reference_natural_stop(omni_runner: OmniRunner) -> None:
    """The native no-voice prompt generates complete audio without reference conditioning."""
    prompt = {"prompt": render_vibevoice_prompt("Hello, this is a short test.", reference=False)}
    (output,) = omni_runner.omni.generate([prompt], [SamplingParams(max_tokens=512)])
    completion = output.outputs[0]
    audio = _audio(completion)
    assert completion.finish_reason == "stop"
    assert audio.numel() == completion.token_ids.count(AUDIO_TOKEN_ID) * SAMPLES_PER_TOKEN
    assert 0.5 < audio.numel() / 24000 < 30.0
